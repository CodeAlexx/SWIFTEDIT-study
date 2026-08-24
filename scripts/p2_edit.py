# P2 (RESEARCH): mask-rescaled EDIT on LTX video, training-free (SwiftEdit region-blend
# in latent space). invert source under src prompt -> re-denoise under EDIT prompt with
# per-step blending: inside mask follow edit (blue car), outside lock to source trajectory.
# Reuses cached T5 embeds. OOM-safe: 2B DiT, tiny res.
import os, time, math, inspect
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch, numpy as np
from PIL import Image
from diffusers import (LTXPipeline, LTXVideoTransformer3DModel,
                       AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler)

REPO="Lightricks/LTX-Video-0.9.1"; DEV,DT="cuda",torch.bfloat16
F,H,W,N = 25,256,256,48
CLAMP_RATE=3.0
OUTDIR="/home/alex/swiftedit-ltx-research/p2_out"; os.makedirs(OUTDIR,exist_ok=True)

def main():
    torch.manual_seed(0); t0=time.time()
    E=torch.load("/home/alex/swiftedit-ltx-research/p1_embeds.pt",map_location="cpu")
    src=E["src"].to(DEV,DT); edit=E["edit"].to(DEV,DT)
    smsk=E["src_mask"].to(DEV); emsk=E["edit_mask"].to(DEV); zero=torch.zeros_like(src)
    vae=AutoencoderKLLTXVideo.from_pretrained(REPO,subfolder="vae",torch_dtype=DT).to(DEV).eval()
    tr =LTXVideoTransformer3DModel.from_pretrained(REPO,subfolder="transformer",torch_dtype=DT).to(DEV).eval()
    sch=FlowMatchEulerDiscreteScheduler.from_pretrained(REPO,subfolder="scheduler")
    pipe=LTXPipeline(scheduler=sch,vae=vae,text_encoder=None,tokenizer=None,transformer=tr)
    pipe.set_progress_bar_config(disable=True)
    sig=inspect.signature(LTXVideoTransformer3DModel.forward)
    mean=vae.latents_mean.view(1,-1,1,1,1).to(DEV,DT); std=vae.latents_std.view(1,-1,1,1,1).to(DEV,DT)
    sf=getattr(vae.config,"scaling_factor",1.0)

    # source clip (same seed/params as P1 -> same red car)
    with torch.no_grad():
        srcvid=pipe(prompt_embeds=src,prompt_attention_mask=smsk,negative_prompt_embeds=zero,
                    negative_prompt_attention_mask=smsk,width=W,height=H,num_frames=F,
                    num_inference_steps=30,guidance_scale=3.0,output_type="pt").frames

    # capture DiT call args
    cap=[]; orig=tr.forward
    def spy(*a,**k):
        b=sig.bind(tr,*a,**k); b.apply_defaults(); cap.append(dict(b.arguments)); return orig(*a,**k)
    tr.forward=spy
    with torch.no_grad():
        pipe(prompt_embeds=src,prompt_attention_mask=smsk,negative_prompt_embeds=zero,
             negative_prompt_attention_mask=smsk,width=W,height=H,num_frames=F,
             num_inference_steps=N,guidance_scale=1.0,output_type="latent")
    tr.forward=orig
    STAT={kk:cap[0][kk] for kk in ['num_frames','height','width','rope_interpolation_scale',
          'video_coords','attention_kwargs'] if kk in cap[0]}
    TS=[c['timestep'] for c in cap]; sigmas=sch.sigmas.to(DEV)

    def vel(z,i,cond,cmsk,gs=1.0):
        with torch.no_grad():
            vc=tr(hidden_states=z,timestep=TS[i],encoder_hidden_states=cond,
                  encoder_attention_mask=cmsk,return_dict=False,**STAT)[0]
            if gs==1.0: return vc
            vu=tr(hidden_states=z,timestep=TS[i],encoder_hidden_states=zero,
                  encoder_attention_mask=cmsk,return_dict=False,**STAT)[0]
            return vu+gs*(vc-vu)
    # invert + source-recon stay at gs=1 (keeps the RF-ODE invertible / source preserved);
    # only the EDIT denoise uses CFG so "blue" actually takes.
    EDIT_GS=float(os.environ.get("P2_EDIT_GS","3.0"))
    def step_down(z,i,cond,cmsk,gs=1.0):
        ds=sigmas[i+1]-sigmas[i]; v=vel(z,i,cond,cmsk,gs)
        zp=z+ds*v; v2=vel(zp,min(i+1,N-1),cond,cmsk,gs); return z+ds*0.5*(v+v2)
    def invert(cond,cmsk):
        z=x0.clone()
        for i in reversed(range(N)):
            ds=sigmas[i]-sigmas[i+1]; v=vel(z,i,cond,cmsk,1.0)
            zp=z+ds*v; v2=vel(zp,max(i-1,0),cond,cmsk,1.0); z=z+ds*0.5*(v+v2)
        return z

    # encode source -> x0
    vin=(srcvid*2-1).permute(0,2,1,3,4).to(DEV,DT)
    with torch.no_grad(): lat=vae.encode(vin).latent_dist.sample()
    Fl,Hl,Wl=lat.shape[2],lat.shape[3],lat.shape[4]
    global x0; x0=pipe._pack_latents((lat-mean)*sf/std,1,1)

    # inversions + mask
    inv_src=invert(src,smsk); inv_edit=invert(edit,emsk)
    subed=(inv_src-inv_edit).abs().float().mean(dim=2)[0]
    mnorm=((subed-subed.min())/(subed.max()-subed.min()+1e-6))
    MPOW=float(os.environ.get("P2_MPOW","3.0"))            # sharpen: push background weight -> 0
    mg=(mnorm**MPOW).view(1,1,Fl,Hl,Wl)
    DIL=int(os.environ.get("P2_DILATE","0"))               # grow mask by DIL cells (cover whole car)
    for _ in range(DIL):
        mg=torch.nn.functional.max_pool3d(mg,kernel_size=3,stride=1,padding=1)
    mtok=mg.reshape(1,-1,1).to(DT)                         # [1,256,1] blend weight (1=edit,0=src)
    print(f"blend mask: frac>0.5={float(((mnorm**MPOW)>0.5).float().mean()):.3f}  rawmax={mnorm.max():.2f}  EDIT_GS={EDIT_GS} MPOW={MPOW}",flush=True)

    # masked SDEdit / blended diffusion: re-noise the car region and regenerate it under the
    # EDIT prompt; outside the mask, composite the source noised to each sigma level so it is
    # preserved. STRENGTH controls how much of the car is re-generated (higher = more edit).
    STRENGTH=float(os.environ.get("P2_STRENGTH","0.8"))
    eps=torch.randn_like(x0)
    k0=max(0,min(N-1,int((1-STRENGTH)*N)))
    def snoise(i): return (1-sigmas[i])*x0+sigmas[i]*eps    # source on the RF path at sigma[i]
    ze=snoise(k0)
    for i in range(k0,N):
        ze=mtok*ze+(1-mtok)*snoise(i)                       # outside mask = source path
        ze=step_down(ze,i,edit,emsk,EDIT_GS)
    ze=mtok*ze+(1-mtok)*x0                                  # final: outside = clean source
    x0_edit=ze
    print(f"  SDEdit strength={STRENGTH} start_step={k0}/{N} (sigma0={sigmas[k0]:.2f})",flush=True)

    # decode source & edit
    def decode(tok):
        ln=pipe._unpack_latents(tok,Fl,Hl,Wl,1,1); l=ln*std/sf+mean
        tz=torch.zeros(1,device=DEV,dtype=DT) if getattr(vae.config,"timestep_conditioning",False) else None
        with torch.no_grad(): return vae.decode(l,tz).sample
    dec_src=decode(x0); dec_edit=decode(x0_edit)
    editvid=((dec_edit.float()+1)/2).clamp(0,1).permute(0,2,1,3,4)  # [1,F,3,H,W]

    # metrics: how much changed inside vs outside mask (latent)
    mfull=mtok.float()
    inside=((x0_edit-x0).float().abs()*mfull).sum()/ (mfull.sum()*x0.shape[2]+1e-6)
    outside=((x0_edit-x0).float().abs()*(1-mfull)).sum()/((1-mfull).sum()*x0.shape[2]+1e-6)
    print(f"latent |edit-src|: inside-mask={inside:.4f}  outside-mask={outside:.4f}  (want inside>>outside)",flush=True)

    # montage: source (top) vs edited (bottom), 4 frames
    canvas=Image.new("RGB",(W*Fl,H*2),(0,0,0))
    for k in range(Fl):
        idx=min(k*(F//Fl),F-1)
        s=(srcvid[0,idx].clamp(0,1).permute(1,2,0).cpu().float().numpy()*255).astype(np.uint8)
        e=(editvid[0,idx].clamp(0,1).permute(1,2,0).cpu().float().numpy()*255).astype(np.uint8)
        canvas.paste(Image.fromarray(s),(k*W,0)); canvas.paste(Image.fromarray(e),(k*W,H))
    out=os.path.join(OUTDIR,"p2_edit_montage.png"); canvas.save(out)
    print(f"  saved {out}",flush=True)
    print(f"  done in {time.time()-t0:.0f}s  GPU peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB",flush=True)

if __name__=="__main__": main()
