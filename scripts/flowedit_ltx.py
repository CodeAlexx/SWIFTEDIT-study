# FlowEdit for LTX video (RESEARCH) — inversion-FREE text edit on a rectified-flow model.
# (Kulikov et al. 2025, "FlowEdit", built for FLUX/SD3 — same RF family as LTX.)
#
# Why this instead of P2/P2b: the wall was INVERSION (detailed clips invert at cos 0.35-0.62).
# FlowEdit never inverts. From the source latent x0 it integrates the DELTA velocity
#   V_delta = v_theta(z_tgt, t, c_edit) - v_theta(z_src, t, c_src)
# where z_src = (1-sigma)x0 + sigma*n  and  z_tgt = Z_fe + (z_src - x0).
# Where the two prompts agree, V_delta ~= 0 -> that region does not move (natural localization,
# no mask needed). Where they disagree (the car/color), Z_fe drifts -> the edit.
# Reuses p1_embeds.pt + the captured-kwargs velocity. OOM-safe: 2B, bf16.
import os, time, inspect
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch, numpy as np
from PIL import Image
from diffusers import (LTXPipeline, LTXVideoTransformer3DModel,
                       AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler)

REPO="Lightricks/LTX-Video-0.9.1"; DEV,DT="cuda",torch.bfloat16
F_=int(os.environ.get("FE_FRAMES","25")); H=int(os.environ.get("FE_H","512"))
W =int(os.environ.get("FE_W","512"));     N=int(os.environ.get("FE_N","48"))
OUTDIR="/home/alex/swiftedit-ltx-research/fe_out"; os.makedirs(OUTDIR, exist_ok=True)

def main():
    torch.manual_seed(0); t0=time.time()
    E=torch.load("/home/alex/swiftedit-ltx-research/p1_embeds.pt", map_location="cpu")
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

    # source clip (gs=3 clean car)
    with torch.no_grad():
        srcvid=pipe(prompt_embeds=src,prompt_attention_mask=smsk,negative_prompt_embeds=zero,
                    negative_prompt_attention_mask=smsk,width=W,height=H,num_frames=F_,
                    num_inference_steps=30,guidance_scale=3.0,output_type="pt").frames
    # capture exact DiT kwargs (rope/packing/timestep)
    cap=[]; orig=tr.forward
    def spy(*a,**k):
        b=sig.bind(tr,*a,**k); b.apply_defaults(); cap.append(dict(b.arguments)); return orig(*a,**k)
    tr.forward=spy
    with torch.no_grad():
        pipe(prompt_embeds=src,prompt_attention_mask=smsk,negative_prompt_embeds=zero,
             negative_prompt_attention_mask=smsk,width=W,height=H,num_frames=F_,
             num_inference_steps=N,guidance_scale=1.0,output_type="latent")
    tr.forward=orig
    STAT={kk:cap[0][kk] for kk in ['num_frames','height','width','rope_interpolation_scale',
          'video_coords','attention_kwargs'] if kk in cap[0]}
    TS=[c['timestep'] for c in cap]; sigmas=sch.sigmas.to(DEV)

    def vel(z,i,cond,cmsk,gs):
        with torch.no_grad():
            vc=tr(hidden_states=z,timestep=TS[i],encoder_hidden_states=cond,
                  encoder_attention_mask=cmsk,return_dict=False,**STAT)[0]
            if gs==1.0: return vc
            vu=tr(hidden_states=z,timestep=TS[i],encoder_hidden_states=zero,
                  encoder_attention_mask=cmsk,return_dict=False,**STAT)[0]
            return vu+gs*(vc-vu)

    # encode source -> normalized packed x0
    vin=(srcvid*2-1).permute(0,2,1,3,4).to(DEV,DT)
    with torch.no_grad(): lat=vae.encode(vin).latent_dist.sample()
    Fl,Hl,Wl=lat.shape[2],lat.shape[3],lat.shape[4]
    x0=pipe._pack_latents((lat-mean)*sf/std,1,1)

    # ---- FlowEdit integration (inversion-free) ----
    SRC_GS=float(os.environ.get("FE_SRC_GS","1.5")); TGT_GS=float(os.environ.get("FE_TGT_GS","5.5"))
    NAVG=int(os.environ.get("FE_NAVG","1"))
    NMIN=int(os.environ.get("FE_NMIN","0")); NMAX=int(os.environ.get("FE_NMAX",str(N)))
    print(f"FlowEdit: N={N} band=[{NMIN},{NMAX}) src_gs={SRC_GS} tgt_gs={TGT_GS} navg={NAVG} "
          f"res={W}x{H} grid=[{Fl},{Hl},{Wl}]",flush=True)
    Zfe=x0.clone()
    for i in range(N):
        if not (NMIN <= i < NMAX): continue
        ds=sigmas[i+1]-sigmas[i]                      # <0 (sigma high->low)
        Vd=torch.zeros_like(x0)
        for _ in range(NAVG):
            n=torch.randn_like(x0)
            z_src=(1-sigmas[i])*x0 + sigmas[i]*n      # source noised to sigma[i]
            z_tgt=Zfe + (z_src - x0)                  # target path = source path shifted by accumulated edit
            Vsrc=vel(z_src,i,src,smsk,SRC_GS)
            Vtgt=vel(z_tgt,i,edit,emsk,TGT_GS)
            Vd += (Vtgt - Vsrc)
        Vd /= NAVG
        Zfe = Zfe + ds*Vd
    x0_edit=Zfe

    # decode source & edit
    def decode(tok):
        ln=pipe._unpack_latents(tok,Fl,Hl,Wl,1,1); l=ln*std/sf+mean
        tz=torch.zeros(1,device=DEV,dtype=DT) if getattr(vae.config,"timestep_conditioning",False) else None
        with torch.no_grad(): return vae.decode(l,tz).sample
    dec_edit=decode(x0_edit)
    editvid=((dec_edit.float()+1)/2).clamp(0,1).permute(0,2,1,3,4)

    d=(x0_edit-x0).float().abs().mean().item()
    print(f"  mean |edit-src| latent = {d:.4f}",flush=True)
    canvas=Image.new("RGB",(W*Fl,H*2),(0,0,0))
    for k in range(Fl):
        idx=min(k*(F_//Fl),F_-1)
        s=(srcvid[0,idx].clamp(0,1).permute(1,2,0).cpu().float().numpy()*255).astype(np.uint8)
        e=(editvid[0,idx].clamp(0,1).permute(1,2,0).cpu().float().numpy()*255).astype(np.uint8)
        canvas.paste(Image.fromarray(s),(k*W,0)); canvas.paste(Image.fromarray(e),(k*W,H))
    out=os.path.join(OUTDIR,"flowedit_montage.png"); canvas.save(out)
    print(f"  saved {out}",flush=True)
    print(f"  done in {time.time()-t0:.0f}s  GPU peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB",flush=True)

if __name__=="__main__": main()
