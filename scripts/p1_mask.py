# P1 stage-2 (RESEARCH): prompt-diff edit mask on LTX video (SwiftEdit mechanism).
#   gen source clip from src embeds -> encode -> RF-ODE invert (Heun) under src AND edit
#   prompts -> mask = |inv_src - inv_edit| -> clamp/normalize/binarize -> 3D edit mask.
# Reuses cached T5 embeds (p1_embeds.pt); no text encoder loaded here. OOM-safe: 2B DiT.
import os, time, math, inspect
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch, numpy as np
from PIL import Image
from diffusers import (LTXPipeline, LTXVideoTransformer3DModel,
                       AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler)

REPO="Lightricks/LTX-Video-0.9.1"; DEV,DT="cuda",torch.bfloat16
F,H,W,N = 25,256,256,48
CLAMP_RATE, MASK_THRESH = 3.0, 0.5
OUTDIR="/home/alex/swiftedit-ltx-research/p1_out"; os.makedirs(OUTDIR, exist_ok=True)

def main():
    torch.manual_seed(0); t0=time.time()
    E=torch.load("/home/alex/swiftedit-ltx-research/p1_embeds.pt", map_location="cpu")
    src=E["src"].to(DEV,DT); edit=E["edit"].to(DEV,DT)
    smsk=E["src_mask"].to(DEV); emsk=E["edit_mask"].to(DEV)
    zero=torch.zeros_like(src)
    vae=AutoencoderKLLTXVideo.from_pretrained(REPO,subfolder="vae",torch_dtype=DT).to(DEV).eval()
    tr =LTXVideoTransformer3DModel.from_pretrained(REPO,subfolder="transformer",torch_dtype=DT).to(DEV).eval()
    sch=FlowMatchEulerDiscreteScheduler.from_pretrained(REPO,subfolder="scheduler")
    pipe=LTXPipeline(scheduler=sch,vae=vae,text_encoder=None,tokenizer=None,transformer=tr)
    pipe.set_progress_bar_config(disable=True)
    sig=inspect.signature(LTXVideoTransformer3DModel.forward)
    mean=vae.latents_mean.view(1,-1,1,1,1).to(DEV,DT); std=vae.latents_std.view(1,-1,1,1,1).to(DEV,DT)
    sf=getattr(vae.config,"scaling_factor",1.0)

    # ---- 1) generate a source clip from the src prompt (real CFG gen) ----
    print("== gen source clip from src prompt ==",flush=True)
    with torch.no_grad():
        gen=pipe(prompt_embeds=src,prompt_attention_mask=smsk,negative_prompt_embeds=zero,
                 negative_prompt_attention_mask=smsk,width=W,height=H,num_frames=F,
                 num_inference_steps=30,guidance_scale=3.0,output_type="pt")
    srcvid=gen.frames  # [1,F,3,H,W] in [0,1]
    print(f"  source clip {tuple(srcvid.shape)}",flush=True)

    # ---- capture exact DiT call args (rope/packing/timestep) from a real denoise ----
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

    def velocity(z,i,cond,cmsk):
        with torch.no_grad():
            return tr(hidden_states=z,timestep=TS[i],encoder_hidden_states=cond,
                      encoder_attention_mask=cmsk,return_dict=False,**STAT)[0]

    # ---- 2) encode source -> normalized packed latents x0 ----
    vin=(srcvid*2-1).permute(0,2,1,3,4).to(DEV,DT)         # [1,3,F,H,W] in [-1,1]
    with torch.no_grad(): lat=vae.encode(vin).latent_dist.sample()
    Fl,Hl,Wl=lat.shape[2],lat.shape[3],lat.shape[4]
    x0=pipe._pack_latents((lat-mean)*sf/std,1,1)

    # ---- 3) RF-ODE inversion x0 -> noise, conditioned per prompt (Heun) ----
    def invert(cond,cmsk):
        z=x0.clone()
        for i in reversed(range(N)):
            ds=sigmas[i]-sigmas[i+1]; v=velocity(z,i,cond,cmsk)
            zp=z+ds*v; v2=velocity(zp,max(i-1,0),cond,cmsk); z=z+ds*0.5*(v+v2)
        return z
    print("== invert under src and edit prompts ==",flush=True)
    inv_src =invert(src, smsk)
    inv_edit=invert(edit,emsk)

    # ---- 4) prompt-diff mask (SwiftEdit math on LTX tokens) ----
    subed=(inv_src-inv_edit).abs().float().mean(dim=2)[0]   # [N tokens]
    max_v=(subed.mean()*CLAMP_RATE).item()
    m=(subed.clamp(0,max_v)/max_v)
    mbin=(m>MASK_THRESH).float()
    mask3d=mbin.view(Fl,Hl,Wl)                              # [4,8,8] spatiotemporal
    frac=mbin.mean().item()
    print(f"== P1 MASK ==",flush=True)
    print(f"  tokens={subed.numel()}  latent grid=[{Fl},{Hl},{Wl}]  masked fraction={frac:.3f}",flush=True)
    print(f"  soft-mask range [{m.min():.2f},{m.max():.2f}]  mean {m.mean():.2f}",flush=True)

    # ---- 5) visualize: source frames (top) + upsampled mask (bottom) ----
    def up(t2d):  # [Hl,Wl] -> [H,W] nearest, to uint8 rgb
        a=torch.nn.functional.interpolate(t2d[None,None],(H,W),mode="nearest")[0,0]
        return (a.clamp(0,1).cpu().numpy()*255).astype(np.uint8)
    cols=Fl
    canvas=Image.new("RGB",(W*cols,H*2),(0,0,0))
    for k in range(Fl):
        sf_idx=min(k*(F//Fl),F-1)
        sfrm=(srcvid[0,sf_idx].clamp(0,1).permute(1,2,0).cpu().float().numpy()*255).astype(np.uint8)
        canvas.paste(Image.fromarray(sfrm),(k*W,0))
        mk=up(mask3d[k]); rgb=np.stack([mk,mk//3,mk//3],-1)  # red-tinted mask
        canvas.paste(Image.fromarray(rgb),(k*W,H))
    montage=os.path.join(OUTDIR,"p1_mask_montage.png"); canvas.save(montage)
    # also dump soft mask (pre-binarize) montage for insight
    soft=m.view(Fl,Hl,Wl)
    c2=Image.new("L",(W*cols,H)); [c2.paste(Image.fromarray(up(soft[k])),(k*W,0)) for k in range(Fl)]
    softpng=os.path.join(OUTDIR,"p1_mask_soft.png"); c2.save(softpng)
    print(f"  saved {montage}",flush=True)
    print(f"  saved {softpng}",flush=True)
    print(f"  done in {time.time()-t0:.0f}s  GPU peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB",flush=True)

if __name__=="__main__": main()
