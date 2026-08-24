# Decisive diagnostic: what caps the P2b recon gate at 0.62?
# Measure Heun RF-ODE recon cos on the SAME gs=3 car x0 for:
#   (A) UNCONDITIONAL inversion (zero text, like P0b which hit 0.99 on a synthetic clip)
#   (B) SRC-CONDITIONAL inversion (real src text, as P2b does)
# If A>>B  -> the 0.62 is the conditioning (curved conditional field), machinery is sound.
# If A~=B~low -> it's content-dependent inversion error on detailed frames.
# Either way it is NOT the attn processor (stock vs _sdpa gave identical 0.6199).
import os, time, inspect
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch
from diffusers import (LTXPipeline, LTXVideoTransformer3DModel,
                       AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler)
REPO="Lightricks/LTX-Video-0.9.1"; DEV,DT="cuda",torch.bfloat16
F_,H,W,N = 25,256,256,48

def main():
    torch.manual_seed(0); t0=time.time()
    E=torch.load("/home/alex/swiftedit-ltx-research/p1_embeds.pt",map_location="cpu")
    src=E["src"].to(DEV,DT); smsk=E["src_mask"].to(DEV); zero=torch.zeros_like(src)
    vae=AutoencoderKLLTXVideo.from_pretrained(REPO,subfolder="vae",torch_dtype=DT).to(DEV).eval()
    tr =LTXVideoTransformer3DModel.from_pretrained(REPO,subfolder="transformer",torch_dtype=DT).to(DEV).eval()
    sch=FlowMatchEulerDiscreteScheduler.from_pretrained(REPO,subfolder="scheduler")
    pipe=LTXPipeline(scheduler=sch,vae=vae,text_encoder=None,tokenizer=None,transformer=tr)
    pipe.set_progress_bar_config(disable=True)
    sig=inspect.signature(LTXVideoTransformer3DModel.forward)
    mean=vae.latents_mean.view(1,-1,1,1,1).to(DEV,DT); std=vae.latents_std.view(1,-1,1,1,1).to(DEV,DT)
    sf=getattr(vae.config,"scaling_factor",1.0)

    # gs=3 car source -> x0
    with torch.no_grad():
        srcvid=pipe(prompt_embeds=src,prompt_attention_mask=smsk,negative_prompt_embeds=zero,
                    negative_prompt_attention_mask=smsk,width=W,height=H,num_frames=F_,
                    num_inference_steps=30,guidance_scale=3.0,output_type="pt").frames
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
    vin=(srcvid*2-1).permute(0,2,1,3,4).to(DEV,DT)
    with torch.no_grad(): lat=vae.encode(vin).latent_dist.sample()
    Fl,Hl,Wl=lat.shape[2],lat.shape[3],lat.shape[4]
    x0=pipe._pack_latents((lat-mean)*sf/std,1,1)

    def vel(z,i,cond,cmsk):
        with torch.no_grad():
            return tr(hidden_states=z,timestep=TS[i],encoder_hidden_states=cond,
                      encoder_attention_mask=cmsk,return_dict=False,**STAT)[0]
    def roundtrip(cond,cmsk,label):
        z=x0.clone()                                   # invert x0 -> noise (Heun)
        for i in reversed(range(N)):
            ds=sigmas[i]-sigmas[i+1]; v=vel(z,i,cond,cmsk)
            zp=z+ds*v; v2=vel(zp,max(i-1,0),cond,cmsk); z=z+ds*0.5*(v+v2)
        for i in range(N):                             # noise -> x0' (Heun)
            ds=sigmas[i+1]-sigmas[i]; v=vel(z,i,cond,cmsk)
            zp=z+ds*v; v2=vel(zp,min(i+1,N-1),cond,cmsk); z=z+ds*0.5*(v+v2)
        cos=float(torch.nn.functional.cosine_similarity(z.float().flatten(),x0.float().flatten(),dim=0))
        print(f"  {label:26s} recon cos vs x0 = {cos:.4f}",flush=True); return cos

    print("== recon on the SAME gs=3 car x0 ==",flush=True)
    a=roundtrip(zero,smsk,"(A) UNCONDITIONAL (zero)")
    b=roundtrip(src, smsk,"(B) SRC-CONDITIONAL")
    print(f"== verdict: {'conditioning is the cap (machinery sound)' if a-b>0.15 else 'content-dependent (both similar)'} ==",flush=True)
    print(f"  A(uncond)={a:.4f}  B(cond)={b:.4f}  done {time.time()-t0:.0f}s  GPU {torch.cuda.max_memory_allocated()/1e9:.2f}GB",flush=True)

if __name__=="__main__": main()
