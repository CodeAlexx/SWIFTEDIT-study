# SwiftEdit->LTX  P0b: RF-ODE inversion round-trip (RESEARCH, not main code)
# Prove: encode a real clip -> integrate RF velocity field x0->noise->x0' -> reconstruct.
# Correctness guard: the velocity(x_t, t) callable reuses the EXACT kwargs (packing/rope/
# timestep) that a real LTXPipeline denoise passes to the DiT -- captured, not hand-rolled.
# OOM-safe: LTX-0.9.1 2B, tiny res, zero text-embeds (no T5).
import os, time, math, inspect, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from diffusers import (LTXPipeline, LTXVideoTransformer3DModel,
                       AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler)

REPO="Lightricks/LTX-Video-0.9.1"; DEV,DT="cuda",torch.bfloat16
F,H,W = 25,256,256
N     = int(sys.argv[1]) if len(sys.argv)>1 else 48     # ODE nodes
METHOD= sys.argv[2] if len(sys.argv)>2 else "heun"      # heun (2nd-order) REQUIRED; euler fails (~17dB)
def vram(t):
    torch.cuda.synchronize()
    print(f"  [VRAM] {t:22s} peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

def main():
    torch.manual_seed(0); t0=time.time()
    vae=AutoencoderKLLTXVideo.from_pretrained(REPO,subfolder="vae",torch_dtype=DT).to(DEV).eval()
    tr =LTXVideoTransformer3DModel.from_pretrained(REPO,subfolder="transformer",torch_dtype=DT).to(DEV).eval()
    sch=FlowMatchEulerDiscreteScheduler.from_pretrained(REPO,subfolder="scheduler")
    pipe=LTXPipeline(scheduler=sch,vae=vae,text_encoder=None,tokenizer=None,transformer=tr)
    pipe.set_progress_bar_config(disable=True)
    cap=tr.config.caption_channels
    zemb=torch.zeros(1,128,cap,device=DEV,dtype=DT); zmsk=torch.ones(1,128,device=DEV,dtype=torch.int64)
    sig=inspect.signature(LTXVideoTransformer3DModel.forward)

    # ---- capture the exact DiT call args from a real N-step denoise ----
    cap_calls=[]
    orig=tr.forward
    def spy(*a,**k):
        b=sig.bind(tr,*a,**k); b.apply_defaults(); cap_calls.append(dict(b.arguments)); return orig(*a,**k)
    tr.forward=spy
    with torch.no_grad():
        pipe(prompt_embeds=zemb,prompt_attention_mask=zmsk,negative_prompt_embeds=zemb,
             negative_prompt_attention_mask=zmsk,width=W,height=H,num_frames=F,
             num_inference_steps=N,guidance_scale=1.0,output_type="latent")
    tr.forward=orig
    print(f"captured {len(cap_calls)} real DiT calls (expect {N})",flush=True)
    STAT={kk:cap_calls[0][kk] for kk in ['encoder_hidden_states','encoder_attention_mask',
          'num_frames','height','width','rope_interpolation_scale','video_coords','attention_kwargs']
          if kk in cap_calls[0]}
    TS=[c['timestep'] for c in cap_calls]            # per-node timestep tensors (as pipeline sends them)
    sigmas=sch.sigmas.to(DEV)                          # len N+1, descending: sigmas[0]~1 (noise) .. sigmas[N]=0
    print(f"  sigmas[0]={sigmas[0].item():.3f} .. sigmas[-1]={sigmas[-1].item():.3f}  (len {len(sigmas)})",flush=True)

    def velocity(z, i):
        with torch.no_grad():
            return tr(hidden_states=z, timestep=TS[i], return_dict=False, **STAT)[0]

    # ---- encode a real source clip -> normalized packed latents x0 ----
    xs=torch.linspace(-1,1,W); ys=torch.linspace(-1,1,H); grid=(ys[:,None]+xs[None,:]).clamp(-1,1)
    vid=torch.stack([torch.roll(grid,shifts=f*4,dims=1) for f in range(F)],0)[None,None].repeat(1,3,1,1,1).to(DEV,DT)
    with torch.no_grad(): lat=vae.encode(vid).latent_dist.sample()     # [1,128,4,8,8]
    Fl,Hl,Wl=lat.shape[2],lat.shape[3],lat.shape[4]
    mean=vae.latents_mean.view(1,-1,1,1,1).to(DEV,DT); std=vae.latents_std.view(1,-1,1,1,1).to(DEV,DT)
    sf=getattr(vae.config,"scaling_factor",1.0)
    latn=(lat-mean)*sf/std
    x0=pipe._pack_latents(latn,1,1)                                    # [1,256,128]
    vram("after encode+pack")

    # ---- RF-ODE inversion  x0 -> noise (integrate sigma UP, reverse node order) ----
    z=x0.clone()
    for i in reversed(range(N)):
        ds=sigmas[i]-sigmas[i+1]                                       # >0, sigma increases
        v=velocity(z,i)
        if METHOD=="heun":
            zp=z+ds*v; v2=velocity(zp, max(i-1,0)); z=z+ds*0.5*(v+v2)  # 2nd-order corrector
        else:
            z=z+ds*v
    noise=z.clone()
    # ---- reconstruct  noise -> x0'  (integrate sigma DOWN, forward node order) ----
    for i in range(N):
        ds=sigmas[i+1]-sigmas[i]                                       # <0, sigma decreases
        v=velocity(z,i)
        if METHOD=="heun":
            zp=z+ds*v; v2=velocity(zp, min(i+1,N-1)); z=z+ds*0.5*(v+v2)
        else:
            z=z+ds*v
    x0p=z
    vram("after inversion round-trip")

    # ---- metrics: latent round-trip + decoded video PSNR ----
    cos=torch.nn.functional.cosine_similarity(x0.float().flatten(),x0p.float().flatten(),dim=0).item()
    lmse=torch.mean((x0.float()-x0p.float())**2).item()
    def decode(tok):
        ln=pipe._unpack_latents(tok,Fl,Hl,Wl,1,1); l=ln*std/sf+mean
        tz=torch.zeros(1,device=DEV,dtype=DT) if getattr(vae.config,"timestep_conditioning",False) else None
        with torch.no_grad(): return vae.decode(l,tz).sample
    rec0=decode(x0); recp=decode(x0p)
    mse=torch.mean((recp.float()-rec0.float())**2).item()
    psnr=10*math.log10(4.0/mse) if mse>0 else 99.0
    print(f"\n== P0b RESULT ==",flush=True)
    print(f"  latent round-trip:  cos={cos:.4f}   MSE={lmse:.4e}",flush=True)
    print(f"  decoded video x0' vs x0:  PSNR={psnr:.1f} dB   (invert->reconstruct fidelity)",flush=True)
    print(f"  done in {time.time()-t0:.0f}s",flush=True); vram("FINAL")

if __name__=="__main__": main()
