# SwiftEdit->LTX  P0 feasibility (RESEARCH, not main code)
# Goal: prove the engine turns over on the SMALL OOM-safe LTX (0.9.1, 2B):
#   (a) temporal VAE video round-trip (encode->decode) recon
#   (b) full few-step generation through the 2B DiT with ZERO text-embeds (no T5)
#   (c) peak-VRAM measured throughout  -> shows int8 is NOT needed at this scale
# No editing yet. No text encoder downloaded. OOM-safe by construction (tiny res).
import os, time, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from diffusers import (LTXPipeline, LTXVideoTransformer3DModel,
                       AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler)

REPO = "Lightricks/LTX-Video-0.9.1"
DEV, DT = "cuda", torch.bfloat16
def vram(tag):
    torch.cuda.synchronize()
    a=torch.cuda.memory_allocated()/1e9; p=torch.cuda.max_memory_allocated()/1e9
    print(f"  [VRAM] {tag:28s} alloc={a:5.2f}GB  peak={p:5.2f}GB", flush=True)

def main():
    torch.manual_seed(0)
    t0=time.time()
    print("== load 2B DiT + temporal VAE (bf16) ==", flush=True)
    vae = AutoencoderKLLTXVideo.from_pretrained(REPO, subfolder="vae", torch_dtype=DT).to(DEV).eval()
    tr  = LTXVideoTransformer3DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=DT).to(DEV).eval()
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(REPO, subfolder="scheduler")
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True); vram("after load")

    # ---- (a) VAE temporal round-trip on a synthetic clip ----
    F,H,W = 25, 256, 256          # 25 = 8*3+1 (LTX temporal); tiny res => OOM-safe
    print(f"== (a) VAE round-trip on synthetic clip [1,3,{F},{H},{W}] ==", flush=True)
    # a moving gradient so recon is meaningful, range [-1,1]
    xs = torch.linspace(-1,1,W); ys = torch.linspace(-1,1,H)
    grid = (ys[:,None]+xs[None,:]).clamp(-1,1)
    vid = torch.stack([torch.roll(grid, shifts=f*4, dims=1) for f in range(F)], 0)  # [F,H,W]
    vid = vid[None,None].repeat(1,3,1,1,1).to(DEV,DT)                                # [1,3,F,H,W]
    with torch.no_grad():
        lat = vae.encode(vid).latent_dist.sample()
        vram("after VAE encode")
        # LTX VAE decoder is timestep-conditioned; pipeline passes decode_timestep (0.0) per batch
        tz = None
        if getattr(vae.config, "timestep_conditioning", False):
            tz = torch.zeros(lat.shape[0], device=DEV, dtype=DT)
        rec = vae.decode(lat, tz).sample
        vram("after VAE decode")
    mse = torch.mean((rec.float()-vid.float())**2).item()
    psnr = 10*math.log10(4.0/mse) if mse>0 else 99.0   # signal range 2 -> peak^2=4
    print(f"  latents {tuple(lat.shape)}  recon MSE={mse:.4e}  PSNR={psnr:.1f} dB", flush=True)
    del rec, vid; torch.cuda.empty_cache()

    # ---- (b) few-step generation through the DiT, ZERO text-embeds ----
    print("== (b) few-step generation via LTXPipeline, zero text-embeds (no T5) ==", flush=True)
    pipe = LTXPipeline(scheduler=sch, vae=vae, text_encoder=None, tokenizer=None, transformer=tr)
    pipe.set_progress_bar_config(disable=True)
    L = 128
    cap = getattr(tr.config, "caption_channels", 4096)
    zemb = torch.zeros(1, L, cap, device=DEV, dtype=DT)
    zmsk = torch.ones(1, L, device=DEV, dtype=torch.int64)
    torch.cuda.reset_peak_memory_stats()
    tg=time.time()
    with torch.no_grad():
        out = pipe(prompt_embeds=zemb, prompt_attention_mask=zmsk,
                   negative_prompt_embeds=zemb, negative_prompt_attention_mask=zmsk,
                   width=W, height=H, num_frames=F, num_inference_steps=4,
                   guidance_scale=1.0, output_type="pt")
    vram("after generation")
    frames = out.frames  # tensor [B,F,C,H,W] for output_type=pt
    print(f"  generated frames tensor {tuple(frames.shape)} in {time.time()-tg:.0f}s", flush=True)
    print(f"== P0 DONE in {time.time()-t0:.0f}s  (peak VRAM below is the OOM headroom on 24GB) ==", flush=True)
    vram("FINAL peak")

if __name__ == "__main__":
    main()
