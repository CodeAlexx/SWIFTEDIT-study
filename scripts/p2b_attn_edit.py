# P2b (RESEARCH): ATTENTION-LEVEL SwiftEdit edit on LTX video, training-free.
#
# P2 (p2_edit.py) blended in LATENT space and could not recolor cleanly (either kept the
# original color or regenerated a new object). SwiftEdit's real localization is at the
# ATTENTION OUTPUT: inside the edit mask use the EDIT-prompt cross-attention output, outside
# use the SOURCE-prompt cross-attention output (MaskController.fwd_ip). This script ports that
# to the LTX DiT: it hooks attn2 (text cross-attention) on a late block range with a mask
# controller that runs the block's own cross-attn twice (source text K/V and edit text K/V
# against the SAME video-token queries) and blends per token by our [Fl,Hl,Wl] mask.
#
# TWO edit routes (P2B_EDIT_MODE):
#   sdedit  (DEFAULT) -- start from the source noised to sigma[k0] (STRENGTH), denoise forward
#                        with the controller ON, compositing the source path OUTSIDE the mask
#                        each step. Background preserved WITHOUT depending on inversion fidelity.
#   invert            -- start from the source's own Heun-inverted noise inv_src and denoise with
#                        the controller ON (optional latent lock outside the mask). Cleaner in
#                        principle but bounded by the RECON GATE (see below).
#
# FIDELITY FINDING (measured, do not re-litigate): a decisive diagnostic on the gs=3 car x0 at
# Heun N=48 gave source-recon cos 0.517 UNCONDITIONAL and 0.620 SRC-CONDITIONAL -- and the stock
# processor equals a reimplemented SDPA to 4 dp. So the recon shortfall is NOT the attention
# processor and NOT the conditioning; it is the CONTENT-DEPENDENT RF-ODE inversion ceiling on
# detailed real frames (P0b's 0.99 was a smooth synthetic clip). Lifting it needs a TRAINED
# one-step inverter (Option B), not a code change here. The RECON GATE below is kept as a
# tripwire; ~0.62 on this content is EXPECTED, not a regression. The `invert` route is quality-
# bounded by it; the `sdedit` route sidesteps it, which is why it is the default.
#
# The processor delegates to the block's ORIGINAL LTXVideoAttnProcessor per branch, so the
# controller-INACTIVE path is bit-identical to no-hook (correctness/hygiene, not a fidelity fix).
#
# Reuses p1_embeds.pt (src/edit T5 embeds + masks), the captured-kwargs velocity, and Heun
# RF-ODE inversion from P0b/P1/P2. OOM-safe: LTX-Video-0.9.1 (2B), 256x256x25, bf16.
# See DESIGN_P2b_attention.md for the full derivation + diffusers hook points (file:line).
#
# Env knobs:
#   P2B_EDIT_MODE                sdedit | invert   (default sdedit)
#   P2B_STRENGTH                 sdedit start strength 0..1 (default 0.8; higher = more regenerated)
#   P2B_BLOCK_LO / P2B_BLOCK_HI  block range to hook (default: [num_layers//3, num_layers) = late half)
#   P2B_SCALE_EDIT               gain on edit-branch output inside mask  (SwiftEdit scale_ip_fg; default 1.0)
#   P2B_SCALE_NON_EDIT           gain on source-branch output outside mask (SwiftEdit scale_ip_bg; default 1.0)
#   P2B_EDIT_GS                  CFG on the controlled denoise (default 1.0 = attention control only)
#   P2B_LATENT_LOCK              invert mode: re-composite outside-mask with the true source trajectory each step
#   P2_MPOW / P2_DILATE          mask sharpen / dilate (as in p2_edit.py)
#
# NOTE: research script, run by a human on the GPU. Do NOT vendor into serenitymojo/.

import os, time, inspect
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch, numpy as np
import torch.nn.functional as F
from PIL import Image
from diffusers import (LTXPipeline, LTXVideoTransformer3DModel,
                       AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler)

REPO = "Lightricks/LTX-Video-0.9.1"; DEV, DT = "cuda", torch.bfloat16
F_ = int(os.environ.get("P2B_FRAMES", "25"))
H  = int(os.environ.get("P2B_H", "256"))
W  = int(os.environ.get("P2B_W", "256"))
N  = int(os.environ.get("P2B_N", "48"))
OUTDIR = "/home/alex/swiftedit-ltx-research/p2_out"; os.makedirs(OUTDIR, exist_ok=True)


# ---------------------------------------------------------------------------------------------
# The mask controller: one shared object, referenced by every hooked attn2 processor (mirrors
# SwiftEdit's single MaskController set on many processors, infer.py:78-81 / models.py:241-250).
# ---------------------------------------------------------------------------------------------
class LTXMaskController:
    def __init__(self):
        self.active = False          # gate: off for source recon + mask-build inversions, on for the edit
        self.tok_mask = None         # [1, Ntok, 1] blend weight (1=edit, 0=source), f/h/w token order
        self.edit_kv = None          # [1, Ledit, inner_dim] CAPTION-PROJECTED edit text (edit-branch K/V src)
        self.emsk = None             # raw edit padding mask; edit_bias is derived from it to match the source
        self.edit_bias = None        # additive attn bias for the edit text, matched to the source-branch form
        self._bias_built = False
        self.scale_edit = float(os.environ.get("P2B_SCALE_EDIT", "1.0"))          # SwiftEdit scale_ip_fg
        self.scale_non_edit = float(os.environ.get("P2B_SCALE_NON_EDIT", "1.0"))  # SwiftEdit scale_ip_bg

    def build_edit_bias(self, src_am):
        # src_am = the attention_mask the block passes for the SOURCE branch. LTX's model.forward
        # (transformer_ltx.py:512-514) has already turned a [B,L] mask into an additive bias
        # (1-mask)*-1e4 of shape [B,1,L]. Reproduce EXACTLY that transform on emsk so the edit
        # branch is fed the identical form (shape/dtype/device); the stock processor's own
        # prepare_attention_mask then handles it just as it does for the source. Deriving from the
        # captured source form (not a hand-built shape) guards against smsk/emsk padding diffs.
        self._bias_built = True
        if src_am is None:
            self.edit_bias = None
            return
        em = self.emsk
        while em.ndim > 2:
            em = em.squeeze(1)                                   # -> [B, Ledit]
        bias = (1.0 - em.to(src_am.dtype)) * -10000.0           # [B, Ledit]
        for _ in range(src_am.ndim - bias.ndim):                # match rank (model does unsqueeze(1))
            bias = bias.unsqueeze(1)
        self.edit_bias = bias.to(device=src_am.device, dtype=src_am.dtype)


# Custom attn2 processor. Matches LTXVideoAttnProcessor.__call__ signature
# (transformer_ltx.py:63-70) so LTXAttention.forward passes it the right args.
class LTXMaskAttnProcessor:
    # Delegates to `stock` (the block's ORIGINAL LTXVideoAttnProcessor) per branch, so the
    # INACTIVE path is bit-identical to no-hook. (The processor was never the fidelity cause: a
    # diagnostic showed stock == a reimplemented SDPA == recon cos 0.62; the shortfall is the
    # content-dependent inversion ceiling, not attention. Stock delegation is kept for
    # correctness/hygiene.) When active, calls stock twice (source vs edit text K/V, SAME
    # video-token query) and blends the two OUTPUTS per token. to_out is linear, so blending
    # post-projection == a pre-projection blend when the two weights sum to 1 (default 1.0/1.0).
    def __init__(self, controller: LTXMaskController, stock):
        self.controller = controller
        self.stock = stock

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        c = self.controller
        if (not c.active) or (encoder_hidden_states is None):
            return self.stock(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
        if not c._bias_built:
            c.build_edit_bias(attention_mask)                   # match the edit bias to the source form, once
        out_src = self.stock(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
        out_edit = self.stock(attn, hidden_states, c.edit_kv, c.edit_bias, image_rotary_emb)
        m = c.tok_mask.to(out_src.dtype)                        # [1, Ntok, 1]
        # fwd_ip blend (mask_ip_controller.py:140): edit-attn inside mask, source-attn outside.
        return m * (c.scale_edit * out_edit) + (1.0 - m) * (c.scale_non_edit * out_src)


def main():
    torch.manual_seed(0); t0 = time.time()
    CTRL = LTXMaskController()

    E = torch.load("/home/alex/swiftedit-ltx-research/p1_embeds.pt", map_location="cpu")
    src = E["src"].to(DEV, DT); edit = E["edit"].to(DEV, DT)
    smsk = E["src_mask"].to(DEV); emsk = E["edit_mask"].to(DEV); zero = torch.zeros_like(src)
    CTRL.emsk = emsk

    vae = AutoencoderKLLTXVideo.from_pretrained(REPO, subfolder="vae", torch_dtype=DT).to(DEV).eval()
    tr = LTXVideoTransformer3DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=DT).to(DEV).eval()
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(REPO, subfolder="scheduler")
    pipe = LTXPipeline(scheduler=sch, vae=vae, text_encoder=None, tokenizer=None, transformer=tr)
    pipe.set_progress_bar_config(disable=True)
    sig = inspect.signature(LTXVideoTransformer3DModel.forward)
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(DEV, DT); std = vae.latents_std.view(1, -1, 1, 1, 1).to(DEV, DT)
    sf = getattr(vae.config, "scaling_factor", 1.0)
    inner_dim = tr.config.num_attention_heads * tr.config.attention_head_dim
    NL = len(tr.transformer_blocks)

    # ---- install the mask processor on a LATE block range (attn2 = text cross-attn) ----
    # Capture each block's ORIGINAL attn2 processor and delegate to it -> inactive path is
    # provably identical to no-hook (also preserves any backend/parallel config already set).
    LO = int(os.environ.get("P2B_BLOCK_LO", str(NL // 3)))
    HI = int(os.environ.get("P2B_BLOCK_HI", str(NL)))
    for i in range(LO, HI):
        a2 = tr.transformer_blocks[i].attn2
        a2.set_processor(LTXMaskAttnProcessor(CTRL, a2.processor))
    print(f"hooked attn2 on blocks [{LO},{HI}) of {NL}  (scale_edit={CTRL.scale_edit} "
          f"scale_non_edit={CTRL.scale_non_edit})", flush=True)

    # ---- source clip at guidance 3 (clean red car). Source-gen guidance does NOT affect the
    # recon gate (it round-trips the encoded x0), so keep gs=3 for a clean source; inversion gs=1. ----
    with torch.no_grad():
        srcvid = pipe(prompt_embeds=src, prompt_attention_mask=smsk, negative_prompt_embeds=zero,
                      negative_prompt_attention_mask=smsk, width=W, height=H, num_frames=F_,
                      num_inference_steps=30, guidance_scale=3.0, output_type="pt").frames

    # ---- capture exact DiT call kwargs from a real denoise (controller stays OFF) ----
    cap = []; orig = tr.forward
    def spy(*a, **k):
        b = sig.bind(tr, *a, **k); b.apply_defaults(); cap.append(dict(b.arguments)); return orig(*a, **k)
    tr.forward = spy
    with torch.no_grad():
        pipe(prompt_embeds=src, prompt_attention_mask=smsk, negative_prompt_embeds=zero,
             negative_prompt_attention_mask=smsk, width=W, height=H, num_frames=F_,
             num_inference_steps=N, guidance_scale=1.0, output_type="latent")
    tr.forward = orig
    STAT = {kk: cap[0][kk] for kk in ['num_frames', 'height', 'width', 'rope_interpolation_scale',
            'video_coords', 'attention_kwargs'] if kk in cap[0]}
    TS = [c['timestep'] for c in cap]; sigmas = sch.sigmas.to(DEV)

    def vel(z, i, cond, cmsk, gs=1.0):
        with torch.no_grad():
            vc = tr(hidden_states=z, timestep=TS[i], encoder_hidden_states=cond,
                    encoder_attention_mask=cmsk, return_dict=False, **STAT)[0]
            if gs == 1.0:
                return vc
            # CFG: uncond = zero text. Controller stays active on both (blend uses its own edit K/V).
            vu = tr(hidden_states=z, timestep=TS[i], encoder_hidden_states=zero,
                    encoder_attention_mask=cmsk, return_dict=False, **STAT)[0]
            return vu + gs * (vc - vu)

    def step_down(z, i, cond, cmsk, gs=1.0):
        ds = sigmas[i + 1] - sigmas[i]; v = vel(z, i, cond, cmsk, gs)
        zp = z + ds * v; v2 = vel(zp, min(i + 1, N - 1), cond, cmsk, gs); return z + ds * 0.5 * (v + v2)

    def invert(cond, cmsk):
        # RF-ODE inversion at gs=1 (the raw model velocity IS the ODE field; a CFG velocity does
        # not invert cleanly -- that is why inversion stays gs=1 regardless of source-gen gs).
        z = x0.clone()
        for i in reversed(range(N)):
            ds = sigmas[i] - sigmas[i + 1]; v = vel(z, i, cond, cmsk, 1.0)
            zp = z + ds * v; v2 = vel(zp, max(i - 1, 0), cond, cmsk, 1.0); z = z + ds * 0.5 * (v + v2)
        return z

    # ---- encode source -> normalized packed x0 ----
    vin = (srcvid * 2 - 1).permute(0, 2, 1, 3, 4).to(DEV, DT)
    with torch.no_grad(): lat = vae.encode(vin).latent_dist.sample()
    Fl, Hl, Wl = lat.shape[2], lat.shape[3], lat.shape[4]
    global x0; x0 = pipe._pack_latents((lat - mean) * sf / std, 1, 1)
    Ntok = Fl * Hl * Wl
    assert x0.shape[1] == Ntok, f"packed tokens {x0.shape[1]} != Fl*Hl*Wl {Ntok}"

    # ---- prompt-diff mask (controller OFF during these inversions => bit-identical forwards) ----
    CTRL.active = False
    inv_src = invert(src, smsk); inv_edit = invert(edit, emsk)
    subed = (inv_src - inv_edit).abs().float().mean(dim=2)[0]
    mnorm = ((subed - subed.min()) / (subed.max() - subed.min() + 1e-6))
    MPOW = float(os.environ.get("P2_MPOW", "3.0"))
    mg = (mnorm ** MPOW).view(1, 1, Fl, Hl, Wl)
    DIL = int(os.environ.get("P2_DILATE", "0"))
    for _ in range(DIL):
        mg = F.max_pool3d(mg, kernel_size=3, stride=1, padding=1)
    # token mask: [Fl,Hl,Wl] flattens DIRECTLY onto the packed-token axis (f/h/w order, same as
    # _pack_latents permute) -> no interpolation. See DESIGN_P2b_attention.md sec 1.
    CTRL.tok_mask = mg.reshape(1, Ntok, 1).to(DT)
    print(f"mask: frac>0.5={float((mg.reshape(-1) > 0.5).float().mean()):.3f}  "
          f"grid=[{Fl},{Hl},{Wl}] Ntok={Ntok} MPOW={MPOW} DIL={DIL}", flush=True)

    # ---- prepare the EDIT branch K/V once: caption-project the edit text. edit_bias is built
    # lazily on the first active call to match the source branch's mask form. model.forward
    # applies caption_projection (transformer_ltx.py:528) to the source text before the blocks;
    # reproduce it so attn2.to_k/to_v see the same [B,L,inner_dim] shape. ----
    with torch.no_grad():
        CTRL.edit_kv = tr.caption_projection(edit).view(1, -1, inner_dim).to(DT)

    # ---- FIDELITY GATE (always on) + source-recon trajectory (used by the invert route's lock).
    # Denoise inv_src forward with NO control -> should return to x0. ~0.99 on smooth clips (P0b);
    # ~0.62 on these detailed car frames = the KNOWN content-dependent RF-ODE inversion ceiling,
    # NOT a regression (needs a trained one-step inverter, Option B). The invert route is bounded
    # by this; the sdedit route is not. ----
    CTRL.active = False
    zr = inv_src.clone(); src_traj = [zr.clone()]
    for i in range(N):
        zr = step_down(zr, i, src, smsk, 1.0); src_traj.append(zr.clone())
    recon_cos = float(torch.nn.functional.cosine_similarity(zr.float().flatten(), x0.float().flatten(), dim=0))
    print(f"  RECON GATE [{'PASS' if recon_cos > 0.97 else 'FAIL'}]: inactive source-recon cos vs x0 = "
          f"{recon_cos:.4f} (want >0.97; ~0.62 here = known inversion ceiling, not a regression)", flush=True)

    # ---- THE EDIT ----------------------------------------------------------------------------
    EDIT_MODE = os.environ.get("P2B_EDIT_MODE", "sdedit").lower()
    EDIT_GS = float(os.environ.get("P2B_EDIT_GS", "1.0"))
    m = CTRL.tok_mask.to(x0.dtype)

    if EDIT_MODE == "sdedit":
        # SDEdit start (DEFAULT): re-noise the source to sigma[k0] and denoise forward with the
        # controller ON. OUTSIDE the mask, composite the source noised to each sigma (preservation
        # proven clean in P2); INSIDE the mask, attn2 injects edit-text cross-attention to change
        # appearance. Sidesteps the inversion ceiling (no inv_src dependence). STRENGTH: how far
        # back to re-noise (higher = more of the masked region regenerated).
        STRENGTH = float(os.environ.get("P2B_STRENGTH", "0.8"))
        eps = torch.randn_like(x0)
        k0 = max(0, min(N - 1, int((1 - STRENGTH) * N)))
        def snoise(i): return (1 - sigmas[i]) * x0 + sigmas[i] * eps    # source on the RF path at sigma[i]
        ze = snoise(k0)
        CTRL.active = True
        for i in range(k0, N):
            ze = m * ze + (1.0 - m) * snoise(i)          # outside mask -> source path (preserve)
            ze = step_down(ze, i, src, smsk, EDIT_GS)    # controller ON: edit-text attn INSIDE mask
        ze = m * ze + (1.0 - m) * x0                      # final: outside -> clean source
        CTRL.active = False
        print(f"  edit mode=sdedit strength={STRENGTH} start={k0}/{N} EDIT_GS={EDIT_GS}", flush=True)
    else:
        # invert start: denoise from the source's own inverted noise inv_src with the controller
        # ON; optional latent lock holds the background on the true source trajectory. Cleaner in
        # principle but quality is bounded by the RECON GATE above.
        LATENT_LOCK = os.environ.get("P2B_LATENT_LOCK", "0") == "1"
        ze = inv_src.clone()
        CTRL.active = True
        for i in range(N):
            ze = step_down(ze, i, src, smsk, EDIT_GS)
            if LATENT_LOCK:
                ze = m * ze + (1.0 - m) * src_traj[i + 1]   # hold background on the true source path
        CTRL.active = False
        print(f"  edit mode=invert EDIT_GS={EDIT_GS} LATENT_LOCK={LATENT_LOCK} "
              f"(quality bounded by recon gate {recon_cos:.2f})", flush=True)
    x0_edit = ze

    # ---- decode source & edit ----
    def decode(tok):
        ln = pipe._unpack_latents(tok, Fl, Hl, Wl, 1, 1); l = ln * std / sf + mean
        tz = torch.zeros(1, device=DEV, dtype=DT) if getattr(vae.config, "timestep_conditioning", False) else None
        with torch.no_grad(): return vae.decode(l, tz).sample
    dec_edit = decode(x0_edit)
    editvid = ((dec_edit.float() + 1) / 2).clamp(0, 1).permute(0, 2, 1, 3, 4)  # [1,F,3,H,W]

    # ---- metrics: latent change inside vs outside mask (want inside >> outside) ----
    mfull = CTRL.tok_mask.float()
    inside = ((x0_edit - x0).float().abs() * mfull).sum() / (mfull.sum() * x0.shape[2] + 1e-6)
    outside = ((x0_edit - x0).float().abs() * (1 - mfull)).sum() / ((1 - mfull).sum() * x0.shape[2] + 1e-6)
    print(f"latent |edit-src|: inside-mask={inside:.4f}  outside-mask={outside:.4f}  (want inside>>outside)", flush=True)

    # ---- montage: source (top) vs edited (bottom), one column per latent frame ----
    canvas = Image.new("RGB", (W * Fl, H * 2), (0, 0, 0))
    for k in range(Fl):
        idx = min(k * (F_ // Fl), F_ - 1)
        s = (srcvid[0, idx].clamp(0, 1).permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        e = (editvid[0, idx].clamp(0, 1).permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(s), (k * W, 0)); canvas.paste(Image.fromarray(e), (k * W, H))
    out = os.path.join(OUTDIR, "p2b_attn_edit_montage.png"); canvas.save(out)
    print(f"  saved {out}", flush=True)
    print(f"  done in {time.time()-t0:.0f}s  GPU peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
