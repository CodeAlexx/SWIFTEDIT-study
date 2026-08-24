# SwiftEdit → LTX (research design, NOT for main code)

Adapting Qualcomm-AI-research **SwiftEdit** (one-step, mask-free, text-guided *image* editing, CVPR'25, BSD-3-Clause-Clear) to **LTX-2** *video* editing. This is an exploratory research note; nothing here should land in the Mojo production stack (`serenitymojo/`) — target is PyTorch/diffusers.

Local clone: `/home/alex/swiftedit-ltx-research/SwiftEdit`.

---

## 1. What SwiftEdit actually is (from the code, not the abstract)

Three parts, all one forward pass each:

1. **One-step inversion UNet** (`InverseModel.unet_inverse`, an SD-turbo UNet fine-tuned).
   `predict_inverted_code = unet_inverse(dub_latents, t=500, [src_emb, edit_emb])` → per-prompt "inverted noise". Trained (2-stage, synthetic→real, reconstruction loss) so that `α_t·latents + σ_t·inverted_noise` is a noisy latent the one-step generator maps back to the source.

2. **Self-guided edit mask** (no user mask):
   `mask = |inverted_noise(src) − inverted_noise(edit)|` averaged over channels, clamped/normalized (`clamp_rate=3`), binarized (`mask_threshold=0.5`). Where the two prompts disagree ⇒ the region to edit.

3. **Mask-rescaled one-step generation** (`IPSBV2Model`, SwiftBrush-v2 UNet + IP-Adapter):
   - x0 in one step: `x0 = (noise − σ_t·ε)/α_t` at t=999.
   - IP-Adapter injects the **source image** as extra cross-attn tokens (identity/structure preservation).
   - `MaskController` rescales attention per region (`src/mask_ip_controller.py:fwd_ip`):
     `out = scale_ip_fg·(edit attn)·mask + scale_ip_bg·(source attn)·(1−mask)`
     with `scale_edit=0.2` (weaken source in the edit region so text takes over), `scale_non_edit=1` (keep source verbatim outside). Text branch boosts edit-text in the mask (`scale_text_hiddenstate`).

Net effect: invert → diff-mask → one-step regenerate, preserving everything outside the mask, applying the edit-prompt inside it. ~0.23 s because every step is a single forward.

---

## 2. LTX target (what's local, PyTorch)

- **diffusers 0.38.0.dev0** exposes: `LTX2Pipeline`, `LTX2VideoTransformer3DModel`, `AutoencoderKLLTX2Video`, `LTX2ConditionPipeline`/`LTXConditionPipeline` (conditioning/i2v), `LTXEulerAncestralRFScheduler` (**rectified-flow / flow-matching**, not DDPM).
- **Distilled few-step LTX-2** exists locally (`serenity-inference/reference/video_ltx2_t2v_distilled.json`, `LTX-2.3_…_Distilled.json`) — this is the analog of "SwiftBrush one-step generator."
- Checkouts: `/home/alex/LTX-2`, `LTX-2-official`, `LTX-2-upstream`, `LTX-Desktop`.

Two structural differences from SwiftEdit's SD base that drive the whole port:
- **Video**, not image: latents are `[B, C, T, H, W]` from a *causal temporal* VAE. Masks and attention rescaling become **spatiotemporal**.
- **Rectified flow**, not DDPM: the reverse process is a (near-)deterministic ODE. This *helps* — inversion is natural — but changes the `x0 = (noise−σε)/α` bookkeeping to an RF velocity/`v`-parameterization.

---

## 3. Component-by-component mapping

| SwiftEdit piece | LTX analog | Reuse / build |
|---|---|---|
| SwiftBrush v2 (one-step gen) | **LTX-2 distilled** (few-step RF) | reuse weights; run 1–4 steps |
| SD VAE (image) | `AutoencoderKLLTX2Video` (video, temporal) | reuse; encode/decode video |
| DDPM `α_t,σ_t`, t∈{500,999} | RF sigmas from `LTXEulerAncestralRFScheduler` | remap; `x0 = z − σ·v` (RF), not `(z−σε)/α` |
| `unet_inverse` (trained inversion) | **two options**, see §4 | Option A: no training (RF-ODE invert). Option B: train an inversion DiT. |
| IP-Adapter (CLIP-image → cross-attn tokens) for source preservation | LTX **native conditioning** (init/ICLoRA/i2v latents) *or* a small video-IP module | prefer native conditioning first |
| `MaskController` spatial attn rescale over UNet mid/up blocks | rescale over **LTX DiT transformer blocks'** text-cross-attn + conditioning, mask flattened over `T·h·w` tokens | port `fwd_ip` math to DiT token layout |
| prompt-diff mask (2D) | prompt-diff over the **video velocity field**, per (t,h,w) | same idea, 3D mask + temporal smoothing |

---

## 4. The inversion — the key fork

SwiftEdit's trained one-step inversion is what makes it *instant*, but it's also the expensive part (2-stage training). Flow-matching gives a shortcut:

- **Option A — RF-ODE inversion (no training, prototype-first).** LTX's RF sampler is a deterministic ODE; run it *backwards* (few steps) from the encoded source video to a noisy latent, then forward-edit. Slower than one-step but validates the *mask + attention-rescaling edit* on LTX video with zero training. **This is the cheapest way to prove the concept.**
- **Option B — learned one-step inversion DiT (the real SwiftEdit contribution).** Fine-tune a copy of the LTX-2 DiT to predict inverted noise in one pass (2-stage: synthetic LTX-generated clips → real clips, reconstruction loss through the frozen distilled generator). This buys the speed but is a training campaign (data + compute).

Recommendation: **A to prove it, then B for speed.**

---

## 5. Video-specific problems SwiftEdit never had

1. **Temporal mask coherence** — a per-frame prompt-diff mask will flicker. Need temporal smoothing / a single 3D mask propagated across T (or compute the diff on the temporally-pooled velocity).
2. **Temporal identity preservation** — the "background" (outside the mask) must stay consistent *across frames*, not just per-frame. Native LTX conditioning (init latents / motion context) is a better preservation vehicle than a CLIP-image IP-Adapter, which has no temporal notion.
3. **DiT vs UNet attention** — LTX has no `down/mid/up` blocks; it's a stack of transformer blocks with text cross-attention. The `MaskController` region-rescale must be re-expressed on flattened `[T·h·w, dim]` tokens, choosing *which* blocks to hook (early blocks = structure, late = texture — needs an ablation).
4. **RF parameterization** — redo the `x0`/mask-blend algebra for velocity prediction and the scheduler's sigma schedule.

---

## 6. Phased plan

- **P0 (feasibility, days):** diffusers `LTX2Pipeline` + distilled weights. Implement **RF-ODE inversion** (Option A) on a short clip; get x0 reconstruction of a source video. No editing yet — just prove invert→regenerate round-trips.
  - **P0a ENGINE-CHECK — DONE + VERIFIED (2026-08-23, RTX 3090 Ti).** On the OOM-safe small model **LTX-Video-0.9.1 (2B)** — NOT 0.9.7 (that's 13B, my earlier mis-size) and NOT the 22B LTX-2 (when-home). `p0_run.py`: temporal VAE round-trips a [1,3,25,256,256] clip → latents (1,128,4,8,8) → decode @ **PSNR 35.5 dB**; few-step gen with **zero text-embeds (no 19GB T5)** drives the RF DiT **8/8 real forward calls** on 256 packed tokens (~78 ms/step) → correct-shape video (1,25,3,256,256); **peak VRAM 5.97 GB / 24 GB**. Text encoder skipped entirely for P0. int8 confirmed unnecessary at 2B.
  - **P0b RF-ODE INVERSION round-trip — DONE + VERIFIED (2026-08-23).** `p0b_invert.py`: encode a real clip → integrate the RF velocity field x0→noise→x0'. Velocity(x_t,t) reuses the EXACT kwargs (packing/rope/timestep) captured from a real `LTXPipeline` denoise — not hand-rolled. **Key finding: the integrator matters, not step count.** Forward Euler round-trips at only cos≈0.87 / **17 dB** (and *degrades* at N=96 from bf16 accumulation); a **2nd-order Heun corrector** gives **cos 0.99 / video PSNR 33.5 dB @N=24, 35.3 dB @N=48** — i.e. ~0 dB above the VAE's own 35.5 dB ceiling, so inversion is effectively lossless. Peak 5.99 GB. **Implication: Option A (training-free RF-ODE inversion) is viable → P1/P2 edit can be prototyped WITHOUT training the one-step inversion DiT (Option B/P3).**
- **P1 (mask): DONE + VERIFIED (2026-08-23).** `p1_encode.py` (T5-XXL encoder-only, bf16, in a MemoryMax-capped scope → 9.6 GB GPU, embeds cached to `p1_embeds.pt`) + `p1_mask.py`. Gen source clip from src prompt (2B 0.9.1 produced a clean **red sports car** on a coastal road at 256²) → encode → RF-ODE invert (Heun) under src("red car") AND edit("blue car") → `mask=|inv_src−inv_edit|` mean-over-channel, clamp_rate 3 / thresh 0.5 → 3D mask [Fl,Hl,Wl]=[4,8,8]. **Result: the mask localizes on the CAR and tracks its motion across frames** (montage `p1_out/p1_mask_montage.png`). Masked fraction 1.2% is *correct* (only the car changes; car ≈ one 8×8 latent cell/frame). 11 s, peak 5.97 GB. Caveats: blocky (latent-res 8×8), frame-1 gap (car at edge) → temporal smoothing still wanted (§5.1); validated on a *generated* source, real-video source next.
- **P2 (edit): ATTEMPTED + BOUNDED (2026-08-23).** Two training-free routes tried, both VERIFIED on LTX-0.9.1:
  - **P2 latent SDEdit** (`p2_edit.py`): mask-blended re-noise/regenerate. Background preserved perfectly (inside/outside latent-change ratio up to 20×), but the recolor either doesn't take (car stays red) or over-regenerates (new car + artifacts) at high strength. Color is contextual → latent blending can't isolate it.
  - **P2b attention-level** (`p2b_attn_edit.py`): ports SwiftEdit `MaskController.fwd_ip` onto the LTX DiT — hooks `attn2` (text cross-attn) on late blocks [9,28), runs the STOCK `LTXVideoAttnProcessor` per branch (source-text vs edit-text K/V, shared video queries) and blends outputs by the [Fl,Hl,Wl] mask. Hook is faithful (source gen through it is clean; stock-processor call is bit-identical). But: (1) the inversion route is capped by RF-ODE fidelity — training-free Heun inversion round-trips well only at gs≈1 (cos 0.99) which yields garbage sources; CFG-3 sources (clean car) invert at cos 0.62 (gs=1 invert) / 0.39 (gs=3 invert). (2) The SDEdit-start route (sidesteps inversion) preserves background perfectly but the car **stays red even at scale_edit=4 / strength 0.92 / EDIT_GS=2** — text cross-attn can't override the color entangled in the preserved latent.
  - **DECISIVE DIAGNOSTIC** (`p2b_recon_diag.py`): the recon-gate 0.62 is CONTENT-DEPENDENT inversion error, not processor/conditioning/guidance. Same gs=3 car x0, Heun N=48: unconditional recon cos **0.517**, src-conditional **0.620** (both low; uncond is *worse*, so conditioning isn't the cap). Stock processor == `_sdpa` == 0.6199 (processor ruled out). P0b's 0.99 was on a *smooth synthetic* clip → the detailed car frames are what break training-free RF-ODE inversion. This is the fundamental training-free-inversion ceiling on real content.
  - **RESOLUTION ruled out too (2026-08-23):** re-ran the SDEdit+attention edit at **512²** (mask grid [4,16,16], 1024 tokens, inside/outside 47×) — source clip is much crisper but the **car still won't recolor**; higher res even worsens the mask (inversion recon drops to 0.35 at 512²). So it's not a latent-resolution limit. Byproduct: **512² LTX render is OOM-safe (peak 6.58 GB)** on the 24 GB card.
  - **CONCLUSION:** the SwiftEdit **mask + region-blend mechanism ports** (localize + preserve verified), but **training-free appearance edit does not transfer** on this 2B RF model — confirmed exhaustively across latent-blend, attention-level, invert/sdedit routes, and 256²/512². This is SwiftEdit's own rationale for **training** the one-step inversion + IP-adapter. The clean recolor needs Option B / P3, not more knob-tuning. Env knobs + all montages in the repo; `p2b_attn_edit.py` is the attention-level scaffold to reuse once a trained inversion/adapter exists.
- **P3 (speed, optional):** train the one-step inversion DiT (Option B) to collapse P0's few-step invert into one pass — the actual SwiftEdit contribution.

Each phase is a standalone PyTorch script under `/home/alex/swiftedit-ltx-research/` with a parity/visual gate; nothing touches `serenitymojo/`.

---

## 7. Reuse vs build

- **Reuse as-is:** the mask math (`|inv(src) − inv(edit)|`, clamp/binarize), the region-blend formula, the "which blocks to hook" idea, SwiftEdit's scale hyperparameters as starting points.
- **Rewrite:** VAE/latent shapes (video), scheduler algebra (RF), attention hook (DiT tokens vs UNet spatial), source-preservation vehicle (LTX conditioning instead of CLIP IP-Adapter).
- **New:** temporal mask coherence, temporal preservation, (optional) the trained one-step inversion DiT.

## 8. INT8 (quantized) LTX for OOM-safe prototyping — 24 GB target

Memory is the binding constraint: LTX-2 (22 B) DiT ≈ **44 GB bf16 / ~22 GB fp8**, and SwiftEdit's inversion runs a **doubled `[src, edit]` batch**, so activations are ~2×. On a 24 GB 3090 Ti we quantize the DiT.

Footprint (DiT weights only, +activations/VAE on top):
- bf16 ≈ 44 GB — no.
- fp8 ≈ 22 GB — fits base but no headroom for the doubled inversion + VAE.
- **int8 weight-only ≈ 11 GB — the sweet spot** (headroom for 2× activations + video VAE).
- int4 ≈ 5.5 GB — safest, more quality risk (reference the existing SVDQuant INT4 A/B).

**Path (verified available, load not yet run):**
- `BitsAndBytesConfig(load_in_8bit=True)` on `LTX2VideoTransformer3DModel.from_pretrained(..., quantization_config=...)` — **bitsandbytes int8 is the recommended backend** (torchao cpp extensions are skipped under torch 2.10; revisit if torch ≥2.11).
- Keep the **VAE + text encoder in bf16/fp16** (small); quantize only the DiT.
- Alternatives to try: load the existing **fp8** checkpoint directly; or GGUF via `GGUFQuantizationConfig`; or reuse the **INT4 SVDQuant** recipe from `~/samples/ltx2_int4_svdquant/`.

**Quant × SwiftEdit compatibility:**
- Editing is inference-only ⇒ int8 **weight-only** is fine; the mask-rescale hooks the *attention outputs*, orthogonal to weight quantization.
- P3 (train the one-step inversion DiT): keep the int8 base **frozen** and train a small inversion adapter on top (QLoRA-style) — never dequantize the base.

**Not yet verified:** that `LTX2VideoTransformer3DModel` loads cleanly under bitsandbytes int8 and that the RF sampler numerics hold at int8 — that's the first thing P0 should check (a load + one-step forward), and it needs the GPU, so it's a when-home run.

## 8b. FlowEdit — THE WORKING EDIT (2026-08-24) ✅

After P2/P2b showed training-free editing capped by inversion fidelity, we tried **FlowEdit**
(Kulikov et al. 2025), an **inversion-free** RF editing method — and it **works**. It never
inverts: from the source latent x0 it integrates the DELTA velocity
`V_Δ = v_θ(z_tgt, t, c_edit) − v_θ(z_src, t, c_src)` where `z_src=(1−σ)x0+σn`,
`z_tgt=Z_fe+(z_src−x0)`. Where the prompts agree V_Δ≈0 (region frozen); where they disagree the
edit accumulates. Sidesteps the inversion ceiling entirely. `flowedit_ltx.py`.

RESULTS (LTX-0.9.1 2B, bf16, gs3 red-car source, red→blue):
- **256², src_gs 1.5 / tgt_gs 5.5:** car turns blue but morphs bigger (edit too strong). 21 s, 6.0 GB.
- **512², src_gs 1.5 / tgt_gs 4.5:** **CLEAN recolor — same car shape/size/position, red→blue,
  background preserved.** 29 s, 6.6 GB. This is the working editor. Minor: slight red residue on the
  car's lower front; ocean slightly more saturated (mild non-local spillover).
- Knobs: `FE_SRC_GS` (~1.5), `FE_TGT_GS` (edit strength; 4–5 clean, 5.5+ morphs), `FE_NAVG`
  (variance reduction), `FE_NMIN/NMAX` (edit band), `FE_H/W/N`.
- **No training, no mask, no inversion.** Option B (trained inverter) is now UNNECESSARY for this.

## 9. Licensing note
SwiftEdit is BSD-3-Clause-Clear (permissive) but it's Qualcomm research code — keep it as a **reference/research** dependency here, re-implement the adaptation cleanly, and do not vendor it into the production Mojo stack. Matches the "research, not main code" instruction.
