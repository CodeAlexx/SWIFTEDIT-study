# P2b — attention-level SwiftEdit for LTX video (design)

Research design note. Target: PyTorch/diffusers, `LTX-Video-0.9.1` (2B), 256×256×25, OOM-safe.
Nothing here touches `serenitymojo/`. Companion runnable script: `p2b_attn_edit.py`.

**Why P2b exists.** P2 (`p2_edit.py`) did the SwiftEdit region blend in *latent* space (blend the
noised source vs the edit trajectory per token). That can't change appearance while preserving
structure: raise the edit weight and you regenerate a new object + artifacts, lower it and it stays
the source color. SwiftEdit's actual localization is at the **attention output**, not the latent.
This note ports that.

---

## 0. What SwiftEdit's region blend really is (correcting a subtlety)

`MaskController.attn_batch` (`SwiftEdit/src/mask_ip_controller.py:24-51`) *looks* like it masks the
softmax (`sim_fg`/`sim_bg`, lines 42-45). It does not, in any meaningful way: the mask is shape
`[1, i, 1]` over the **query** axis `i`, broadcast across the key axis `j`, and it is added to `sim`
*before* `softmax(-1)` (over `j`). Softmax is shift-invariant along `j`, so adding a per-row constant
is a no-op on the attention weights. The `sim_fg`/`sim_bg` split only serves to produce **two copies**
(`out_target_fg`, `out_target_bg`) that get chunked and blended in `fwd_ip`.

The part that actually localizes is the **output blend** in `fwd_ip`
(`mask_ip_controller.py:140-142`):

```python
out_target = scale_ip_fg * out_target_fg * mask + scale_ip_bg * out_source * (1 - mask)
```

i.e. **per query token**: inside the mask use the *edit* branch's attention output, outside use the
*source* branch's. That is the whole mechanism worth porting. Our LTX port therefore does exactly
this output blend and skips the (no-op) softmax masking.

In SwiftEdit the two branches are (a) text cross-attn `q·k` over the **edit text** and (b) an
IP-adapter cross-attn `q·ip_k` over the **source image** (identity preservation). SwiftEdit has a
trained IP-adapter for the "source" branch. We are **training-free**, so our two branches are both
the DiT's own text cross-attention, fed **source text** vs **edit text**. Inside the mask → edit
text output; outside → source text output. This is the natural training-free specialization of
`fwd_ip`, and it needs no extra weights.

---

## 1. LTX DiT attention layout (the hook points, with file:line)

Installed diffusers: `/home/alex/.local/lib/python3.12/site-packages/diffusers`,
file `models/transformers/transformer_ltx.py`.

- **Model:** `LTXVideoTransformer3DModel` (`transformer_ltx.py:385`). For 0.9.1: `num_layers=28`,
  `num_attention_heads=32`, `attention_head_dim=64` → `inner_dim=2048`, `cross_attention_dim=2048`,
  `caption_channels=4096`.
- **Block:** `LTXVideoTransformerBlock` (`transformer_ltx.py:282`). Each block, in order:
  - `attn1` — **self-attention** over video tokens (`transformer_ltx.py:317-326`,
    `cross_attention_dim=None`). Called with `encoder_hidden_states=None` and the video RoPE
    `image_rotary_emb` (`:362-366`). **We do NOT hook this.** (It mixes tokens across the mask
    boundary — see §6 leakage note — but hooking/sharing it is a structure-transfer tool, out of
    scope for a recolor.)
  - `attn2` — **text cross-attention** (`transformer_ltx.py:329-338`, `cross_attention_dim=2048`).
    Called (`:369-374`) with `encoder_hidden_states=<text>`, `image_rotary_emb=None`,
    `attention_mask=encoder_attention_mask`. **THIS is our hook point.** Query = video tokens,
    Key/Value = text tokens. No RoPE on cross-attn.
  - `ff` — feed-forward (`:378`).
- **Attention module:** `LTXAttention` (`transformer_ltx.py:115`). Its `forward`
  (`:161-176`) filters kwargs to the processor signature and calls
  `self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)`.
- **Default processor:** `LTXVideoAttnProcessor` (`transformer_ltx.py:48-112`). Signature to match
  (`:63-70`): `__call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
  image_rotary_emb=None)`. It does: `q=to_q(hs)`, `k=to_k(ehs)`, `v=to_v(ehs)`, `q=norm_q(q)`,
  `k=norm_k(k)`, RoPE only if `image_rotary_emb` (None here), unflatten to heads, SDPA, `to_out`.
- **How to override — per block, cleanly:**
  `tr.transformer_blocks[i].attn2.set_processor(proc)` via `AttentionModuleMixin.set_processor`
  (`models/attention.py:127`). (The model also has `AttentionMixin.set_attn_processor` at
  `attention.py:64` which takes a full name→proc dict; per-block `set_processor` is simpler for a
  block *range*.)

### Two facts that make the port exact

1. **Token order = mask order.** `LTXPipeline._pack_latents`
   (`pipelines/ltx/pipeline_ltx.py:420-440`) with `patch_size=patch_size_t=1` permutes
   `(0,2,4,6,1,3,5,7)` → `(B, F, H, W, C,1,1,1)` then flattens → **`[B, F*H*W, C]`, f-major then h
   then w**. Our P1 mask `mask3d = m.view(Fl,Hl,Wl)` (`p1_mask.py:82`) has the identical order, so
   `mask.reshape(1, Fl*Hl*Wl, 1)` broadcasts **directly** onto the video-token axis — **no
   `F.interpolate` needed** (SwiftEdit interpolated because its UNet worked at pixel-spatial
   `H=W=sqrt(N)`; LTX tokens already live on the latent grid). For 256×256×25 → latent grid
   `[Fl,Hl,Wl]=[4,8,8]` → `Ntok=256`.
2. **Caption projection + mask-bias happen once in `model.forward`, before the blocks.**
   `encoder_hidden_states = caption_projection(...)` (`transformer_ltx.py:528`,
   `PixArtAlphaTextProjection`, `models/embeddings.py:2191`) maps `4096→inner_dim`; and
   `encoder_attention_mask` is turned into an additive bias `(1-mask)*-10000` shape `[B,1,L]`
   (`transformer_ltx.py:512-514`). So the **source** text reaching `attn2` is already projected and
   its mask already a bias. For the **edit** branch we must reproduce both **once, up front**:
   `edit_proj = tr.caption_projection(edit_raw).view(1,-1,inner_dim)` and
   `edit_bias = ((1-emsk)*-10000).unsqueeze(1)`, and stash them on the controller.

---

## 2. The controller (LTX MaskController analog)

A single shared object (like SwiftEdit's one `MaskController` set on many processors,
`infer.py:78-81`, `models.py:241-250`). It holds:

| field | shape / type | meaning |
|---|---|---|
| `active` | bool | gate; off during source recon & mask build, on during the edit denoise |
| `tok_mask` | `[1, Ntok, 1]` bf16 | blend weight per video token (1=edit, 0=source), f/h/w order |
| `edit_kv` | `[1, Ledit, inner_dim]` | **caption-projected** edit text (KV source for the edit branch) |
| `edit_bias` | `[1, 1, Ledit]` | additive attn bias for the edit text (padding mask) |
| `scale_edit` | float (env `P2B_SCALE_EDIT`, default 1.0) | ↔ SwiftEdit `scale_ip_fg`; gain on edit-branch output inside mask |
| `scale_non_edit` | float (env `P2B_SCALE_NON_EDIT`, default 1.0) | ↔ SwiftEdit `scale_ip_bg`; gain on source-branch output outside mask |

**Semantic note on the scales (do not copy SwiftEdit's 0.2 blindly).** In SwiftEdit `scale_ip_fg=0.2`
*weakens the source-image IP branch inside the edit region so the edit text can take over*
(`infer.py:36`, `DESIGN_SwiftEdit_for_LTX.md` §1.3). Here the foreground branch **is** the edit text,
so we want it at full strength: default `scale_edit=1.0`, `scale_non_edit=1.0`. Treat `scale_edit>1`
as "push the edit harder", `scale_non_edit<1` as "let the background drift toward the edit". Start at
(1.0, 1.0).

### The blend, on LTX tokens

Inside the hooked `attn2` processor, with shared video-token query `q` (from the single hidden-state
stream) and the block's own `to_k/to_v/norm_k`:

```
out_src  = sdpa(q, norm_k(to_k(ehs_src )), to_v(ehs_src ), bias_src )   # source text
out_edit = sdpa(q, norm_k(to_k(edit_kv )), to_v(edit_kv ), edit_bias)   # edit text
out = tok_mask * (scale_edit * out_edit) + (1 - tok_mask) * (scale_non_edit * out_src)   # then to_out
```

This is `fwd_ip`'s `scale_ip_fg*fg*mask + scale_ip_bg*src*(1-mask)` (`mask_ip_controller.py:140`)
re-expressed on `[Ntok, dim]` video tokens. `q` is **shared** across branches (prompt2prompt style:
same queries, two key/value sets) — so only `attn2` forks; the residual stream stays single and the
self-attention / FF are untouched.

---

## 3. Which blocks to hook (early=structure, late=texture)

SwiftEdit hooks **mid + up** blocks only (`infer.py:81`, `where=["mid_blocks","up_blocks"]`) — the
*later* half of the UNet — because a **recolor** is an appearance/texture change; the early
(down) blocks that fix geometry are left alone so structure is preserved. LTX has no down/mid/up, just
28 uniform blocks, so the analog is a **contiguous late-block range**:

- **Default (recolor / appearance):** hook blocks `[num_layers//3, num_layers)` → **`[9, 28)`** for
  0.9.1. Early blocks 0-8 (structure) run the plain source cross-attn, so geometry/motion is held and
  only appearance changes inside the mask.
- **To also change shape/object:** extend the low end toward 0 (`P2B_BLOCK_LO=0`). Expect more drift
  outside the mask; pair with the latent lock (§5).
- Env knobs: `P2B_BLOCK_LO`, `P2B_BLOCK_HI` (defaults `9`, `28`). This is the main **ablation axis**
  the human should sweep first.

---

## 4. Composition with the captured-kwargs velocity + Heun inversion

Reuses P0b/P1/P2 plumbing verbatim:

1. **Same source clip** (src prompt, CFG 3.0) → encode → normalized packed `x0` (`p2_edit.py:31-76`).
2. **Capture exact DiT kwargs** (`num_frames/height/width/rope_interpolation_scale/video_coords/
   attention_kwargs`) from a real `LTXPipeline` denoise via the `spy` (`p2_edit.py:37-48`) →
   `STAT`, `TS` (per-step timesteps), `sigmas`. Velocity is `tr(hidden_states=z, timestep=TS[i],
   encoder_hidden_states=<text>, encoder_attention_mask=<mask>, **STAT)[0]`.
3. **Mask** exactly as P1/P2 (`p2_edit.py:79-88`): `inv_src`, `inv_edit` via **Heun RF-ODE inversion**
   (`p2_edit.py:65-70`), `|inv_src-inv_edit|` mean-over-channel, normalize, `**MPOW` sharpen, optional
   `max_pool3d` dilate → `tok_mask = mg.reshape(1,Ntok,1)`.
4. **The edit denoise IS attention-level now.** Start from the **source's own inverted noise**
   `inv_src` (structure-preserving; P0b round-trips it to `x0` at cos 0.99). Turn the controller
   **on** and integrate the RF-ODE **forward** (`sigma` high→low) with the **source** prompt as
   `encoder_hidden_states`. Because `attn2` blends per token, the velocity field is automatically
   "edit inside the mask, source outside" in a single forward — no latent blend. Heun step reused
   from `p2_edit.py:62-64`. Controller **off** for the mask-building inversions and for the optional
   source-recon pass.
5. Decode, montage (source top / edit bottom), latent inside/outside-mask change metric — all from
   `p2_edit.py:105-128`.

Guidance: default `P2B_EDIT_GS=1.0` — the attention swap *is* the guidance, so plain gs=1 should
already recolor inside the mask. `P2B_EDIT_GS>1` adds CFG (controller-active conditional vs a
zero-text uncond) for a harder push; keep it optional.

---

## 5. Optional latent lock (guard against self-attention leak)

`attn1` self-attention (not hooked) mixes edited tokens into background tokens across the mask
boundary, so over N steps the edit can bleed outward. Optional `P2B_LATENT_LOCK=1`: run a source
reconstruction denoise first (controller **off**, from `inv_src`, src prompt), store `src_traj[i]`
at every step, then in the edit denoise re-composite outside the mask each step:
`z = tok_mask*z + (1-tok_mask)*src_traj[i+1]`. Accurate (uses the true source trajectory, not a
straight-line approx) and cheap at this res. Off by default so the attention-only result is visible
first (that is the point of P2b). The source-recon pass also gives a free sanity number (recon PSNR
vs the source, should match P0b ~33-35 dB).

---

## 6. Uncertainties for the human to verify at run time

1. **Token/mask alignment** — verified from source (`_pack_latents` permute vs `mask.view(Fl,Hl,Wl)`),
   but confirm visually: with `scale_edit=0, scale_non_edit=1` the edit denoise must reproduce the
   source (mask does nothing) — if instead it corrupts a *stripe*, the f/h/w flatten order is
   transposed and `tok_mask` needs reshaping.
2. **`num_layers`** — assumed 28 for 0.9.1; script reads `len(tr.transformer_blocks)` and derives the
   range, so a different depth self-adjusts. Confirm the printed block range looks late-half.
3. **Self-attention leakage** — how much the edit bleeds outside the mask without `P2B_LATENT_LOCK`.
   If bad, turn the lock on (§5). This is the main quality risk unique to a DiT (SwiftEdit's one-step
   UNet never integrated N steps, so it never accumulated leak).
4. **`caption_projection` reuse for the edit branch** — we call `tr.caption_projection(edit)` once and
   cache it; confirm it is not stochastic (it is a plain MLP, so fine) and that `edit`'s token length
   `Ledit` matches its `emsk` length.
5. **SDPA path** — the custom processor uses `F.scaled_dot_product_attention` directly (transpose to
   `[B,heads,S,hd]`) rather than diffusers' `dispatch_attention_fn`, to stay self-contained and avoid
   backend-layout surprises. Numerically equivalent (default scale `1/sqrt(hd)`); just noting the
   deviation from the stock processor.
6. **Block range for a recolor vs an object swap** — start at the default late range for the blue-car
   recolor; if the edit is too weak, lower `P2B_BLOCK_LO`; if the background drifts, raise it or
   enable the lock.

---

## 7. Deliverable map

- Design: this file.
- Runnable (by the human, not the author): `p2b_attn_edit.py` — hooks `attn2` on the late block range
  with the mask controller, denoises from `inv_src` under the source prompt with the controller on,
  env knobs `P2B_BLOCK_LO/HI`, `P2B_SCALE_EDIT`, `P2B_SCALE_NON_EDIT`, `P2B_EDIT_GS`,
  `P2B_LATENT_LOCK`, plus the P2 mask knobs `P2_MPOW`, `P2_DILATE`. OOM-safe (2B, 256×256×25, bf16).
</content>
</invoke>
