# BitBLAS W8A8 pi0.5 success, synchronous and lazy

`fatdove/pi05-libero-w8a8` (Hugging Face revision `2c9896f6`) is a public
INT8 quantization of PI's `pi05_libero`. Its INT8 weights are our bf16
weights re-quantized per channel, and every other tensor is identical. Activations are quantized on the fly to INT8 with one absmax scale per
tensor, and the matmuls run on BitBLAS 0.1.0.post1 INT8×INT8→INT32
kernels, following the author's runtime (`JingxuanZhang77/openpi_duquant`
at `7d4d158`). Its arm actions sit 0.027 from JAX's (mean abs on the 96
recorded plate requests), against 0.006 for the bf16 stacks and FoldQuant
W8A8. The same INT8 weights with unquantized activations give 0.006, so
the per-tensor activation scale causes the gap. Served from an RTX 3090
with the same episodes, tasks, speeds and seeds as the other stacks.

| Mode | BitBLAS W8A8 | JAX bf16 | Δ [paired 95% CI] | Discordant | Model ms |
| --- | ---: | ---: | --- | ---: | ---: |
| synchronous 0 ms (2,400 episodes) | 72.8% | 71.6% | +1.2 [−0.8, +3.2] | 24.5% | 495 |
| lazy 100 ms (2,400) | 64.7% | 66.2% | | | |
| lazy 200 ms (2,400) | 58.9% | 61.5% | | | |
| lazy pooled (4,800) | 61.8% | 63.8% | **−2.0 [−3.4, −0.7]** | 23.2% | 493 |

## Findings

- Synchronous: BitBLAS holds the production success rate, like FoldQuant
  W8A8, despite actions four times further from JAX's.
- Lazy: it loses 2.0 points. The loss sits on near-static and slow points:
  - basket 0.001 m/s: −8.5 at 100 ms, −10.5 at 200 ms;
  - drawer 0.001 m/s: −5.5 and −10.5;
  - drawer 0.025 m/s: −5.5 and −9.5.

  The same points cost it only 2 to 3.5 points synchronously. FlashRT
  NVFP4 shows the same pattern in [`../success-lazy/`](../success-lazy/README.md). When the arm keeps moving on a stale chunk, small action errors at the
  grasp compound in a way the synchronous setting forgives.
- Every 8-bit and 4-bit scheme with calibrated or per-channel activations
  (FlashRT FP8, FoldQuant W8A8 and W4A4) holds success in both modes. Both
  stacks that lose under lazy execution have the largest action errors
  (NVFP4 0.036, BitBLAS 0.027).

## Files

- `sync_vs_jax_bf16.csv`, `lazy_vs_jax_bf16_lazy.csv`: pooled (task empty)
  and per task, speed and latency against JAX bf16 on the same GPU type.

Per-episode outcomes (BitBLAS and JAX, sync and lazy), sha256
`fa95bc54bbe6d9c352c68d216c12e379a6f008dbb36f210e1c6dec0eaa944462`. The BitBLAS
engine and its specs are not part of this release; these tables are kept as
measured.
