# Batch-1 latency: added GPUs and stacks

Same protocol as [`../latency-batch1/`](../latency-batch1/README.md): 20
warm-up calls, then 500 timed calls on recorded LIBERO requests, alone on the
GPU. This folder adds the following:

- FoldQuant bf16/W8A8/W4A4 on the RTX 5090 and A40, with the rest of the suite rerun on both.
- A full suite on the L4.
- BitBLAS W8A8 (`fatdove/pi05-libero-w8a8`) on the RTX 3090, RTX 5090 and L4.

These records name the host CPU (`cpu_model`).

| GPU | Host CPU |
| --- | --- |
| RTX 5090 | Ryzen 9 9950X |
| A40 | Xeon Gold 6342 |
| L4 | EPYC 9254 |
| RTX 3090 | Xeon Gold 6226R (RTX 3090 server) |

Median request latency (ms) for pi0.5:

| Stack | RTX 5090 | A40 | L4 | RTX 3090 |
| --- | ---: | ---: | ---: | ---: |
| FlashRT NVFP4 | 16.7 | — | — | — |
| FlashRT FP8 | 18.9 | — | 62.0 | — |
| FlashRT bf16 | 27.4 | 51.8 | 89.8 | |
| openpi PyTorch bf16 | 49.6 | 105.4 | 171.2 | |
| openpi JAX bf16 | 58.3 | 140.4 | 199.7 | |
| FoldQuant bf16 | 61.8 | 125.2 | 173.2 | |
| FoldQuant W8A8 | 56.1 | 119.8 | 142.0 | |
| FoldQuant W4A4 | 109.9 | 108.6 | 123.3 | |
| BitBLAS W8A8 | 151.8 | — | 414.1 | 490.1 |

Empty cells are in `../latency-batch1/`; "—" is unsupported (FP8 needs
sm_89+, NVFP4 sm_120) or not run (BitBLAS on the A40).

- FoldQuant W4A4 is fastest among FoldQuant arms only where INT4 tensor
  cores exist (A40, L4, 3090, 4090). On the 5090 it is twice as slow as W8A8.
- BitBLAS W8A8 is the slowest stack everywhere. It runs eagerly, as its
  author's loader does, and launches about 11,000 small kernels per call. On the 3090 its BitBLAS target is `nvidia/geforce-rtx-3090`. On the
  5090 and L4, BitBLAS has no named target and builds for the generic
  `cuda` target. Its actions match the 3090's (arm mean abs 0.026 vs 0.027,
  against 0.006 for the bf16 stacks).
- Two A40 records (FoldQuant bf16 and W8A8) carry a false `shared GPU`
  note: nvidia-smi in that container listed host PIDs, so the run could not
  recognise its own process (fixed in 9f0347b2). The pre-run idle check
  found no other process on the GPU.
