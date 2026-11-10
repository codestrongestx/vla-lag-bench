# pi0.5 inference stacks and quantizations

Does pi0.5 keep its success on the paper's four moving LIBERO tasks when it
is served by a different inference stack or quantized, and how fast is each
option on common GPUs? GR00T N1.7 and DynamicVLA are timed alongside it
for reference.

| Folder | Question | Status |
| --- | --- | --- |
| [`success-sync/`](success-sync/README.md) | Output quality at synchronous 0 ms, 2,400 paired episodes per stack | done |
| [`success-lazy/`](success-lazy/README.md) | Output quality in lazy async at 100 and 200 ms, 4,800 paired episodes per stack | done |
| [`success-bitblas/`](success-bitblas/README.md) | The public BitBLAS W8A8 checkpoint, synchronous and lazy | done |
| [`latency-batch1/`](latency-batch1/README.md) | Batch-1 request latency on seven GPUs | done; H100 rows superseded by `latency-batch1-h100-quiet/` |
| [`latency-batch1-h100-quiet/`](latency-batch1-h100-quiet/README.md) | H100 latency rerun on a quiet host | done |
| [`latency-batch1-additions/`](latency-batch1-additions/README.md) | FoldQuant on the 5090 and A40, a full L4 suite, BitBLAS W8A8 | done |

Stacks: openpi JAX bf16 (production), openpi PyTorch bf16, FlashRT
(bf16, FP8, NVFP4), FoldQuant TensorRT (bf16, W8A8, W4A4), BitBLAS W8A8.
The paper uses the bf16 stacks; this release includes their specs
(`configs/studies/pi05_stacks_20261014/`) and keeps the quantized results for
reference only.
