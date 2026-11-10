# Batch-1 latency on the H100 with a quiet host

The H100 rows in [`../latency-batch1/`](../latency-batch1/README.md) were
taken while other jobs loaded the shared host (load average 20 to 125 on 208
threads). The arms that launch many kernels from Python moved with that load. This rerun waited until the 1-minute load average had stayed below 25 for 20
minutes. It then measured every arm on GPU 1, with CPUs pinned to cores 44–51
on the GPU's NUMA node, using the same protocol: 20 warm-up and 500 timed
calls, recorded requests, GPU unshared. The host CPU is a Xeon Platinum 8481C.

| Stack | Precision | p50 ms | p95 ms | p99 ms | Host load | Earlier busy-host p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| openpi JAX | bf16 | 47.8 | 55.7 | 56.6 | 15 | 50 to 60 |
| openpi JAX, websocket | bf16 | 48.5 | 56.1 | 58.2 | 10 | 49 to 51 |
| openpi PyTorch | bf16 | 59.2 | 64.3 | 66.3 | 12 | 63 to 77 |
| FoldQuant | bf16 | 61.5 | 64.7 | 66.1 | 12 | 63 to 70 |
| FoldQuant | W8A8 | 73.2 | 78.2 | 84.3 | 7 | 76 to 91 |
| FoldQuant | W4A4 | 176.1 | 183.3 | 209.0 | 30 | 176 to 180 |
| GR00T N1.7 | bf16 | 104.3 | 112.5 | 118.6 | 21 | 109 to 201 |
| GR00T N1.7, websocket | bf16 | 104.5 | 116.4 | 184.1 | 18 | 113 to 180 |
| DynamicVLA | fp32 | 213.9 | 226.0 | 273.7 | 16 | 244 to 262 |

FlashRT bf16 (17.6 ms) is unaffected by host load and was not rerun; FlashRT
FP8 does not run on Hopper. These rows supersede the H100 rows of
`../latency-batch1/latency.csv` for papers. The tails above p95 are now within
about 10 ms of the median for every arm except the GR00T websocket and
DynamicVLA p99. FoldQuant W4A4 stays the slowest pi0.5 arm on the H100,
which has no INT4 tensor cores.

`latency.csv` holds the rows with each record's host load and CPU model.