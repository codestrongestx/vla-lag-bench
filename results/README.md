# Results

Published success tables behind the paper. Each folder holds, per model, a
`plot_data.csv` with one row per verified sweep point (successes, episodes,
Wilson 95% interval, run ID, and evaluating commit), the study's three-panel
`success_rates.png`, and, where a sweep pools several studies, a provenance
record. Traces, videos, and per-episode rows are not committed; rerunning a
study writes them under its `output_root`.

`tests/test_results_hygiene.py` keeps these folders to compact conclusion
files.

Synchronous execution holds the arm still while it waits for inference. The
first runs of every synchronous point with a nonzero delay did not (LIBERO
stepped a zero action, which lets robosuite's controller keep turning the arm;
DynamicVLA re-sent the last pose target), so those points were rerun with the
arm held and only the reruns are published. Each task's
`<model>-sync-hold-pose/` folder (`dynamicvla-sync-hold-pose/` for DynamicVLA)
holds the full grid the paper reports: the reruns plus the original
synchronous 0 ms, lazy and continuous points. The other folders keep each
run's remaining points; synchronous points with no rerun (pi0.5 basket above
0.2 m/s, bowl above 0.2 m/s on the October 10 grid, the RTX 3090 and FlashRT
set-down runs) are not published.

## LIBERO

Paper task names: release = moving basket, set-down = moving plate,
grasping = moving bowl (carried by the moving cabinet), articulated = moving
drawer.

| Folder | Task | Models | Paper grid success (`*-sync-hold-pose`) |
| --- | --- | --- | --- |
| [`20260908_moving-basket`](libero/20260908_moving-basket/README.md) | Release: Object Task 1, basket on its ellipse | `*-sync-hold-pose`; `pi05`, `pi05-latency-grid`, `gr00t`, `molmoact2` | pi0.5 67.0%, GR00T 36.7%, MolmoAct2 71.6% |
| [`20260926_moving-drawer-fixed`](libero/20260926_moving-drawer-fixed/README.md) | Articulated: Goal Task 0, drawer of the moving cabinet | `*-sync-hold-pose`; `pi05`, `pi05-latency-grid`, `gr00t`, `molmoact2` | pi0.5 27.3%, GR00T 19.2%, MolmoAct2 31.7% |
| [`20261001_task9-moving-plate`](libero/20261001_task9-moving-plate/README.md) | Set-down: Spatial Task 9, plate moving | `*-sync-hold-pose`; `pi05-xtra3090-single` (RTX 3090, single requests), `pi05-h100-b32`, `pi05-latency-grid`, `pi05-flashrt-bf16`, `gr00t`, `molmoact2` | pi0.5 51.7%, GR00T 62.4%, MolmoAct2 55.8% |
| [`20261010_task9-moving-cabinet-bowl-centered`](libero/20261010_task9-moving-cabinet-bowl-centered/README.md) | Grasping: Spatial Task 9, bowl centered on the moving cabinet | `*-sync-hold-pose`; `pi05`, `pi05-latency-grid`, `gr00t`, `molmoact2`, `*-speed-fill` (0–0.2 m/s grid) | pi0.5 70.9%, GR00T 66.8%, MolmoAct2 63.3% |
| [`20261013_libero-10-90-official`](libero/20261013_libero-10-90-official/README.md) | Static LIBERO-10 and LIBERO-90 (task selection) | `pi05`, `gr00t` | LIBERO-10: pi0.5 92.8%, GR00T 89.2%; LIBERO-90: 28.3%, 16.8% |
| [`20261014_pi05-inference-stacks`](libero/20261014_pi05-inference-stacks/README.md) | pi0.5 inference stacks on the four tasks, and batch-1 latency per GPU | `pi05` stacks | JAX bf16 71.6%; FlashRT bf16 within about ±2 points |
| [`gr00t_speed_sweep_20260914`](libero/gr00t_speed_sweep_20260914/README.md) | Provenance of the GR00T moving-basket sweep | `gr00t` | see `20260908_moving-basket/gr00t` |

## DynamicVLA (Isaac Sim, DOM)

| Folder | Task | Paper grid (`dynamicvla-sync-hold-pose`) |
| --- | --- | --- |
| [`fcan04_native200_20261001`](dynamicvla/fcan04_native200_20261001/README.md) | Release: food can into a moving box, 200 conditions per setting | 7,183 of 14,000 (51.3%) |
| [`fcan04_moving_can200_20261008`](dynamicvla/fcan04_moving_can200_20261008/README.md) | Grasping: food can on an invisible moving carrier, 200 conditions per setting | 5,502 of 12,600 (43.7%) |

Each pools a 20-condition pilot study and a 180-condition remainder study by
adding success counts per setting; `sources.json` lists every contributing
run.
