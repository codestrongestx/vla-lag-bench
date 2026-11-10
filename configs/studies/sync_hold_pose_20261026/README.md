# Synchronous LIBERO reruns with the arm holding its pose during waits

Every synchronous point with a nonzero delay behind the paper's four LIBERO
tasks. The first runs of these points stepped each inference-wait tick with a
zero action; under robosuite's delta OSC that resets the position goal but
keeps the previous orientation goal, so the arm kept turning toward its last
commanded orientation for the whole wait (about 2° and 5 mm per 200 ms in the
π0.5 drawer traces). These reruns hold the pose where each wait began, which
is now the only wait behaviour. Synchronous 0 ms never waits and was not
rerun.

Each study matches the cohort it replaces except the wait: same base
experiment, speeds 0.001 and 0.025–0.2 m/s by 0.025, 200 episodes per point
over the 50 official states with the four phase/direction conditions, 20 Hz,
10 warm-up steps, execution horizon 5, 600 acting steps, seeds 7/7/7, batched
serving (up to 32 requests per pass).

| Policy | Delays (ms) | Replaces the synchronous points of | Episodes |
| --- | --- | --- | ---: |
| π0.5 (`pi05/`) | 50, 100, 150, 200, 250 | `pi05_basket_20260908`, `pi05_drawer_fixed_20260926`, `libero_spatial_task9_20261001`, `libero_spatial_task9_bowl_centered_20261010`, `bowl_speed_fill_20261021`, `libero_latency_grid_20261014` | 36,000 |
| GR00T N1.7 (`gr00t/`) | 100, 200 | `gr00t_speed_sweep_20260914`, `gr00t_drawer_fixed_20261004`, `libero_spatial_task9_20261001`, `libero_spatial_task9_bowl_centered_20261010`, `bowl_speed_fill_20261021` | 14,400 |
| MolmoAct2 (`molmoact2/`) | 50, 100, 150, 200, 250 | `molmoact2_libero_20261015`, `bowl_speed_fill_20261021` | 36,000 |

Evaluated on H100s, October 26–27, 2026. The specs no longer set
`inference.idle_control: hold_pose`, the opt-in setting they were evaluated
with; `../evaluated_hashes.yaml` records their study hashes.

```bash
uv run vlalagbench study run configs/studies/sync_hold_pose_20261026/pi05/drawer.yaml \
  --placement configs/placements/libero_batched.yaml
```

Results: the `*-sync-hold-pose` folders of each task under `results/libero/`.
