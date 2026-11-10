# Configurations

`schema.yaml` documents every field. An **experiment** defines one condition
(task, initial states, target motion, policy, inference mode and latency,
seeds); a **study** sweeps experiment fields; a **placement** says where a study
runs; a **runtime** configures a standalone `vlalagbench eval`.

## Tasks

| Experiment | Paper task | Target motion |
| --- | --- | --- |
| `experiments/libero/libero_object_task1_native_basket_ellipse/basket_ellipse_paired_200.yaml` | Release | The basket on a fixed world ellipse (center (-0.144, 0.327) m, radii (0.11, 0.0698) m); only the basket's start position is overridden. |
| `experiments/libero/libero_spatial_task9_bowl_to_plate/plate_ellipse_paired_200.yaml` | Set-down | The plate on a closed path clear of the scene's other objects. |
| `experiments/libero/libero_spatial_task9_bowl_to_plate/cabinet_ellipse_paired_200.yaml` | Grasping | The cabinet carrying the bowl, centered on its top; friction carries the bowl (`slide_joints`). |
| `experiments/libero/libero_goal_task0_moving_drawer/drawer_ellipse_paired_200.yaml` | Articulated | The whole cabinet; the drawer's damped prismatic joint stays physical. |
| `experiments/dynamicvla/fcan04_compressed_path_20260928/candidate-200.yaml` | Release (DOM) | The box on a compressed ellipse; source of both box studies. |
| `experiments/dynamicvla/fcan04_moving_can_20261006/ribbed-experiment.yaml` | Grasping (DOM) | An invisible ribbed carrier under the can; source of both moving-can studies. |

Every LIBERO task uses the 50 official initial states, two path start phases
per state, and both directions: 200 conditions, episode `i`, `i+50`, `i+100`,
`i+150` sharing state `i` (`vlalagbench.authoring.paired_trajectory_schedule`).
Targets are driven in X/Y position and velocity at every physics substep;
height, rotation, and contacts stay physical. `release_manifest.json` and
`layout_validation.json` record the full-cycle contact audits of the basket
and drawer layouts. The basket audit reruns with the command below, which
checks LIBERO-Object closed-path tasks only:

```bash
uv run python -m vlalagbench.benchmarks.libero.suite CONFIG.yaml --output audit.json
```

## Studies behind the published results

| Results folder | Studies |
| --- | --- |
| `libero/20260908_moving-basket/pi05` | `pi05_basket_20260908`: `native_ellipse_600_*` (0.001–0.2 m/s), `extension_20260911_*` (0.4–0.6 m/s; their 0.3 m/s points are not in the published table), `highspeed_20260912_*` (0.8–1.5 m/s) |
| `libero/20260908_moving-basket/gr00t` | `gr00t_speed_sweep_20260914/basket_*` |
| `libero/20260926_moving-drawer-fixed/{pi05,gr00t}` | `pi05_drawer_fixed_20260926`, `gr00t_drawer_fixed_20261004` |
| `libero/20261001_task9-moving-plate/{pi05-*,gr00t}` | `libero_spatial_task9_20261001` |
| `libero/20261010_task9-moving-cabinet-bowl-centered` | `libero_spatial_task9_bowl_centered_20261010`, `bowl_speed_fill_20261021` |
| `*/pi05-latency-grid` | `libero_latency_grid_20261014` |
| `*/molmoact2`, `*/molmoact2-speed-fill` | `molmoact2_libero_20261015`, `bowl_speed_fill_20261021` |
| `libero/20261001_task9-moving-plate/pi05-flashrt-bf16` | `flashrt_setdown_grid_20261018` |
| `libero/*/*-sync-hold-pose`, `dynamicvla/*/dynamicvla-sync-hold-pose` | synchronous points at nonzero delay: `sync_hold_pose_20261026`, `dynamicvla_sync_hold_pose_20261027`; the rest from the studies of the folders they name |
| `libero/20261013_libero-10-90-official` | `libero_10_90_official_20261013` |
| `libero/20261014_pi05-inference-stacks` | `pi05_stacks_20261014` (JAX and FlashRT bf16 specs) |
| `dynamicvla/fcan04_native200_20261001` | `fcan04_native_pilot20_20260928`, `fcan04_native_remaining180_20260928`, `fcan04_native200_speed_extension_20261001` |
| `dynamicvla/fcan04_moving_can200_20261008` | `fcan04_moving_can_pilot20_20261008`, `fcan04_moving_can_remaining180_20261008` |

Specs that write a local checkpoint as `$VLALAGBENCH_CHECKPOINTS/<name>` hash
differently from their evaluated studies, which named an absolute path, and the
synchronous reruns were evaluated with the opt-in setting
`inference.idle_control: hold_pose` that their specs no longer state.
`studies/evaluated_hashes.yaml` lists each one's evaluated path and setting and
its recorded study hash, and `scripts/studies/evaluated_hashes.py` recomputes
them.

`studies/native_ellipse_600.yaml` and `studies/dynamicvla_screen.yaml` are small
examples used by the tests.

## Placements

`placements/libero.yaml` runs four environment workers; `libero_batched.yaml`
is the paper's H100 layout (one batched server, 48 environment workers,
rendering limited to policy frames); `dynamicvla.yaml` holds the Isaac and
policy installations. Placements never change a trajectory and do not enter
the study hash, so the same study can be resumed on a different layout.
