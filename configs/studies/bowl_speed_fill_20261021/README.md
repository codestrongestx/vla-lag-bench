# Moving bowl: in-between speeds

The moving bowl (`libero_spatial_task9_bowl_to_plate/cabinet_ellipse_paired_200.yaml`,
bowl centered) was swept at 0.001 and 0.05–0.4 m/s in steps of 0.05. The
other three paper tasks use 0.001 and 0.025–0.2 m/s in steps of 0.025. Above
about 0.2 m/s bowl success stops falling: the looping bowl comes back to a
waiting hand, so those speeds measure interception. Capped at 0.2 m/s the
bowl keeps five speeds, too few to fit the falloff. These studies add
0.025, 0.075, 0.125 and 0.175 m/s so the bowl gets the same nine-speed grid.

| Model | Synchronous (ms) | Lazy, continuous (ms) | Points |
| --- | --- | --- | --- |
| pi0.5 | 0, 50, 100, 150, 200, 250 | 50, 100, 150, 200, 250 | 64 |
| GR00T N1.7 | 0, 100, 200 | 100, 200 | 28 |
| MolmoAct2 | 0, 50, 100, 150, 200, 250 | 50, 100, 150, 200, 250 | 64 |

156 points × 200 episodes = 31,200 episodes. Every other setting copies the
runs being filled (`libero_spatial_task9_bowl_centered_20261010`,
`libero_latency_grid_20261014`, `molmoact2_libero_20261015`): 20 Hz, 10
warm-up steps, 600 acting steps, execution horizon 5, seeds 7/7/7, dynamic
batching up to 32. Traces record states only.
