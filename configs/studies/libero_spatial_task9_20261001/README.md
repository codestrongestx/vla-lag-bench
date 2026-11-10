# LIBERO-Spatial Task 9: moving plate (set-down)

The moving plate (`libero_spatial_task9_bowl_to_plate/plate_ellipse_paired_200.yaml`)
at 0.001 and 0.025–0.2 m/s by 0.025. Synchronous runs use 0, 100, and 200 ms;
lazy and continuous use 100 and 200 ms. The 0.001 m/s point is near-static,
not stationary.

Every point uses the task's 200-episode schedule over all 50 official initial
states, seeds 7, 20 Hz control, 10 warm-up steps, execution horizon 5, and a
600-acting-step limit. pi0.5 uses action chunks of 10; GR00T uses its required
chunks of 16 and the LIBERO-Spatial checkpoint.

| Directory | Policy and serving | Published in |
| --- | --- | --- |
| `pi05/` | pi0.5, one request per pass (RTX 3090) | `results/libero/20261001_task9-moving-plate/pi05-xtra3090-single/` |
| `pi05_b32/` | pi0.5, batches of up to 32 (H100) | `results/libero/20261001_task9-moving-plate/pi05-h100-b32/` |
| `gr00t/` | GR00T N1.7, batches of up to 32 | `results/libero/20261001_task9-moving-plate/gr00t/` |

Batched serving gives each request its own seeded noise; on the moving plate
it reproduced single-request results within seed-to-seed noise.
