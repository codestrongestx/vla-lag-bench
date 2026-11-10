# GR00T fixed-placement drawer speed sweep

Rerun of the GR00T N1.7 drawer cohort (`gr00t_speed_sweep_repeat_20260915`,
specs `configs/studies/gr00t_speed_sweep_20260914/drawer_ellipse_600*.yaml`)
on the fixed world placement from commit `d1e43e4b`: the cabinet ellipse sits
at (0.044606, -0.25403, 0.905) for all 50 states. It pairs with the pi0.5
rerun in `configs/studies/pi05_drawer_fixed_20260926`.

Nine speeds (0.001, 0.025–0.2 by 0.025 m/s); synchronous and lazy at 0/100/200
ms, continuous at 100/200 ms: 72 points × 200 episodes = 14,400 episodes.

Unchanged from the 20260914 GR00T drawer specs: checkpoint
`nvidia/GR00T-N1.7-LIBERO` at revision `2ea293aa…`, subdir `libero_goal`,
embodiment `LIBERO_PANDA`; chunk 16, horizon 5, 20 Hz, 10 warm-up steps, 600
max steps; seeds 7/7/7; eager PyTorch. Points run speed ascending, then
latency ascending.

- Synchronous and lazy are separate studies (the original combined them), so
  two GPUs can share the work.
- Serving uses GR00T dynamic batching (up to 32 requests), the setup verified
  on the Task 9 GR00T H100 run
  (equivalent to single-request serving within noise; per-request seeded noise
  is preserved).
