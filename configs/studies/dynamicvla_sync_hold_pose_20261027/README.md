# Synchronous DynamicVLA reruns with the arm holding its pose during waits

Every synchronous point with a nonzero delay on both DOM tasks. The first runs
re-sent the last absolute pose target on every wait step, so the hand kept
moving toward it (median 28 mm over a 240 ms wait). These reruns send the
hand's current pose with the last gripper command from the first wait step
after a model action, which is now the only wait behaviour. Same checkpoint,
renderer settings, K = 20, H = 14 and 750 active actions as the runs they
replace.

| Spec | Task | Episodes | Delays (ms) | Speeds (m/s) |
| --- | --- | --- | --- | --- |
| `box.yaml` | Release (moving box) | 200 (speed-extension bank) | 120, 240 | 0.001–0.4 |
| `can_pilot20_120.yaml`, `can_remaining180_120.yaml` | Grasping (moving can) | 20 + 180 | 120 | 0.001–0.2 |
| `can_pilot20_240_slow.yaml`, `can_remaining180_240_slow.yaml` | Grasping | 20 + 180 | 240 | 0.025–0.1 |
| `can_pilot20_240.yaml`, `can_remaining180_240.yaml` | Grasping | 20 + 180 | 240 | 0.001, 0.125–0.2 |

As with the other moving-can studies, add the 20- and 180-episode success
counts per point. Evaluated on RTX 3090s, October 27–29, 2026. The specs no
longer set `inference.idle_control: hold_pose`, the opt-in setting they were
evaluated with; `../evaluated_hashes.yaml` records their study hashes. The two
`*_240_slow` studies also ran the old wait as a second arm, which their specs
leave out.

Results: the `dynamicvla-sync-hold-pose` folders under `results/dynamicvla/`.
