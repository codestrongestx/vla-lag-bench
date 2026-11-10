# LIBERO-10 and LIBERO-90, official initial states

Static LIBERO-10 (10 tasks) and LIBERO-90 (90 tasks) for pi0.5 and GR00T at
synchronous inference with 0 ms latency: 50 episodes per task, one per
official initial state, 5,000 episodes per model. Step limits follow the
openpi LIBERO example (520 for LIBERO-10, 400 for LIBERO-90) after 10 warmup
steps; seeds 7/7/7.

Neither model was trained on LIBERO-90. `pi05_libero` was trained on the
LIBERO-10, Goal, Object and Spatial demos, and GR00T-N1.7-LIBERO releases one
checkpoint per suite (`libero_10`, `libero_goal`, `libero_object`,
`libero_spatial`), none for LIBERO-90. LIBERO-90 results are therefore on
unseen tasks. The `gr00t_libero90_pilot` studies run each GR00T checkpoint on
LIBERO-90 at the first five initial states per task (450 episodes each) to
choose the checkpoint for the full GR00T LIBERO-90 study.

Policies are served exactly as in the H100 Task 9 runs: pi0.5 batched up to 32
(max-autotune), GR00T batched up to 32 (eager).

## GR00T LIBERO-90 checkpoint pilot

First five official initial states of every LIBERO-90 task, 450 episodes per
checkpoint (01:55–02:22 UTC, October 13):

| GR00T checkpoint | Successes |
| --- | --- |
| `libero_10` | 72/450 (16.0%) |
| `libero_goal` | 6/450 (1.3%) |
| `libero_spatial` | 6/450 (1.3%) |
| `libero_object` | 5/450 (1.1%) |

`gr00t/libero_90.yaml` therefore runs the `libero_10` checkpoint.
