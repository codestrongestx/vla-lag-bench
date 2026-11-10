# DynamicVLA on DOM (Isaac Sim)

The cross-simulator check runs the released `dynamic-vla-DOM` checkpoint of
[DynamicVLA](https://github.com/hzxie/DynamicVLA) on its Dynamic Object
Manipulation (DOM) benchmark, through the same study runner, scheduler,
verification, and result format as LIBERO. Integration is applied in memory;
the upstream checkout stays untouched and clean.

| Interaction | Task | Studies | Published in |
| --- | --- | --- | --- |
| Release | drop the food can into a moving box | `fcan04_native_pilot20_20260928`, `fcan04_native_remaining180_20260928`, `fcan04_native200_speed_extension_20261001` | `results/dynamicvla/fcan04_native200_20261001` |
| Grasping | pick the can off an invisible moving carrier, place it in a static box | `fcan04_moving_can_pilot20_20261008`, `fcan04_moving_can_remaining180_20261008` | `results/dynamicvla/fcan04_moving_can200_20261008` |

Each task has 200 conditions (can or box placement, path start phase, and
direction), split into a 20-condition pilot and the remaining 180; a published
point adds the two studies' success counts. Control runs at 25 Hz with chunks
of 20, execution horizon 14, and 750 acting steps. Latencies are 0, 120, and
240 ms, aligned to the control grid.

## Setup

- DynamicVLA source at revision `a27a06d2ca74d0e987a5e552e01013073e93cfd8`
  (`benchmark.upstream_revision`; preflight rejects other revisions and
  untracked files).
- Two environments following upstream's instructions: Isaac Sim 4.5 with
  Isaac Lab for the simulator, and the policy environment.
- The `dynamic-vla-DOM` checkpoint under `$VLALAGBENCH_CHECKPOINTS`; its SHA-256
  is pinned in every experiment.
- The official DOM assets, extracted, then the assets root the studies read:

```bash
python scripts/dynamicvla/prepare_assets.py /path/to/DynamicVLA-assets/extracted assets/dynamicvla
```

It links the official `objects/` and `scenes/` and copies the two derived
environments and the ribbed carrier, checking each against the evaluated
checksum (environment checksums enter every episode's identity).

Edit `configs/placements/dynamicvla.yaml` for the host: installation paths,
GPU, ports, and CPU set. The runner generates each shard's runtime from the
study and placement. Isaac camera rendering needs an RTX-capable GPU.

```bash
uv run vlalagbench study plan configs/studies/fcan04_moving_can_pilot20_20261008/synchronous.yaml \
  --placement configs/placements/dynamicvla.yaml
uv run vlalagbench study run configs/studies/fcan04_moving_can_pilot20_20261008/synchronous.yaml \
  --placement configs/placements/dynamicvla.yaml \
  --select trajectory.speed_mps=0.1 --select inference.latency_ms=0
```

Only one environment per simulator is supported. Independent shards need
separate GPU ownership and port pairs, or `allow_gpu_sharing: true` with
disjoint CPU sets. A shard's `shared_endpoint: tcp://127.0.0.1:PORT` lets
simulators on one GPU share one model backend; histories, request IDs, and
scheduling stay per client and model calls run one at a time.
`kit_tasking_threads` (default 4) caps Isaac's task pool. None of these
placement settings enter the study hash.

`scripts/studies/plot_fcan04_native200_20261001.py` and
`plot_fcan04_moving_can200_20261008.py` pool the pilot and remainder studies
into the published tables, checking study hashes and episode checksums.

## Rendering

Studies pin the original renderer: `render_mode: RaytracedLighting`,
`antialiasing_mode: DLSS`, `dlss_mode: Balanced`, `enable_dl_denoiser: true`,
`enable_dlssg: false`. These fields enter the study hash. The simulator applies
and verifies each setting at launch and after environment creation, because
Isaac can overwrite them with its preset; a mismatch fails the run before any
observation is captured. Effective settings are saved in every trace.

## Timing

`src/vlalagbench/timing.py` schedules synchronous, lazy, and continuous inference
for both simulators. Each observation is published once; the simulator waits
for its reply without advancing physics and then advances one control step, so
wall-clock inference time affects throughput, not the injected latency.
Waiting steps hold the last target and do not consume the 750-action budget.
For asynchronous modes, latency steps plus the horizon must fit in the chunk.

Policy noise is seeded by policy seed, task, environment checksum, replica, and
request. Receipts record request and release schedules, applied action
indices, waiting steps, outcomes, and trace checksums, and an independent
protocol auditor checks them. Invalid or timed-out episodes keep their
failure evidence and cannot become accepted results.

## Traces, replay, and continuation

Each episode writes `dynamicvla_taskNN_epNNN.npz`, bound to its receipt by
SHA-256: T+1 Isaac states, the end-effector state sent to the policy, T
actions, outcomes, scheduler snapshots, and pending inference. Traces hold no
camera frames (`trace_images: false`), about 1 MB per episode.

```bash
uv run vlalagbench dynamicvla-trace TRACE.npz studies/STUDY/runtimes/shard00.yaml \
  --video replay.mp4 --verify --rejudge
uv run vlalagbench dynamicvla-continue TRACE.npz studies/STUDY/runtimes/shard00.yaml continued.npz \
  --additional-steps 250
```

Replay re-renders saved states; Isaac keeps DLSS, denoiser, and antialiasing
history outside the state, so frames are not guaranteed pixel-identical to what
the policy saw. Verification re-simulates recorded actions and reports drift
against a tolerance; rejudging evaluates the pinned termination functions on
recorded states. `study videos` replays three successes and three failures per
point (`--runtime` for a copied study on another machine).

Continuation restores physical, control, and RNG state, observation history,
scheduler state, and pending chunks, and extends a failed episode by at most
`--additional-steps` control steps. History images are re-rendered from
states, so a continuation is exact in state and protocol but not in the
policy's first inputs; the output records this.
