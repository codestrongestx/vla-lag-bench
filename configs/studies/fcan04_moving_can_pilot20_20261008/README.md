# Moving can (ribbed): random-20 pilot

The second DynamicVLA task, the food can carried by the invisible ribbed
carrier (`configs/experiments/dynamicvla/fcan04_moving_can_20261006/`), on the
same protocol as the food-can moving-box study
(`../fcan04_native_pilot20_20260928`, `../fcan04_native_remaining180_20260928`):
checkpoint, renderer settings, K=20, H=14, 750 active actions, latencies and
inference modes. Only the speed grid differs: it is the LIBERO grid.

| Mode | Latency (ms) | Speeds (m/s) |
|---|---|---|
| synchronous | 0, 120, 240 | 0.001, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2 |
| lazy | 120, 240 | same |
| continuous | 120, 240 | same |

63 cells × 20 episodes = 1,260 episodes.

`base.yaml` holds 20 episodes of the ribbed 200-episode bank
(`ribbed-experiment.yaml`), chosen by `split.py` with the food-can pilot's
recipe and seed 20261008: 20 distinct box placements sampled from `range(50)`,
plus shuffled offsets `[0, 50, 100, 150] * 5`, so each start-phase half and
direction contributes five episodes. Each keeps its bank box placement, phase
and direction; local state indices are remapped to 0–19 in
`selection.json` order. Policy seed is 0.

The complementary 180 episodes are in `../fcan04_moving_can_remaining180_20261008`.
Combine per `(mode, latency_ms, speed_mps)` cell by adding success counts and
dividing by 200 (equivalently `0.1 * pilot_rate + 0.9 * remaining_rate`).
