# Native-friction 20-condition pilot

The base uses 20 distinct can placements sampled from the corrected-path
200-condition suite in
`configs/experiments/dynamicvla/fcan04_compressed_path_20260928/candidate-200.yaml`.
The derived close-box environment remains the pilot scene.

Selection uses Python `random.Random(20260928)`: first sample 20 placement
indices without replacement from `range(50)`, then shuffle
`[0, 50, 100, 150] * 5` and add those offsets to the sampled indices.
The four offsets select the source schedule's two phase halves and two
travel directions, with five conditions per combination. Each selected
placement retains its source condition's phase and direction. Local state
indices are remapped to 0–19 in the following episode order:

```text
93, 135, 117, 194, 152, 13, 56, 28, 172, 139, 62, 21, 82, 145, 179, 151, 16, 75, 49, 134
```

All speed, latency, and inference-mode cells share this selection. There are
10 conditions per direction, with starting phases spanning both halves of
the path. This is a stratified random subset, not an independent uniform
sample of 20 source replicas. Policy seed remains 0; remapped episode IDs
do not preserve the full suite's per-episode policy noise identity.

The study names use `random20` so these results have separate directories
from the earlier `first20` selection. The sweep still contains 49 cells and
980 total episodes, each with a 750-active-action budget (30 seconds at 25 Hz).

The complementary 180 conditions are in
`../fcan04_native_remaining180_20260928`. Its `selection.json` maps both
subsets to source episode IDs; its README explains how to combine results.
