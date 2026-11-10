# Remaining 180 existing conditions

`base.yaml` contains exactly the 180 existing conditions not selected by
`../fcan04_native_pilot20_20260928`. Every condition retains its combination
of can position, cup path phase, and travel direction from
`configs/experiments/dynamicvla/fcan04_compressed_path_20260928/candidate-200.yaml`.
No conditions are regenerated.

Conditions are ordered by ascending source episode ID. Local state indices
are remapped to 0–179, with each placement entry paired with the trajectory
schedule entry at the same index. The 180 entries reuse the source's 50 can
positions; repeated can positions are valid and do not imply duplicate full
conditions. Use the validator fix that allows repeated placement values
across distinct state indices. No new state-selection mechanism is needed.

`selection.json` maps local zero-based episode IDs (array indices) to source
episode IDs (array values), for both the pilot and remainder. Together the
two arrays cover source IDs 0–199 exactly once. Keep the study identity
alongside local episode IDs when joining results.

The remainder matches the pilot's derived close-box scene, checkpoint, path,
native friction, renderer settings, 750-active-action budget, seven speeds,
and latency/mode combinations. There are 49 sweep cells and 8,820 episodes;
including the pilot gives 9,800 episodes, with 200 conditions per cell.
Each source phase-half/direction stratum contributes 45 remaining conditions
and five pilot conditions.

For each matching `(inference mode, latency_ms, speed_mps)` cell, add the
pilot and remainder success counts, then divide by 200 once all episodes
are complete. Equivalently, use `0.1 * pilot_rate + 0.9 * remaining_rate`.
Do not average the two rates equally. As with the pilot, policy seed is 0
but local episode IDs are remapped; this preserves the existing condition
coverage, not the original full suite's per-episode policy noise identity.
