"""Offline helpers for authoring paired trajectory conditions."""

import hashlib
import random


def paired_trajectory_schedule(
    family: str, *, seed: int = 7, layout_count: int = 50
) -> list[dict[str, float | int]]:
    """Pair two stratified phases with both directions for each base layout.

    Layout i uses episodes i, i + N, i + 2N, i + 3N. The family is a seed
    namespace; keep it unchanged when comparing paths, speeds, or latencies
    with an existing schedule (the released native ellipses use ``ellipse``).
    """
    if layout_count < 1:
        raise ValueError("layout_count must be positive")
    point_count = layout_count * 2
    design_rng = random.Random(_stable_seed(seed, family, -1))
    phases = [
        round((phase_bin + design_rng.random()) / point_count, 12)
        for phase_bin in range(point_count)
    ]
    phase_pairs = [
        (phases[index], phases[index + layout_count])
        for index in range(layout_count)
    ]
    design_rng.shuffle(phase_pairs)
    schedule: list[dict[str, float | int]] = [{} for _ in range(layout_count * 4)]
    for layout_index, paired_phases in enumerate(phase_pairs):
        rng = random.Random(_stable_seed(seed, family, layout_index))
        variants = [
            {"initial_phase": phase, "initial_direction": direction}
            for phase in paired_phases
            for direction in (1, -1)
        ]
        rng.shuffle(variants)
        for repeat_index, variant in enumerate(variants):
            schedule[layout_index + repeat_index * layout_count] = variant
    return schedule


def _stable_seed(seed: int, family: str, layout_index: int) -> int:
    payload = f"{seed}:{family}:{layout_index}:paired-trajectory".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
