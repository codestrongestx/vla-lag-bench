from __future__ import annotations

from vlalagbench.benchmarks.dynamicvla.prompts import (
    derive_prompt_seed,
    literal_prompt,
)


def test_official_instruction_metadata_maps_to_paired_literal_prompt():
    metadata = {
        "task": "place",
        "objects": ["red apple", "apple"],
        "containers": ["white box", "box"],
    }
    seed = derive_prompt_seed(0, "MG", "a" * 64, 3)

    assert seed == 7542329097118177879
    assert literal_prompt(metadata, seed) == "grasp the apple and set on the box."
    assert literal_prompt(metadata, seed) == literal_prompt(metadata, seed)


def test_explicit_literal_prompt_is_preserved():
    assert literal_prompt("literal prompt", 123) == "literal prompt"
