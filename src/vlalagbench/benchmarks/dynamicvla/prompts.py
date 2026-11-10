"""Deterministic literal prompts for official DynamicVLA instruction metadata."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any


PROMPT_SCHEME = "sha256-policy-task-environment-replica-prompt-v1"
PICK_ACTIONS = ("pick up", "grasp", "catch", "grab", "get hold of")
PLACE_ACTIONS = (
    "place on",
    "put on",
    "set on",
    "position on",
    "return to",
    "deposit in",
)


def derive_prompt_seed(
    policy_seed: int, task_name: str, environment_sha256: str, replica_id: int
) -> int:
    payload = json.dumps(
        {
            "scheme": PROMPT_SCHEME,
            "policy_seed": policy_seed,
            "task_name": task_name,
            "environment_sha256": environment_sha256,
            "replica_id": replica_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def literal_prompt(instruction: Any, prompt_seed: int) -> str:
    if isinstance(instruction, str):
        if not instruction:
            raise ValueError("literal DynamicVLA instruction must not be empty")
        return instruction
    if not isinstance(instruction, dict):
        raise ValueError("DynamicVLA instruction must be a string or metadata object")
    task = instruction.get("task")
    objects = instruction.get("objects", [""])
    containers = instruction.get("containers", [""])
    if (
        task not in {"pick", "place", "long-horizon"}
        or not isinstance(objects, list)
        or not objects
        or not all(isinstance(value, str) for value in objects)
        or not isinstance(containers, list)
        or not containers
        or not all(isinstance(value, str) for value in containers)
    ):
        raise ValueError("invalid official DynamicVLA instruction metadata")
    generator = random.Random(prompt_seed)
    pick_action = generator.choice(PICK_ACTIONS)
    place_action = generator.choice(PLACE_ACTIONS)
    object_description = generator.choice(objects)
    container_description = generator.choice(containers)
    if task == "pick":
        return f"{pick_action} the {object_description}."
    return (
        f"{pick_action} the {object_description} and "
        f"{place_action} the {container_description}."
    )
