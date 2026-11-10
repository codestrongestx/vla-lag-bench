"""Optional idle pacing around the unchanged upstream streaming inference loop."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

ENV_BACKOFF_MS = "VLALAGBENCH_DYNAMICVLA_IDLE_BACKOFF_MS"


def backoff_seconds(value: str) -> float:
    seconds = float(value) / 1000
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{ENV_BACKOFF_MS} must be a positive finite number")
    return seconds


class IdleInput:
    def __init__(self, source: Any, seconds: float):
        self.source = source
        self.seconds = seconds

    def get(self, key):
        value = self.source.get(key)
        if value is None:
            time.sleep(self.seconds)
        return value

    def clear(self):
        return self.source.clear()


def paced_inference_loop(pretrained_model, config, q_in, q_out):
    """Spawn imports the pristine vendor class, retaining its original no-grad loop."""
    from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy

    seconds = backoff_seconds(os.environ[ENV_BACKOFF_MS])
    return DynamicVLAPolicy._inference_loop(
        pretrained_model, config, IdleInput(q_in, seconds), q_out
    )


def with_idle_backoff(upstream: Any, get_model: Callable, *args, **kwargs):
    backoff_seconds(os.environ[ENV_BACKOFF_MS])
    checkpoint = Path(args[0] if args else kwargs["pretrained_model"])
    model_type = json.loads((checkpoint / "config.json").read_text())["type"]
    model_class = upstream.utils.helpers.get_policy_class(model_type)
    original = model_class.__dict__["_inference_loop"]
    model_class._inference_loop = staticmethod(paced_inference_loop)
    try:
        return get_model(*args, **kwargs)
    finally:
        model_class._inference_loop = original
