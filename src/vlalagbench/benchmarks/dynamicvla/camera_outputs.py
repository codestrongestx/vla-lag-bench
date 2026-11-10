"""Choose which camera annotators DynamicVLA environments render."""
from __future__ import annotations

import os
from functools import wraps
from typing import Any

ENV_CAMERA_OUTPUTS = "VLALAGBENCH_DYNAMICVLA_CAMERA_OUTPUTS"
RGB_ONLY = "rgb"
UPSTREAM = "upstream"
_active: str | None = None


def simulator_camera_outputs() -> str:
    """Evaluation keeps upstream annotators unless RGB-only rendering is requested.

    RGB-only rendering is experimental: replays are not byte-deterministic, and
    its equivalence to upstream rendering has not been established.
    """

    value = os.environ.get(ENV_CAMERA_OUTPUTS, UPSTREAM)
    if value not in (RGB_ONLY, UPSTREAM):
        raise ValueError(f"{ENV_CAMERA_OUTPUTS} must be {RGB_ONLY} or {UPSTREAM}")
    return value


def recorded_camera_outputs(versions: dict[str, str]) -> str:
    """Traces without the field were recorded with upstream annotators."""

    value = versions.get("renderer_camera_outputs", UPSTREAM)
    if value not in (RGB_ONLY, UPSTREAM):
        raise ValueError(f"unsupported recorded camera outputs: {value}")
    return value


def active_camera_outputs() -> str | None:
    return _active


def install_camera_outputs(upstream: Any, mode: str) -> None:
    """Drop unused camera annotators when building environment configs.

    Evaluation consumes only RGB. Upstream test environments also request
    semantic segmentation, which is rendered every step and then discarded.
    """

    global _active
    if mode not in (RGB_ONLY, UPSTREAM):
        raise ValueError(f"unknown camera outputs mode: {mode}")
    if _active is not None and _active != mode:
        raise ValueError(f"camera outputs already installed as {_active}")
    _active = mode
    original = upstream._get_env_cfg
    if getattr(original, "_vlalagbench_camera_outputs", False) or mode == UPSTREAM:
        return

    @wraps(original)
    def configured(source, *args, **kwargs):
        cfg = original(source, *args, **kwargs)
        for value in vars(cfg.scene).values():
            data_types = getattr(value, "data_types", None)
            if type(value).__name__ == "CameraCfg" and data_types is not None:
                if "rgb" not in data_types:
                    raise ValueError("DynamicVLA cameras must render rgb")
                value.data_types = ["rgb"]
        return cfg

    configured._vlalagbench_camera_outputs = True
    upstream._get_env_cfg = configured
