"""Single-request OpenPI inference with episode-scoped diffusion noise."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


# Stable wire keys, shared with historical batched servers.
REQUEST_KEY = "_mujoco_dynamic_request"
RESPONSE_KEY = "mujoco_dynamic_request"


def deterministic_noise(
    request_identity: dict[str, Any], *, action_horizon: int, action_dim: int
) -> np.ndarray:
    encoded = json.dumps(request_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal((action_horizon, action_dim), dtype=np.float32)


class SingleRequestPolicy:
    """Use OpenPI's public inference API, without padding or a batching queue.

    Seeds make policy randomness independent of arrival order. They do not
    promise bitwise equality across GPU kernels or devices.
    """

    def __init__(self, policy: Any, *, action_horizon: int, action_dim: int) -> None:
        self.policy = policy
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        observation = dict(observation)
        identity = observation.pop(REQUEST_KEY)
        if not isinstance(identity, dict):
            raise ValueError(f"{REQUEST_KEY} must be a request identity")
        noise = deterministic_noise(identity, action_horizon=self.action_horizon, action_dim=self.action_dim)
        result = dict(self.policy.infer(observation, noise=noise))
        result[RESPONSE_KEY] = {**identity, "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest()}
        return result


def request_seed(identity: dict[str, Any]) -> int:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


RNG_SCHEME = "sha256-request-python-numpy-torch-v1"


def validate_response_identity(response: dict[str, Any], identity: dict[str, Any], *, backend: str = "openpi") -> None:
    audit = response.get(RESPONSE_KEY)
    valid = isinstance(audit, dict) and all(audit.get(key) == value for key, value in identity.items())
    if valid and backend in ("gr00t", "molmoact2"):
        valid = audit.get("seed") == request_seed(identity) and audit.get("rng_scheme") == RNG_SCHEME
    elif valid:
        valid = isinstance(audit.get("noise_sha256"), str) and len(audit["noise_sha256"]) == 64
    if not valid:
        raise RuntimeError("policy response identity does not match its request")


def validate_actions(actions: Any, horizon: int) -> np.ndarray:
    chunk = np.asarray(actions)
    if chunk.shape != (horizon, 7) or not (np.issubdtype(chunk.dtype, np.floating) or np.issubdtype(chunk.dtype, np.integer)) or not np.isfinite(chunk).all():
        raise ValueError(f"policy actions must be finite with shape ({horizon}, 7); got {chunk.shape}")
    return chunk
