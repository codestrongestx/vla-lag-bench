"""Pinned GR00T public-policy adapter for LIBERO's simulator conventions."""

from __future__ import annotations

import contextlib
import json
import random
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from vlalagbench.benchmarks.libero.policy import (
    REQUEST_KEY, RESPONSE_KEY, RNG_SCHEME, request_seed, validate_actions,
)
from vlalagbench.config import Gr00tModelSpec
from vlalagbench.files import sha256

UPSTREAM_REVISION = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
CHECKPOINT_REVISION = "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21"
AXES = ("x", "y", "z", "roll", "pitch", "yaw")
_RNG_LOCK = threading.Lock()


def format_observation(obs: dict[str, Any], language: str) -> dict[str, Any]:
    quat = np.array(obs["robot0_eef_quat"], dtype=np.float64, copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] ** 2)
    angle = np.zeros(3) if den == 0 else quat[:3] * 2 * np.arccos(quat[3]) / den
    pose = np.concatenate((obs["robot0_eef_pos"], angle))
    state = {key: np.asarray([[[value]]], dtype=np.float32) for key, value in zip(AXES, pose, strict=True)}
    state["gripper"] = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(1, 1, 2)
    video = {}
    for key, camera in (("image", "agentview_image"), ("wrist_image", "robot0_eye_in_hand_image")):
        frame = np.asarray(obs[camera])
        if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
            raise ValueError("GR00T requires native 256x256 uint8 RGB cameras")
        video[key] = np.ascontiguousarray(frame[::-1, ::-1][None, None])
    return {"video": video, "state": state,
            "language": {"annotation.human.action.task_description": [[str(language)]]}}


def decode_actions(actions: dict[str, Any]) -> np.ndarray:
    components = [np.asarray(actions[key]) for key in (*AXES, "gripper")]
    if any(value.shape != (1, 16, 1) for value in components):
        raise ValueError("GR00T decoded modalities must each have shape (1, 16, 1)")
    chunk = validate_actions(np.concatenate(components, axis=-1)[0], 16).copy()
    chunk[:, -1] = -np.sign(2 * chunk[:, -1] - 1)
    return chunk


@contextlib.contextmanager
def isolated_rng(seed: int):
    """Serialize requests and restore caller RNG state even when inference fails."""
    import torch

    with _RNG_LOCK:
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng():
                random.seed(seed)
                np.random.seed(seed % (2 ** 32))
                torch.manual_seed(seed)
                yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class Gr00tRequestPolicy:
    def __init__(self, policy: Any):
        self.policy = policy

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        observation = dict(observation)
        identity = observation.pop(REQUEST_KEY)
        if not isinstance(identity, dict):
            raise ValueError(f"{REQUEST_KEY} must be a request identity")
        seed = request_seed(identity)
        with isolated_rng(seed):
            actions, _ = self.policy.get_action(observation)
        return {"actions": decode_actions(actions),
                RESPONSE_KEY: {**identity, "seed": seed, "rng_scheme": RNG_SCHEME}}


def checkpoint_path(model: Gr00tModelSpec) -> Path:
    root = Path(model.checkpoint).expanduser()
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        subdir = model.checkpoint_subdir
        root = Path(snapshot_download(
            model.checkpoint, revision=model.revision,
            allow_patterns=[f"{subdir}/config.json"],
        ))
        check_backbone_access(root / subdir)
        root = Path(snapshot_download(
            model.checkpoint, revision=model.revision,
            allow_patterns=[f"{subdir}/config.json", f"{subdir}/model*.safetensors",
                            f"{subdir}/model.safetensors.index.json", f"{subdir}/processor_config.json",
                            f"{subdir}/statistics.json", f"{subdir}/embodiment_id.json",
                            f"{subdir}/processor/*"],
        ))
    path = root / model.checkpoint_subdir
    if not (path / "config.json").is_file():
        raise ValueError(f"missing GR00T checkpoint config: {path}")
    return path


def checkpoint_identity(model: Gr00tModelSpec, path: Path, upstream: Path) -> dict[str, Any]:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"GR00T upstream revision must be {UPSTREAM_REVISION}, got {revision}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=upstream, text=True).strip():
        raise ValueError("GR00T upstream checkout must be read-only and clean")
    files = [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix in (".json", ".safetensors")
             and "experiment_cfg" not in p.parts]
    digests = {}
    for file in files:
        digests[file.relative_to(path).as_posix()] = sha256(file)
    if not any("processor_config.json" in key for key in digests) or not any("statistics.json" in key for key in digests):
        raise ValueError("checkpoint processor and statistics files are required")
    return {**model.model_dump(), "upstream_revision": revision, "files_sha256": digests,
            "action_chunk_steps": 16, "implementation": "single", "precision": "bfloat16",
            "inference": "eager", "max_batch_size": 1, "rng_scheme": RNG_SCHEME,
            "model_config": json.loads((path / "config.json").read_text())}


def check_backbone_access(path: Path) -> dict[str, Any]:
    """Check the configured local or Hub backbone before allocating GPU memory."""
    from huggingface_hub import hf_hub_download

    config = json.loads((path / "config.json").read_text())
    backbone = config["model_name"]
    local = Path(backbone)
    resolved = local.resolve() / "config.json" if local.is_dir() else Path(hf_hub_download(backbone, "config.json"))
    return {backbone: {"revision": resolved.parent.name if resolved.parent.parent.name == "snapshots" else None,
                       "config_sha256": sha256(resolved)}}


def backbone_processor_identity(policy: Any) -> dict[str, str]:
    """Hash the loaded image/text processors, including their tokenizers and templates."""
    digests = {}
    with tempfile.TemporaryDirectory(prefix="vlalagbench-processor-") as directory:
        root = Path(directory)
        for name, processor in (("observation", policy.processor.processor),
                                ("collator", policy.collate_fn.processor)):
            processor.save_pretrained(root / name)
        for file in sorted(root.rglob("*")):
            if file.is_file():
                digests[file.relative_to(root).as_posix()] = sha256(file)
    if not digests:
        raise ValueError("loaded backbone processors did not serialize any files")
    return digests
