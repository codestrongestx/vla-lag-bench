"""DynamicVLA study identity and offline verification of recorded evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from vlalagbench.benchmarks.dynamicvla.adapter import _preflight
from vlalagbench.benchmarks.dynamicvla.protocol import receipt_errors
from vlalagbench.config import (
    DynamicVLAExperiment, DynamicVLARuntime, DynamicVLAStudyPolicy,
    DynamicVLAObservationSettings,
)
from vlalagbench.files import sha256


def execution_identity(experiment: DynamicVLAExperiment, runtime: DynamicVLARuntime, repo_root: Path) -> dict:
    paths = _preflight(experiment, runtime, check_ports=False)
    root = repo_root / "src/vlalagbench"
    digest = hashlib.sha256()
    sources = [*sorted((root / "benchmarks/dynamicvla").glob("*.py")),
               *sorted((root / "simulators/isaaclab").glob("*.py")),
               *sorted((root / "study").glob("*.py")),
               *(root / name for name in ("config.py", "runner.py", "timing.py", "files.py"))]
    for path in sources:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    query = """import importlib.metadata as m, json, sys
versions = {}
for name in ('numpy', 'torch', 'torchvision', 'transformers', 'isaaclab', 'isaacsim', 'pydantic'):
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: pass
print(json.dumps({'python': sys.version.split()[0], 'packages': versions}, sort_keys=True))
"""
    stacks = {}
    for role in ("policy_python", "simulator_python"):
        output = subprocess.run([str(paths[role]), "-c", query], check=True,
                                capture_output=True, text=True, timeout=60)
        stacks[role] = json.loads(output.stdout)
    return {"evaluator_sha256": digest.hexdigest(), "python": sys.version.split()[0],
            "stacks": stacks, "upstream_revision": experiment.benchmark.upstream_revision,
            "checkpoint_config_sha256": sha256(paths["checkpoint"] / "config.json"),
            "environments": {row["relative_path"]: row["sha256"] for row in paths["manifest"]["environments"]}}


def expected_ids(experiment: DynamicVLAExperiment) -> set[tuple[int, int]]:
    return {(task_id, episode) for task_id, task in enumerate(experiment.benchmark.tasks)
            for episode in range(len(task.environments) * experiment.benchmark.episodes_per_environment)}


def verify_runtime(runtime: DynamicVLARuntime, policy: DynamicVLAStudyPolicy) -> None:
    if not isinstance(policy, DynamicVLAStudyPolicy):
        raise ValueError("DynamicVLA point needs DynamicVLA study semantics")
    if not isinstance(runtime, DynamicVLARuntime):
        raise ValueError("DynamicVLA point has a different runtime kind")
    if any(
        getattr(runtime, field) != getattr(policy, field)
        for field in DynamicVLAObservationSettings.model_fields
    ):
        raise ValueError("DynamicVLA runtime differs from study observation semantics")


def verify_trace(meta: dict, episode: dict, experiment: DynamicVLAExperiment) -> None:
    task = experiment.benchmark.tasks[episode["task_id"]]
    environment_index, replica_id = divmod(episode["episode_id"], experiment.benchmark.episodes_per_environment)
    expected = {
        "task_id": episode["task_id"], "episode": episode["episode_id"], "task_name": task.name,
        "environment_index": environment_index, "replica_id": replica_id,
        "success": episode["success"], "status": episode["status"], "num_steps": episode["control_steps"],
        "upstream_revision": experiment.benchmark.upstream_revision,
        "checkpoint_sha256": experiment.model.checkpoint_sha256,
        "policy_seed": experiment.seeds.policy, "inference_mode": experiment.inference.mode,
        "action_index_origin": "release" if experiment.inference.mode == "synchronous" else "observation", "latency_steps": experiment.effective_latency_steps,
        "action_chunk_steps": experiment.inference.action_chunk_steps,
        "execution_horizon_steps": experiment.resolved_execution_horizon_steps,
        "execution_horizon_basis": "post_latency_action_count",
    }
    if experiment.benchmark.max_episode_steps is not None:
        expected['episode_length_s'] = experiment.benchmark.max_episode_steps / experiment.benchmark.control_hz
        expected['deadline_basis'] = 'active_action_steps'
    if episode["error"] is not None or any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("DynamicVLA trace disagrees with its episode or experiment")
    if meta["status"] not in (0, 1, 2) or meta["success"] != (meta["status"] == 0):
        raise ValueError("DynamicVLA trace has an inconsistent outcome")
    motion = meta.get("motion")
    if motion is not None:
        expected_friction = None if experiment.benchmark.driven_friction == "native" else 0.0
        if motion.get("driven_friction") != expected_friction:
            raise ValueError("DynamicVLA trace has different driven friction")


def verify_evidence(trace_dir: Path, episodes: list[dict], experiment: DynamicVLAExperiment) -> dict[str, str]:
    """Keep protocol receipts and the environment manifest in the immutable inventory."""
    manifest_path = trace_dir / "dynamicvla-manifest.json"
    receipts_path = trace_dir / "dynamicvla-receipts.jsonl"
    manifest = json.loads(manifest_path.read_text())
    identities = {(row["task_id"], row["episode_id"]) for row in manifest["episodes"]}
    if identities != expected_ids(experiment) or len(identities) != len(manifest["episodes"]):
        raise ValueError("DynamicVLA manifest has different episode identities")
    rows = [json.loads(line) for line in receipts_path.read_text().splitlines() if line]
    receipts = {(row["task_id"], row["episode"]): row for row in rows}
    if set(receipts) != identities or len(rows) != len(receipts):
        raise ValueError("DynamicVLA receipts have missing or duplicate episodes")
    video_files = {}
    for episode in episodes:
        receipt = receipts[episode["task_id"], episode["episode_id"]]
        if (receipt.get("trace_sha256") != episode["trace_sha256"]
                or receipt.get("trace") != Path(episode["trace"]).name
                or receipt.get("status") != episode["status"]
                or receipt.get("success") != episode["success"]
                or receipt.get("n_steps") != episode["control_steps"]
                or not receipt.get("exact") or receipt.get("errors")):
            raise ValueError("DynamicVLA receipt differs from recorded episode")
        if receipt.get("video") is not None:
            relative = Path(receipt["video"])
            path = (trace_dir / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(trace_dir.resolve()):
                raise ValueError("recorded video escapes the trace directory")
            if receipt.get("video_frames") != episode["control_steps"] + 1:
                raise ValueError("recorded video has an incorrect frame count")
            if not path.is_file() or sha256(path) != receipt.get("video_sha256"):
                raise ValueError("recorded video digest differs from receipt")
            video_files[relative.as_posix()] = receipt["video_sha256"]
        if receipt.get("execution_horizon_basis") != "post_latency_action_count":
            raise ValueError("DynamicVLA receipt uses a different execution horizon basis")
        if experiment.benchmark.max_episode_steps is not None and (
            receipt.get('deadline_basis') != 'active_action_steps'
            or receipt.get('active_action_deadline_steps') != experiment.benchmark.max_episode_steps
        ):
            raise ValueError('DynamicVLA receipt has a different active-action budget')
        errors = receipt_errors(receipt, mode=experiment.inference.mode,
            latency_steps=experiment.effective_latency_steps,
            action_chunk_steps=experiment.inference.action_chunk_steps,
            execution_horizon_steps=experiment.resolved_execution_horizon_steps)
        if errors:
            raise ValueError(f"DynamicVLA protocol receipt failed: {errors}")
    return {**{path.name: sha256(path) for path in (manifest_path, receipts_path)}, **video_files}
