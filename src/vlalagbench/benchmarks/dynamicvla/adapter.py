"""DynamicVLA/DOM adapter for the shared VLALagBench run lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from vlalagbench.benchmarks.base import EpisodeOutput
from vlalagbench.benchmarks.dynamicvla.protocol import (
    POLICY_NOISE_SCHEME,
    derive_episode_seed,
    receipt_errors,
)
from vlalagbench.benchmarks.dynamicvla.motion import episode_actors
from vlalagbench.benchmarks.dynamicvla.prompts import (
    derive_prompt_seed,
    literal_prompt,
)
from vlalagbench.benchmarks.dynamicvla.scene import authored_environment, rigid_assets
from vlalagbench.benchmarks.dynamicvla.tracing import DynamicVLATrace
from vlalagbench.config import (
    DynamicVLAExperiment,
    DynamicVLARuntime,
    ExperimentDocument,
    RuntimeDocument,
)
from vlalagbench.files import local_path, sha256 as _sha256


class DynamicVLAAdapter:
    def validate(
        self, experiment: ExperimentDocument, runtime: RuntimeDocument
    ) -> None:
        if not isinstance(experiment, DynamicVLAExperiment):
            raise TypeError("dynamicvla adapter requires a DynamicVLA experiment")
        if not isinstance(runtime, DynamicVLARuntime):
            raise TypeError("dynamicvla experiments require a DynamicVLA runtime")
        if runtime.host not in {"127.0.0.1", "localhost"}:
            raise ValueError("DynamicVLA simulator and policy processes must share one host")
        if runtime.replica_batch_size != 1:
            raise ValueError(
                "DynamicVLA requires replica_batch_size=1; "
                "use independent study shards for concurrency"
            )

    def evaluate(
        self,
        experiment: ExperimentDocument,
        runtime: RuntimeDocument,
        artifact_dir: Path,
        *,
        episode_ids: set[tuple[int, int]] | None = None,
    ) -> list[EpisodeOutput]:
        self.validate(experiment, runtime)
        assert isinstance(experiment, DynamicVLAExperiment)
        assert isinstance(runtime, DynamicVLARuntime)
        paths = _preflight(experiment, runtime)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        receipts_path = artifact_dir / "dynamicvla-receipts.jsonl"
        if receipts_path.exists():
            raise ValueError("DynamicVLA artifact directory already contains receipts")
        paths["manifest"] = _build_manifest(
            experiment, paths["assets"], artifact_dir / "initial-states"
        )
        if episode_ids is not None:
            paths["manifest"] = select_manifest_episodes(paths["manifest"], episode_ids)
        manifest_path = artifact_dir / "dynamicvla-manifest.json"
        manifest_path.write_text(
            json.dumps(paths["manifest"], sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        environment_list = artifact_dir / "environments.txt"
        environment_list.write_text(
            "".join(
                f"{environment['path']}\n"
                for environment in paths["manifest"]["environments"]
            ),
            encoding="utf-8",
        )
        _run_processes(
            experiment, runtime, paths, artifact_dir, environment_list,
            manifest_path, receipts_path,
        )
        return _load_episodes(
            experiment, receipts_path, artifact_dir, manifest=paths["manifest"]
        )


def select_manifest_episodes(manifest: dict[str, Any], episode_ids: set[tuple[int, int]]) -> dict[str, Any]:
    """Select execution work without renumbering states, replicas, prompts, or seeds."""
    available = {(row["task_id"], row["episode_id"]) for row in manifest["episodes"]}
    if not episode_ids or not episode_ids <= available:
        raise ValueError("episode selection must be nonempty and contained in the original manifest")
    episodes = [row for row in manifest["episodes"] if (row["task_id"], row["episode_id"]) in episode_ids]
    return {**manifest, "episodes": episodes,
            "tasks": [{**task, "resolved_episode_count": sum(row["task_id"] == task["task_id"] for row in episodes)}
                      for task in manifest["tasks"]]}


def _preflight(
    experiment: DynamicVLAExperiment, runtime: DynamicVLARuntime, *, check_ports: bool = True
) -> dict[str, Any]:
    root = Path(runtime.dynamicvla_root).expanduser().resolve()
    assets = Path(runtime.assets_root).expanduser().resolve()
    checkpoint = local_path(experiment.model.checkpoint).resolve()
    simulator_python = _runtime_executable(runtime.simulator_python)
    policy_python = _runtime_executable(runtime.policy_python)
    required = [
        root / "simulations/evaluate.py",
        root / "scripts/inference.py",
        root / "policies/dynamicvla/modeling_dynamicvla.py",
        assets / "scenes",
        assets / "objects",
        checkpoint / "config.json",
        checkpoint / "model.safetensors",
        simulator_python,
        policy_python,
    ]
    environments = [
        (assets / environment).resolve()
        for task in experiment.benchmark.tasks
        for environment in task.environments
    ]
    if len(environments) != len(set(environments)):
        raise ValueError(
            "DynamicVLA environment paths must resolve to globally unique files"
        )
    required.extend(environments)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing DynamicVLA paths: " + ", ".join(missing))
    if any(
        not path.is_file() or not os.access(path, os.X_OK)
        for path in [simulator_python, policy_python]
    ):
        raise ValueError("DynamicVLA Python executables must be executable files")

    revision = _command(
        ["git", "-C", str(root), "rev-parse", "HEAD"], "read DynamicVLA revision"
    )
    if revision != experiment.benchmark.upstream_revision:
        raise ValueError(
            f"DynamicVLA checkout is {revision}, expected "
            f"{experiment.benchmark.upstream_revision}"
        )
    dirty = _command(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        "inspect DynamicVLA checkout",
    )
    if checkout_changes(dirty):
        raise ValueError("DynamicVLA checkout has tracked or untracked modifications")

    checkpoint_hash = _sha256(checkpoint / "model.safetensors")
    if checkpoint_hash != experiment.model.checkpoint_sha256:
        raise ValueError(
            f"DynamicVLA checkpoint SHA-256 is {checkpoint_hash}, expected "
            f"{experiment.model.checkpoint_sha256}"
        )
    model_config = json.loads((checkpoint / "config.json").read_text())
    if model_config.get("type") != "dynamicvla":
        raise ValueError("checkpoint config is not DynamicVLA")
    if model_config.get("n_action_steps") != experiment.inference.action_chunk_steps:
        raise ValueError("configured action chunk does not match the checkpoint")
    timestamps = (model_config.get("delta_timestamps") or {}).get("observation")
    if timestamps != [-2, 0]:
        raise ValueError("DynamicVLA checkpoint must use the verified [-2, 0] observation window")

    for environment in environments:
        if not environment.is_relative_to(assets):
            raise ValueError(f"environment escapes assets_root: {environment}")
        data = json.loads(environment.read_text())
        if not isinstance(data.get("seed"), int) or "instruction" not in data:
            raise ValueError(f"invalid DOM environment config: {environment}")
        literal_prompt(data["instruction"], 0)
        _check_authored_targets(experiment, environment, data)
    manifest = _build_manifest(experiment, assets)
    if check_ports:
        _require_available_ports(runtime.host, runtime.image_port, runtime.action_port)
    return {
        "root": root,
        "assets": assets,
        "checkpoint": checkpoint,
        "simulator_python": simulator_python,
        "policy_python": policy_python,
        "environments": environments,
        "manifest": manifest,
    }


def _check_authored_targets(
    experiment: DynamicVLAExperiment, environment: Path, data: dict[str, Any]
) -> None:
    assets = set(rigid_assets(data))
    targets = {actor.target for actor in experiment.dynamics}
    if experiment.trajectory is not None:
        targets.add(experiment.trajectory.target)
    for actor in [*experiment.dynamics, experiment.trajectory]:
        targets.update(getattr(actor, "carried_objects", []))
    for override in experiment.initial_state_overrides:
        targets.update(override.object_positions_m)
        targets.update(override.object_yaws_rad)
    unknown = sorted(targets - assets)
    if unknown:
        raise ValueError(
            f"{environment} has no rigid assets named: {', '.join(unknown)}"
        )


def _initial_state_environment(
    experiment: DynamicVLAExperiment,
    *,
    relative_path: str,
    path: Path,
    data: dict[str, Any],
    name: str,
    state_index: int,
    output_dir: Path | None,
) -> tuple[str | None, str]:
    """Return the environment JSON path and SHA-256 for one initial state.

    Official states run the configured file. Authored states are written as
    derived environment JSONs so Isaac builds them through the upstream
    reset path; before materialization only their SHA-256 is known.
    """

    derived = authored_environment(experiment, relative_path, data, state_index)
    if derived is None:
        return str(path), _sha256(path)
    text = json.dumps(derived, indent=2) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    if output_dir is None:
        return None, digest
    output_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("VLALAGBENCH_DYNAMICVLA_REUSE_IDENTICAL_SCENES") == "1":
        # Upstream caches its environment by filename, then performs its normal
        # seeded reset. Equal payloads may share a filename without merging
        # their episode identities, prompts, or motion schedules.
        target = output_dir / f"{name}-state-{digest}.json"
    else:
        target = output_dir / f"{name}-state{state_index}.json"
    target.write_text(text, encoding="utf-8")
    return str(target.resolve()), digest


def _build_manifest(
    experiment: DynamicVLAExperiment,
    assets_root: Path,
    initial_state_dir: Path | None = None,
) -> dict[str, Any]:
    episodes_per_environment = experiment.benchmark.episodes_per_environment
    state_count = experiment.benchmark.initial_states_per_environment
    tasks = []
    environments = []
    episodes = []
    for task_id, task in enumerate(experiment.benchmark.tasks):
        tasks.append(
            {
                "task_id": task_id,
                "name": task.name,
                "resolved_episode_count": len(task.environments)
                * episodes_per_environment,
            }
        )
        for environment_index, relative_path in enumerate(task.environments):
            path = (assets_root / relative_path).resolve()
            environment_sha256 = _sha256(path)
            environment_data = json.loads(path.read_text())
            seed = environment_data.get("seed")
            instruction_metadata = environment_data.get("instruction")
            environments.append(
                {
                    "task_id": task_id,
                    "task_name": task.name,
                    "environment_index": environment_index,
                    "path": str(path),
                    "relative_path": relative_path,
                    "sha256": environment_sha256,
                    "seed": seed,
                    "instruction_metadata": instruction_metadata,
                }
            )
            states = [
                _initial_state_environment(
                    experiment,
                    relative_path=relative_path,
                    path=path,
                    data=environment_data,
                    name=f"task{task_id:02d}-env{environment_index:03d}-{path.stem}",
                    state_index=state_index,
                    output_dir=initial_state_dir,
                )
                for state_index in range(state_count)
            ]
            for replica_id in range(episodes_per_environment):
                state_index = replica_id % state_count
                episode_environment, episode_sha256 = states[state_index]
                prompt_seed = derive_prompt_seed(
                    experiment.seeds.policy,
                    task.name,
                    episode_sha256,
                    replica_id,
                )
                episode = {
                    "task_id": task_id,
                    "task_name": task.name,
                    "environment_index": environment_index,
                    "environment": episode_environment,
                    "environment_sha256": episode_sha256,
                    "seed": seed,
                    "prompt_seed": prompt_seed,
                    "instruction": literal_prompt(instruction_metadata, prompt_seed),
                    "replica_id": replica_id,
                    "episode_id": environment_index * episodes_per_environment
                    + replica_id,
                }
                if experiment.benchmark.max_episode_steps is not None:
                    episode['max_episode_steps'] = experiment.benchmark.max_episode_steps
                if experiment.has_authored_scene:
                    episode["initial_state_index"] = state_index
                    episode["motion_actors"] = [
                        actor.model_dump(mode="json")
                        for actor in episode_actors(experiment, replica_id)
                    ]
                    if experiment.benchmark.driven_friction == "native":
                        episode["driven_friction"] = "native"
                episodes.append(episode)
    return {
        "schema_version": 1,
        "episodes_per_environment": episodes_per_environment,
        "tasks": tasks,
        "environments": environments,
        "episodes": episodes,
    }


def _run_processes(
    experiment: DynamicVLAExperiment,
    runtime: DynamicVLARuntime,
    paths: dict[str, Any],
    artifact_dir: Path,
    environment_list: Path,
    manifest_path: Path,
    receipts_path: Path,
) -> None:
    src_root = Path(__file__).resolve().parents[3]
    common_env = os.environ.copy()
    common_env["PYTHONPATH"] = os.pathsep.join(
        [str(src_root), str(paths["root"]), common_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    if runtime.policy_gpu is not None:
        common_env["CUDA_VISIBLE_DEVICES"] = str(runtime.policy_gpu)
    simulator_env = common_env.copy()
    if runtime.simulator_gpu is not None:
        simulator_env["CUDA_VISIBLE_DEVICES"] = str(runtime.simulator_gpu)
    simulator_env["VLALAGBENCH_DYNAMICVLA_ROOT"] = str(paths["root"])
    simulator_env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    library_paths = [
        *runtime.simulator_library_paths,
        *_discover_isaac_library_paths(paths["simulator_python"]),
    ]
    if library_paths:
        simulator_env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*library_paths, simulator_env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)

    simulator_command = [
        str(paths["simulator_python"]),
        "-m",
        "vlalagbench.benchmarks.dynamicvla.simulator",
        "--scene-dir",
        str(paths["assets"] / "scenes"),
        "--object-dir",
        str(paths["assets"] / "objects"),
        "--output-dir",
        str(artifact_dir / "evaluation"),
        "--env-cfg",
        str(environment_list),
        "--manifest",
        str(manifest_path),
        "--receipts",
        str(receipts_path),
        "--trace-dir",
        str(artifact_dir),
        "--upstream-revision",
        experiment.benchmark.upstream_revision,
        "--checkpoint",
        str(paths["checkpoint"]),
        "--checkpoint-sha256",
        experiment.model.checkpoint_sha256,
        "--checkpoint-config-sha256",
        _sha256(paths["checkpoint"] / "config.json"),
        "--host",
        runtime.host,
        "--img-port",
        str(runtime.image_port),
        "--act-port",
        str(runtime.action_port),
        "--latency-steps",
        str(experiment.effective_latency_steps),
        "--mode",
        experiment.inference.mode,
        "--execution-horizon-steps",
        str(experiment.resolved_execution_horizon_steps),
        "--action-chunk-steps",
        str(experiment.inference.action_chunk_steps),
        "--policy-seed",
        str(experiment.seeds.policy),
        "--reply-timeout-s",
        str(runtime.reply_timeout_s),
        "--startup-timeout-s",
        str(runtime.startup_timeout_s),
        "--trace-writer-workers",
        str(runtime.trace_writer_workers),
        "--enable_cameras",
        "--headless",
        "--num_envs",
        str(runtime.replica_batch_size),
    ]
    simulator_command.extend(_renderer_gpu_args(runtime))
    simulator_command.extend(_renderer_setting_args(runtime))
    if runtime.disable_fabric:
        simulator_command.append("--disable_fabric")
    if runtime.save_videos:
        simulator_command.append("--save")
    if runtime.video_episode_limit is not None:
        simulator_command.extend(["--video-episode-limit", str(runtime.video_episode_limit)])
    if runtime.trace_images:
        simulator_command.append("--trace-images")
    policy_command = [
        str(paths["policy_python"]),
        "-m",
        "vlalagbench.benchmarks.dynamicvla.policy",
        "--dynamicvla-root",
        str(paths["root"]),
        "--weights",
        str(paths["checkpoint"]),
        "--host",
        runtime.host,
        "--img-port",
        str(runtime.image_port),
        "--act-port",
        str(runtime.action_port),
        "--mode",
        experiment.inference.mode,
        "--execution-horizon-steps",
        str(experiment.inference.execution_horizon_steps),
        "--latency-steps",
        str(experiment.effective_latency_steps),
        "--reply-timeout-s",
        str(runtime.reply_timeout_s),
    ]
    simulator_log = artifact_dir / "simulator.log"
    policy_log = artifact_dir / "policy.log"
    simulator = policy = None
    with simulator_log.open("w", encoding="utf-8") as simulator_stream, policy_log.open(
        "w", encoding="utf-8"
    ) as policy_stream:
        try:
            simulator = subprocess.Popen(
                simulator_command,
                cwd=paths["root"],
                env=simulator_env,
                stdout=simulator_stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            _wait_for_ports(
                simulator,
                runtime.host,
                (runtime.image_port, runtime.action_port),
                runtime.startup_timeout_s,
            )
            policy = subprocess.Popen(
                policy_command,
                cwd=paths["root"],
                env=common_env,
                stdout=policy_stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            _wait_for_processes(
                simulator,
                policy,
                runtime.suite_timeout_s,
                failure_flush_timeout_s=2 * runtime.reply_timeout_s + 30.0,
            )
        except Exception as exc:
            raise RuntimeError(
                f"{exc}\nsimulator log:\n{_tail(simulator_log)}"
                f"\npolicy log:\n{_tail(policy_log)}"
            ) from exc
        finally:
            _stop_process(policy)
            _stop_process(simulator)


def _renderer_gpu_args(runtime: DynamicVLARuntime) -> list[str]:
    settings = []
    if runtime.simulator_gpu is not None:
        # Vulkan rendering is not restricted by CUDA_VISIBLE_DEVICES. The Kit GPU
        # table must be checked against the host's device identities at deployment.
        settings += ["--/renderer/multiGpu/enabled=false", f"--/renderer/activeGpu={runtime.simulator_gpu}"]
    if runtime.kit_tasking_threads is not None:
        settings.append(f"--/plugins/carb.tasking.plugin/threadCount={runtime.kit_tasking_threads}")
    return ["--kit_args", " ".join(settings)] if settings else []


def _load_episodes(
    experiment: DynamicVLAExperiment,
    receipts_path: Path,
    artifact_dir: Path,
    *,
    manifest: dict[str, Any],
) -> list[EpisodeOutput]:
    if not receipts_path.is_file():
        raise RuntimeError("DynamicVLA simulator produced no episode receipts")
    receipts = [json.loads(line) for line in receipts_path.read_text().splitlines() if line]
    expected_episodes = {
        (episode["task_id"], episode["episode_id"]): episode
        for episode in manifest["episodes"]
    }
    expected = len(expected_episodes)
    if len(receipts) != expected:
        raise RuntimeError(f"DynamicVLA produced {len(receipts)} receipts, expected {expected}")

    expected_keys = set(expected_episodes)
    seen: set[tuple[int, int]] = set()
    validated: list[tuple[tuple[int, int], dict[str, Any], Path, dict[str, Any]]] = []
    for receipt in receipts:
        try:
            key = (int(receipt["task_id"]), int(receipt["episode"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("DynamicVLA receipt has an invalid episode identity") from exc
        if key in seen:
            raise RuntimeError(f"duplicate DynamicVLA episode receipt: {key}")
        seen.add(key)
        if key not in expected_keys:
            raise RuntimeError(f"unexpected DynamicVLA episode receipt: {key}")
        expected_episode = expected_episodes[key]
        expected_environment = Path(expected_episode["environment"]).resolve()
        expected_environment_sha256 = expected_episode["environment_sha256"]
        if _sha256(expected_environment) != expected_environment_sha256:
            raise RuntimeError(
                f"DynamicVLA environment for receipt {key} changed after manifest creation"
            )
        try:
            actual_environment = Path(receipt["environment"]).resolve()
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"DynamicVLA receipt {key} has no valid environment") from exc
        if actual_environment != expected_environment:
            raise RuntimeError(
                f"DynamicVLA receipt {key} used {actual_environment}, "
                f"expected {expected_environment}"
            )
        for field in ("task_name", "environment_index", "replica_id"):
            if receipt.get(field) != expected_episode[field]:
                raise RuntimeError(
                    f"DynamicVLA receipt {key} has the wrong {field} mapping"
                )
        expected_episode_seed = derive_episode_seed(
            experiment.seeds.policy,
            expected_episode["task_name"],
            expected_environment_sha256,
            expected_episode["replica_id"],
        )
        if receipt.get("episode_seed") != expected_episode_seed:
            raise RuntimeError(f"DynamicVLA receipt {key} has the wrong episode seed")
        expected_environment_data = json.loads(expected_environment.read_text())
        expected_seed = expected_environment_data["seed"]
        if receipt.get("seed") != expected_seed:
            raise RuntimeError(f"DynamicVLA receipt {key} has the wrong environment seed")
        expected_instruction = expected_episode["instruction"]
        if receipt.get("instruction") != expected_instruction:
            raise RuntimeError(f"DynamicVLA receipt {key} has the wrong literal prompt")
        status = receipt.get("status")
        success = receipt.get("success")
        if status not in (0, 1, 2):
            raise RuntimeError(
                f"DynamicVLA receipt {key} has invalid completed status {status!r}"
            )
        if not isinstance(success, bool) or success != (status == 0):
            raise RuntimeError(f"DynamicVLA receipt {key} has inconsistent outcome fields")
        if receipt.get("injected_steps") != experiment.effective_latency_steps:
            raise RuntimeError(f"DynamicVLA receipt {key} has the wrong injected latency")
        errors = receipt_errors(
            receipt,
            mode=experiment.inference.mode,
            latency_steps=experiment.effective_latency_steps,
            action_chunk_steps=experiment.inference.action_chunk_steps,
            execution_horizon_steps=(
                experiment.resolved_execution_horizon_steps
            ),
        )
        expected_origin = (
            "release" if experiment.inference.mode == "synchronous" else "observation"
        )
        if receipt.get("action_index_origin") != expected_origin:
            errors.append("receipt uses a different action index origin")
        if receipt.get("execution_horizon_basis") != "post_latency_action_count":
            errors.append("receipt uses a different execution horizon basis")
        if errors or receipt.get("errors") or not receipt.get("exact"):
            details = errors or receipt["errors"]
            raise RuntimeError(f"invalid DynamicVLA latency receipt {key}: {details}")
        validated.append((key, receipt, expected_environment, expected_environment_data))

    if seen != expected_keys:
        missing = sorted(expected_keys - seen)
        raise RuntimeError(f"DynamicVLA receipts are missing configured episodes: {missing}")

    episodes: list[EpisodeOutput] = []
    for key, receipt, expected_environment, expected_environment_data in validated:
        expected_environment_sha256 = expected_episodes[key]["environment_sha256"]
        status = receipt["status"]
        success = receipt["success"]
        trace_name = receipt.get("trace")
        if not isinstance(trace_name, str) or Path(trace_name).name != trace_name:
            raise RuntimeError(f"DynamicVLA receipt {key} has an invalid trace path")
        trace_path = artifact_dir / trace_name
        if not trace_path.is_file():
            raise RuntimeError(f"DynamicVLA receipt {key} is missing trace {trace_name}")
        if receipt.get("trace_sha256") != _sha256(trace_path):
            raise RuntimeError(f"DynamicVLA receipt {key} has the wrong trace checksum")
        try:
            trace = DynamicVLATrace.load(trace_path)
        except (OSError, KeyError, OverflowError, TypeError, ValueError) as exc:
            raise RuntimeError(f"DynamicVLA trace {trace_name} is invalid: {exc}") from exc
        if (
            (trace.meta.task_id, trace.meta.episode) != key
            or trace.meta.task_name != receipt["task_name"]
            or trace.meta.environment_index != receipt["environment_index"]
            or trace.meta.replica_id != receipt["replica_id"]
            or trace.meta.environment_path != receipt["environment"]
            or trace.meta.seed != receipt["seed"]
            or trace.meta.instruction != receipt["instruction"]
            or trace.meta.status != status
            or trace.meta.success != success
            or trace.meta.upstream_revision
            != experiment.benchmark.upstream_revision
            or trace.meta.checkpoint
            != str(local_path(experiment.model.checkpoint).resolve())
            or trace.meta.checkpoint_sha256
            != experiment.model.checkpoint_sha256
            or trace.meta.policy_noise_scheme != POLICY_NOISE_SCHEME
            or trace.meta.policy_seed != experiment.seeds.policy
            or trace.meta.policy_episode_seed != receipt["episode_seed"]
            or trace.meta.inference_mode != experiment.inference.mode
            or trace.meta.action_index_origin
            != ("release" if experiment.inference.mode == "synchronous" else "observation")
            or trace.meta.latency_steps != experiment.effective_latency_steps
            or trace.meta.action_chunk_steps != experiment.inference.action_chunk_steps
            or trace.meta.execution_horizon_steps
            != experiment.resolved_execution_horizon_steps
            or trace.meta.execution_horizon_basis != "post_latency_action_count"
            or trace.meta.environment_sha256 != expected_environment_sha256
            or trace.meta.environment != expected_environment_data
            or trace.meta.episode_length_s
            != (experiment.benchmark.max_episode_steps / experiment.benchmark.control_hz
                if experiment.benchmark.max_episode_steps is not None
                else expected_environment_data["episode_length_s"])
        ):
            raise RuntimeError(f"DynamicVLA trace {trace_name} disagrees with its receipt")
        control_steps = int(receipt["n_steps"])
        if trace.meta.motion is not None:
            expected_friction = None if experiment.benchmark.driven_friction == "native" else 0.0
            if trace.meta.motion.driven_friction != expected_friction:
                raise RuntimeError(f"DynamicVLA trace {trace_name} has different driven friction")
            if expected_friction is None and set(trace.meta.motion.native_materials) != {
                actor.target for actor in trace.meta.motion.actors
            }:
                raise RuntimeError(f"DynamicVLA trace {trace_name} lacks native materials")
        if trace.num_steps != control_steps:
            raise RuntimeError(f"DynamicVLA trace {trace_name} has the wrong step count")
        if trace.actions.tolist() != [
            event.get("action") for event in receipt.get("control_events", [])
        ]:
            raise RuntimeError(
                f"DynamicVLA trace {trace_name} disagrees with receipt control actions"
            )
        trace_requests = [
            {"step": step, **protocol["request"]}
            for step, protocol in enumerate(trace.protocol)
            if protocol.get("request") is not None
        ]
        if trace_requests != receipt["request_events"]:
            raise RuntimeError(
                f"DynamicVLA trace {trace_name} disagrees with receipt request events"
            )
        applied = int(receipt["applied"])
        episodes.append(
            {
                "task_id": key[0],
                "episode": key[1],
                "seed": int(receipt["seed"]),
                "success": bool(receipt["success"]),
                "status": status,
                "steps": applied,
                "control_steps": control_steps,
                "wait_steps": control_steps - applied,
                "policy_requests": int(receipt["requests"]),
                "trace": trace_name,
                "error": None,
            }
        )
    return sorted(episodes, key=lambda episode: (episode["task_id"], episode["episode"]))


# Python's tempfile falls back to the working directory (the checkout) when it
# cannot write to the system temp directory; PyTorch then leaves a random
# tmpXXXXXXXX/ directory there. Nothing imports it, so it is not a code change.
_STRAY_TEMP_DIRECTORY = re.compile(r"\?\? tmp[a-z0-9_]{8}/")


def checkout_changes(porcelain: str) -> list[str]:
    """`git status --porcelain` lines that change the checkout's code."""
    return [line for line in porcelain.splitlines()
            if line and not _STRAY_TEMP_DIRECTORY.match(line)]


def _command(command: list[str], action: str) -> str:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"failed to {action}: {exc}") from exc


def _runtime_executable(path: str | Path) -> Path:
    """Make an executable path absolute without dereferencing virtualenv symlinks."""
    return Path(os.path.abspath(Path(path).expanduser()))


def _require_available_ports(host: str, *ports: int) -> None:
    for port in ports:
        with socket.socket() as probe:
            try:
                probe.bind((host, port))
            except OSError as exc:
                raise ValueError(f"DynamicVLA port {host}:{port} is unavailable") from exc


def _wait_for_ports(
    process: subprocess.Popen[str], host: str, ports: tuple[int, ...], timeout_s: float
) -> None:
    pending = set(ports)
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"DynamicVLA simulator exited with status {process.returncode}")
        for port in tuple(pending):
            try:
                with socket.create_connection((host, port), timeout=0.25):
                    pending.remove(port)
            except OSError:
                pass
        if pending:
            time.sleep(0.25)
    if pending:
        raise TimeoutError(f"DynamicVLA simulator ports did not open: {sorted(pending)}")


def _wait_for_processes(
    simulator: subprocess.Popen[str],
    policy: subprocess.Popen[str],
    timeout_s: float,
    *,
    failure_flush_timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    policy_failure = None
    failure_deadline = None
    while time.monotonic() < deadline:
        simulator_status, policy_status = simulator.poll(), policy.poll()
        if simulator_status not in (None, 0):
            raise RuntimeError(f"DynamicVLA simulator exited with status {simulator_status}")
        if policy_status not in (None, 0):
            policy_failure = policy_status
            if failure_deadline is None:
                failure_deadline = min(
                    deadline, time.monotonic() + failure_flush_timeout_s
                )
            if simulator_status == 0 or time.monotonic() >= failure_deadline:
                raise RuntimeError(
                    f"DynamicVLA policy exited with status {policy_status}"
                )
        if simulator_status == policy_status == 0:
            return
        time.sleep(0.5)
    if policy_failure is not None:
        raise RuntimeError(f"DynamicVLA policy exited with status {policy_failure}")
    raise TimeoutError(f"DynamicVLA suite exceeded {timeout_s:.1f}s")


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _discover_isaac_library_paths(python: Path) -> list[str]:
    environment_root = python.parent.parent
    paths: list[Path] = [environment_root / "lib"]
    for site_packages in environment_root.glob("lib/python*/site-packages"):
        paths.extend(site_packages.glob("isaacsim/extscache/omni.usd.libs-*/bin"))
        paths.append(site_packages / "isaacsim/extsPhysics/omni.usd.schema.physx/bin")
        paths.append(site_packages / "omni")
    return [str(path) for path in paths if path.is_dir()]


def _tail(path: Path, lines: int = 80) -> str:
    if not path.is_file():
        return "<missing>"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


ADAPTER = DynamicVLAAdapter()


def _renderer_setting_args(runtime: DynamicVLARuntime) -> list[str]:
    args = []
    for field, flag in (("antialiasing_mode", "antialiasing-mode"), ("dlss_mode", "dlss-mode"), ("render_mode", "render-mode")):
        value = getattr(runtime, field)
        if value is not None:
            args.extend([f"--{flag}", value])
    for field, flag in (("enable_dl_denoiser", "enable-dl-denoiser"), ("enable_dlssg", "enable-dlssg")):
        value = getattr(runtime, field)
        if value is not None:
            args.append(f"--{flag}" if value else f"--no-{flag}")
    return args
