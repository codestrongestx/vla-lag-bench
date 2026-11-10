"""Resume failed LIBERO traces without repeating VLA inference for their prefixes."""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from vlalagbench.benchmarks.libero.policy import REQUEST_KEY, RESPONSE_KEY
from vlalagbench.benchmarks.libero.envs import make_env_from_trace
from vlalagbench.benchmarks.libero.tracing import load_libero_trace
from vlalagbench.benchmarks.libero.evaluator import (
    EvalConfig,
    _policy_element,
    _record_request,
    _record_step,
    idle_step,
    make_policy,
)
from vlalagbench.timing import ChunkScheduler, TimingConfig
from vlalagbench.simulators.mujoco.tracing import (
    EpisodeTrace,
    TraceMeta,
    TraceRecordingEnv,
    get_mj_sim,
    get_sim_state,
    set_sim_state,
)

logger = logging.getLogger(__name__)

Scheme = Literal["synchronous", "lazy", "continuous"]


@dataclasses.dataclass(frozen=True)
class ContinuationSpec:
    trace: Path
    scheme: Scheme
    max_active_steps: int

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ContinuationSpec":
        return cls(
            trace=Path(value["trace"]),
            scheme=value["scheme"],
            max_active_steps=int(value["max_active_steps"]),
        )


@dataclasses.dataclass(frozen=True)
class PolicyConnection:
    host: str = "127.0.0.1"
    port: int = 8000
    model: Any = None
    action_chunk_steps: int = 10
    openpi_config: str = "pi05_libero"
    checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero"
    compile_mode: str = "max-autotune-no-cudagraphs"
    deterministic_ops: bool = False
    policy_seed: int = 7

    @classmethod
    def from_trace(cls, trace: EpisodeTrace, *, host: str = "127.0.0.1", port: int = 8000) -> "PolicyConnection":
        from vlalagbench.config import Gr00tModelSpec, ModelSpec, MolmoAct2ModelSpec
        data = getattr(trace.meta, "model", None)
        spec = {"gr00t": Gr00tModelSpec, "molmoact2": MolmoAct2ModelSpec}.get(None if data is None else data.get("backend"), ModelSpec)
        model = None if data is None else spec.model_validate(data)
        identity = getattr(trace.meta, "policy_identity", {})
        return cls(host=host, port=port, model=model, action_chunk_steps=trace.meta.action_chunk_steps,
                   openpi_config=getattr(trace.meta, "openpi_config", "pi05_libero"),
                   checkpoint=getattr(trace.meta, "checkpoint", "gs://openpi-assets/checkpoints/pi05_libero"),
                   policy_seed=getattr(trace.meta, "policy_seed", 7),
                   compile_mode=(identity.get("inference", "eager") if getattr(model, "backend", "openpi") in ("gr00t", "molmoact2")
                                 else identity.get("pytorch_compile_mode", "max-autotune-no-cudagraphs")),
                   deterministic_ops=identity.get("deterministic_ops", False))


@dataclasses.dataclass
class RestoredEpisode:
    scheduler: ChunkScheduler[np.ndarray]
    observation: Any
    active_steps: int
    control_steps: int
    last_gripper: float
    max_state_error: float
    done_mismatches: int


def timing_from_trace(trace: EpisodeTrace, scheme: Scheme) -> TimingConfig:
    return TimingConfig(
        scheme=scheme,
        action_chunk_steps=int(trace.meta.action_chunk_steps),
        execution_horizon_steps=int(trace.meta.execution_horizon_steps),
        latency_steps=int(trace.meta.latency_steps),
    )


def replay_prefix(
    env: Any,
    trace: EpisodeTrace,
    scheme: Scheme,
    *,
    state_atol: float | None = None,
) -> RestoredEpisode:
    """Replay a trace and reconstruct its controller, dynamics, and scheduler state."""

    if trace.meta.success:
        raise ValueError("only failed traces may be deadline-extended")
    wait_steps = int(trace.meta.num_steps_wait)
    if wait_steps > trace.num_steps:
        raise ValueError("trace is shorter than its warmup")

    env.reset()
    _restore_dynamic_actor_origins(env, trace)
    observation = env.set_init_state(trace.states[0])
    max_state_error = float(np.max(np.abs(get_sim_state(env) - trace.states[0])))
    done_mismatches = 0
    image_observables = _image_observable_names(env)
    _set_observables_enabled(env, image_observables, False)
    images_enabled = False
    prepare_dynamic_step = getattr(env, "prepare_dynamic_step", None)
    restore_prepared = getattr(env, "restore_prepared_dynamic_step", None)
    prepared_before_warmup = bool(
        getattr(trace.meta, "dynamic_step_zero_prepared_before_warmup", False)
    )
    state_zero_prepared = prepared_before_warmup or wait_steps == 0
    prepared = (
        bool(restore_prepared(0))
        if state_zero_prepared and restore_prepared
        else False
    )
    hold_dynamic_time = getattr(env, "hold_dynamic_time", None)
    try:
        for trace_step in range(wait_steps):
            if trace_step == trace.num_steps - 1:
                _set_observables_enabled(env, image_observables, True)
                images_enabled = True
            if prepared_before_warmup and hold_dynamic_time is not None:
                hold_dynamic_time(0.0)
            else:
                env.set_dynamic_step(None)
            observation, _, done, _ = env.step(trace.actions[trace_step].tolist())
            if trace_step + 1 == wait_steps and prepare_dynamic_step is not None:
                prepared = bool(env.prepare_dynamic_step(0))
            if done != bool(trace.dones[trace_step]):
                done_mismatches += 1
            max_state_error = max(
                max_state_error,
                _snap_to_recorded_state(env, trace.states[trace_step + 1]),
            )

        scheduler: ChunkScheduler[np.ndarray] = ChunkScheduler(timing_from_trace(trace, scheme))
        waiting = False
        active_steps = 0
        last_gripper = -1.0
        control_steps = trace.num_steps - wait_steps
        for control_step in range(control_steps):
            trace_step = wait_steps + control_step
            if trace_step == trace.num_steps - 1 and not images_enabled:
                _set_observables_enabled(env, image_observables, True)
                images_enabled = True
            request = scheduler.request(control_step)
            if request is not None:
                if request.id >= len(trace.chunks):
                    raise RuntimeError(f"missing recorded chunk {request.id}")
                if request.sent_step != int(trace.request_steps[request.id]):
                    raise RuntimeError(f"request-step mismatch for chunk {request.id}")
                if request.available_step != int(trace.arrival_steps[request.id]):
                    raise RuntimeError(f"arrival-step mismatch for chunk {request.id}")
                chunk = trace.chunks[request.id]
                scheduler.deliver(request, chunk)
                _record_request(
                    env,
                    request.id,
                    request.sent_step,
                    request.available_step,
                    chunk,
                    _recorded_timing(trace, request.id),
                    getattr(trace.meta, "request_audits", [None] * len(trace.chunks))[request.id],
                )

            scheduled = scheduler.action(control_step)
            expected = (
                bool(trace.acting[trace_step]),
                _optional_index(trace.chunk_ids[trace_step]),
                _optional_index(trace.action_indices[trace_step]),
                _optional_index(trace.observation_steps[trace_step]),
            )
            actual = (
                scheduled.active,
                scheduled.chunk_id,
                scheduled.index_in_chunk,
                scheduled.observation_step,
            )
            if actual != expected:
                raise RuntimeError(
                    f"scheduler mismatch at control step {control_step}: expected {expected}, got {actual}"
                )
            if scheduled.active:
                active_steps += 1
                last_gripper = float(trace.actions[trace_step, -1])
                if not np.allclose(scheduled.action, trace.actions[trace_step], atol=1e-10, rtol=0.0):
                    raise RuntimeError(f"recorded action mismatch at control step {control_step}")
            _record_step(env, scheduled)
            if control_step != 0 or not prepared:
                env.set_dynamic_step(control_step)
            if scheduled.active:
                observation, _, done, _ = env.step(trace.actions[trace_step].tolist())
            else:
                observation, _, done, _ = idle_step(
                    env, trace.actions[trace_step].tolist(), wait_started=not waiting
                )
            waiting = not scheduled.active
            if done != bool(trace.dones[trace_step]):
                done_mismatches += 1
            max_state_error = max(
                max_state_error,
                _snap_to_recorded_state(env, trace.states[trace_step + 1]),
            )
    finally:
        if not images_enabled:
            _set_observables_enabled(env, image_observables, True)

    if state_atol is not None and max_state_error > state_atol:
        raise RuntimeError(
            f"prefix replay drift {max_state_error:.3g} exceeds tolerance {state_atol:.3g}"
        )
    observation = _force_observations(env, observation)
    if active_steps != int(trace.acting.sum()):
        raise RuntimeError("active-step accounting mismatch")
    return RestoredEpisode(
        scheduler=scheduler,
        observation=observation,
        active_steps=active_steps,
        control_steps=control_steps,
        last_gripper=last_gripper,
        max_state_error=max_state_error,
        done_mismatches=done_mismatches,
    )


def continue_trace(
    spec: ContinuationSpec,
    env: TraceRecordingEnv,
    policy: Any,
    connection: PolicyConnection,
    out_path: Path,
) -> dict[str, Any]:
    """Replay one failed prefix, continue to its new deadline, and save one full trace."""

    source = load_libero_trace(spec.trace)
    source_model = getattr(source.meta, "model", None)
    target_model = connection.model.model_dump(mode="json") if connection.model else {
        "openpi_config": connection.openpi_config, "checkpoint": connection.checkpoint,
    }
    if source_model is None:
        source_model = {"openpi_config": getattr(source.meta, "openpi_config", connection.openpi_config),
                        "checkpoint": getattr(source.meta, "checkpoint", connection.checkpoint)}
    if source_model != target_model or source.meta.action_chunk_steps != connection.action_chunk_steps:
        raise ValueError("continuation checkpoint or action chunk does not match source trace")
    if getattr(source.meta, "policy_seed", connection.policy_seed) != connection.policy_seed:
        raise ValueError("continuation policy seed does not match source trace")
    source_identity = getattr(source.meta, "policy_identity", None)
    if source_identity is not None:
        from vlalagbench.benchmarks.libero.policy_server import METADATA_KEY
        if policy.get_server_metadata().get(METADATA_KEY) != source_identity:
            raise ValueError("continuation policy identity does not match source trace")
    original_active_steps = int(source.acting.sum())
    if spec.max_active_steps <= original_active_steps:
        raise ValueError("extended deadline must exceed the recorded active-step count")
    restored = replay_prefix(env, source, spec.scheme)
    task_id = int(source.meta.task_id)
    episode_id = _episode_id(spec.trace)
    observation = restored.observation
    scheduler = restored.scheduler
    active_steps = restored.active_steps
    control_steps = restored.control_steps
    last_gripper = restored.last_gripper
    new_policy_requests = 0
    success = False
    waiting = source.num_steps > int(source.meta.num_steps_wait) and not bool(source.acting[-1])

    while active_steps < spec.max_active_steps:
        request = scheduler.request(control_steps)
        if request is not None:
            element = _policy_element(observation, str(source.meta.task_description), 224, connection.model)
            element[REQUEST_KEY] = {
                "policy_seed": connection.policy_seed,
                "task_id": task_id,
                "episode_id": episode_id,
                "request_id": request.id,
            }
            response = policy.infer(element)
            _validate_response_identity(response, element[REQUEST_KEY], backend=getattr(connection.model, "backend", "openpi"))
            from vlalagbench.benchmarks.libero.policy import validate_actions
            chunk = validate_actions(response["actions"], connection.action_chunk_steps)
            scheduler.deliver(request, chunk)
            _record_request(
                env,
                request.id,
                request.sent_step,
                request.available_step,
                chunk,
                response.get("server_timing", {}),
                response.get(RESPONSE_KEY),
            )
            new_policy_requests += 1

        scheduled = scheduler.action(control_steps)
        if scheduled.active:
            action = np.asarray(scheduled.action, dtype=np.float64)
            active_steps += 1
            last_gripper = float(action[-1])
        else:
            action = np.zeros_like(source.actions[0], dtype=np.float64)
            action[-1] = last_gripper
        _record_step(env, scheduled)
        env.set_dynamic_step(control_steps)
        if scheduled.active:
            observation, _, done, _ = env.step(action.tolist())
        else:
            observation, _, done, _ = idle_step(env, action.tolist(), wait_started=not waiting)
        waiting = not scheduled.active
        control_steps += 1
        if done:
            success = True
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    extended = env.finish(
        extra_meta={
            **_source_experiment_meta(source),
            "policy": f"websocket://{connection.host}:{connection.port}",
            "deadline_extension": {
                "source_trace": str(spec.trace),
                "scheme": spec.scheme,
                "original_active_steps": original_active_steps,
                "max_active_steps": spec.max_active_steps,
                "new_policy_requests": new_policy_requests,
                "policy_implementation": "single",
                "policy_seed": connection.policy_seed,
                "deterministic_ops": connection.deterministic_ops,
                "pytorch_compile_mode": connection.compile_mode,
                "prefix_replay_max_state_error": restored.max_state_error,
                "prefix_replay_done_mismatches": restored.done_mismatches,
            },
        }
    )
    extended.meta = TraceMeta.model_validate(
        {
            **extended.meta.model_dump(mode="json"),
            "environment": source.meta.environment.model_dump(mode="json"),
            "success": success,
        }
    )
    _replace_recorded_prefix(extended, source)
    extended.save(out_path)
    return {
        "source_trace": str(spec.trace),
        "extended_trace": str(out_path),
        "task_suite": str(source.meta.task_suite),
        "task_id": task_id,
        "episode_id": episode_id,
        "scheme": spec.scheme,
        "latency_steps": int(source.meta.latency_steps),
        "original_active_steps": original_active_steps,
        "max_active_steps": spec.max_active_steps,
        "success": success,
        "first_success_active_step": active_steps if success else None,
        "total_active_steps": active_steps,
        "original_control_steps": source.num_steps - int(source.meta.num_steps_wait),
        "total_control_steps": control_steps,
        "new_policy_requests": new_policy_requests,
        "prefix_replay_max_state_error": restored.max_state_error,
        "prefix_replay_done_mismatches": restored.done_mismatches,
        "error": None,
    }


_WORKER_POLICY: Any = None
_WORKER_CONNECTION: PolicyConnection | None = None
_WORKER_OUT: Path | None = None
_WORKER_TAG: str | None = None


def run_manifest(
    manifest: Path,
    out_dir: Path,
    connection: PolicyConnection,
    *,
    workers: int,
    shard_size: int = 24,
    worker_tag: str = "local",
    force: bool = False,
) -> dict[str, Any]:
    """Run an idempotent, process-parallel continuation manifest."""

    if workers <= 0 or shard_size <= 0:
        raise ValueError("workers and shard_size must be positive")
    specs = [ContinuationSpec.from_json(json.loads(line)) for line in manifest.read_text().splitlines()]
    completed = _completed_sources(out_dir)
    completed_in_manifest = {str(spec.trace) for spec in specs if str(spec.trace) in completed}
    remaining = list(specs) if force else [
        spec for spec in specs if str(spec.trace) not in completed_in_manifest
    ]
    shards = _homogeneous_shards(remaining, shard_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = out_dir / "progress"
    progress_dir.mkdir(exist_ok=True)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"

    counts = {
        "completed_before": 0 if force else len(completed_in_manifest),
        "attempted": 0,
        "successes": 0,
        "errors": 0,
    }
    if shards:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(shards)),
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(connection, out_dir, worker_tag),
        ) as executor:
            futures = [executor.submit(_run_continuation_shard, shard) for shard in shards]
            for future in as_completed(futures):
                result = future.result()
                counts["attempted"] += result["attempted"]
                counts["successes"] += result["successes"]
                counts["errors"] += result["errors"]
                logger.info("continuation progress: %s", counts)
    rows = _read_progress(out_dir)
    rows_by_source = {row["source_trace"]: row for row in rows}
    consolidated = [rows_by_source[str(spec.trace)] for spec in specs if str(spec.trace) in rows_by_source]
    (out_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in consolidated),
        encoding="utf-8",
    )
    return {
        "manifest": str(manifest),
        "out_dir": str(out_dir),
        "requested": len(specs),
        "completed": len(consolidated),
        "successes": sum(bool(row["success"]) for row in consolidated),
        "errors": sum(row.get("error") is not None for row in consolidated),
        **counts,
    }


def _initialize_worker(connection: PolicyConnection, out_dir: Path, worker_tag: str) -> None:
    global _WORKER_POLICY, _WORKER_CONNECTION, _WORKER_OUT, _WORKER_TAG
    config = EvalConfig(
        model=connection.model,
        action_chunk_steps=connection.action_chunk_steps,
        host=connection.host,
        port=connection.port,
        openpi_config=connection.openpi_config,
        checkpoint=connection.checkpoint,
        pytorch_compile_mode=connection.compile_mode,
        deterministic_ops=connection.deterministic_ops,
        policy_seed=connection.policy_seed,
    )
    _WORKER_POLICY = make_policy(config)
    _WORKER_CONNECTION = connection
    _WORKER_OUT = out_dir
    _WORKER_TAG = worker_tag


def _run_continuation_shard(specs: Sequence[ContinuationSpec]) -> dict[str, int]:
    if _WORKER_POLICY is None or _WORKER_CONNECTION is None or _WORKER_OUT is None:
        raise RuntimeError("continuation worker was not initialized")
    progress_path = _WORKER_OUT / "progress" / f"{_WORKER_TAG}-{os.getpid()}.jsonl"
    counts = {"attempted": 0, "successes": 0, "errors": 0}
    for spec in specs:
        counts["attempted"] += 1
        env: TraceRecordingEnv | None = None
        try:
            source = load_libero_trace(spec.trace)
            env = TraceRecordingEnv(
                make_env_from_trace(source.meta, hard_reset=True, ignore_done=True)
            )
            out_path = (
                _WORKER_OUT
                / "traces"
                / spec.trace.parent.name
                / f"{spec.trace.stem}_extended.npz"
            )
            row = continue_trace(spec, env, _WORKER_POLICY, _WORKER_CONNECTION, out_path)
            counts["successes"] += int(row["success"])
        except Exception as error:
            logger.exception("continuation failed for %s", spec.trace)
            counts["errors"] += 1
            row = {
                "source_trace": str(spec.trace),
                "extended_trace": None,
                "success": False,
                "error": f"{type(error).__name__}: {error}",
            }
        finally:
            if env is not None:
                env.close()
        with progress_path.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return counts


def _homogeneous_shards(
    specs: Sequence[ContinuationSpec], shard_size: int
) -> list[tuple[ContinuationSpec, ...]]:
    groups: dict[tuple[str, int], list[ContinuationSpec]] = {}
    for spec in specs:
        key = (str(spec.trace.parent), _task_id(spec.trace))
        groups.setdefault(key, []).append(spec)
    shards = []
    for group in groups.values():
        for start in range(0, len(group), shard_size):
            shards.append(tuple(group[start : start + shard_size]))
    return shards


def _completed_sources(out_dir: Path) -> set[str]:
    return {
        row["source_trace"]
        for row in _read_progress(out_dir)
        if row.get("error") is None and row.get("extended_trace") and Path(row["extended_trace"]).exists()
    }


def _read_progress(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((out_dir / "progress").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            rows.append(json.loads(line))
    return rows


def _source_experiment_meta(trace: EpisodeTrace) -> dict[str, Any]:
    source = trace.meta.model_dump(mode="json")
    for key in (
        "schema_version",
        "environment",
        "control_freq",
        "num_steps",
        "success",
        "recorded_at",
        "platform",
        "python_version",
        "mujoco_version",
        "robosuite_version",
        "numpy_version",
        "repo_commit",
        "nq",
        "nv",
        "state_dim",
        "error",
        "request_audits",
    ):
        source.pop(key, None)
    return source


def _validate_response_identity(
    response: dict[str, Any], request_identity: dict[str, int], *, backend: str = "openpi"
) -> None:
    from vlalagbench.benchmarks.libero.policy import validate_response_identity
    try:
        validate_response_identity(response, request_identity, backend=backend)
    except RuntimeError as exc:
        raise RuntimeError("policy response identity does not match its continuation request") from exc


def _optional_index(value: np.integer) -> int | None:
    parsed = int(value)
    return None if parsed < 0 else parsed


def _image_observable_names(env: Any) -> tuple[str, ...]:
    names = getattr(env, "observation_names", ())
    return tuple(str(name) for name in names if str(name).endswith("_image"))


def _set_observables_enabled(env: Any, names: Sequence[str], enabled: bool) -> None:
    modify = getattr(env, "modify_observable", None)
    if modify is None:
        return
    for name in names:
        modify(name, "enabled", enabled)


def _restore_dynamic_actor_origins(env: Any, trace: EpisodeTrace) -> None:
    """Restore jointless model poses that flattened MuJoCo state does not contain."""

    method = getattr(env, "restore_actor_origins", None)
    if method is not None:
        method(trace.model_xml)
        return
    if trace.model_xml is None:
        return
    actors = getattr(trace.meta, "dynamics", ())
    if not actors:
        return
    root = ET.fromstring(trace.model_xml)
    sim = get_mj_sim(env)
    for actor in actors:
        target = str(actor["target"])
        body = root.find(f".//body[@name='{target}']")
        if body is None:
            raise RuntimeError(f"dynamic body {target!r} is missing from the recorded model XML")
        body_id = sim.model.body_name2id(target)
        position = np.fromstring(body.attrib.get("pos", ""), sep=" ")
        quaternion = np.fromstring(body.attrib.get("quat", ""), sep=" ")
        if position.shape == (3,):
            sim.model.body_pos[body_id] = position
        if quaternion.shape == (4,):
            sim.model.body_quat[body_id] = quaternion
    sim.forward()


def _snap_to_recorded_state(env: Any, recorded_state: np.ndarray) -> float:
    error = float(np.max(np.abs(get_sim_state(env) - recorded_state)))
    set_sim_state(env, recorded_state)
    return error


def _force_observations(env: Any, fallback: Any) -> Any:
    method = getattr(env, "_get_observations", None)
    return fallback if method is None else method(force_update=True)


def _recorded_timing(trace: EpisodeTrace, request_id: int) -> dict[str, float]:
    return {
        name: getattr(trace, name)[request_id]
        for name in ("queue_ms", "infer_ms", "batch_size", "max_batch_size")
    }


def _replace_recorded_prefix(extended: EpisodeTrace, source: EpisodeTrace) -> None:
    """Keep the saved prefix byte-for-byte equal to its original trace arrays."""

    steps = source.num_steps
    requests = len(source.chunks)
    extended.states[: steps + 1] = source.states
    extended.actions[:steps] = source.actions
    extended.rewards[:steps] = source.rewards
    extended.dones[:steps] = source.dones
    extended.acting[:steps] = source.acting
    extended.chunk_ids[:steps] = source.chunk_ids
    extended.action_indices[:steps] = source.action_indices
    extended.observation_steps[:steps] = source.observation_steps
    extended.request_steps[:requests] = source.request_steps
    extended.arrival_steps[:requests] = source.arrival_steps
    extended.chunks[:requests] = source.chunks
    for name in ("queue_ms", "infer_ms", "batch_size", "max_batch_size"):
        getattr(extended, name)[:requests] = getattr(source, name)


def _episode_id(path: Path) -> int:
    marker = path.stem.rsplit("_ep", 1)
    if len(marker) != 2 or not marker[1].isdigit():
        raise ValueError(f"cannot derive episode id from {path.name}")
    return int(marker[1])


def _task_id(path: Path) -> int:
    marker = path.stem.rsplit("_task", 1)
    if len(marker) != 2:
        raise ValueError(f"cannot derive task id from {path.name}")
    value = marker[1].split("_", 1)[0]
    if not value.isdigit():
        raise ValueError(f"cannot derive task id from {path.name}")
    return int(value)
