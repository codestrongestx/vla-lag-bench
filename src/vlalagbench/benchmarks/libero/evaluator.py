"""Evaluate a pi0.5-style policy on LIBERO while recording reproducible traces.

The rollout loop mirrors ``third_party/openpi/examples/libero/main.py`` (same
observation preprocessing, action chunking, and wait steps) with every episode
wrapped in :class:`~vlalagbench.simulators.mujoco.tracing.TraceRecordingEnv`, so each
run leaves behind one reproducible ``.npz`` trace per episode.

The policy is anything with an ``infer(element) -> {"actions": ...}`` method.
The default websocket client verifies the server identity before evaluation;
the built-in ``dummy`` policy exercises the pipeline without a model server.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import logging
import multiprocessing
import os
import pathlib
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from vlalagbench.benchmarks.libero import _vendor
from vlalagbench.benchmarks.libero import policy_server
from vlalagbench.benchmarks.libero.policy import REQUEST_KEY, RESPONSE_KEY
from vlalagbench.config import InitialStateOverride, TrajectorySpec
from vlalagbench.simulators.mujoco.camera_gate import CameraGate
from vlalagbench.simulators.mujoco.dynamics import DynamicEnv
from vlalagbench.benchmarks.libero.envs import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    make_env,
    reset_to_init_state,
    warm_up,
)
from vlalagbench.timing import ChunkScheduler, TimingConfig
from vlalagbench.simulators.mujoco.tracing import TraceRecordingEnv, get_sim_state, git_commit
from vlalagbench.trajectory import resolve_dynamic_phases, sample_trajectory

logger = logging.getLogger(__name__)

_WORKER_CONFIG: EvalConfig | None = None
_WORKER_POLICY: Any = None
_WORKER_TASK_SUITE: Any = None


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    task_suite: str = "libero_spatial"
    task_ids: Sequence[int] | None = None  # None = all tasks in the suite
    trials_per_task: int = 1
    episode_ids: Sequence[int] | None = None  # None = every episode below trials_per_task
    policy: str = "websocket"  # "websocket" (openpi policy server) or "dummy"
    model: Any = None
    openpi_config: str = "pi05_libero"
    checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero"
    mode: Literal["synchronous", "lazy", "continuous"] = "synchronous"
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    action_chunk_steps: int = 10
    execution_horizon_steps: int = 5
    latency_steps: int = 0
    unavailable_action: Sequence[float] = tuple(LIBERO_DUMMY_ACTION)
    dynamics: Sequence[Any] = ()
    dynamics_by_task: Mapping[int, Sequence[Any]] = dataclasses.field(default_factory=dict)
    trajectory: TrajectorySpec | None = None
    dynamics_seed: int = 7
    num_steps_wait: int = 10  # let objects settle before the policy acts
    max_steps: int | None = None  # required to roll out; see resolved_max_steps
    seed: int = 7
    policy_seed: int = 7
    pytorch_compile_mode: str = "max-autotune-no-cudagraphs"
    deterministic_ops: bool = False
    # Server request shaping; None means single-request serving.
    batching: dict[str, Any] | None = None
    workers: int = 1
    initial_state_indices: Sequence[int] | None = None
    initial_state_overrides: Sequence[InitialStateOverride] = ()
    process_start_method: Literal["spawn", "fork"] = "spawn"
    env_resolution: int = LIBERO_ENV_RESOLUTION
    render_policy_observations_only: bool = False
    dynamic_episode_queue: bool = False
    out_dir: str | pathlib.Path | None = None  # None = artifacts/<timestamp>_eval_<suite>

    def resolved_out_dir(self) -> pathlib.Path:
        if self.out_dir is not None:
            return pathlib.Path(self.out_dir)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return _vendor.REPO_ROOT / "artifacts" / f"{stamp}_eval_{self.task_suite}"

    def resolved_max_steps(self) -> int:
        # A deadline is stated or the run does not happen. Falling back to a
        # per-suite constant would let a result carry a horizon nobody chose
        # and nothing records as a default rather than a decision.
        if self.max_steps is None:
            raise ValueError("max_steps must be set: an episode deadline cannot be defaulted")
        return self.max_steps


class DummyPolicy:
    """No-op stand-in for a policy server (arm still, gripper open)."""

    def __init__(self, action_chunk_steps: int = 10):
        self.action_chunk_steps = action_chunk_steps

    def infer(self, element: dict[str, Any]) -> dict[str, Any]:
        return {"actions": np.tile(np.asarray(LIBERO_DUMMY_ACTION), (self.action_chunk_steps, 1))}


def make_policy(config: EvalConfig):
    if config.policy == "dummy":
        return DummyPolicy(config.action_chunk_steps)
    if config.policy == "websocket":
        from openpi_client import websocket_client_policy

        policy = websocket_client_policy.WebsocketClientPolicy(config.host, config.port)
        policy_server.validate_metadata(
            policy.get_server_metadata(),
            model=config.model,
            openpi_config=config.openpi_config,
            checkpoint=config.checkpoint,
            action_chunk_steps=config.action_chunk_steps,
            pytorch_compile_mode=config.pytorch_compile_mode,
            deterministic_ops=config.deterministic_ops,
            batching=config.batching,
        )
        return policy
    raise ValueError(f"unknown policy: {config.policy!r}")


def _policy_element(obs, task_description: str, resize_size: int, model=None) -> dict[str, Any]:
    """Apply the selected checkpoint's LIBERO observation conventions."""
    from vlalagbench.config import Gr00tModelSpec
    if isinstance(model, Gr00tModelSpec):
        from vlalagbench.benchmarks.libero.gr00t_adapter import format_observation
        return format_observation(obs, task_description)
    if getattr(model, "backend", None) == "molmoact2":
        from vlalagbench.benchmarks.libero.molmoact2_adapter import format_observation
        return format_observation(obs, task_description)
    from openpi_client import image_tools
    from robosuite.utils.transform_utils import quat2axisangle

    # Rotate 180 degrees to match the training preprocessing.
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, resize_size, resize_size)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
        ),
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"].copy()),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(task_description),
    }


def run_episode(
    env,
    task_description: str,
    policy,
    config: EvalConfig,
    obs=None,
    *,
    task_id: int = 0,
    episode_id: int = 0,
) -> dict[str, Any]:
    """Roll out one episode on an already reset+initialized env."""
    max_steps = config.resolved_max_steps()
    success = False
    error = None
    policy_requests = 0
    acting_steps = 0
    control_steps = 0
    last_gripper = float(config.unavailable_action[-1])
    try:
        prepare_dynamic_step = getattr(env, "prepare_dynamic_step", None)
        dynamic_step_prepared = bool(
            prepare_dynamic_step(0) if prepare_dynamic_step is not None else False
        )
        if dynamic_step_prepared:
            env.refresh_current_state()
            obs = env.regenerate_obs_from_state(get_sim_state(env))
        obs = warm_up(
            env,
            obs,
            config.num_steps_wait,
            hold_dynamic_time=0.0 if dynamic_step_prepared else None,
        )
        if config.num_steps_wait and prepare_dynamic_step is not None:
            dynamic_step_prepared = bool(prepare_dynamic_step(0))
            if dynamic_step_prepared:
                env.refresh_current_state()
                obs = env.regenerate_obs_from_state(get_sim_state(env))

        gate = CameraGate(env) if config.render_policy_observations_only else None
        scheduler = ChunkScheduler(
            TimingConfig(
                scheme=config.mode,
                action_chunk_steps=config.action_chunk_steps,
                execution_horizon_steps=config.execution_horizon_steps,
                latency_steps=config.latency_steps,
            )
        )
        waiting = False
        while acting_steps < max_steps:
            request = scheduler.request(control_steps)
            if request is not None:
                if gate is not None:
                    gate.require_fresh()
                element = _policy_element(obs, task_description, config.resize_size, config.model)
                element[REQUEST_KEY] = {
                    "policy_seed": config.policy_seed,
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "request_id": request.id,
                }
                response = policy.infer(element)
                from vlalagbench.benchmarks.libero.policy import validate_response_identity, validate_actions
                if config.policy == "websocket":
                    validate_response_identity(response, element[REQUEST_KEY], backend=getattr(config.model, "backend", "openpi"))
                action_chunk = validate_actions(response["actions"], config.action_chunk_steps)
                policy_requests += 1
                scheduler.deliver(request, action_chunk)
                _record_request(
                    env,
                    request.id,
                    request.sent_step,
                    request.available_step,
                    action_chunk,
                    response.get("server_timing", {}),
                    response.get(RESPONSE_KEY),
                )
            scheduled = scheduler.action(control_steps)
            if scheduled.active:
                action = np.asarray(scheduled.action, dtype=np.float64)
                acting_steps += 1
                last_gripper = float(action[-1])
            else:
                action = np.asarray(config.unavailable_action, dtype=np.float64).copy()
                action[:-1] = 0.0
                action[-1] = last_gripper
            if gate is not None:
                # Render only when the next control step will read this observation.
                gate.open = acting_steps < max_steps and (
                    copy.deepcopy(scheduler).request(control_steps + 1) is not None
                )
            _record_step(env, scheduled)
            if control_steps != 0 or not dynamic_step_prepared:
                env.set_dynamic_step(control_steps)
            if scheduled.active:
                obs, _, done, _ = env.step(action.tolist())
            else:
                obs, _, done, _ = idle_step(env, action.tolist(), wait_started=not waiting)
            waiting = not scheduled.active
            control_steps += 1
            if done:
                success = True
                break
    except Exception as e:  # record the partial episode rather than losing the run
        logger.exception("episode aborted at control step %d", control_steps)
        error = f"{type(e).__name__}: {e}"
    return {
        "success": success,
        "steps": acting_steps,
        "control_steps": control_steps,
        "wait_steps": control_steps - acting_steps,
        "policy_requests": policy_requests,
        "error": error,
    }


def _record_request(
    env,
    request_id: int,
    sent_step: int,
    arrival_step: int,
    chunk: np.ndarray,
    timing: Mapping[str, float],
    audit: dict | None = None,
) -> None:
    method = getattr(env, "record_request", None)
    if method is not None:
        method(request_id, sent_step, arrival_step, chunk, timing)
    audit_method = getattr(env, "record_request_audit", None)
    if audit is not None and audit_method is not None:
        audit_method(request_id, audit)


def idle_step(env, action, *, wait_started: bool):
    """Step one control tick without a policy action, holding the arm still.

    robosuite's delta OSC resets the position goal on a zero action but keeps the
    previous orientation goal, so a zero action alone lets the arm keep turning.
    On the first tick of a wait both goals are set to the current pose, and
    set_goal is skipped on every wait tick so they stay there.
    """
    controllers = [robot.controller for robot in _robots(env)]
    if wait_started:
        for controller in controllers:
            controller.update(force=True)
            controller.goal_pos = np.array(controller.ee_pos)
            controller.goal_ori = np.array(controller.ee_ori_mat)
    with _keep_goals(controllers):
        return env.step(action)


@contextlib.contextmanager
def _keep_goals(controllers):
    # robosuite calls set_goal once per env.step; skipping it keeps the held pose.
    originals = [controller.set_goal for controller in controllers]
    for controller in controllers:
        controller.set_goal = lambda *args, **kwargs: None
    try:
        yield
    finally:
        for controller, original in zip(controllers, originals):
            controller.set_goal = original


def _robots(env) -> list[Any]:
    # Wrappers forward attribute lookups, so find the env that owns ``robots``.
    current = env
    for _ in range(8):
        if "robots" in getattr(current, "__dict__", {}):
            return list(current.robots)
        current = getattr(current, "__dict__", {}).get("env")
        if current is None:
            break
    raise AttributeError(f"could not find robosuite robots on {type(env).__name__}")


def _record_step(env, scheduled) -> None:
    method = getattr(env, "record_step", None)
    if method is not None:
        method(
            acting=scheduled.active,
            chunk_id=scheduled.chunk_id,
            action_index=scheduled.index_in_chunk,
            observation_step=scheduled.observation_step,
        )


def run_eval(config: EvalConfig) -> dict[str, Any]:
    libero = _vendor.libero()
    np.random.seed(config.seed)
    task_suite = libero.benchmark.get_benchmark_dict()[config.task_suite]()
    task_ids = list(config.task_ids) if config.task_ids is not None else list(range(task_suite.n_tasks))
    out_dir = config.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    if config.workers <= 0:
        raise ValueError("workers must be positive")
    config.resolved_max_steps()  # fail before the first episode, not inside it
    episode_ids = (
        list(config.episode_ids)
        if config.episode_ids is not None
        else list(range(config.trials_per_task))
    )
    shards = (
        _episode_queue(task_ids, episode_ids)
        if config.dynamic_episode_queue and config.workers > 1
        else _episode_shards(task_ids, episode_ids, config.workers)
    )
    if config.workers == 1:
        policy = make_policy(config)
        episodes = [
            episode
            for task_id, episode_ids in shards
            for episode in _run_shard(task_suite, policy, config, out_dir, task_id, episode_ids)
        ]
    else:
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[variable] = "1"
        context = multiprocessing.get_context(config.process_start_method)
        with ProcessPoolExecutor(
            max_workers=min(config.workers, len(shards)),
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(config,),
        ) as executor:
            groups = list(executor.map(_worker_shard, shards, chunksize=1))
        episodes = [episode for group in groups for episode in group]
    episodes.sort(key=lambda episode: (episode["task_id"], episode["episode"]))

    summary = {
        "config": {**dataclasses.asdict(config), "model": config.model.model_dump(mode="json") if config.model else None, "out_dir": str(out_dir), "task_ids": task_ids},
        "num_episodes": len(episodes),
        "num_successes": sum(e["success"] for e in episodes),
        "episodes": episodes,
    }
    logger.info(
        "done: %d/%d episodes succeeded; traces in %s",
        summary["num_successes"],
        summary["num_episodes"],
        out_dir,
    )
    return summary


def _episode_shards(
    task_ids: Sequence[int],
    episode_ids: Sequence[int],
    workers: int,
) -> list[tuple[int, tuple[int, ...]]]:
    per_task = max(1, workers // len(task_ids))
    shards = []
    for task_id in task_ids:
        task_shards = min(per_task, len(episode_ids))
        for shard in range(task_shards):
            shards.append((task_id, tuple(episode_ids[shard::task_shards])))
    return shards


def _episode_queue(
    task_ids: Sequence[int], episode_ids: Sequence[int]
) -> list[tuple[int, tuple[int, ...]]]:
    """One work item per episode; an idle worker takes the next one."""
    return [(task_id, (episode_id,)) for task_id in task_ids for episode_id in episode_ids]


def _initialize_worker(config: EvalConfig) -> None:
    global _WORKER_CONFIG, _WORKER_POLICY, _WORKER_TASK_SUITE
    libero = _vendor.libero()
    np.random.seed(config.seed)
    _WORKER_CONFIG = config
    _WORKER_POLICY = make_policy(config)
    _WORKER_TASK_SUITE = libero.benchmark.get_benchmark_dict()[config.task_suite]()


def _worker_shard(shard: tuple[int, tuple[int, ...]]) -> list[dict[str, Any]]:
    if _WORKER_CONFIG is None or _WORKER_POLICY is None or _WORKER_TASK_SUITE is None:
        raise RuntimeError("evaluation worker was not initialized")
    return _run_shard(
        _WORKER_TASK_SUITE,
        _WORKER_POLICY,
        _WORKER_CONFIG,
        _WORKER_CONFIG.resolved_out_dir(),
        *shard,
    )


def _run_shard(
    task_suite: Any,
    policy: Any,
    config: EvalConfig,
    out_dir: pathlib.Path,
    task_id: int,
    episode_ids: Sequence[int],
) -> list[dict[str, Any]]:
    task = task_suite.get_task(task_id)
    init_states = task_suite.get_task_init_states(task_id)
    env = None
    episodes = []
    try:
        for episode_idx in episode_ids:
            # Construction can consume RNGs before LIBERO's seed() runs.
            # Reset both streams before constructing an isolated episode.
            np.random.seed(config.seed)
            random.seed(config.seed)
            env = _recording_env(task, config, task_id)
            logger.info("task %d (%s), episode %d", task_id, task.language, episode_idx)
            sampled_trajectory = sample_trajectory(
                config.trajectory,
                base_seed=config.dynamics_seed,
                task_id=task_id,
                episode_id=episode_idx,
            )
            base_dynamics = config.dynamics_by_task.get(task_id, config.dynamics)
            init_state_index, environment_seed = _select_initial_state(
                config, episode_idx, len(init_states)
            )
            resolved_dynamics = resolve_dynamic_phases(
                base_dynamics,
                base_seed=config.dynamics_seed,
                task_id=task_id,
                initial_state_id=f"official:{init_state_index}",
            )
            if sampled_trajectory is not None:
                resolved_dynamics.append(sampled_trajectory)
            env.set_dynamic_actors(resolved_dynamics)
            initial_override = next(
                (
                    override
                    for override in config.initial_state_overrides
                    if override.task_id == task_id
                    and override.initial_state_index == init_state_index
                ),
                None,
            )
            obs = _initialize_episode(
                env,
                init_states,
                init_state_index,
                initial_override.object_positions_m if initial_override else {},
                initial_override.object_yaws_rad if initial_override else {},
            )
            result = run_episode(
                env,
                task.language,
                policy,
                config,
                obs=obs,
                task_id=task_id,
                episode_id=episode_idx,
            )
            trace = env.finish(
                extra_meta={
                    **({"policy_identity": policy.get_server_metadata()[policy_server.METADATA_KEY]}
                       if config.model is not None and hasattr(policy, "get_server_metadata") else {}),
                    "policy_seed": config.policy_seed,
                    "init_state_strategy": "cycle_official",
                    "episode_id": episode_idx,
                    "episode_reset": "fresh",
                    "init_state_index": init_state_index,
                    "environment_seed": environment_seed,
                    "initial_state_override": (
                        initial_override.model_dump(mode="json")
                        if initial_override
                        else None
                    ),
                    "dynamics": [
                        actor.model_dump(mode="json") for actor in resolved_dynamics
                    ],
                    "error": result["error"],
                    **({"render_policy_observations_only": True}
                       if config.render_policy_observations_only else {}),
                }
            )
            trace_path = trace.save(
                out_dir / f"{config.task_suite}_task{task_id:02d}_ep{episode_idx:03d}.npz"
            )
            episodes.append(
                {
                    "task_id": task_id,
                    "episode": episode_idx,
                    "init_state_index": init_state_index,
                    "environment_seed": environment_seed,
                    "trace": trace_path.name,
                    **result,
                }
            )
            logger.info(
                "  -> success=%s acting=%d control=%d trace=%s",
                result["success"],
                result["steps"],
                result["control_steps"],
                trace_path.name,
            )
            env.close()
            env = None
    finally:
        if env is not None:
            env.close()
    return episodes


def _select_initial_state(
    config: EvalConfig, episode_idx: int, state_count: int
) -> tuple[int, int]:
    if config.initial_state_indices is not None:
        init_state_index = config.initial_state_indices[
            episode_idx % len(config.initial_state_indices)
        ]
        if init_state_index >= state_count:
            raise IndexError(
                f"official init state {init_state_index} is outside "
                f"[0, {state_count})"
            )
        return init_state_index, config.seed

    return episode_idx % state_count, config.seed


def _initialize_episode(
    env: Any,
    init_states: Sequence[Any],
    init_state_index: int,
    object_positions_m: Mapping[str, Sequence[float]] | None = None,
    object_yaws_rad: Mapping[str, float] | None = None,
) -> Any:
    env.set_initial_object_positions(object_positions_m or {})
    env.set_initial_object_yaws(object_yaws_rad or {})
    return reset_to_init_state(env, init_states[init_state_index])


def _recording_env(task: Any, config: EvalConfig, task_id: int) -> TraceRecordingEnv:
    raw_env = make_env(
        task,
        config.env_resolution,
        config.seed,
        hard_reset=True,
        # LIBERO replaces robosuite's horizon ``done`` with task success, but
        # robosuite still marks the environment terminated internally.  A
        # latency-aware rollout can legitimately need more than the default
        # 1,000 control steps to consume its configured acting-step budget.
        ignore_done=True,
    )
    dynamics = config.dynamics_by_task.get(task_id, config.dynamics)
    return TraceRecordingEnv(
        DynamicEnv(raw_env, dynamics),
        meta={
            "environment": {
                "kind": "libero",
                "problem_folder": task.problem_folder,
                "bddl_file": task.bddl_file,
                "resolution": config.env_resolution,
                "seed": config.seed,
            },
            "task_suite": config.task_suite,
            "task_id": task_id,
            "task_name": task.name,
            "task_description": task.language,
            "libero_commit": git_commit(_vendor.LIBERO_ROOT),
            "num_steps_wait": config.num_steps_wait,
            "action_chunk_steps": config.action_chunk_steps,
            "execution_horizon_steps": config.execution_horizon_steps,
            "latency_steps": config.latency_steps,
            "idle_control": "hold_pose",
            **({"model": config.model.model_dump(mode="json")} if config.model is not None else {}),
            "openpi_config": config.openpi_config,
            "checkpoint": config.checkpoint,
            "trajectory": (
                config.trajectory.model_dump(mode="json")
                if config.trajectory is not None
                else None
            ),
            "dynamic_step_zero_prepared_before_warmup": True,
            "policy": config.policy
            if config.policy != "websocket"
            else f"websocket://{config.host}:{config.port}",
            "dynamics": [actor.model_dump(mode="json") for actor in dynamics],
        },
    )
