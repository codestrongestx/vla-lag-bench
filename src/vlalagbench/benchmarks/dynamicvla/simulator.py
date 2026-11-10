"""Run one pinned DOM suite with exact, control-step DynamicVLA latency."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
import importlib.metadata
import json
import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from vlalagbench.files import sha256 as _sha256, tree_sha256

from vlalagbench.benchmarks.dynamicvla.motion import (
    PlanarDriver,
    actors_from_json,
    capture_episode_motion,
)
from vlalagbench.benchmarks.dynamicvla.protocol import (
    derive_episode_seed,
    receipt_errors,
)
from vlalagbench.benchmarks.dynamicvla.runtime import (
    ENV_REUSE_CPU_OBSERVATION,
    action_tensor,
    episode_length_s_for_steps as _episode_length_s_for_steps,
    evaluate_termination_terms,
    flatten_policy_observation,
    hold_pose_action,
    owned_cpu_policy_state,
    policy_observation_message,
    read_policy_observation,
    receive_reply,
    refresh_sensors,
    step_and_capture_terminal,
    wait_for_action,
)
from vlalagbench.benchmarks.dynamicvla.tracing import (
    DynamicVLATrace,
    DynamicVLATraceRecorder,
)


def _reset_active_action_deadline(upstream: Any, env: Any) -> int:
    """Restore the configured or native per-episode deadline before applying pause extensions."""

    step_dt = float(env.unwrapped.step_dt)
    configured = getattr(upstream.get_sim_results, "cfg", None)
    episode_length_s = (
        configured.get("episode_length_s") if isinstance(configured, dict) else None
    )
    override = getattr(upstream.get_sim_results, "vlalagbench_max_episode_steps", None)
    if override is not None or episode_length_s is not None:
        deadline_steps = int(override) if override is not None else round(float(episode_length_s) / step_dt)
        if deadline_steps <= 0:
            raise ValueError("DynamicVLA environment has no positive episode deadline")
        env.unwrapped.cfg.episode_length_s = _episode_length_s_for_steps(
            deadline_steps, step_dt
        )
    active_action_deadline_steps = int(env.unwrapped.max_episode_length)
    if active_action_deadline_steps <= 0:
        raise ValueError("DynamicVLA environment has no positive episode deadline")
    return active_action_deadline_steps


def simulate_injected(
    upstream: Any,
    env: Any,
    obs_socket: Any,
    act_socket: Any,
    init_poses: dict[str, Any],
    *,
    mode: str,
    execution_horizon_steps: int,
    latency_steps: int,
    reply_timeout_s: float,
    action_chunk_steps: int,
    trace_images: bool = False,
) -> dict[str, Any]:
    """Advance once per unique observation and release chunks on an index grid."""

    import configs.robot_cfg
    import configs.termination_cfg

    received_action = False
    last_action = None
    result = {"status": -1, "cam_views": [], "ee_path": []}
    term_mgr = env.env.termination_manager
    done_term = configs.termination_cfg.get_done_term(term_mgr.active_terms)
    result["success_term"] = done_term
    result["timeout_terms"] = [
        name
        for name in term_mgr.active_terms
        if term_mgr.get_term_cfg(name).time_out
    ]
    poller = upstream.zmq.Poller()
    poller.register(act_socket, upstream.zmq.POLLIN)
    task_ack = receive_reply(upstream, poller, act_socket, reply_timeout_s)
    if not isinstance(task_ack, dict) or task_ack.get("task_ack") is not True:
        raise TimeoutError("DynamicVLA policy did not acknowledge the task")
    result["instruction"] = task_ack.get("instruction")
    result["policy_noise_scheme"] = task_ack.get("policy_noise_scheme")
    result["policy_torch_version"] = task_ack.get("policy_torch_version")
    result["policy_episode_seed"] = task_ack.get("episode_seed")
    literal_instruction = getattr(
        upstream.get_sim_results, "vlalagbench_expected_instruction", None
    )
    if literal_instruction is None:
        configured_instruction = getattr(upstream.get_sim_results, "cfg", {}).get(
            "instruction"
        )
        if isinstance(configured_instruction, str):
            literal_instruction = configured_instruction
    if not isinstance(literal_instruction, str) or not literal_instruction:
        raise RuntimeError("DynamicVLA simulator has no paired literal prompt")
    if result["instruction"] != literal_instruction:
        raise RuntimeError("DynamicVLA policy used a different literal prompt")
    if (
        task_ack.get("inference_mode") != mode
        or task_ack.get("action_index_origin") != ("release" if mode == "synchronous" else "observation")
        or task_ack.get("execution_horizon_steps") != execution_horizon_steps
        or task_ack.get("execution_horizon_basis") != "post_latency_action_count"
        or task_ack.get("action_chunk_steps") != action_chunk_steps
    ):
        raise RuntimeError("DynamicVLA policy uses a different inference schedule")
    active_action_deadline_steps = _reset_active_action_deadline(upstream, env)
    driver = None
    motion_actors = getattr(upstream.get_sim_results, "vlalagbench_motion_actors", None)
    if motion_actors:
        motion = capture_episode_motion(
            env.unwrapped,
            actors_from_json(motion_actors),
            1.0 / env.unwrapped.step_dt,
            driven_friction=(None if getattr(upstream.get_sim_results, "vlalagbench_driven_friction", "zero") == "native" else 0.0),
        )
        driver = PlanarDriver(motion)
        driver.prepare(env.unwrapped)
        # Step 0 places each moving target on its path before the first observation.
        driver.apply(env.unwrapped, 0)
        driver.place_carried(env.unwrapped)
        refresh_sensors(env.unwrapped)
        result["motion"] = motion.model_dump(mode="json")
    recorder = DynamicVLATraceRecorder(env.unwrapped, images=trace_images)
    deadline_paused_steps = 0
    stats: dict[str, Any] = {
        "applied": 0,
        "gaps": [],
        "request_events": [],
        "release_events": [],
        "applied_events": [],
        "control_events": [],
        "blocked_s": 0.0,
        "violations": 0,
        "timeouts": 0,
    }

    reuse_cpu_observation = os.environ.get(ENV_REUSE_CPU_OBSERVATION) == "1"
    while result["status"] == -1:
        cameras, state = read_policy_observation(upstream, env)
        if reuse_cpu_observation:
            state = owned_cpu_policy_state(state)
        recorder.record_observation(flatten_policy_observation(cameras, state))
        result["cam_views"].append(cameras)
        result["ee_path"].append(state["end_effector"]["pos"].cpu().numpy())
        index = len(result["cam_views"]) - 1
        obs_socket.send_pyobj(
            policy_observation_message(
                cameras,
                state,
                index=index,
                seed=int(getattr(upstream.get_sim_results, "cfg")["seed"]),
                latency_steps=latency_steps,
            )
        )

        reply = receive_reply(upstream, poller, act_socket, reply_timeout_s)
        if reply is None:
            stats["timeouts"] += 1
            result["status"] = 3
            break

        meta = reply.get("meta") if isinstance(reply, dict) else None
        if not isinstance(meta, dict) or meta.get("reply_to") != index:
            stats["violations"] += 1
            meta = None
        else:
            request = meta.get("request")
            if request is not None:
                stats["request_events"].append({**request, "step": index})
            release = meta.get("release")
            if release is not None:
                stats["release_events"].append({**release, "step": index})
            stats["blocked_s"] += float(meta.get("blocked_s") or 0.0)

        action = None
        if isinstance(reply, dict) and reply.get("action") is not None:
            action = action_tensor(upstream, reply["action"], env)
            if action is None:
                stats["violations"] += 1
        if action is not None:
            last_action = action
            received_action = True
            stats["applied"] += 1
            source = meta.get("observation_step") if meta is not None else None
            position = meta.get("action_index") if meta is not None else None
            chunk_index = meta.get("chunk_index") if meta is not None else None
            expected_position = (
                index - source - (latency_steps if mode == "synchronous" else 0)
                if isinstance(source, int)
                else None
            )
            if (
                not isinstance(source, int)
                or not isinstance(position, int)
                or not isinstance(chunk_index, int)
                or position != expected_position
            ):
                stats["violations"] += 1
                source = position = chunk_index = -1
            stats["applied_events"].append(
                {
                    "step": index,
                    "chunk_index": chunk_index,
                    "observation_step": source,
                    "action_index": position,
                }
            )
        else:
            if stats["applied_events"] and stats["applied_events"][-1]["step"] == index - 1:
                last_action = hold_pose_action(upstream.torch, env, last_action)
            stats["gaps"].append(index)
            deadline_paused_steps += 1
            env.unwrapped.cfg.episode_length_s = _episode_length_s_for_steps(
                active_action_deadline_steps + deadline_paused_steps,
                env.unwrapped.step_dt,
            )

        if last_action is None:
            robot_name = configs.robot_cfg.get_robot_name(
                env.unwrapped.scene["robot"].cfg.spawn.usd_path
            )
            last_action = init_poses[robot_name].repeat(env.unwrapped.num_envs, 1)
        stats["control_events"].append(
            {
                "step": index,
                "action": _json_action(last_action),
                "reply_to": meta.get("reply_to") if meta is not None else None,
                "wait_reason": (
                    meta.get("wait_reason") if meta is not None else "invalid_reply"
                ),
                "chunk_index": (
                    meta.get("chunk_index") if meta is not None else None
                ),
                "observation_step": (
                    meta.get("observation_step") if meta is not None else None
                ),
                "action_index": (
                    meta.get("action_index") if meta is not None else None
                ),
            }
        )
        terminal_terminations = None

        def terminal_observation() -> dict:
            nonlocal terminal_terminations
            terminal_terminations = evaluate_termination_terms(
                term_mgr, env.unwrapped
            )
            terminal_cameras, terminal_state = read_policy_observation(upstream, env)
            result["terminal_cameras"] = terminal_cameras
            return flatten_policy_observation(terminal_cameras, terminal_state)

        if driver is not None:
            driver.apply(env.unwrapped, index)
        was_terminal = step_and_capture_terminal(
            env,
            last_action,
            recorder,
            terminal_observation=terminal_observation,
        )
        terminations = terminal_terminations or {
            name: False for name in term_mgr.active_terms
        }
        success = terminations[done_term]
        done = any(terminations.values())
        if was_terminal != done:
            raise RuntimeError("DOM termination terms disagree with Isaac Lab")
        recorder.record_step(
            last_action,
            success=success,
            done=done,
            terminations=terminations,
            protocol=meta,
        )
        if success:
            result["status"] = 0
        elif done:
            result["status"] = 1 if received_action else 2

    startup_gap = 0
    for expected, gap in enumerate(stats["gaps"]):
        if gap != expected:
            break
        startup_gap += 1
    staleness_hist: dict[int, int] = {}
    for event in stats["applied_events"]:
        staleness = event["step"] - event["observation_step"]
        staleness_hist[staleness] = staleness_hist.get(staleness, 0) + 1
    receipt = {
        "episode_seed": result["policy_episode_seed"],
        "mode": mode,
        "action_index_origin": "release" if mode == "synchronous" else "observation",
        "execution_horizon_steps": execution_horizon_steps,
        "execution_horizon_basis": "post_latency_action_count",
        "injected_steps": latency_steps,
        "status": result["status"],
        "success": result["status"] == 0,
        "deadline_basis": "active_action_steps",
        "active_action_deadline_steps": active_action_deadline_steps,
        "deadline_paused_steps": deadline_paused_steps,
        "n_steps": len(stats["control_events"]),
        "applied": stats["applied"],
        "coverage": stats["applied"] / max(len(stats["control_events"]), 1),
        "startup_gap": startup_gap,
        "steady_gaps": len(stats["gaps"]) - startup_gap,
        "gap_steps": stats["gaps"],
        "requests": len(stats["request_events"]),
        "request_events": stats["request_events"],
        "releases": len(stats["release_events"]),
        "release_events": stats["release_events"],
        "applied_events": stats["applied_events"],
        "control_events": stats["control_events"],
        "staleness_hist": dict(sorted(staleness_hist.items())),
        "blocked_s": round(stats["blocked_s"], 4),
        "violations": stats["violations"],
        "timeouts": stats["timeouts"],
    }
    errors = receipt_errors(
        receipt,
        mode=mode,
        latency_steps=latency_steps,
        action_chunk_steps=action_chunk_steps,
        execution_horizon_steps=execution_horizon_steps,
    )
    receipt["errors"] = errors
    receipt["exact"] = not errors
    result["latency"] = receipt
    result["trace_recorder"] = recorder
    logging.info("[InjectedLatency] %s", json.dumps(receipt, sort_keys=True))
    return result


def _json_action(action: Any) -> Any:
    detach = getattr(action, "detach", None)
    if detach is not None:
        action = detach().cpu().numpy()
    return action.tolist()


def _write_receipt(
    path: Path,
    *,
    task_id: int,
    task_name: str,
    episode: int,
    environment_index: int,
    replica_id: int,
    environment: str,
    seed: int,
    instruction: str,
    trace: str,
    trace_sha256: str,
    result: dict[str, Any],
) -> None:
    receipt = {
        "task_id": task_id,
        "task_name": task_name,
        "episode": episode,
        "environment_index": environment_index,
        "replica_id": replica_id,
        "environment": environment,
        "seed": seed,
        "instruction": instruction,
        "success": result["status"] == 0,
        "status": result["status"],
        "trace": trace,
        "trace_sha256": trace_sha256,
        **result["latency"],
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


def _persist_trace(
    trace: DynamicVLATrace,
    trace_path: Path,
    receipt_args: dict[str, Any],
    result: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    started = time.perf_counter()
    trace.save(trace_path)
    trace_sha256 = _sha256(trace_path)
    result["latency"].update(
        {
            "trace_codec": "npz-deflate-level-1-v1",
            "trace_bytes": trace_path.stat().st_size,
            "trace_persist_s": round(time.perf_counter() - started, 4),
        }
    )
    return receipt_args, result, trace_sha256


class _TraceReceiptWriter:
    """Persist traces concurrently while committing receipts in episode order."""

    def __init__(self, max_workers: int = 1) -> None:
        if max_workers <= 0:
            raise ValueError("trace writer workers must be positive")
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="trace-writer"
        )
        self._pending: list[
            tuple[Future[tuple[dict[str, Any], dict[str, Any], str]], Path]
        ] = []

    def _commit_oldest(self) -> None:
        future, receipts_path = self._pending.pop(0)
        receipt_args, result, trace_sha256 = future.result()
        _write_receipt(
            receipts_path,
            **receipt_args,
            trace_sha256=trace_sha256,
            result=result,
        )

    def submit(
        self,
        trace: DynamicVLATrace,
        trace_path: Path,
        receipts_path: Path,
        receipt_args: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        if len(self._pending) >= self._max_workers:
            self._commit_oldest()
        future = self._executor.submit(
            _persist_trace,
            trace,
            trace_path,
            receipt_args,
            result,
        )
        self._pending.append((future, receipts_path))

    def flush(self) -> None:
        while self._pending:
            self._commit_oldest()

    def finish(self) -> None:
        try:
            self.flush()
        finally:
            self._executor.shutdown(wait=True)


def ordered_episode_replicas(
    episode_specs: dict[tuple[int, int, int], dict[str, Any]],
    task_id: int,
    environment_index: int,
    *,
    group_identical_scenes: bool = False,
) -> list[int]:
    """Group identical scenes in first-occurrence order without changing identities."""
    replicas = sorted(
        [replica for task, environment, replica in episode_specs
         if task == task_id and environment == environment_index],
        key=lambda replica: (
            episode_specs[(task_id, environment_index, replica)].get("initial_state_index", 0),
            replica,
        ),
    )
    if not group_identical_scenes:
        return replicas
    groups: dict[str, list[int]] = {}
    for replica in replicas:
        digest = episode_specs[(task_id, environment_index, replica)]["environment_sha256"]
        groups.setdefault(digest, []).append(replica)
    return [replica for group in groups.values() for replica in group]


def _run_suite(upstream: Any, simulation_app: Any, args: argparse.Namespace) -> None:
    with open(args.sim_cfg_file) as stream:
        simulator_config = upstream.yaml.load(stream, Loader=upstream.yaml.FullLoader)
    initial_poses = {
        key: upstream.torch.tensor(
            [value["init_pose"] + [1]], dtype=upstream.torch.float32, device=args.device
        )
        for key, value in simulator_config["robots"].items()
    }
    obs_socket, act_socket = upstream.get_zmq_sockets(
        args.host, args.img_port, args.act_port
    )
    handshake = wait_for_action(upstream, act_socket, args.startup_timeout_s)
    if not isinstance(handshake, dict) or "vla" not in handshake:
        raise TimeoutError("DynamicVLA policy did not complete its handshake")

    output_dir = Path(args.output_dir) / handshake["vla"] / f"{handshake['epoch']:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    environment_paths = [line for line in Path(args.env_cfg).read_text().splitlines() if line]
    manifest = json.loads(Path(args.manifest).read_text())
    environment_specs = manifest["environments"]
    if [spec["path"] for spec in environment_specs] != environment_paths:
        raise ValueError("DynamicVLA manifest does not match the simulator environment list")
    for spec in environment_specs:
        if _sha256(Path(spec["path"])) != spec["sha256"]:
            raise ValueError(
                f"DynamicVLA environment changed after manifest creation: {spec['path']}"
            )
    episodes_per_environment = int(manifest["episodes_per_environment"])
    episode_specs = {
        (
            episode["task_id"],
            episode["environment_index"],
            episode["replica_id"],
        ): episode
        for episode in manifest["episodes"]
    }
    if len(episode_specs) != len(manifest["episodes"]):
        raise ValueError("DynamicVLA manifest contains duplicate episode identities")
    recorded_video_count = 0
    success_rates: dict[str, float] = {}
    sim_config = {
        "num_envs": args.num_envs,
        "scene_dir": args.scene_dir,
        "object_dir": args.object_dir,
        "physics_time_step": args.physics_time_step,
        "tolerance": args.tolerance,
        "device": args.device,
        "disable_fabric": args.disable_fabric,
        "path_tracing": args.path_tracing,
        "init_poses": initial_poses,
    }
    receipts_path = Path(args.receipts)
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    if receipts_path.exists():
        raise ValueError("DynamicVLA artifact directory already contains receipts")
    trace_writer = _TraceReceiptWriter(
        max_workers=int(getattr(args, "trace_writer_workers", 1))
    )
    for environment_spec in environment_specs:
        task_id = int(environment_spec["task_id"])
        task_name = str(environment_spec["task_name"])
        environment_index = int(environment_spec["environment_index"])
        environment_path = str(environment_spec["path"])
        environment_name = Path(environment_path).stem
        successes = 0
        attempted = 0
        ordered_replicas = ordered_episode_replicas(
            episode_specs, task_id, environment_index,
            group_identical_scenes=os.environ.get("VLALAGBENCH_DYNAMICVLA_REUSE_IDENTICAL_SCENES") == "1",
        )
        for replica_id in ordered_replicas:
            if not simulation_app.is_running():
                break
            episode_spec = episode_specs[(task_id, environment_index, replica_id)]
            environment_path = str(episode_spec["environment"])
            environment_sha256 = episode_spec["environment_sha256"]
            if _sha256(Path(environment_path)) != environment_sha256:
                raise ValueError(
                    f"DynamicVLA episode environment changed after manifest creation: {environment_path}"
                )
            expected_episode_seed = derive_episode_seed(
                args.policy_seed,
                task_name,
                environment_sha256,
                replica_id,
            )
            obs_socket.send_pyobj(
                {
                    "episode_identity": {
                        "policy_seed": args.policy_seed,
                        "task_name": task_name,
                        "environment_sha256": environment_sha256,
                        "replica_id": replica_id,
                    }
                }
            )
            identity_ack = wait_for_action(
                upstream, act_socket, args.reply_timeout_s
            )
            if (
                not isinstance(identity_ack, dict)
                or not identity_ack.get("identity_ack")
                or identity_ack.get("episode_seed") != expected_episode_seed
            ):
                raise RuntimeError("DynamicVLA policy rejected the episode identity")
            setattr(upstream.get_sim_results, "vlalagbench_max_episode_steps", episode_spec.get("max_episode_steps"))
            setattr(upstream.get_sim_results, "vlalagbench_driven_friction", episode_spec.get("driven_friction", "zero"))
            literal_instruction = episode_spec["instruction"]
            if not isinstance(literal_instruction, str) or not literal_instruction:
                raise ValueError("DynamicVLA manifest has no paired literal prompt")
            setattr(
                upstream.get_sim_results,
                "vlalagbench_expected_instruction",
                literal_instruction,
            )
            setattr(
                upstream.get_sim_results,
                "vlalagbench_motion_actors",
                episode_spec.get("motion_actors") or [],
            )
            instruction_generator = getattr(upstream, "InstructionGenerator", None)
            original_generator = (
                instruction_generator.generate_instruction
                if instruction_generator is not None
                else None
            )
            if instruction_generator is not None:
                instruction_generator.generate_instruction = staticmethod(
                    lambda _metadata: literal_instruction
                )
            try:
                result = upstream.get_sim_results(
                    sim_config, environment_path, obs_socket, act_socket
                )
            finally:
                if instruction_generator is not None:
                    instruction_generator.generate_instruction = staticmethod(
                        original_generator
                    )
            if result.get("instruction") != literal_instruction:
                raise RuntimeError("DynamicVLA result used a different literal prompt")
            if result.get("policy_episode_seed") != expected_episode_seed:
                raise RuntimeError("DynamicVLA policy used the wrong episode seed")
            environment_config = json.loads(Path(environment_path).read_text())
            ack = None
            if result["status"] != 3:
                episode_name = upstream.get_episode_name(
                    environment_name, result["status"]
                )
                obs_socket.send_pyobj(
                    {
                        "env_name": environment_name,
                        "eps_name": episode_name[:-4],
                        "success": result["status"] == 0,
                        "ee_path": upstream.np.array(result["ee_path"]),
                    }
                )
                video_limit = getattr(args, "video_episode_limit", None)
                if args.save and (video_limit is None or recorded_video_count < video_limit):
                    video_path = output_dir / f"task{task_id:02d}_env{environment_index:03d}_replica{replica_id:03d}.mp4"
                    views = result["cam_views"]
                    if result.get("terminal_cameras") is not None:
                        views = [*views, result["terminal_cameras"]]
                    upstream.sim.dump_video(
                        upstream.sim.get_frames(
                            upstream.get_frames(views), state_keys=[]
                        ),
                        video_path,
                        fps=1.0 / args.physics_time_step,
                    )
                    recorded_video_count += 1
                    result["latency"].update(
                        video=str(video_path.relative_to(Path(args.trace_dir))),
                        video_sha256=_sha256(video_path),
                        video_frames=len(views),
                        video_cameras=list(views[0]),
                    )
                ack = wait_for_action(upstream, act_socket, args.reply_timeout_s)
            acknowledged = isinstance(ack, dict) and "ack" in ack
            if result["status"] != 3 and not acknowledged:
                result["latency"]["errors"].append(
                    "policy did not acknowledge the episode"
                )
                result["latency"]["exact"] = False
            episode_id = environment_index * episodes_per_environment + replica_id
            trace_name = f"dynamicvla_task{task_id:02d}_ep{episode_id:03d}.npz"
            recorder = result.pop("trace_recorder")
            trace = recorder.finish(
                {
                    "task_id": task_id,
                    "task_name": task_name,
                    "episode": episode_id,
                    "environment_index": environment_index,
                    "replica_id": replica_id,
                    "environment_path": environment_path,
                    "environment_sha256": _sha256(Path(environment_path)),
                    "environment": environment_config,
                    "asset_sha256": _asset_hashes(
                        environment_config,
                        scene_dir=Path(args.scene_dir),
                        object_dir=Path(args.object_dir),
                    ),
                    "instruction": result["instruction"],
                    "seed": int(environment_config["seed"]),
                    "upstream_revision": args.upstream_revision,
                    "checkpoint": args.checkpoint,
                    "checkpoint_sha256": args.checkpoint_sha256,
                    "checkpoint_config_sha256": args.checkpoint_config_sha256,
                    "policy_noise_scheme": result["policy_noise_scheme"],
                    "policy_torch_version": result["policy_torch_version"],
                    "policy_seed": args.policy_seed,
                    "policy_episode_seed": result["policy_episode_seed"],
                    "episode_length_s": (
                        episode_spec["max_episode_steps"] * args.physics_time_step
                        if episode_spec.get("max_episode_steps") is not None
                        else float(environment_config["episode_length_s"])
                    ),
                    "deadline_basis": "active_action_steps",
                    "control_hz": 1.0 / args.physics_time_step,
                    "physics_time_step": args.physics_time_step,
                    "termination_tolerance": args.tolerance,
                    "disable_fabric": args.disable_fabric,
                    "path_tracing": args.path_tracing,
                    "simulator_versions": _simulator_versions(),
                    "inference_mode": args.mode,
                    "action_index_origin": "release" if args.mode == "synchronous" else "observation",
                    "latency_steps": args.latency_steps,
                    "action_chunk_steps": args.action_chunk_steps,
                    "execution_horizon_steps": args.execution_horizon_steps,
                    "execution_horizon_basis": "post_latency_action_count",
                    "idle_control": "hold_pose",
                    "initial_state_index": int(
                        episode_spec.get("initial_state_index", 0)
                    ),
                    "motion": result.get("motion"),
                    "success_term": result["success_term"],
                    "timeout_terms": result["timeout_terms"],
                    "status": result["status"],
                    "success": result["status"] == 0,
                },
                final_policy=ack["final_policy"] if acknowledged else {},
            )
            trace_writer.submit(
                trace,
                trace_dir / trace_name,
                receipts_path,
                {
                    "task_id": task_id,
                    "task_name": task_name,
                    "episode": episode_id,
                    "environment_index": environment_index,
                    "replica_id": replica_id,
                    "environment": environment_path,
                    "seed": int(environment_config["seed"]),
                    "instruction": literal_instruction,
                    "trace": trace_name,
                },
                result,
            )
            if result["status"] == 3 or not acknowledged:
                trace_writer.finish()
            if result["status"] == 3:
                raise TimeoutError("DynamicVLA policy reply timed out")
            if not acknowledged:
                raise TimeoutError("DynamicVLA policy did not acknowledge the episode")
            attempted += 1
            successes += int(result["status"] == 0)
        success_rates[f"{task_name}/{environment_name}"] = (
            successes / attempted if attempted else 0.0
        )

    trace_writer.finish()
    obs_socket.send_pyobj({"vla": handshake["vla"], "success_rates": success_rates})
    suite_ack = wait_for_action(upstream, act_socket, args.reply_timeout_s)
    if not isinstance(suite_ack, dict) or not suite_ack.get("suite_ack"):
        raise TimeoutError("DynamicVLA policy did not acknowledge suite completion")
    env = getattr(upstream.get_sim_results, "env", None)
    if env is not None:
        env.close()
    obs_socket.close()
    act_socket.close()


_tree_sha256 = lru_cache(maxsize=128)(tree_sha256)


def _asset_hashes(
    environment: dict[str, Any], *, scene_dir: Path, object_dir: Path
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for usd_path in _usd_paths(environment):
        if usd_path.startswith(("http://", "https://", "omniverse://")):
            continue
        source = Path(usd_path)
        candidates = [
            ("scenes", scene_dir, scene_dir / source.name),
            (
                "objects",
                object_dir,
                object_dir / source.parent.name / source.name,
            ),
        ]
        matches = [candidate for candidate in candidates if candidate[2].is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"cannot uniquely resolve DOM asset {usd_path!r} under "
                f"{scene_dir} or {object_dir}"
            )
        label, root, candidate = matches[0]
        hashes[str(Path(label) / candidate.relative_to(root))] = _sha256(candidate)
        bundle = scene_dir / "textures" if label == "scenes" else candidate.parent
        if bundle.is_dir():
            hashes[str(Path(label) / bundle.relative_to(root))] = _tree_sha256(bundle)
    return hashes


def _usd_paths(value: Any) -> list[str]:
    if isinstance(value, dict):
        paths = [value["usd_path"]] if isinstance(value.get("usd_path"), str) else []
        return paths + [path for child in value.values() for path in _usd_paths(child)]
    if isinstance(value, list):
        return [path for child in value for path in _usd_paths(child)]
    return []


def _simulator_versions() -> dict[str, str]:
    versions = {
        package: importlib.metadata.version(package)
        for package in ("isaaclab", "isaacsim")
    }
    from vlalagbench.benchmarks.dynamicvla.camera_outputs import active_camera_outputs
    camera_outputs = active_camera_outputs()
    if camera_outputs is not None:
        versions["renderer_camera_outputs"] = camera_outputs
    try:
        import carb

        settings = carb.settings.get_settings()
        aa_op = int(settings.get("/rtx/post/aa/op"))
        dlss_mode = int(settings.get("/rtx/post/dlss/execMode"))
    except (ImportError, TypeError, ValueError):
        return versions
    aa_names = {0: "Off", 1: "TAA", 2: "FXAA", 3: "DLSS", 4: "DLAA"}
    dlss_names = {0: "Performance", 1: "Balanced", 2: "Quality", 3: "Auto"}
    versions["renderer_active_gpu"] = str(settings.get("/renderer/activeGpu"))
    versions["renderer_multi_gpu"] = str(settings.get("/renderer/multiGpu/enabled"))
    versions["renderer_mode"] = str(settings.get("/rtx/rendermode"))
    versions["renderer_dlssg"] = str(settings.get("/rtx-transient/dlssg/enabled"))
    versions["renderer_dl_denoiser"] = str(settings.get("/rtx-transient/dldenoiser/enabled"))
    versions["renderer_antialiasing"] = f"{aa_op}:{aa_names.get(aa_op, 'Unknown')}"
    versions["renderer_dlss_mode"] = (
        f"{dlss_mode}:{dlss_names.get(dlss_mode, 'Unknown')}"
    )
    return versions


def _verify_antialiasing(mode: str) -> dict[str, str]:
    import carb

    expected = {"Off": 0, "TAA": 1, "FXAA": 2, "DLSS": 3, "DLAA": 4}[mode]
    settings = carb.settings.get_settings()
    actual = int(settings.get("/rtx/post/aa/op"))
    if actual != expected:
        raise RuntimeError(
            f"renderer anti-aliasing is {actual}, expected {expected} ({mode})"
        )
    return _simulator_versions()


def _configure_antialiasing(upstream: Any, mode: str) -> None:
    _configure_renderer(upstream, antialiasing=mode)


def _configure_renderer(
    upstream: Any, *, antialiasing: str | None = None, dl_denoiser: bool | None = None,
    dlss_mode: str | None = None, enable_dlssg: bool | None = None, render_mode: str | None = None,
) -> None:

    def apply_and_verify() -> dict[str, str]:
        if antialiasing is not None:
            import omni.replicator.core as rep

            rep.settings.set_render_rtx_realtime(antialiasing=antialiasing)
            _verify_antialiasing(antialiasing)
        requested = {
            "/rtx-transient/dldenoiser/enabled": dl_denoiser,
            "/rtx-transient/dlssg/enabled": enable_dlssg,
            "/rtx/rendermode": render_mode,
            "/rtx/post/dlss/execMode": (
                {"Performance": 0, "Balanced": 1, "Quality": 2, "Auto": 3}[dlss_mode]
                if dlss_mode is not None else None
            ),
        }
        import carb

        settings = carb.settings.get_settings()
        for key, value in requested.items():
            if value is not None:
                settings.set(key, value)
        if antialiasing is not None:
            _verify_antialiasing(antialiasing)
        for key, value in requested.items():
            if value is not None:
                if settings.get(key) != value:
                    raise RuntimeError(f"renderer did not retain requested {key}={value!r}")
        return _simulator_versions()

    configured = apply_and_verify()
    logging.info(
        "Configured DynamicVLA renderer: %s",
        json.dumps(configured, sort_keys=True),
    )
    original_get_test_env = upstream.get_test_env

    def get_test_env_verified(*args: Any, **kwargs: Any) -> Any:
        env = original_get_test_env(*args, **kwargs)
        # Isaac Lab's environment construction reapplies its render preset and can
        # silently restore DLSS.  Apply the requested mode only after construction,
        # before the first policy observation is captured, and fail closed if the
        # setting still does not stick.
        effective = apply_and_verify()
        logging.info(
            "Re-applied DynamicVLA renderer after environment creation: %s",
            json.dumps(effective, sort_keys=True),
        )
        return env

    upstream.get_test_env = get_test_env_verified


def _add_renderer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dlss-mode", choices=("Performance", "Balanced", "Quality", "Auto"))
    parser.add_argument("--enable-dlssg", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--render-mode", choices=("RaytracedLighting",))


def main(argv: list[str] | None = None) -> int:
    from vlalagbench.benchmarks.dynamicvla.profiling import ENV_PROFILE, HostTimings

    profile = HostTimings() if os.environ.get(ENV_PROFILE) == "1" else None
    try:
        with profile.span("simulator.main") if profile is not None else nullcontext():
            return _main(argv, profile)
    finally:
        if profile is not None:
            profile.restore()
            profile.write()


def _main(argv: list[str] | None = None, profile: Any = None) -> int:
    timing = profile.span if profile is not None else lambda _name: nullcontext()
    root_parser = argparse.ArgumentParser(description=__doc__)
    root_parser.add_argument("--disable_fabric", action="store_true")
    root_parser.add_argument("--num_envs", type=int, choices=(1,), default=1)
    root_parser.add_argument("--save", action="store_true")
    root_parser.add_argument("--video-episode-limit", type=int)

    root_parser.parse_known_args(argv)

    root = Path(os.environ["VLALAGBENCH_DYNAMICVLA_ROOT"]).resolve()
    sys.path.insert(0, str(root))
    with timing("startup.upstream_import"):
        import simulations.evaluate as upstream

    upstream.AppLauncher.add_app_launcher_args(root_parser)
    isaac_args, remaining = root_parser.parse_known_args(argv)
    root_parser.add_argument("--path-tracing", action="store_true")
    root_parser.add_argument("--physics-time-step", type=float, default=0.04)
    root_parser.add_argument("--tolerance", type=float, default=0.07)
    root_parser.add_argument("--scene-dir", required=True)
    root_parser.add_argument("--object-dir", required=True)
    root_parser.add_argument("--output-dir", required=True)
    root_parser.add_argument("--host", default="127.0.0.1")
    root_parser.add_argument("--img-port", type=int, default=3186)
    root_parser.add_argument("--act-port", type=int, default=3188)
    root_parser.add_argument("--sim-cfg-file", default=root / "simulations/configs/sim_cfg.yaml")
    root_parser.add_argument("--env-cfg", required=True)
    root_parser.add_argument("--manifest", required=True)
    root_parser.add_argument("--receipts", required=True)
    root_parser.add_argument("--trace-dir", required=True)
    root_parser.add_argument("--upstream-revision", required=True)
    root_parser.add_argument("--checkpoint", required=True)
    root_parser.add_argument("--checkpoint-sha256", required=True)
    root_parser.add_argument("--checkpoint-config-sha256", required=True)
    root_parser.add_argument(
        "--mode", choices=("synchronous", "lazy", "continuous"), required=True
    )
    root_parser.add_argument("--execution-horizon-steps", type=int, required=True)
    root_parser.add_argument("--latency-steps", type=int, required=True)
    root_parser.add_argument("--action-chunk-steps", type=int, required=True)
    root_parser.add_argument("--policy-seed", type=int, required=True)
    root_parser.add_argument("--reply-timeout-s", type=float, required=True)
    root_parser.add_argument("--startup-timeout-s", type=float, required=True)
    root_parser.add_argument("--trace-writer-workers", type=int, default=1)
    root_parser.add_argument(
        "--trace-images",
        action="store_true",
        help="also store every policy camera frame in each trace",
    )
    root_parser.add_argument(
        "--antialiasing-mode",
        choices=("Off", "TAA", "FXAA", "DLSS", "DLAA"),
    )
    root_parser.add_argument("--enable-dl-denoiser", action=argparse.BooleanOptionalAction, default=None)
    _add_renderer_args(root_parser)
    args = root_parser.parse_args(remaining)
    if args.video_episode_limit is not None and args.video_episode_limit <= 0:
        root_parser.error("--video-episode-limit must be positive")
    for key, value in vars(isaac_args).items():
        setattr(args, key, value)
    if profile is not None:
        profile.output = Path(args.trace_dir) / "host-timing.json"

    upstream.simulate = lambda env, obs, act, poses: simulate_injected(
        upstream,
        env,
        obs,
        act,
        poses,
        mode=args.mode,
        execution_horizon_steps=args.execution_horizon_steps,
        latency_steps=args.latency_steps,
        reply_timeout_s=args.reply_timeout_s,
        action_chunk_steps=args.action_chunk_steps,
        trace_images=args.trace_images,
    )
    with timing("startup.launcher"):
        launcher = upstream.AppLauncher(isaac_args)
    try:
        from vlalagbench.benchmarks.dynamicvla.camera_outputs import (
            install_camera_outputs, simulator_camera_outputs,
        )
        from vlalagbench.benchmarks.dynamicvla.carriers import install_carriers
        from vlalagbench.benchmarks.dynamicvla.contact_diagnostics import install_contact_diagnostics
        install_camera_outputs(upstream, simulator_camera_outputs())
        install_contact_diagnostics(upstream)
        install_carriers(upstream)
        if args.render_mode == "RaytracedLighting" and args.path_tracing:
            raise ValueError("RaytracedLighting conflicts with --path-tracing")
        renderer = dict(antialiasing=args.antialiasing_mode, dl_denoiser=args.enable_dl_denoiser,
                        dlss_mode=args.dlss_mode, enable_dlssg=args.enable_dlssg, render_mode=args.render_mode)
        if any(value is not None for value in renderer.values()):
            _configure_renderer(upstream, **renderer)
        if profile is not None:
            from vlalagbench.benchmarks.dynamicvla.profiling import install_simulator_timing
            install_simulator_timing(profile, sys.modules[__name__], upstream)
        try:
            with timing("suite.run"):
                _run_suite(upstream, launcher.app, args)
        except BaseException:
            logging.exception("DynamicVLA simulator suite failed before Isaac shutdown")
            raise
    finally:
        with profile.episode_scope(None) if profile is not None else nullcontext():
            with timing("launcher.shutdown"):
                launcher.app.close(wait_for_replicator=False)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s", level=logging.INFO
    )
    raise SystemExit(main())
