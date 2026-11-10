"""Inspect, verify, rejudge, or render a DynamicVLA trace in Isaac Lab."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import random
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from vlalagbench.files import sha256 as _sha256, tree_sha256 as _tree_sha256
from vlalagbench.benchmarks.dynamicvla.adapter import checkout_changes

import numpy as np

from vlalagbench.benchmarks.dynamicvla.replay import (
    rejudge_trace,
    render_trace_video,
    render_diagnostic_video,
    state_summary,
    verify_trace_actions,
)
from vlalagbench.benchmarks.dynamicvla.continuation import (
    merge_continuation,
    preflight_continuation,
)
from vlalagbench.benchmarks.dynamicvla.protocol import (
    audited_trace_receipt,
)
from vlalagbench.benchmarks.dynamicvla.runtime import (
    action_tensor,
    episode_length_s_for_steps,
    evaluate_termination_terms,
    flatten_policy_observation,
    hold_pose_action,
    policy_observation_message,
    read_policy_observation,
    receive_reply,
    refresh_sensors,
    step_and_capture_terminal,
    wait_for_action,
)
from vlalagbench.benchmarks.dynamicvla.motion import PlanarDriver
from vlalagbench.benchmarks.dynamicvla.tracing import (
    DynamicVLATrace,
    DynamicVLATraceRecorder,
)
from vlalagbench.simulators.isaaclab.tracing import restore_env_state


def _parser(upstream: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamicvla-root", required=True)
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--video")
    parser.add_argument("--camera")
    parser.add_argument("--separate-cameras", action="store_true")
    parser.add_argument("--camera-outputs", choices=("recorded", "rgb", "upstream"), default="recorded",
                        help="camera annotators to render; defaults to the trace's recorded outputs")
    parser.add_argument("--frame-hashes", action="store_true",
                        help="report SHA-256 of every rendered camera RGB frame")
    parser.add_argument("--video-path-tracing-spp", type=int,
                        help="Use path tracing for video only, with this many samples per pixel")
    parser.add_argument("--diagnostic-mode", choices=("states", "actions", "frozen", "frozen-restore"))
    parser.add_argument("--diagnostic-frames", type=int, default=250)
    parser.add_argument("--diagnostic-samples", type=int, default=1)
    parser.add_argument("--diagnostic-start", type=int, default=0)
    parser.add_argument("--diagnostic-png-frames", type=int, nargs="*", default=[])
    parser.add_argument("--antialiasing-mode", choices=("Off", "TAA", "FXAA", "DLSS", "DLAA"))
    parser.add_argument("--enable-dl-denoiser", action=argparse.BooleanOptionalAction, default=None)
    from vlalagbench.benchmarks.dynamicvla.simulator import _add_renderer_args
    _add_renderer_args(parser)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--rejudge", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--continue-steps", type=int)
    parser.add_argument("--resume-step", type=int)
    parser.add_argument("--out-trace")
    parser.add_argument("--checkpoint")
    parser.add_argument("--report")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--img-port", type=int, default=3186)
    parser.add_argument("--act-port", type=int, default=3188)
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--reply-timeout-s", type=float, default=300.0)
    upstream.AppLauncher.add_app_launcher_args(parser)
    return parser


def _make_env(upstream: Any, trace: DynamicVLATrace, assets_root: Path, args: Any) -> Any:
    import torch

    random.seed(trace.meta.seed)
    np.random.seed(trace.meta.seed)
    torch.manual_seed(trace.meta.seed)
    environment = copy.deepcopy(trace.meta.environment)
    paused_steps = (
        sum(event.get("wait_reason") != "model" for event in trace.protocol)
        if trace.meta.deadline_basis == "active_action_steps"
        else 0
    )
    environment["episode_length_s"] = episode_length_s_for_steps(
        round(trace.meta.episode_length_s * trace.meta.control_hz) + paused_steps,
        1.0 / trace.meta.control_hz,
    )
    from vlalagbench.benchmarks.dynamicvla.camera_outputs import (
        install_camera_outputs, recorded_camera_outputs,
    )
    from vlalagbench.benchmarks.dynamicvla.carriers import install_carriers
    from vlalagbench.benchmarks.dynamicvla.contact_diagnostics import install_contact_diagnostics
    camera_outputs = getattr(args, "camera_outputs", "recorded")
    if camera_outputs == "recorded":
        camera_outputs = recorded_camera_outputs(trace.meta.simulator_versions)
    elif args.continue_steps is not None:
        raise ValueError("continuation cannot override recorded camera outputs")
    install_camera_outputs(upstream, camera_outputs)
    install_contact_diagnostics(upstream)
    install_carriers(upstream)
    env = upstream.get_test_env(
        environment,
        1,
        str(assets_root / "scenes"),
        str(assets_root / "objects"),
        trace.meta.physics_time_step,
        trace.meta.termination_tolerance,
        args.device,
        trace.meta.disable_fabric,
        trace.meta.path_tracing,
    )
    _restore_recorded_dlss_mode(trace, override=getattr(args, "dlss_mode", None))
    env.reset(seed=trace.meta.seed)
    return env


def _restore_recorded_dlss_mode(trace: DynamicVLATrace, *, override: str | None = None) -> None:
    recorded = trace.meta.simulator_versions.get("renderer_dlss_mode")
    if override is not None:
        recorded = str({"Performance": 0, "Balanced": 1, "Quality": 2, "Auto": 3}[override])
    if recorded is None:
        return
    import carb

    mode = int(recorded.split(":", 1)[0])
    if mode not in {0, 1, 2, 3}:
        raise ValueError(f"unsupported recorded DLSS mode: {recorded}")
    settings = carb.settings.get_settings()
    settings.set("/rtx/post/dlss/execMode", mode)
    if settings.get("/rtx/post/dlss/execMode") != mode:
        raise RuntimeError("replay did not retain recorded DLSS execution mode")


def _validate_inputs(trace: DynamicVLATrace, root: Path, assets_root: Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != trace.meta.upstream_revision:
        raise ValueError(
            f"DynamicVLA checkout is {revision}, trace requires "
            f"{trace.meta.upstream_revision}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if checkout_changes(dirty):
        raise ValueError("DynamicVLA checkout has tracked or untracked modifications")
    required_versions = {
        key: value for key, value in trace.meta.simulator_versions.items()
        if not key.startswith("renderer_")
    }
    versions = {
        package: importlib.metadata.version(package)
        for package in required_versions
    }
    if versions != required_versions:
        raise ValueError(
            f"simulator versions are {versions}, trace requires "
            f"{required_versions}"
        )
    for relative, expected in trace.meta.asset_sha256.items():
        # Study asset roots symlink shared objects/scenes; reject only lexical escapes.
        path = Path(os.path.normpath(assets_root / relative))
        if not path.is_relative_to(assets_root):
            raise ValueError(f"asset path escapes assets root: {relative}")
        actual = _tree_sha256(path) if path.is_dir() else _sha256(path)
        if actual != expected:
            raise ValueError(f"asset checksum mismatch: {path}")


def _observation(upstream: Any, env: Any, trace: DynamicVLATrace, index: int) -> dict:
    cameras, state = read_policy_observation(upstream, env)
    return policy_observation_message(
        cameras,
        state,
        index=index,
        seed=trace.meta.seed,
        latency_steps=trace.meta.latency_steps,
    )


def _recorded_observation(
    trace: DynamicVLATrace, index: int, images: Mapping[str, Any] | None = None
) -> dict:
    values = trace.observation(index)
    if images is None:
        images = {
            path[1]: value
            for path, value in values.items()
            if len(path) == 2 and path[0] == "images"
        }
    end_effector = {
        path[2]: value
        for path, value in values.items()
        if len(path) == 3 and path[:2] == ("state", "end_effector")
    }
    if not images or not end_effector:
        raise ValueError("trace observation is missing images or end-effector state")
    return {
        "dt_scale": 1.0,
        "index": index,
        "seed": trace.meta.seed,
        "latency_steps": trace.meta.latency_steps,
        "observation.state": {"end_effector": end_effector},
        **{
            f"observation.images.{name}": value for name, value in images.items()
        },
    }


def _rendered_observation(
    upstream: Any, env: Any, trace: DynamicVLATrace, index: int
) -> dict:
    """Recorded end-effector state with camera images re-rendered from saved state.

    Isaac's render history is not part of the saved state, so these images need
    not be bitwise identical to the ones the policy saw during the episode.
    """

    restore_env_state(env.unwrapped, trace.state(index))
    refresh_sensors(env.unwrapped)
    cameras, _ = read_policy_observation(upstream, env)
    return _recorded_observation(
        trace, index, {name: outputs["rgb"] for name, outputs in cameras.items()}
    )


def _resume_history(
    trace: DynamicVLATrace,
    resume_step: int,
    observe: Callable[[int], dict] | None = None,
) -> list[dict]:
    observe = observe or (lambda index: _recorded_observation(trace, index))
    history = [
        observe(index) for index in range(max(0, resume_step - 2), resume_step)
    ]
    for observation in history:
        observation["task"] = trace.meta.instruction
    if resume_step == 1:
        history.insert(0, {"task": trace.meta.instruction})
    return history


def _recorder_metadata(
    source: DynamicVLATrace,
    *,
    episode_length_s: float,
    status: int,
    success: bool,
) -> dict[str, Any]:
    metadata = source.meta.model_dump(mode="json")
    for generated in (
        "schema_version",
        "num_steps",
        "recorded_at",
        "python_version",
        "numpy_version",
        "continuation",
    ):
        metadata.pop(generated)
    metadata.update(
        episode_length_s=episode_length_s, status=status, success=success,
        deadline_basis="control_steps",
    )
    return metadata


def _audit_continuation(trace: DynamicVLATrace) -> dict[str, Any]:
    receipt = audited_trace_receipt(trace)
    if receipt["errors"]:
        raise RuntimeError(
            f"invalid continuation latency receipt: {receipt['errors']}"
        )
    return receipt


def _continue_trace(
    upstream: Any,
    env: Any,
    trace: DynamicVLATrace,
    trace_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import configs.termination_cfg
    import torch

    additional_steps = int(args.continue_steps)
    if additional_steps <= 0 or args.out_trace is None:
        raise ValueError("continuation requires positive --continue-steps and --out-trace")
    resume_step = trace.num_steps if args.resume_step is None else args.resume_step
    snapshot = preflight_continuation(trace, resume_step)
    # A trace without camera frames resumes from images re-rendered at the
    # restored history states.
    observe: Callable[[int], dict] = (
        (lambda index: _recorded_observation(trace, index))
        if trace.has_images
        else (lambda index: _rendered_observation(upstream, env, trace, index))
    )
    history = _resume_history(trace, resume_step, observe)
    restore_env_state(env.unwrapped, trace.state(resume_step))
    refresh_sensors(env.unwrapped)
    resume_observation = observe(resume_step)
    extended_deadline = resume_step + additional_steps
    env.unwrapped.cfg.episode_length_s = episode_length_s_for_steps(
        extended_deadline, env.unwrapped.step_dt
    )
    recorder = DynamicVLATraceRecorder(env.unwrapped)
    obs_socket, act_socket = upstream.get_zmq_sockets(
        args.host, args.img_port, args.act_port
    )
    handshake = wait_for_action(upstream, act_socket, args.startup_timeout_s)
    if not isinstance(handshake, dict) or "vla" not in handshake:
        raise TimeoutError("DynamicVLA policy did not complete its handshake")
    obs_socket.send_pyobj(
        {
            "episode_identity": {
                "policy_seed": trace.meta.policy_seed,
                "task_name": trace.meta.task_name,
                "environment_sha256": trace.meta.environment_sha256,
                "replica_id": trace.meta.replica_id,
            }
        }
    )
    identity_ack = wait_for_action(upstream, act_socket, args.reply_timeout_s)
    if (
        not isinstance(identity_ack, dict)
        or not identity_ack.get("identity_ack")
        or identity_ack.get("episode_seed") != trace.meta.policy_episode_seed
    ):
        raise ValueError("continuation policy rejected the trace episode identity")
    obs_socket.send_pyobj(
        {
            "task": trace.meta.instruction,
            "resume": {"history": history, "policy": snapshot},
        }
    )
    task_ack = wait_for_action(upstream, act_socket, args.reply_timeout_s)
    if not isinstance(task_ack, dict) or not task_ack.get("task_ack"):
        raise TimeoutError("DynamicVLA policy did not restore the trace")
    if (
        task_ack.get("policy_noise_scheme") != trace.meta.policy_noise_scheme
        or task_ack.get("policy_torch_version") != trace.meta.policy_torch_version
        or task_ack.get("episode_seed") != trace.meta.policy_episode_seed
        or task_ack.get("inference_mode") != trace.meta.inference_mode
        or task_ack.get("action_index_origin") != trace.meta.action_index_origin
        or (
            trace.meta.inference_mode != "synchronous"
            and task_ack.get("execution_horizon_basis")
            != trace.meta.execution_horizon_basis
        )
        or task_ack.get("execution_horizon_steps")
        != trace.meta.execution_horizon_steps
        or task_ack.get("action_chunk_steps") != trace.meta.action_chunk_steps
    ):
        raise ValueError("continuation policy runtime does not match the trace")

    poller = upstream.zmq.Poller()
    poller.register(act_socket, upstream.zmq.POLLIN)
    manager = env.unwrapped.termination_manager
    success_term = configs.termination_cfg.get_done_term(manager.active_terms)
    timeout_terms = [
        name for name in manager.active_terms if manager.get_term_cfg(name).time_out
    ]
    if success_term != trace.meta.success_term or set(timeout_terms) != set(trace.meta.timeout_terms):
        raise ValueError("trace termination configuration does not match the environment")
    last_action = torch.as_tensor(
        trace.actions[resume_step - 1], device=env.unwrapped.device
    )
    status = -1
    executed = 0
    previous_was_model = (trace.protocol[resume_step - 1] or {}).get("wait_reason") == "model"
    ee_path = []
    driver = PlanarDriver(trace.meta.motion) if trace.meta.motion else None
    if driver is not None:
        driver.prepare(env.unwrapped)
    for index in range(resume_step, resume_step + additional_steps):
        observation = (
            resume_observation
            if index == resume_step
            else _observation(upstream, env, trace, index)
        )
        recorder.record_observation(
            {
                **{
                    ("images", key.removeprefix("observation.images.")): value
                    for key, value in observation.items()
                    if key.startswith("observation.images.")
                },
                **{
                    ("state", "end_effector", key): value
                    for key, value in observation["observation.state"][
                        "end_effector"
                    ].items()
                },
            }
        )
        ee_path.append(observation["observation.state"]["end_effector"]["pos"])
        obs_socket.send_pyobj(observation)
        reply = receive_reply(
            upstream, poller, act_socket, args.reply_timeout_s
        )
        if not isinstance(reply, dict) or reply.get("meta", {}).get("reply_to") != index:
            raise RuntimeError(f"invalid continuation policy reply at step {index}")
        meta = reply["meta"]
        action = action_tensor(upstream, reply.get("action"), env)
        if (action is not None) != (meta.get("wait_reason") == "model"):
            raise RuntimeError(f"invalid continuation policy action at step {index}")
        if action is not None:
            last_action = action
        elif previous_was_model:
            last_action = hold_pose_action(torch, env, last_action)
        previous_was_model = action is not None
        terminal_terminations = None

        def terminal_observation() -> dict:
            nonlocal terminal_terminations
            terminal_terminations = evaluate_termination_terms(
                manager, env.unwrapped
            )
            return flatten_policy_observation(
                *read_policy_observation(upstream, env)
            )

        if driver is not None:
            driver.apply(env.unwrapped, index)
        was_terminal = step_and_capture_terminal(
            env,
            last_action,
            recorder,
            terminal_observation=terminal_observation,
        )
        terminations = terminal_terminations or {
            name: False for name in manager.active_terms
        }
        success = terminations[success_term]
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
        executed += 1
        if success:
            status = 0
            break
        if done:
            status = 1
            break
    if status == -1:
        status = 1

    obs_socket.send_pyobj(
        {
            "env_name": Path(trace.meta.environment_path).stem,
            "eps_name": Path(args.out_trace).stem,
            "success": status == 0,
            "ee_path": np.asarray(ee_path),
        }
    )
    ack = wait_for_action(upstream, act_socket, args.reply_timeout_s)
    if not isinstance(ack, dict) or "final_policy" not in ack:
        raise TimeoutError("DynamicVLA policy did not finalize the continuation")
    obs_socket.send_pyobj({"vla": handshake["vla"], "success_rates": {}})
    suite_ack = wait_for_action(upstream, act_socket, args.reply_timeout_s)
    if not isinstance(suite_ack, dict) or not suite_ack.get("suite_ack"):
        raise TimeoutError("DynamicVLA policy did not acknowledge continuation completion")
    obs_socket.close()
    act_socket.close()

    suffix = recorder.finish(
        _recorder_metadata(
            trace,
            episode_length_s=extended_deadline * env.unwrapped.step_dt,
            status=status,
            success=status == 0,
        ),
        final_policy=ack["final_policy"],
    )
    extended = merge_continuation(
        trace,
        suffix,
        resume_step=resume_step,
        source_path=trace_path,
        source_sha256=_sha256(trace_path),
        max_additional_steps=additional_steps,
    )
    receipt = _audit_continuation(extended)
    out_path = extended.save(args.out_trace)
    return {
        "source_trace": str(trace_path),
        "extended_trace": str(out_path),
        "resume_step": resume_step,
        "additional_steps_executed": executed,
        "success": status == 0,
        "total_steps": extended.num_steps,
        "history_images": extended.meta.continuation.history_images,
        "receipt": receipt,
    }


def _replay_antialiasing(trace: DynamicVLATrace, args: Any) -> str | None:
    recorded = getattr(trace.meta, "simulator_versions", {}).get("renderer_antialiasing")
    mode = recorded.split(":", 1)[-1] if recorded else None
    if mode is not None and mode not in {"Off", "TAA", "FXAA", "DLSS", "DLAA"}:
        raise ValueError(f"unsupported recorded antialiasing: {recorded}")
    override = args.antialiasing_mode
    if args.continue_steps is not None and override is not None and override != mode:
        raise ValueError("continuation cannot override recorded antialiasing")
    return override or mode


def _replay_dl_denoiser(trace: DynamicVLATrace, args: Any) -> bool | None:
    recorded = getattr(trace.meta, "simulator_versions", {}).get("renderer_dl_denoiser")
    if recorded not in {None, "True", "False"}:
        raise ValueError(f"unsupported recorded DL denoiser value: {recorded}")
    mode = None if recorded is None else recorded == "True"
    override = args.enable_dl_denoiser
    if args.continue_steps is not None and override is not None and override != mode:
        raise ValueError("continuation cannot override recorded DL denoiser")
    return mode if override is None else override


def _replay_extra_renderer(trace: DynamicVLATrace, args: Any) -> dict[str, Any]:
    versions = getattr(trace.meta, "simulator_versions", {})
    recorded = {
        "dlss_mode": versions.get("renderer_dlss_mode", "").split(":", 1)[-1] or None,
        "enable_dlssg": {"True": True, "False": False, None: None}.get(versions.get("renderer_dlssg")),
        "render_mode": versions.get("renderer_mode"),
    }
    resolved = {}
    for key, value in recorded.items():
        override = getattr(args, key, None)
        if args.continue_steps is not None and override is not None and override != value:
            raise ValueError(f"continuation cannot override recorded {key}")
        resolved[key] = value if override is None else override
    if getattr(args, "video_path_tracing_spp", None) is not None:
        if (not getattr(args, "video", None) or args.continue_steps is not None
                or getattr(args, "verify", False) or getattr(args, "rejudge", False)):
            raise ValueError("path-tracing override requires video-only replay")
        if not 1 <= args.video_path_tracing_spp <= 128:
            raise ValueError("video path-tracing samples must be between 1 and 128")
        resolved["render_mode"] = "PathTracing"
    return resolved


def _configure_video_samples(samples: int | None) -> dict[str, int]:
    if samples is None:
        return {}
    import carb

    settings = carb.settings.get_settings()
    requested = {"/rtx/pathtracing/spp": samples, "/rtx/pathtracing/totalSpp": samples}
    for key, value in requested.items():
        settings.set(key, value)
        if settings.get(key) != value:
            raise RuntimeError(f"renderer did not retain {key}={value}")
    return requested


def main(argv: list[str] | None = None) -> int:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--dynamicvla-root", required=True)
    known, _ = bootstrap.parse_known_args(argv)
    root = Path(known.dynamicvla_root).resolve()
    sys.path.insert(0, str(root))
    import simulations.evaluate as upstream

    args = _parser(upstream).parse_args(argv)
    if args.diagnostic_mode and (not args.video or args.verify or args.rejudge or args.continue_steps):
        raise ValueError("diagnostic mode requires --video and cannot combine with other operations")
    trace = DynamicVLATrace.load(args.trace)
    assets_root = Path(args.assets_root).resolve()
    _validate_inputs(trace, root, assets_root)
    antialiasing = _replay_antialiasing(trace, args)
    dl_denoiser = _replay_dl_denoiser(trace, args)
    extra_renderer = _replay_extra_renderer(trace, args)
    if args.continue_steps is not None and (args.verify or args.rejudge or args.video):
        raise ValueError("continuation cannot be combined with replay operations")
    if args.continue_steps is not None:
        if args.continue_steps <= 0 or args.out_trace is None:
            raise ValueError(
                "continuation requires positive --continue-steps and --out-trace"
            )
        checkpoint = Path(args.checkpoint or trace.meta.checkpoint).resolve()
        model_path = checkpoint / "model.safetensors"
        if _sha256(model_path) != trace.meta.checkpoint_sha256:
            raise ValueError(f"checkpoint checksum does not match trace: {model_path}")
        config_path = checkpoint / "config.json"
        if _sha256(config_path) != trace.meta.checkpoint_config_sha256:
            raise ValueError(
                f"checkpoint config checksum does not match trace: {config_path}"
            )
        checkpoint_config = json.loads(config_path.read_text())
        resume_step = trace.num_steps if args.resume_step is None else args.resume_step
        preflight_continuation(trace, resume_step, checkpoint_config)
    launcher = upstream.AppLauncher(args)
    report = {"summary": state_summary(trace)}
    env = None
    try:
        if args.verify or args.rejudge or args.video or args.continue_steps is not None:
            from vlalagbench.benchmarks.dynamicvla.simulator import (
                _configure_renderer, _simulator_versions,
            )
            from vlalagbench.benchmarks.dynamicvla.camera_outputs import active_camera_outputs

            if antialiasing is not None or dl_denoiser is not None or any(v is not None for v in extra_renderer.values()):
                _configure_renderer(upstream, antialiasing=antialiasing, dl_denoiser=dl_denoiser, **extra_renderer)
            env = _make_env(upstream, trace, assets_root, args)
            video_samples = _configure_video_samples(args.video_path_tracing_spp)
            report["renderer"] = {
                "recorded": trace.meta.simulator_versions,
                "antialiasing_override": args.antialiasing_mode,
                "dl_denoiser_override": args.enable_dl_denoiser,
                "resolved_extra_settings": extra_renderer,
                "effective": _simulator_versions(),
                "video_path_tracing_settings": video_samples,
                "video_state_capture": "state-restoration-v1" if args.video else None,
            }
        if args.continue_steps is not None:
            report["continuation"] = _continue_trace(
                upstream, env, trace, Path(args.trace), args
            )
        if args.verify:
            report["verification"] = verify_trace_actions(
                trace, env, atol=args.atol
            )
        if args.rejudge:
            import configs.termination_cfg

            manager = env.unwrapped.termination_manager
            success_term = configs.termination_cfg.get_done_term(manager.active_terms)
            report["rejudgement"] = rejudge_trace(
                trace, env, success_term=success_term
            )
        if args.video and args.diagnostic_mode:
            report["diagnostic"] = render_diagnostic_video(
                trace, env, upstream.sim, args.video, camera=args.camera,
                mode=args.diagnostic_mode, frames=args.diagnostic_frames,
                samples=args.diagnostic_samples, start=args.diagnostic_start,
                png_frames=tuple(args.diagnostic_png_frames),
            )
            report["diagnostic"]["trace_sha256"] = _sha256(Path(args.trace))
        elif args.video:
            digests = {} if args.frame_hashes else None
            rendered = render_trace_video(
                trace, env, upstream.sim, args.video, camera=args.camera,
                separate_cameras=args.separate_cameras, digests=digests,
            )
            if digests is not None:
                report["frame_sha256"] = digests
            report["camera_outputs"] = active_camera_outputs()
            if isinstance(rendered, dict):
                report["videos"] = {name: str(path) for name, path in rendered.items()}
            else:
                report["video"] = str(rendered)
        rendered_report = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.report:
            Path(args.report).write_text(rendered_report, encoding="utf-8")
        print(rendered_report, end="")
        verification = report.get("verification")
        rejudgement = report.get("rejudgement")
        return int(
            bool(verification and not verification["within_atol"])
            or bool(rejudgement and not rejudgement["matches_recorded"])
        )
    finally:
        if env is not None:
            env.close()
        launcher.app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
