"""Replay, re-render, and independently rejudge DynamicVLA traces."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from vlalagbench.benchmarks.dynamicvla.motion import PlanarDriver
from vlalagbench.benchmarks.dynamicvla.tracing import DynamicVLATrace
from vlalagbench.benchmarks.dynamicvla.runtime import evaluate_termination_terms
from vlalagbench.simulators.isaaclab.tracing import (
    capture_env_state,
    max_state_error,
    restore_env_state,
    step_and_capture_env_state,
)


def verify_trace_actions(
    trace: DynamicVLATrace, env: Any, *, atol: float = 1e-5
) -> dict[str, Any]:
    """Re-simulate recorded actions and measure drift from saved states."""

    import torch

    base = env.unwrapped
    restore_env_state(base, trace.state(0))
    initial_error, initial_path = max_state_error(
        capture_env_state(base), trace.state(0)
    )
    driver = PlanarDriver(trace.meta.motion) if trace.meta.motion else None
    if driver is not None:
        driver.prepare(base)
    per_step = []
    for index, action in enumerate(trace.actions):
        if driver is not None:
            driver.apply(base, index)
        actual = step_and_capture_env_state(
            env,
            torch.as_tensor(action, device=base.device),
        )
        error, path = max_state_error(actual, trace.state(index + 1))
        per_step.append((error, path))
        if bool(trace.done[index]):
            break

    all_errors = [(initial_error, initial_path), *per_step]
    max_error, max_path = max(all_errors, key=lambda item: item[0])
    first_divergence = next(
        (index for index, (error, _) in enumerate(per_step) if error > atol), None
    )
    return {
        "steps": len(per_step),
        "initial_state_error": initial_error,
        "max_state_error": max_error,
        "max_error_path": list(max_path) if max_path is not None else None,
        "exact": max_error == 0.0,
        "within_atol": max_error <= atol,
        "first_step_exceeding_atol": first_divergence,
    }


def rejudge_trace(
    trace: DynamicVLATrace, env: Any, *, success_term: str
) -> dict[str, Any]:
    """Evaluate every configured termination directly on saved states."""

    base = env.unwrapped
    manager = base.termination_manager
    term_names = list(trace.terminations[0]) if trace.terminations else list(manager.active_terms)
    if success_term != trace.meta.success_term or set(term_names) != set(
        manager.active_terms
    ):
        raise ValueError("trace termination configuration does not match the environment")
    timeout_terms = [
        name for name in term_names if manager.get_term_cfg(name).time_out
    ]
    if set(timeout_terms) != set(trace.meta.timeout_terms):
        raise ValueError("trace timeout terms do not match the environment")
    outcomes = []
    for index in range(1, trace.num_steps + 1):
        restore_env_state(base, trace.state(index))
        outcomes.append(evaluate_termination_terms(manager, base))
    decisions = [terms[success_term] for terms in outcomes]
    dones = [any(terms.values()) for terms in outcomes]
    first_success = next(
        (index for index, success in enumerate(decisions) if success), None
    )
    termination_mismatches = [
        {"step": index, "term": name, "recorded": recorded[name], "actual": actual[name]}
        for index, (actual, recorded) in enumerate(
            zip(outcomes, trace.terminations, strict=True)
        )
        for name in term_names
        if actual[name] != recorded[name]
    ]
    done_mismatches = [
        index
        for index, (actual, recorded) in enumerate(
            zip(dones, trace.done, strict=True)
        )
        if actual != bool(recorded)
    ]
    success_mismatches = [
        index
        for index, (actual, recorded) in enumerate(
            zip(decisions, trace.success, strict=True)
        )
        if actual != bool(recorded)
    ]
    return {
        "steps": trace.num_steps,
        "success": any(decisions),
        "first_success_step": first_success,
        "recorded_success": trace.meta.success,
        "matches_recorded": not (
            termination_mismatches or done_mismatches or success_mismatches
        ),
        "termination_mismatches": termination_mismatches,
        "done_mismatches": done_mismatches,
        "success_mismatches": success_mismatches,
    }


def render_trace_video(
    trace: DynamicVLATrace,
    env: Any,
    upstream_sim: Any,
    out_path: str | Path,
    *,
    camera: str | None = None,
    separate_cameras: bool = False,
    digests: dict[str, list[str]] | None = None,
) -> Path | dict[str, Path]:
    """Restore every saved state and render camera frames after the run.

    ``digests`` collects the SHA-256 of each rendered RGB frame by camera.
    """

    base = env.unwrapped
    frames: dict[str, list[Any]] = {}
    for index in range(trace.num_steps + 1):
        restore_env_state(base, trace.state(index))
        # Advance sensor timestamps as env.step does so reads fetch the new
        # render. Camera.reset() resets sensor bookkeeping; that is not a
        # documented guarantee of clearing the RTX denoiser's history.
        base.sim.render()
        base.scene.update(base.step_dt)
        views = upstream_sim.get_camera_views(base.scene.sensors, ["rgb"])
        for name, outputs in views.items():
            if camera is None or name == camera:
                frames.setdefault(f"{name}_rgb", []).append(
                    outputs["rgb"].squeeze(0).copy()
                )
                if digests is not None:
                    digests.setdefault(name, []).append(
                        hashlib.sha256(outputs["rgb"].tobytes()).hexdigest()
                    )
    if not frames:
        raise ValueError(f"trace environment has no camera named {camera!r}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if separate_cameras:
        paths = {}
        for name, images in frames.items():
            camera_name = name.removesuffix("_rgb")
            path = out_path.with_name(f"{out_path.stem}_{camera_name}{out_path.suffix}")
            upstream_sim.dump_video(upstream_sim.get_frames({name: images}, state_keys=[]),
                                    path, fps=trace.meta.control_hz)
            paths[camera_name] = path
        return paths
    video_frames = upstream_sim.get_frames(frames, state_keys=[])
    upstream_sim.dump_video(video_frames, out_path, fps=trace.meta.control_hz)
    return out_path


def state_summary(trace: DynamicVLATrace) -> dict[str, Any]:
    """Return a compact inventory without starting Isaac Sim."""

    scheduler = trace.final_policy.get("scheduler")
    pending = scheduler.get("pending") if isinstance(scheduler, dict) else None
    pending_chunk = pending.get("chunk") if isinstance(pending, dict) else None
    if pending_chunk is None:
        pending_chunk = trace.final_policy.get("pending_chunk")

    return {
        "task_id": trace.meta.task_id,
        "episode": trace.meta.episode,
        "steps": trace.num_steps,
        "success": trace.meta.success,
        "episode_length_s": trace.meta.episode_length_s,
        "state_tensors": {
            "/".join(path): list(values.shape[1:])
            for path, values in sorted(trace.states.items())
        },
        "action_shape": list(trace.actions.shape[1:]),
        "released_chunks": sum(
            step.get("released_chunk") is not None for step in trace.protocol
        ),
        "pending_chunk": pending_chunk is not None,
    }


def render_diagnostic_video(
    trace: DynamicVLATrace, env: Any, upstream_sim: Any, out_path: str | Path,
    *, mode: str, frames: int, samples: int = 1, start: int = 0,
    camera: str | None = None, png_frames: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Controlled rendering comparison, separate from production replay.

    ``states`` follows the production restoration path; ``frozen`` restores
    once, while ``frozen-restore`` repeats that same restore each frame.
    ``actions`` uses normal env.step and reports drift against recorded states.
    Extra samples render a held pose without advancing physics. They test
    temporal convergence, not equivalence to the experiment's render cadence.
    """
    import time
    import torch

    if mode not in {"states", "frozen", "frozen-restore", "actions"}:
        raise ValueError(f"unknown diagnostic mode: {mode}")
    if frames < 3 or samples < 1 or not 0 <= start <= trace.num_steps:
        raise ValueError("diagnostics require frames >=3, samples >=1 and a valid start")
    if mode == "actions" and (start != 0 or samples != 1):
        raise ValueError("action control requires start=0 and samples=1")
    base = env.unwrapped
    restore_env_state(base, trace.state(start))
    driver = PlanarDriver(trace.meta.motion) if mode == "actions" and trace.meta.motion else None
    if driver is not None:
        driver.prepare(base)
    views_by_camera: dict[str, list[Any]] = {}
    raw_paths = []
    errors = []
    error_paths = []
    scene_errors = []
    started = time.monotonic()
    count = frames if mode.startswith("frozen") else min(frames, trace.num_steps+1-start)
    for offset in range(count):
        index = start+offset
        if mode == "actions" and offset:
            if driver is not None:
                driver.apply(base, index-1)
            actual = step_and_capture_env_state(env, torch.as_tensor(trace.actions[index-1], device=base.device))
            expected = trace.state(index)
            error, path = max_state_error(actual, expected)
            errors.append(error)
            error_paths.append(list(path) if path is not None else None)
            scene_error, _ = max_state_error(
                {key: value for key, value in actual.items() if key[0] == "scene"},
                {key: value for key, value in expected.items() if key[0] == "scene"},
            )
            scene_errors.append(scene_error)
            # Auto-reset replaces the terminal scene, so do not film that frame.
            if bool(trace.done[index-1]):
                break
        else:
            if mode in {"states", "frozen-restore"}:
                restore_env_state(base, trace.state(start if mode == "frozen-restore" else index))
            for _ in range(samples):
                base.sim.render()
                base.scene.update(base.step_dt / samples)
        for name, outputs in upstream_sim.get_camera_views(base.scene.sensors, ["rgb"]).items():
            if camera is None or name == camera:
                pixels = outputs["rgb"].squeeze(0).copy()
                views_by_camera.setdefault(name, []).append(pixels)
    if not views_by_camera:
        raise ValueError(f"trace has no camera {camera!r}")
    rendered_s = time.monotonic()-started
    paths = {}
    for name, images in views_by_camera.items():
        for offset in png_frames:
            if not 0 <= offset < len(images):
                continue
            import imageio.v3 as iio

            raw = Path(out_path).parent / "raw" / f"{name}_{offset:05d}.png"
            raw.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(raw, images[offset])
            raw_paths.append(str(raw))
        path = Path(out_path)
        if camera is None:
            path = path.with_name(f"{path.stem}_{name}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        upstream_sim.dump_video(upstream_sim.get_frames({f"{name}_rgb": images}, state_keys=[]),
                                path, fps=trace.meta.control_hz)
        paths[name] = str(path)
    return dict(mode=mode, start=start, samples=samples, videos=paths, raw_frames=raw_paths,
                state_capture="state-restoration-v1",
                frames=len(next(iter(views_by_camera.values()))), render_seconds=rendered_s,
                total_seconds=time.monotonic()-started,
                max_state_error=max(errors, default=None), per_step_state_error=errors,
                max_state_error_path=error_paths[errors.index(max(errors))] if errors else None,
                max_scene_state_error=max(scene_errors, default=None),
                physics_dt=base.physics_dt, control_dt=base.step_dt,
                decimation=base.cfg.decimation, render_interval=base.cfg.sim.render_interval)
