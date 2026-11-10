"""Reproduce recorded episodes: render videos, verify determinism.

Env-family-agnostic: both entry points take an ``env_factory(meta,
resolution=None)`` callable that rebuilds the environment a trace was
recorded in from its metadata (for LIBERO:
:func:`vlalagbench.benchmarks.libero.envs.make_env_from_trace`).

Two reproduction modes:

- :func:`render_trace_video` — state playback. Each recorded MuJoCo state is
  loaded into a fresh env and rendered offscreen. This reproduces the run
  exactly (bit-for-bit the recorded trajectory) with no policy or server
  involved, at any camera/resolution.
- :func:`verify_trace` — action replay. Re-runs the recorded actions from the
  recorded initial state and reports how far the re-simulated states drift
  from the recorded ones (0.0 on the same machine/versions means the run is
  exactly re-simulable).
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Callable, Protocol

import numpy as np

from vlalagbench.simulators.mujoco.tracing import EpisodeTrace, TraceMeta, get_mj_sim, get_sim_state, set_sim_state
from vlalagbench.simulators.mujoco.rendering import MjvSceneRenderer, SceneOverlay

logger = logging.getLogger(__name__)


class EnvFactory(Protocol):
    def __call__(self, meta: TraceMeta, resolution: int | None = None) -> Any: ...


def render_trace_video(
    trace: EpisodeTrace,
    out_path: str | pathlib.Path,
    env_factory: EnvFactory,
    camera: str = "agentview",
    resolution: int = 512,
    fps: float | None = None,
    scene_overlay: SceneOverlay | None = None,
) -> pathlib.Path:
    """Render exact state playback, optionally with visual-only scene geoms."""
    import imageio

    env = env_factory(trace.meta, resolution=resolution)
    scene_renderer = None
    try:
        env.reset()
        restore_origins = getattr(env, "restore_actor_origins", None)
        if restore_origins is not None:
            restore_origins(trace.model_xml)
        set_init = getattr(env, "set_init_state", None)
        if set_init is not None:
            set_init(trace.states[0])
        sim = get_mj_sim(env)
        if scene_overlay is not None:
            scene_renderer = MjvSceneRenderer(
                sim,
                camera=camera,
                width=resolution,
                height=resolution,
                overlay=scene_overlay,
            )
        frames = []
        warmup_steps = int(getattr(trace.meta, "num_steps_wait", 0))
        apply_dynamic_time = getattr(env, "apply_dynamic_time", None)
        for index, state in enumerate(trace.states):
            sim.set_state_from_flattened(np.asarray(state, dtype=np.float64))
            sim.forward()
            if apply_dynamic_time is not None:
                apply_dynamic_time(max(0, index - warmup_steps))
            if scene_renderer is None:
                frame = sim.render(width=resolution, height=resolution, camera_name=camera)
                frames.append(frame[::-1])  # robosuite offscreen frames are upside down
            else:
                frames.append(scene_renderer.render())
    finally:
        if scene_renderer is not None:
            scene_renderer.close()
        env.close()

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = fps if fps is not None else trace.meta.control_freq
    imageio.mimwrite(out_path, frames, fps=int(round(fps)))
    logger.info("wrote %d frames to %s", len(frames), out_path)
    return out_path


def verify_trace(
    trace: EpisodeTrace,
    env_factory: EnvFactory,
    atol: float = 1e-9,
    idle_step: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Re-simulate the recorded actions and measure drift from recorded states.

    ``idle_step(env, action, wait_started=...)`` steps the control ticks the
    trace marks as not acting, as the evaluator did while it waited.
    """
    env = env_factory(trace.meta, resolution=None)
    try:
        env.reset()
        # Restore the recorded initial state: LIBERO-style envs expose
        # set_init_state (set_state + forward, no extra physics steps);
        # otherwise write the state into the sim directly.
        set_init = getattr(env, "set_init_state", None)
        if set_init is not None:
            set_init(trace.states[0])
        else:
            set_sim_state(env, trace.states[0])
        init_diff = float(np.max(np.abs(get_sim_state(env) - trace.states[0])))
        per_step: list[float] = []
        warmup_steps = int(getattr(trace.meta, "num_steps_wait", 0))
        set_dynamic_step = getattr(env, "set_dynamic_step", None)
        prepare_dynamic_step = getattr(env, "prepare_dynamic_step", None)
        restore_prepared = getattr(env, "restore_prepared_dynamic_step", None)
        prepared_before_warmup = bool(
            getattr(trace.meta, "dynamic_step_zero_prepared_before_warmup", False)
        )
        state_zero_prepared = prepared_before_warmup or warmup_steps == 0
        prepared = (
            bool(restore_prepared(0))
            if state_zero_prepared and restore_prepared
            else False
        )
        hold_dynamic_time = getattr(env, "hold_dynamic_time", None)
        waiting = False
        for index, action in enumerate(trace.actions):
            if index < warmup_steps:
                if prepared_before_warmup and hold_dynamic_time is not None:
                    hold_dynamic_time(0.0)
                elif set_dynamic_step is not None:
                    set_dynamic_step(None)
            elif set_dynamic_step is not None and (
                index != warmup_steps or not prepared
            ):
                set_dynamic_step(index - warmup_steps)
            if idle_step is not None and index >= warmup_steps and not trace.acting[index]:
                idle_step(env, np.asarray(action).tolist(), wait_started=not waiting)
                waiting = True
            else:
                env.step(np.asarray(action).tolist())
                waiting = False
            if index + 1 == warmup_steps and prepare_dynamic_step is not None:
                prepared = bool(prepare_dynamic_step(0))
            per_step.append(float(np.max(np.abs(get_sim_state(env) - trace.states[len(per_step) + 1]))))
    finally:
        env.close()

    max_diff = max([init_diff, *per_step]) if per_step else init_diff
    first_divergence = next((i for i, d in enumerate(per_step) if d > atol), None)
    return {
        "steps": len(per_step),
        "init_state_diff": init_diff,
        "max_state_diff": max_diff,
        "exact": max_diff == 0.0,
        "within_atol": max_diff <= atol,
        "first_step_exceeding_atol": first_divergence,
    }
