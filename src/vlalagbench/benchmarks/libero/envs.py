"""Build LIBERO environments — from a benchmark task or from trace metadata."""

from __future__ import annotations

import pathlib
from typing import Any

from pydantic import TypeAdapter

from vlalagbench.config import ResolvedDynamicActor
from vlalagbench.simulators.mujoco.dynamics import DynamicEnv
from vlalagbench.benchmarks.libero import _vendor
from vlalagbench.benchmarks.libero.tracing import parse_libero_environment
from vlalagbench.simulators.mujoco.tracing import TraceMeta

LIBERO_ENV_RESOLUTION = 256  # resolution used to render pi0.5 training data
LIBERO_CONTROL_HZ = 20
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
_DYNAMIC_ACTOR_ADAPTER = TypeAdapter(ResolvedDynamicActor)


def require_supported_control_hz(control_hz: int) -> None:
    if control_hz != LIBERO_CONTROL_HZ:
        raise NotImplementedError("stock LIBERO runs at 20 control_hz")


def make_env(
    task,
    resolution: int,
    seed: int,
    *,
    hard_reset: bool = True,
    ignore_done: bool = False,
    camera_observations: bool = True,
):
    """LIBERO env for one benchmark task, with optional camera observations."""
    return _make_env(
        task.problem_folder,
        task.bddl_file,
        resolution,
        seed,
        hard_reset=hard_reset,
        ignore_done=ignore_done,
        camera_observations=camera_observations,
    )


def make_env_from_trace(
    meta: TraceMeta,
    resolution: int | None = None,
    *,
    hard_reset: bool = True,
    ignore_done: bool = False,
):
    """Rebuild the LIBERO env an episode was recorded in.

    Matches the ``env_factory`` signature expected by
    :mod:`vlalagbench.simulators.mujoco.replay`.
    """
    environment = parse_libero_environment(meta)
    env = _make_env(
        environment.problem_folder,
        environment.bddl_file,
        resolution if resolution is not None else environment.resolution,
        environment.seed,
        hard_reset=hard_reset,
        ignore_done=ignore_done,
    )
    dynamics = [
        _DYNAMIC_ACTOR_ADAPTER.validate_python(item)
        for item in getattr(meta, "dynamics", [])
    ]
    return DynamicEnv(env, dynamics)


def _make_env(
    problem_folder: str,
    bddl_file: str,
    resolution: int,
    seed: int,
    *,
    hard_reset: bool,
    ignore_done: bool = False,
    camera_observations: bool = True,
):
    libero = _vendor.libero()

    bddl_path = pathlib.Path(libero.get_libero_path("bddl_files")) / problem_folder / bddl_file
    env = libero.ControlEnv(
        bddl_file_name=str(bddl_path),
        hard_reset=hard_reset,
        ignore_done=ignore_done,
        use_camera_obs=camera_observations,
        has_offscreen_renderer=camera_observations,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)  # seed affects object placement even with fixed init states
    return env


def reset_to_init_state(env: Any, init_state: Any) -> Any:
    env.reset()
    return env.set_init_state(init_state)


def warm_up(
    env: Any, obs: Any, steps: int, *, hold_dynamic_time: float | None = None
) -> Any:
    hold = getattr(env, "hold_dynamic_time", None)
    for _ in range(steps):
        if hold_dynamic_time is None or hold is None:
            env.set_dynamic_step(None)
        else:
            hold(hold_dynamic_time)
        obs, *_ = env.step(LIBERO_DUMMY_ACTION)
    return obs
