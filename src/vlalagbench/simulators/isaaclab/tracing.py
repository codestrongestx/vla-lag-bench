"""Capture and restore the state used by the pinned DynamicVLA DOM tasks.

The scene portion uses :class:`isaaclab.scene.InteractiveScene`'s structured
``get_state``/``reset_to`` API.  A small runtime portion keeps the episode
counter, action-manager buffers, interval-event timers, external wrenches, and
random generators that live outside the physics scene but affect the next
control step. The private fields are isolated here and pinned to Isaac Lab
2.2.1; this module is not a generic snapshot contract for arbitrary tasks.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


StatePath = tuple[str, ...]
FlatState = dict[StatePath, np.ndarray]


def capture_env_state(env: Any) -> FlatState:
    """Return a CPU copy of the state needed at a control-step boundary."""

    state = _flatten_tensors(("scene",), env.scene.get_state(is_relative=False))
    for name in ("episode_length_buf", "common_step_counter", "_sim_step_counter"):
        if hasattr(env, name):
            state[("runtime", name)] = _as_array(getattr(env, name))
    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None:
        for name in ("action", "prev_action"):
            value = getattr(action_manager, name, None)
            if value is not None:
                state[("runtime", "action_manager", name)] = _as_array(value)
    event_manager = getattr(env, "event_manager", None)
    if event_manager is not None:
        for index, time_left in enumerate(event_manager._interval_term_time_left):
            state[("runtime", "event_manager", "interval", str(index))] = _as_array(
                time_left
            )
    # Contact forces are observational diagnostics, not state to restore.
    sensors = getattr(env.scene, "sensors", {})
    if "dynabench_box_contacts" in sensors:
        force = sensors["dynabench_box_contacts"].data.force_matrix_w
        if force is None:
            raise RuntimeError("Instrumented box contact sensor has no filtered forces")
        state[("diagnostics", "box_robot_contact_force_w")] = _as_array(force)
    state.update(_capture_external_wrenches(env.scene))
    state.update(_capture_rng_state(getattr(env, "device", env.scene.device)))
    return state


def restore_env_state(env: Any, state: Mapping[StatePath, np.ndarray]) -> None:
    """Restore one state captured by :func:`capture_env_state`."""

    import torch

    scene_state = _unflatten_tensors(
        {
            path[1:]: torch.as_tensor(value, device=env.scene.device)
            for path, value in state.items()
            if path[:1] == ("scene",)
        }
    )
    env.scene.reset_to(scene_state, is_relative=False)

    for name in ("episode_length_buf", "common_step_counter", "_sim_step_counter"):
        path = ("runtime", name)
        if path in state:
            _restore_value(env, name, state[path])
    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None:
        for name in ("action", "prev_action"):
            path = ("runtime", "action_manager", name)
            if path in state:
                _restore_value(action_manager, name, state[path])
    event_manager = getattr(env, "event_manager", None)
    if event_manager is not None:
        for index, time_left in enumerate(event_manager._interval_term_time_left):
            path = ("runtime", "event_manager", "interval", str(index))
            _copy_tensor(time_left, state[path])
    _restore_external_wrenches(env.scene, state)

    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(0.0)
    _restore_rng_state(state, getattr(env, "device", env.scene.device))


def step_and_capture_env_state(env: Any, action: Any) -> FlatState:
    """Step a gym-wrapped manager environment and preserve its terminal state."""

    base = env.unwrapped
    terminal_state: FlatState | None = None

    def capture() -> None:
        nonlocal terminal_state
        terminal_state = capture_env_state(base)

    step_with_pre_reset_capture(env, action, capture)
    return terminal_state or capture_env_state(base)


def step_with_pre_reset_capture(
    env: Any, action: Any, capture: Callable[[], None]
) -> bool:
    """Run one step and call ``capture`` before any terminal auto-reset."""

    manager = env.unwrapped.recorder_manager
    record_pre_reset = manager.record_pre_reset
    captured = False

    def capture_then_record(env_ids: Any) -> Any:
        nonlocal captured
        capture()
        captured = True
        return record_pre_reset(env_ids)

    manager.record_pre_reset = capture_then_record
    try:
        env.step(action)
    finally:
        manager.record_pre_reset = record_pre_reset
    return captured


def max_state_error(
    actual: Mapping[StatePath, np.ndarray], expected: Mapping[StatePath, np.ndarray]
) -> tuple[float, StatePath | None]:
    """Return the largest absolute tensor error and its state path."""

    if actual.keys() != expected.keys():
        raise ValueError("Isaac Lab state structures differ")
    if any(actual[path].shape != expected[path].shape for path in actual):
        raise ValueError("Isaac Lab state tensor shapes differ")
    errors = {
        path: _max_array_error(actual[path], expected[path])
        for path in actual
        if actual[path].size
    }
    if not errors:
        return 0.0, None
    path = max(errors, key=errors.__getitem__)
    return errors[path], path


def _max_array_error(actual: np.ndarray, expected: np.ndarray) -> float:
    actual_float = actual.astype(np.float64)
    expected_float = expected.astype(np.float64)
    actual_nan = np.isnan(actual_float)
    expected_nan = np.isnan(expected_float)
    if not np.array_equal(actual_nan, expected_nan):
        return float("inf")
    difference = np.where(actual_nan, 0.0, np.abs(actual_float - expected_float))
    return float(np.max(difference))


def _flatten_tensors(prefix: StatePath, value: Any) -> FlatState:
    if isinstance(value, Mapping):
        flattened: FlatState = {}
        for key, child in value.items():
            flattened.update(_flatten_tensors((*prefix, str(key)), child))
        return flattened
    return {prefix: _as_array(value)}


def _unflatten_tensors(values: Mapping[StatePath, Any]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for path, value in values.items():
        if not path:
            raise ValueError("state tensor path must not be empty")
        node = root
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return root


def _as_array(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach().cpu().numpy()
    return np.asarray(value).copy()


def _restore_value(owner: Any, name: str, value: np.ndarray) -> None:
    current = getattr(owner, name)
    copy = getattr(current, "copy_", None)
    if copy is not None:
        _copy_tensor(current, value)
    elif np.isscalar(current):
        setattr(owner, name, np.asarray(value).item())
    else:
        current[...] = value


def _copy_tensor(target: Any, value: np.ndarray) -> None:
    import torch

    target.copy_(torch.as_tensor(value, device=target.device, dtype=target.dtype))


def _external_wrench_assets(scene: Any) -> Any:
    for kind in ("articulations", "rigid_objects"):
        for name, asset in getattr(scene, kind, {}).items():
            yield kind, name, asset


def _capture_external_wrenches(scene: Any) -> FlatState:
    state: FlatState = {}
    for kind, name, asset in _external_wrench_assets(scene):
        prefix = ("runtime", "external_wrench", kind, name)
        state[(*prefix, "force")] = _as_array(asset._external_force_b)
        state[(*prefix, "torque")] = _as_array(asset._external_torque_b)
        state[(*prefix, "position")] = _as_array(
            asset._external_wrench_positions_b
        )
        state[(*prefix, "flags")] = np.asarray(
            [
                asset.has_external_wrench,
                asset.uses_external_wrench_positions,
                asset._use_global_wrench_frame,
            ],
            dtype=np.uint8,
        )
    return state


def _restore_external_wrenches(
    scene: Any, state: Mapping[StatePath, np.ndarray]
) -> None:
    import torch

    for kind, name, asset in _external_wrench_assets(scene):
        prefix = ("runtime", "external_wrench", kind, name)
        flags = state[(*prefix, "flags")]
        device = asset.device
        forces = torch.as_tensor(state[(*prefix, "force")], device=device)
        torques = torch.as_tensor(state[(*prefix, "torque")], device=device)
        positions = (
            torch.as_tensor(state[(*prefix, "position")], device=device)
            if flags[1]
            else None
        )
        asset.set_external_force_and_torque(
            forces,
            torques,
            positions=positions,
            is_global=bool(flags[2]),
        )
        asset.has_external_wrench = bool(flags[0])
        asset.uses_external_wrench_positions = bool(flags[1])


def _capture_rng_state(device: Any) -> FlatState:
    import torch

    numpy_state = np.random.get_state()
    python_state = random.getstate()
    state: FlatState = {
        ("runtime", "rng", "numpy_keys"): numpy_state[1].copy(),
        ("runtime", "rng", "numpy_meta"): np.asarray(
            [numpy_state[2], numpy_state[3]], dtype=np.int64
        ),
        ("runtime", "rng", "numpy_gauss"): np.asarray(
            [numpy_state[4]], dtype=np.float64
        ),
        ("runtime", "rng", "python_keys"): np.asarray(
            python_state[1], dtype=np.uint64
        ),
        ("runtime", "rng", "python_version"): np.asarray(
            python_state[0], dtype=np.int64
        ),
        ("runtime", "rng", "python_gauss"): np.asarray(
            [np.nan if python_state[2] is None else python_state[2]],
            dtype=np.float64,
        ),
        ("runtime", "rng", "torch_cpu"): torch.get_rng_state().cpu().numpy(),
    }
    if str(device).startswith("cuda"):
        state[("runtime", "rng", "torch_device")] = (
            torch.cuda.get_rng_state(device).cpu().numpy()
        )
    return state


def _restore_rng_state(state: Mapping[StatePath, np.ndarray], device: Any) -> None:
    import torch

    numpy_meta = state[("runtime", "rng", "numpy_meta")]
    np.random.set_state(
        (
            "MT19937",
            state[("runtime", "rng", "numpy_keys")],
            int(numpy_meta[0]),
            int(numpy_meta[1]),
            float(state[("runtime", "rng", "numpy_gauss")][0]),
        )
    )
    python_gauss = float(state[("runtime", "rng", "python_gauss")][0])
    random.setstate(
        (
            int(state[("runtime", "rng", "python_version")]),
            tuple(
                int(value)
                for value in state[("runtime", "rng", "python_keys")]
            ),
            None if np.isnan(python_gauss) else python_gauss,
        )
    )
    torch.set_rng_state(
        torch.as_tensor(state[("runtime", "rng", "torch_cpu")], dtype=torch.uint8)
    )
    device_state = state.get(("runtime", "rng", "torch_device"))
    if device_state is not None:
        torch.cuda.set_rng_state(
            torch.as_tensor(device_state, dtype=torch.uint8), device=device
        )
