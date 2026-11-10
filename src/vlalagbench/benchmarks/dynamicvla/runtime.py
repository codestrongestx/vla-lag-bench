"""Shared DynamicVLA simulator and continuation control boundaries."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

from vlalagbench.benchmarks.dynamicvla.tracing import DynamicVLATraceRecorder
from vlalagbench.simulators.isaaclab.tracing import (
    step_with_pre_reset_capture,
)

ENV_REUSE_CPU_OBSERVATION = "VLALAGBENCH_DYNAMICVLA_REUSE_CPU_OBSERVATION"
ENV_RGB_ONLY_CAMERAS = "VLALAGBENCH_DYNAMICVLA_RGB_ONLY_CAMERAS"


def owned_cpu_policy_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Copy end-effector tensors once for the observation's CPU consumers.

    Keep tensor types and original keys so trace, IPC, and path recording can
    retain their existing conversions. CPU sources also require owned storage.
    """
    return {
        **state,
        "end_effector": {
            key: value.detach().to(device="cpu", copy=True)
            for key, value in state["end_effector"].items()
        },
    }


def episode_length_s_for_steps(steps: int, step_dt: float) -> float:
    """Encode an exact control deadline for Isaac Lab's ceil-based property."""

    if steps <= 0 or step_dt <= 0:
        raise ValueError("episode deadline steps and step duration must be positive")
    return math.nextafter(steps * step_dt, 0.0)


def receive_reply(upstream: Any, poller: Any, socket: Any, timeout_s: float) -> Any:
    deadline = time.perf_counter() + timeout_s
    while True:
        remaining_ms = int((deadline - time.perf_counter()) * 1000)
        if remaining_ms <= 0:
            return None
        if poller.poll(max(1, min(remaining_ms, 1000))):
            return socket.recv_pyobj(flags=upstream.zmq.NOBLOCK)


def action_tensor(upstream: Any, value: Any, env: Any) -> Any:
    if isinstance(value, upstream.np.ndarray):
        action = upstream.torch.from_numpy(value).to(env.unwrapped.device)
    elif isinstance(value, upstream.torch.Tensor):
        action = value.to(env.unwrapped.device)
    else:
        return None
    if tuple(action.shape) != (env.unwrapped.num_envs, 8):
        return None
    return action


def hold_pose_action(torch: Any, env: Any, last_action: Any) -> Any:
    """The hand's current pose with the last gripper command.

    Sent on every wait step after a model action, so the arm holds where it is
    instead of continuing toward the last absolute pose target.
    """

    pos, quat = env.unwrapped.action_manager.get_term("arm_action")._compute_frame_pose()
    return torch.cat([pos, quat, last_action[:, 7:]], dim=1)


def step_and_capture_terminal(
    env: Any,
    action: Any,
    recorder: DynamicVLATraceRecorder,
    *,
    terminal_observation: Callable[[], Mapping[tuple[str, ...], Any]] | None = None,
) -> bool:
    def capture() -> None:
        recorder.capture_post_step()
        if terminal_observation is not None:
            recorder.record_observation(terminal_observation())

    return step_with_pre_reset_capture(env, action, capture)


def evaluate_termination_terms(manager: Any, env: Any) -> dict[str, bool]:
    """Evaluate the pinned DOM termination terms against the current state."""

    results = {}
    for name in manager.active_terms:
        term = manager.get_term_cfg(name)
        results[name] = bool(term.func(env, **term.params).all().item())
    return results


def refresh_sensors(env: Any) -> None:
    """Re-render sensors after a state write without advancing physics."""

    for sensor in env.scene.sensors.values():
        sensor.reset()
    env.sim.render()
    env.scene.update(0.0)


def rgb_camera_views(sensors: Mapping[str, Any], convert: Callable[..., dict]) -> dict:
    """Convert only camera RGB outputs, as upstream returns for ``views=["rgb"]``.

    Upstream converts every annotator and decodes semantic segmentation before
    discarding everything except RGB. Rendering and annotators are unchanged.
    """
    return {
        name: convert(
            {key: value for key, value in sensor.data.output.items() if key == "rgb"},
            backend="numpy",
        )
        for name, sensor in sensors.items()
        if type(sensor).__name__ == "Camera"
    }


def _assert_same_camera_views(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    if list(expected) != list(actual):
        raise RuntimeError("RGB-only camera views changed camera order")
    for name, outputs in expected.items():
        if list(outputs) != list(actual[name]):
            raise RuntimeError(f"RGB-only camera views changed outputs for {name}")
        for key, value in outputs.items():
            other = actual[name][key]
            if (type(other) is not type(value) or other.dtype != value.dtype
                    or other.shape != value.shape or other.tobytes() != value.tobytes()):
                raise RuntimeError(f"RGB-only camera view differs for {name}/{key}")


def read_policy_observation(upstream: Any, env: Any) -> tuple[dict, dict]:
    base = env.unwrapped
    rgb_only = os.environ.get(ENV_RGB_ONLY_CAMERAS, "1")
    if rgb_only in ("1", "verify"):
        from isaaclab.utils import convert_dict_to_backend

        cameras = rgb_camera_views(base.scene.sensors, convert_dict_to_backend)
        if rgb_only == "verify":
            _assert_same_camera_views(
                upstream.sim.get_camera_views(base.scene.sensors, ["rgb"]), cameras
            )
    elif rgb_only == "0":
        cameras = upstream.sim.get_camera_views(base.scene.sensors, ["rgb"])
    else:
        raise ValueError(f"{ENV_RGB_ONLY_CAMERAS} must be 1, 0, or verify")
    state = upstream.sim.get_curr_state(
        ee_state=base.scene["ee_frame"].data,
        object_state=base.scene["object"].data,
        env_origins=base.scene["robot"].data.root_pos_w,
        robot_quat=base.scene["robot"].data.root_quat_w,
        device=base.device,
    )
    return cameras, state


def flatten_policy_observation(
    cameras: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[tuple[str, ...], Any]:
    return {
        **{
            ("images", str(name)): outputs["rgb"]
            for name, outputs in cameras.items()
        },
        **{
            ("state", "end_effector", str(name)): value
            for name, value in state["end_effector"].items()
        },
    }


def policy_observation_message(
    cameras: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    index: int,
    seed: int,
    latency_steps: int,
) -> dict[str, Any]:
    end_effector = {
        key: value.cpu().numpy()
        for key, value in state["end_effector"].items()
    }
    if "gripper" not in end_effector and "pos" in end_effector:
        end_effector["gripper"] = end_effector["pos"][:, :0]
    return {
        "dt_scale": 1.0,
        "index": index,
        "seed": seed,
        "latency_steps": latency_steps,
        "observation.state": {
            "end_effector": end_effector
        },
        **{
            f"observation.images.{key}": value["rgb"]
            for key, value in cameras.items()
        },
    }


def wait_for_action(upstream: Any, socket: Any, timeout_s: float) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        action = upstream.get_latest_action(socket)
        if action is not None:
            return action
        time.sleep(0.05)
    return None
