"""Robot-base-frame views of DOM environment initial states.

DOM environment JSONs store every asset's reset state in world coordinates of
its house scene. Authored initial states use the robot base frame instead, so
one layout reads the same in every scene. Quaternions are Isaac's (w, x, y, z)
and yaws follow Isaac's XYZ Euler convention, R = Rz(yaw) Ry(pitch) Rx(roll).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

from pathlib import PurePosixPath

from vlalagbench.config import DynamicVLAExperiment, DynamicVLAInitialStateOverride

RIGID_OBJECT = "isaaclab.assets.rigid_object.rigid_object:RigidObject"
ROBOT = "robot"


def rigid_assets(environment: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return the scene's rigid objects in JSON order."""

    return {
        name: value
        for name, value in environment["scene"].items()
        if isinstance(value, Mapping) and value.get("class_type") == RIGID_OBJECT
    }


def euler_xyz_from_quat(quat: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = _normalized(quat)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ]


class RobotFrame:
    """The upright robot base pose of one DOM environment."""

    def __init__(self, environment: Mapping[str, Any]) -> None:
        state = environment["scene"][ROBOT]["init_state"]
        roll, pitch, yaw = euler_xyz_from_quat(state["rot"])
        if abs(roll) > 1e-9 or abs(pitch) > 1e-9:
            raise ValueError("DOM robot base must be upright")
        self.origin = [float(value) for value in state["pos"]]
        self.yaw = yaw

    def to_robot(self, position: Sequence[float]) -> list[float]:
        dx, dy = position[0] - self.origin[0], position[1] - self.origin[1]
        cos, sin = math.cos(self.yaw), math.sin(self.yaw)
        return [cos * dx + sin * dy, -sin * dx + cos * dy, position[2] - self.origin[2]]

    def to_world(self, position: Sequence[float]) -> list[float]:
        cos, sin = math.cos(self.yaw), math.sin(self.yaw)
        return [
            self.origin[0] + cos * position[0] - sin * position[1],
            self.origin[1] + sin * position[0] + cos * position[1],
            self.origin[2] + position[2],
        ]

    def vector_to_robot(self, vector: Sequence[float]) -> list[float]:
        cos, sin = math.cos(self.yaw), math.sin(self.yaw)
        return [cos * vector[0] + sin * vector[1], -sin * vector[0] + cos * vector[1], vector[2]]


def official_state(environment: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Every rigid asset's official reset state in the robot base frame."""

    frame = RobotFrame(environment)
    state = {}
    for name, asset in rigid_assets(environment).items():
        initial = asset["init_state"]
        roll, pitch, yaw = euler_xyz_from_quat(initial["rot"])
        state[name] = {
            "position_m": frame.to_robot(initial["pos"]),
            "yaw_rad": _wrap(yaw - frame.yaw),
            "roll_rad": roll,
            "pitch_rad": pitch,
            "velocity_mps": frame.vector_to_robot(initial["lin_vel"]),
        }
    return state


def apply_initial_state(
    environment: Mapping[str, Any],
    override: DynamicVLAInitialStateOverride | None,
    *,
    start_at_rest: bool,
) -> dict[str, Any]:
    """Return a copy of the environment JSON with one authored reset state."""

    result = copy.deepcopy(dict(environment))
    frame = RobotFrame(environment)
    assets = rigid_assets(result)
    positions = override.object_positions_m if override else {}
    yaws = override.object_yaws_rad if override else {}
    unknown = sorted({*positions, *yaws} - set(assets))
    if unknown:
        raise ValueError("initial-state override names unknown DOM assets: " + ", ".join(unknown))
    for name, asset in assets.items():
        initial = asset["init_state"]
        if name in positions:
            initial["pos"] = frame.to_world(positions[name])
        if name in yaws:
            roll, pitch, _ = euler_xyz_from_quat(initial["rot"])
            initial["rot"] = quat_from_euler_xyz(roll, pitch, yaws[name] + frame.yaw)
        if start_at_rest:
            initial["lin_vel"] = [0.0, 0.0, 0.0]
            initial["ang_vel"] = [0.0, 0.0, 0.0]
    return result


def authored_environment(
    experiment: DynamicVLAExperiment,
    relative_path: str,
    environment: Mapping[str, Any],
    state_index: int,
) -> dict[str, Any] | None:
    """The environment JSON one authored state runs, or None for the official file."""

    override = next(
        (
            entry
            for entry in experiment.initial_state_overrides
            if PurePosixPath(entry.environment) == PurePosixPath(relative_path)
            and entry.initial_state_index == state_index
        ),
        None,
    )
    if override is None and not experiment.benchmark.start_at_rest:
        return None
    return apply_initial_state(
        environment, override, start_at_rest=experiment.benchmark.start_at_rest
    )


def _normalized(quat: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in quat))
    if not norm:
        raise ValueError("quaternion must be nonzero")
    return [value / norm for value in quat]


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
