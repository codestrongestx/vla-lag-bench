"""Constant-speed planar motion of DOM rigid objects.

Offsets and velocities are authored in the robot base frame relative to each
target's reset position. Before every control step the driver writes the
target's world XY position and XY velocity; height, orientation, and their
velocities stay physical, as in the LIBERO driver. By default, driven targets get zero
static and dynamic friction on their own material: PhysX friction anchors
otherwise hold a resting body against pose writes until it has moved past the
friction correlation distance, so it advances in jumps. Contacts combine
friction by averaging, so a driven container still grips what it carries. The
episode's resolved actors, reset positions, robot base pose, and driven
friction are recorded so replay and continuation reproduce the same writes.
Native-friction experiments instead record and preserve each shape's materials.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from vlalagbench.benchmarks.dynamicvla.scene import euler_xyz_from_quat
from vlalagbench.config import (
    DynamicActor,
    DynamicVLAExperiment,
    ReflectedTrajectorySample,
    ResolvedDynamicActor,
)
from vlalagbench.trajectory import motion_path, offset_and_velocity, sample_trajectory

Position = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]


class EpisodeMotion(BaseModel):
    """Everything needed to reproduce one episode's motion writes."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    control_hz: Annotated[float, Field(gt=0)]
    robot_position_w: Position
    robot_yaw_rad: FiniteFloat
    reset_positions_w: dict[str, Position]
    actors: Annotated[list[ResolvedDynamicActor], Field(min_length=1)]
    driven_friction: Annotated[float, Field(ge=0)] | None
    native_materials: dict[str, list[list[list[FiniteFloat]]]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_native_materials(self):
        if self.driven_friction is None and set(self.native_materials) != {
            actor.target for actor in self.actors
        }:
            raise ValueError("native friction requires recorded materials for every driven target")
        return self


def episode_actors(
    experiment: DynamicVLAExperiment, replica_id: int
) -> list[ResolvedDynamicActor]:
    """The motion of one replica: dynamics plus its scheduled trajectory."""

    actors: list[ResolvedDynamicActor] = list(experiment.dynamics)
    sampled = sample_trajectory(
        experiment.trajectory, base_seed=0, task_id=0, episode_id=replica_id
    )
    if sampled is not None:
        actors.append(sampled)
    return actors


def actors_from_json(values: Sequence[Mapping[str, Any]]) -> list[ResolvedDynamicActor]:
    return [
        ReflectedTrajectorySample.model_validate(value)
        if value.get("kind") == "reflected_trajectory"
        else DynamicActor.model_validate(value)
        for value in values
    ]


class PlanarDriver:
    """Evaluates and writes an episode's planar motion."""

    def __init__(self, motion: EpisodeMotion):
        self.motion = motion
        self._paths = {
            index: motion_path(actor)
            for index, actor in enumerate(motion.actors)
            if isinstance(actor, DynamicActor) and actor.motion is not None
        }
        cos, sin = math.cos(motion.robot_yaw_rad), math.sin(motion.robot_yaw_rad)
        self._rotation = np.array([[cos, -sin], [sin, cos]])

    def command(self, step: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """World XY position and XY velocity of every moving target."""

        commands = {}
        for index, actor in enumerate(self.motion.actors):
            offset, velocity = offset_and_velocity(
                actor, step, self.motion.control_hz, self._paths.get(index)
            )
            origin = np.asarray(self.motion.reset_positions_w[actor.target][:2])
            commands[actor.target] = (
                origin + self._rotation @ offset[:2],
                self._rotation @ velocity[:2],
            )
        return commands

    def prepare(self, env: Any) -> None:
        """Set the driven targets' material friction before the first write."""

        import torch

        for target in {actor.target for actor in self.motion.actors}:
            view = env.scene[target].root_physx_view
            materials = view.get_material_properties()
            if self.motion.driven_friction is None:
                materials = torch.as_tensor(
                    self.motion.native_materials[target], dtype=materials.dtype,
                    device=materials.device,
                )
                if materials.shape != view.get_material_properties().shape:
                    raise ValueError(f"recorded native material shape differs for {target}")
            else:
                materials[..., :2] = self.motion.driven_friction
            view.set_material_properties(
                materials, torch.arange(view.count, dtype=torch.int32)
            )

    def place_carried(self, env: Any) -> None:
        """Start objects resting on a driven target at its scheduled start.

        Run once, right after the step-0 write: it shifts each carried object
        by its carrier's displacement from the reset pose. Height,
        orientation and (zero) velocity are kept; later motion is contact only.
        """

        import torch

        commands = self.command(0)
        for actor in self.motion.actors:
            carried = getattr(actor, "carried_objects", [])
            if not carried:
                continue
            origin = np.asarray(self.motion.reset_positions_w[actor.target][:2])
            shift = commands[actor.target][0] - origin
            for name in carried:
                asset = env.scene[name]
                state = asset.data.root_state_w.clone()
                state[:, 0:2] += torch.as_tensor(shift, dtype=state.dtype, device=state.device)
                asset.write_root_state_to_sim(state)

    def apply(self, env: Any, step: int) -> None:
        import torch

        scene = env.scene
        for target, (position, velocity) in self.command(step).items():
            asset = scene[target]
            state = asset.data.root_state_w.clone()
            state[:, 0:2] = torch.as_tensor(position, dtype=state.dtype, device=state.device)
            state[:, 7:9] = torch.as_tensor(velocity, dtype=state.dtype, device=state.device)
            asset.write_root_state_to_sim(state)


def capture_episode_motion(
    env: Any, actors: Sequence[ResolvedDynamicActor], control_hz: float,
    *, driven_friction: float | None = None,
) -> EpisodeMotion:
    """Read the reset poses that anchor an episode's motion."""

    scene = env.scene
    robot = scene["robot"].data
    quat = robot.root_quat_w[0].tolist()
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    if abs(roll) > 1e-6 or abs(pitch) > 1e-6:
        raise RuntimeError("DOM robot base must be upright to drive planar motion")
    return EpisodeMotion(
        control_hz=float(control_hz),
        driven_friction=driven_friction,
        native_materials={
            target: scene[target].root_physx_view.get_material_properties().cpu().tolist()
            for target in {actor.target for actor in actors}
        } if driven_friction is None else {},
        robot_position_w=[float(value) for value in robot.root_pos_w[0].tolist()],
        robot_yaw_rad=float(yaw),
        reset_positions_w={
            actor.target: [
                float(value) for value in scene[actor.target].data.root_pos_w[0].tolist()
            ]
            for actor in actors
        },
        actors=list(actors),
    )
