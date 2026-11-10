"""Planar MuJoCo actors driven at the physics-substep frequency."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from vlalagbench.trajectory import MotionPath, motion_path, offset_and_velocity



def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


_SLIDE_AXES = (("x", "1 0 0"), ("y", "0 1 0"))


def slide_joint_names(target: str) -> tuple[str, str]:
    """Names of the planar slide joints added to a jointless driven body."""

    return tuple(f"{target}_slide_{axis}" for axis, _ in _SLIDE_AXES)


def add_slide_joints(xml: str, targets: Sequence[str]) -> str:
    """Give jointless bodies X/Y slide joints; bodies that already have them are kept."""

    root = ET.fromstring(xml)
    for target in targets:
        body = root.find(f".//body[@name='{target}']")
        if body is None:
            raise ValueError(f"slide-joint target {target!r} is missing from the model XML")
        names = slide_joint_names(target)
        present = {joint.get("name") for joint in body.findall("joint")}
        if set(names) <= present:
            continue
        if present or body.find("freejoint") is not None:
            raise ValueError(f"slide-joint target {target!r} must be jointless")
        for index, ((_, axis), name) in enumerate(zip(_SLIDE_AXES, names)):
            body.insert(index, ET.Element("joint", {
                "name": name, "type": "slide", "axis": axis, "limited": "false",
                "damping": "0", "armature": "0", "frictionloss": "0", "stiffness": "0",
            }))
    return ET.tostring(root, encoding="unicode")


def _slide_targets(actors: Sequence[Any]) -> tuple[str, ...]:
    return tuple(actor.target for actor in actors if getattr(actor, "slide_joints", False))


@dataclass
class _Driver:
    actor: Any
    body_id: int
    joint: str | None
    initial_position: np.ndarray
    motion_path: MotionPath | None
    grasp_object: Any | None = None
    grasp_steps_required: int | None = None
    grasp_steps: int = 0
    released: bool = False
    slides: tuple[str, str] | None = None
    # World XY direction of each slide joint (columns); the body may be yawed.
    slide_axes: np.ndarray | None = None


class _SubstepSim:
    def __init__(
        self,
        sim: Any,
        drivers: Sequence[_Driver],
        *,
        control_hz: float,
        substeps_per_control: int,
    ) -> None:
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "_drivers", tuple(drivers))
        object.__setattr__(self, "_control_hz", control_hz)
        object.__setattr__(self, "_substeps_per_control", substeps_per_control)
        object.__setattr__(self, "_control_step", None)
        object.__setattr__(self, "_substep", 0)
        object.__setattr__(self, "_prepared_step_zero", False)
        object.__setattr__(self, "_hold_time", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sim, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._sim, name, value)

    def set_control_step(self, step: int | None) -> None:
        if step == 0:
            for driver in self._drivers:
                if driver.slides is not None:
                    position = _slide_position(self._sim, driver)
                elif driver.joint is None:
                    position = self._sim.model.body_pos[driver.body_id]
                else:
                    pose = np.asarray(
                        self._sim.data.get_joint_qpos(driver.joint),
                        dtype=np.float64,
                    )
                    position = pose[:3]
                driver.initial_position = np.asarray(position, dtype=np.float64).copy()
                if self._prepared_step_zero:
                    offset, _ = offset_and_velocity(
                        driver.actor, 0.0, self._control_hz, driver.motion_path
                    )
                    driver.initial_position[:2] -= offset[:2]
            self._prepared_step_zero = True
        self._control_step = step
        self._substep = 0
        self._hold_time = None

    def set_hold_time(self, time_steps: float) -> None:
        self._control_step = None
        self._hold_time = time_steps
        self._substep = 0
        self._apply(time_steps, free_joints=True, zero_planar_velocity=True)
        self._sim.forward()

    def step(self) -> None:
        if self._hold_time is not None:
            self._apply(
                self._hold_time, free_joints=True, zero_planar_velocity=True
            )
            self._sim.step()
            self._apply(
                self._hold_time, free_joints=True, zero_planar_velocity=True
            )
            self._sim.forward()
            return
        if self._control_step is None:
            self._sim.step()
            return
        before = self._control_step + self._substep / self._substeps_per_control
        self._apply(before)
        self._sim.step()
        after = self._control_step + (self._substep + 1) / self._substeps_per_control
        self._apply(after)
        self._sim.forward()
        self._substep += 1

    def _apply(
        self,
        time_steps: float,
        *,
        free_joints: bool = True,
        zero_planar_velocity: bool = False,
    ) -> None:
        for driver in self._drivers:
            jointed = driver.joint is not None or driver.slides is not None
            if jointed and not free_joints:
                continue
            offset, linear_velocity = offset_and_velocity(
                driver.actor, time_steps, self._control_hz, driver.motion_path
            )
            if driver.slides is not None:
                # The body origin stays fixed; the slides carry the offset and
                # the path velocity, so friction acts on objects resting on it.
                origin = self._sim.model.body_pos[driver.body_id]
                coordinates = np.linalg.solve(
                    driver.slide_axes, driver.initial_position[:2] + offset[:2] - origin[:2]
                )
                rates = np.linalg.solve(driver.slide_axes, linear_velocity[:2])
                for index, joint in enumerate(driver.slides):
                    self._sim.data.set_joint_qpos(joint, coordinates[index])
                    self._sim.data.set_joint_qvel(
                        joint, 0.0 if zero_planar_velocity else rates[index]
                    )
                continue
            if driver.joint is None:
                self._sim.model.body_pos[driver.body_id, :2] = (
                    driver.initial_position[:2] + offset[:2]
                )
                continue

            if driver.released:
                continue
            pose = np.asarray(
                self._sim.data.get_joint_qpos(driver.joint), dtype=np.float64
            ).copy()
            pose[:2] = driver.initial_position[:2] + offset[:2]
            velocity = np.asarray(
                self._sim.data.get_joint_qvel(driver.joint), dtype=np.float64
            ).copy()
            velocity[:2] = 0.0 if zero_planar_velocity else linear_velocity[:2]
            # Drive only XY; height, orientation, and their velocities remain
            # governed by physics, including before a grasp-triggered release.
            self._sim.data.set_joint_qpos(driver.joint, pose)
            self._sim.data.set_joint_qvel(driver.joint, velocity)

    def apply_time(self, time_steps: float, *, free_joints: bool = False) -> None:
        """Apply an exact dynamic time without advancing physics."""

        self._apply(time_steps, free_joints=free_joints)
        self._sim.forward()


class DynamicEnv:
    """Apply configured free-joint motion while preserving the wrapped env API."""

    def __init__(self, env: Any, actors: Sequence[Any]):
        self.env = env
        self.actors = tuple(actors)
        self.initial_object_positions_m: dict[str, tuple[float, float, float]] = {}
        self.initial_object_yaws_rad: dict[str, float] = {}
        self._host: Any | None = None
        self._sim: _SubstepSim | None = None
        self._original_sim: Any | None = None
        self._base_xml_processor: Any | None = None
        self._install_xml_processor()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self._detach()
        return self.env.reset(*args, **kwargs)

    def set_dynamic_actors(self, actors: Sequence[Any]) -> None:
        """Replace actors before initializing the next episode."""

        self._detach()
        self.actors = tuple(actors)
        self._install_xml_processor()

    def _install_xml_processor(self) -> None:
        """Add slide joints to driven fixtures whenever the model is rebuilt."""

        try:
            host = _simulation_host(self.env)
        except AttributeError:
            return
        if not hasattr(host, "set_xml_processor"):
            return
        current = getattr(host, "_xml_processor", None)
        if getattr(current, "_vlalagbench_slide_targets", None) is None:
            self._base_xml_processor = current
        base = self._base_xml_processor
        targets = _slide_targets(self.actors)
        if not targets:
            host.set_xml_processor(base)
            return

        def processor(xml: str) -> str:
            return add_slide_joints(base(xml) if base is not None else xml, targets)

        processor._vlalagbench_slide_targets = targets
        host.set_xml_processor(processor)

    def set_initial_object_positions(
        self, positions_m: Mapping[str, Sequence[float]]
    ) -> None:
        """Set per-object world positions for the next initialized episode."""

        self.initial_object_positions_m = {
            target: tuple(float(value) for value in position)
            for target, position in positions_m.items()
        }

    def set_initial_object_yaws(self, yaws_rad: Mapping[str, float]) -> None:
        """Set per-object world yaw angles for the next initialized episode."""

        self.initial_object_yaws_rad = {
            target: float(yaw) for target, yaw in yaws_rad.items()
        }

    def set_init_state(self, init_state: Any, *args: Any, **kwargs: Any) -> Any:
        init_state = self._expand_slide_state(init_state)
        result = self.env.set_init_state(init_state, *args, **kwargs)
        if self.initial_object_positions_m or self.initial_object_yaws_rad:
            self._apply_initial_object_poses()
            result = self.env.regenerate_obs_from_state(self.env.get_sim_state())
        self._attach()
        return result

    def _expand_slide_state(self, state: Any) -> Any:
        """Insert zero slide coordinates into a state recorded without them."""

        targets = _slide_targets(self.actors)
        if not targets:
            return state
        sim = _simulation_host(self.env).sim
        model = sim.model
        size = int(np.asarray(sim.get_state().flatten()).size)
        state = np.asarray(state, dtype=np.float64)
        if state.size == size:
            return state
        joints = [model.joint_name2id(name) for target in targets for name in slide_joint_names(target)]
        added_q = {int(model.jnt_qposadr[joint]) for joint in joints}
        added_v = {int(model.jnt_dofadr[joint]) for joint in joints}
        if state.size != size - len(added_q) - len(added_v):
            raise ValueError(
                f"state has {state.size} values; expected {size} or {size - len(added_q) - len(added_v)}"
            )
        nq, nv = int(model.nq), int(model.nv)
        keep_q = [index for index in range(nq) if index not in added_q]
        keep_v = [index for index in range(nv) if index not in added_v]
        expanded = np.zeros(size)
        expanded[0] = state[0]
        expanded[1 + np.asarray(keep_q)] = state[1:1 + len(keep_q)]
        start = 1 + len(keep_q)
        expanded[1 + nq + np.asarray(keep_v)] = state[start:start + len(keep_v)]
        expanded[1 + nq + nv:] = state[start + len(keep_v):]
        return expanded

    def step(self, *args: Any, **kwargs: Any) -> Any:
        result = self.env.step(*args, **kwargs)
        if self._sim is not None and self._sim._control_step is not None:
            gripper = self._host.robots[0].gripper
            for driver in self._sim._drivers:
                required = driver.grasp_steps_required
                if required is None or driver.released:
                    continue
                if self._host._check_grasp(gripper, driver.grasp_object):
                    driver.grasp_steps += 1
                    driver.released = driver.grasp_steps >= required
                else:
                    driver.grasp_steps = 0
        return result

    def set_dynamic_step(self, step: int | None) -> None:
        if self._sim is None and self.actors:
            raise RuntimeError("set_init_state must be called before dynamic stepping")
        if step is not None and step < 0:
            raise ValueError("dynamic step must be nonnegative")
        if self._sim is not None:
            self._sim.set_control_step(step)

    def prepare_dynamic_step(self, step: int) -> bool:
        """Move free-joint actors before observing the first control step."""

        if self._sim is None:
            return False
        self._sim.set_control_step(step)
        before = {
            id(driver): _slide_position(self._sim, driver)
            for driver in self._sim._drivers
            if driver.slides is not None
        }
        self._sim.apply_time(step, free_joints=True)
        # The scheduled start pose also places what rests on a moved fixture.
        for driver in self._sim._drivers:
            if driver.slides is None:
                continue
            shift = _slide_position(self._sim, driver)[:2] - before[id(driver)][:2]
            for name in getattr(driver.actor, "carried_objects", ()):
                _shift_free_body(self._sim, name, shift)
        self._sim.forward()
        return True

    def hold_dynamic_time(self, time_steps: float) -> bool:
        """Keep actors at one trajectory time while other bodies settle."""

        if self._sim is None:
            return False
        self._sim.set_hold_time(time_steps)
        return True

    def restore_prepared_dynamic_step(self, step: int) -> bool:
        """Recover origins when the restored state already includes dynamic motion."""

        if self._sim is None:
            return False
        self._sim.set_control_step(step)
        for driver in self._sim._drivers:
            if driver.joint is not None or driver.slides is not None:
                offset, _ = offset_and_velocity(
                    driver.actor, step, self._sim._control_hz, driver.motion_path
                )
                driver.initial_position[:2] -= offset[:2]
        self._sim.apply_time(step)
        return True

    def restore_actor_origins(self, model_xml: str | None) -> None:
        """Restore model poses absent from flattened MuJoCo simulator state."""

        if model_xml is None or not self.actors:
            return
        if self._sim is not None:
            raise RuntimeError("restore actor origins before set_init_state attaches dynamics")
        sim = _simulation_host(self.env).sim
        root = ET.fromstring(model_xml)
        for actor in self.actors:
            body = root.find(f".//body[@name='{actor.target}']")
            if body is None:
                raise RuntimeError(
                    f"dynamic body {actor.target!r} is missing from the recorded model XML"
                )
            body_id = sim.model.body_name2id(actor.target)
            position = np.fromstring(body.attrib.get("pos", ""), sep=" ")
            quaternion = np.fromstring(body.attrib.get("quat", ""), sep=" ")
            if position.shape == (3,):
                sim.model.body_pos[body_id] = position
            if quaternion.shape == (4,):
                sim.model.body_quat[body_id] = quaternion
        sim.forward()

    def apply_dynamic_time(
        self, time_steps: float, *, free_joints: bool = False
    ) -> None:
        """Set moving actors for state-only rendering at an exact control time."""

        if self._sim is None and self.actors:
            raise RuntimeError("set_init_state must be called before applying dynamic time")
        if time_steps < 0:
            raise ValueError("dynamic time must be nonnegative")
        if self._sim is not None:
            self._sim.apply_time(time_steps, free_joints=free_joints)

    def close(self) -> None:
        self._detach()
        self.env.close()

    def _attach(self) -> None:
        self._detach()
        if not self.actors:
            return
        host = _simulation_host(self.env)
        sim = host.sim
        control_timestep = float(host.control_timestep)
        model_timestep = float(host.model_timestep)
        substeps = round(control_timestep / model_timestep)
        if substeps <= 0 or not math.isclose(
            substeps * model_timestep,
            control_timestep,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("control timestep must be an integer number of physics substeps")
        drivers = [_driver(host, sim, actor) for actor in self.actors]
        proxy = _SubstepSim(
            sim,
            drivers,
            control_hz=1.0 / control_timestep,
            substeps_per_control=substeps,
        )
        host.sim = proxy
        self._host = host
        self._original_sim = sim
        self._sim = proxy

    def _apply_initial_object_poses(self) -> None:
        host = _simulation_host(self.env)
        sim = host.sim
        targets = (
            self.initial_object_positions_m.keys()
            | self.initial_object_yaws_rad.keys()
        )
        for target in targets:
            position = self.initial_object_positions_m.get(target)
            yaw = self.initial_object_yaws_rad.get(target)
            try:
                body_id = sim.model.body_name2id(target)
            except Exception as error:
                raise ValueError(f"unknown initial-state object {target!r}") from error
            joint_count = int(sim.model.body_jntnum[body_id])
            slides = slide_joint_names(target)
            if joint_count == 0 or (
                target in _slide_targets(self.actors) and joint_count == len(slides)
            ):
                if position is not None:
                    sim.model.body_pos[body_id] = position
                if yaw is not None:
                    sim.model.body_quat[body_id] = _yaw_quaternion(yaw)
                for name in slides if joint_count else ():
                    sim.data.set_joint_qpos(name, 0.0)
                    sim.data.set_joint_qvel(name, 0.0)
                continue
            if joint_count != 1:
                raise ValueError(
                    f"initial-state object {target!r} must be jointless or have one free joint"
                )
            joint_id = int(sim.model.body_jntadr[body_id])
            joint = sim.model.joint_id2name(joint_id)
            pose = np.asarray(sim.data.get_joint_qpos(joint), dtype=np.float64).copy()
            if pose.shape != (7,):
                raise ValueError(f"initial-state object {target!r} must use a free joint")
            if position is not None:
                pose[:3] = position
            if yaw is not None:
                pose[3:] = _yaw_quaternion(yaw)
            sim.data.set_joint_qpos(joint, pose)
        sim.forward()

    def _detach(self) -> None:
        if self._host is not None and self._original_sim is not None:
            self._host.sim = self._original_sim
        self._host = None
        self._sim = None
        self._original_sim = None


def _simulation_host(env: Any) -> Any:
    current = env
    for _ in range(8):
        if all(hasattr(current, name) for name in ("sim", "control_timestep", "model_timestep")):
            return current
        current = current.__dict__.get("env") if hasattr(current, "__dict__") else None
        if current is None:
            break
    raise AttributeError("could not find the robosuite simulation host")


def _driver(host: Any, sim: Any, actor: Any) -> _Driver:
    try:
        body_id = sim.model.body_name2id(actor.target)
    except Exception as error:
        raise ValueError(f"unknown dynamic target body {actor.target!r}") from error
    joint_count = int(sim.model.body_jntnum[body_id])
    path = motion_path(actor) if getattr(actor, "motion", None) is not None else None
    if getattr(actor, "slide_joints", False):
        slides = slide_joint_names(actor.target)
        ids = [int(sim.model.body_jntadr[body_id]) + index for index in range(joint_count)]
        if joint_count != 2 or [sim.model.joint_id2name(index) for index in ids] != list(slides):
            raise ValueError(
                f"slide-joint target {actor.target!r} lacks its slide joints; "
                "the model must be rebuilt (hard reset) after setting the actors"
            )
        if getattr(actor, "release_on_grasp_steps", None) is not None:
            raise ValueError("release_on_grasp_steps requires a free-joint target")
        rotation = np.asarray(sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)
        axes = np.stack([rotation @ np.asarray(sim.model.jnt_axis[index]) for index in ids], axis=1)
        if not np.allclose(axes[2], 0.0, atol=1e-9):
            raise ValueError(f"slide-joint target {actor.target!r} must stay upright")
        driver = _Driver(
            actor=actor, body_id=body_id, joint=None,
            initial_position=np.zeros(3), motion_path=path, slides=slides,
            slide_axes=axes[:2],
        )
        driver.initial_position = _slide_position(sim, driver)
        return driver
    if joint_count == 0:
        if getattr(actor, "release_on_grasp_steps", None) is not None:
            raise ValueError("release_on_grasp_steps requires a free-joint target")
        return _Driver(
            actor=actor,
            body_id=body_id,
            joint=None,
            initial_position=np.asarray(sim.model.body_pos[body_id], dtype=np.float64).copy(),
            motion_path=path,
        )
    if joint_count != 1:
        raise ValueError(f"dynamic target {actor.target!r} must be jointless or have one free joint")
    joint_id = int(sim.model.body_jntadr[body_id])
    joint = sim.model.joint_id2name(joint_id)
    qpos = np.asarray(sim.data.get_joint_qpos(joint), dtype=np.float64)
    qvel = np.asarray(sim.data.get_joint_qvel(joint), dtype=np.float64)
    if qpos.shape != (7,) or qvel.shape != (6,):
        raise ValueError(f"dynamic target {actor.target!r} must use a free joint")
    grasp_steps_required = getattr(actor, "release_on_grasp_steps", None)
    grasp_object = None
    if grasp_steps_required is not None:
        grasp_object = next(
            obj for obj in host.objects_dict.values() if obj.root_body == actor.target
        )
    return _Driver(
        actor=actor,
        body_id=body_id,
        joint=joint,
        initial_position=qpos[:3].copy(),
        motion_path=path,
        grasp_object=grasp_object,
        grasp_steps_required=grasp_steps_required,
    )


def _slide_position(sim: Any, driver: _Driver) -> np.ndarray:
    position = np.asarray(sim.model.body_pos[driver.body_id], dtype=np.float64).copy()
    coordinates = [float(sim.data.get_joint_qpos(joint)) for joint in driver.slides]
    position[:2] += driver.slide_axes @ coordinates
    return position


def _shift_free_body(sim: Any, name: str, shift: np.ndarray) -> None:
    body_id = sim.model.body_name2id(name)
    if int(sim.model.body_jntnum[body_id]) != 1:
        raise ValueError(f"carried object {name!r} must have one free joint")
    joint = sim.model.joint_id2name(int(sim.model.body_jntadr[body_id]))
    pose = np.asarray(sim.data.get_joint_qpos(joint), dtype=np.float64).copy()
    if pose.shape != (7,):
        raise ValueError(f"carried object {name!r} must use a free joint")
    pose[:2] += shift
    sim.data.set_joint_qpos(joint, pose)
