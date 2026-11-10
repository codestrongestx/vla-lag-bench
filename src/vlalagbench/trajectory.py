"""Deterministic episode sampling and trajectory kinematics."""

from __future__ import annotations

import bisect
import functools
import hashlib
import math
from typing import Any, Sequence

import numpy as np

from vlalagbench.config import (
    ClosedPathTrajectory,
    DynamicActor,
    Motion,
    ReflectedTrajectory,
    ReflectedTrajectorySample,
    ResolvedDynamicActor,
    TrajectorySpec,
)


# Path points, arc-length parameters, and cumulative distances.
MotionPath = tuple[np.ndarray, np.ndarray, np.ndarray]


def resolve_dynamic_phases(
    actors: Sequence[Any],
    *,
    base_seed: int,
    task_id: int,
    initial_state_id: int | str,
) -> list[Any]:
    """Resolve random path phases reproducibly for one initial state."""

    resolved = []
    for actor in actors:
        if (
            isinstance(actor, DynamicActor)
            and actor.motion is not None
            and actor.motion.initial_phase == "random"
        ):
            phase = float(
                np.random.default_rng(
                    _stable_seed(base_seed, task_id, initial_state_id, actor.target)
                ).random()
            )
            actor = actor.model_copy(
                update={"motion": actor.motion.model_copy(update={"initial_phase": phase})}
            )
        resolved.append(actor)
    return resolved


def sample_trajectory(
    trajectory: TrajectorySpec | None,
    *,
    base_seed: int,
    task_id: int,
    episode_id: int,
) -> ResolvedDynamicActor | None:
    """Pair both directions at one stratified position for every phase bin."""

    if trajectory is None:
        return None
    if trajectory.episode_schedule is not None:
        try:
            scheduled = trajectory.episode_schedule[episode_id]
        except IndexError as error:
            raise IndexError(
                f"trajectory episode {episode_id} is outside its explicit schedule"
            ) from error
        phase_bin = episode_id
        initial_phase = scheduled.initial_phase
        initial_direction = scheduled.initial_direction
    else:
        cells = trajectory.phase_bins * 2
        replicate, cell = divmod(episode_id, cells)
        if trajectory.shuffle:
            order = np.random.default_rng(
                _stable_seed(base_seed, task_id, replicate, "trajectory-order")
            ).permutation(cells)
            cell = int(order[cell])
        phase_bin, direction_bit = divmod(cell, 2)
        seed = _stable_seed(base_seed, task_id, replicate, phase_bin)
        jitter = float(np.random.default_rng(seed).random())
        initial_phase = (phase_bin + jitter) / trajectory.phase_bins
        initial_direction = 1 if direction_bit == 0 else -1
    if isinstance(trajectory, ReflectedTrajectory):
        return ReflectedTrajectorySample(
            target=trajectory.target,
            start_step=trajectory.start_step,
            path_start_offset_m=trajectory.path_start_offset_m,
            path_end_offset_m=trajectory.path_end_offset_m,
            speed_mps=trajectory.speed_mps,
            phase_bin=phase_bin,
            initial_phase=initial_phase,
            initial_direction=initial_direction,
        )
    if isinstance(trajectory, ClosedPathTrajectory):
        return DynamicActor(
            target=trajectory.target,
            start_step=trajectory.start_step,
            end_step=None,
            loop=True,
            interpolation=trajectory.interpolation,
            release_on_grasp_steps=trajectory.release_on_grasp_steps,
            slide_joints=trajectory.slide_joints,
            carried_objects=trajectory.carried_objects,
            motion=Motion(
                speed_mps=trajectory.speed_mps,
                initial_phase=initial_phase,
                initial_direction=initial_direction,
            ),
            keyframes=trajectory.keyframes,
        )
    raise TypeError(f"unsupported trajectory type {type(trajectory).__name__}")


def reflected_kinematics(
    actor: ReflectedTrajectorySample,
    time_steps: float,
    control_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return relative position and velocity on an exact triangle-wave path."""

    start = np.asarray(actor.path_start_offset_m, dtype=np.float64)
    end = np.asarray(actor.path_end_offset_m, dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    elapsed_s = max(0.0, time_steps - actor.start_step) / control_hz
    distance = (
        actor.initial_phase * length
        + actor.initial_direction * actor.speed_mps * elapsed_s
    ) % (2.0 * length)
    if distance <= length:
        fraction = distance / length
        direction = actor.initial_direction
    else:
        fraction = (2.0 * length - distance) / length
        direction = -actor.initial_direction
    position = start + fraction * delta
    velocity = delta / length * actor.speed_mps * direction
    return position, velocity


def offset_and_velocity(
    actor: Any,
    time_steps: float,
    control_hz: float,
    path: MotionPath | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one dynamic actor's position offset and velocity.

    Both are in the frame its keyframes are authored in; the simulator driver
    maps them into world coordinates. Before start_step the actor holds still
    at its starting point, so a delayed start never jumps; from end_step it
    holds where it stopped.
    """

    zero = np.zeros(3, dtype=np.float64)
    if time_steps < actor.start_step:
        offset, _ = offset_and_velocity(actor, actor.start_step, control_hz, path)
        return offset, zero
    if isinstance(actor, ReflectedTrajectorySample):
        return reflected_kinematics(actor, time_steps, control_hz)
    local = time_steps - actor.start_step
    if actor.end_step is not None and time_steps >= actor.end_step:
        offset, _ = offset_and_velocity(
            actor.model_copy(update={"end_step": None}), actor.end_step, control_hz, path
        )
        return offset, zero

    if actor.motion is not None:
        return _constant_speed_offset_and_velocity(actor, local, control_hz, path)

    steps = [keyframe.step for keyframe in actor.keyframes]
    if actor.loop:
        local %= steps[-1]
    elif local >= steps[-1]:
        return np.asarray(actor.keyframes[-1].position_offset_m, dtype=np.float64), zero

    left_index = max(0, bisect.bisect_right(steps, local) - 1)
    step_span = steps[left_index + 1] - steps[left_index]
    fraction = (local - steps[left_index]) / step_span
    positions = [np.asarray(keyframe.position_offset_m, dtype=np.float64) for keyframe in actor.keyframes]
    position, derivative = _path_state(positions, left_index + fraction, actor.interpolation, actor.loop)
    return position, derivative * control_hz / step_span


_ARC_LENGTH_RELATIVE_TOLERANCE = 1e-3
_ARC_LENGTH_MAX_REFINEMENTS = 20
_ARC_LENGTH_MAX_CELLS_PER_SEGMENT = 2048


def _constant_speed_offset_and_velocity(
    actor: Any,
    local_steps: float,
    control_hz: float,
    path: MotionPath | None,
) -> tuple[np.ndarray, np.ndarray]:
    positions, parameters, distances = path or motion_path(actor)
    distance = (
        actor.motion.initial_phase * distances[-1]
        + actor.motion.initial_direction
        * local_steps
        * actor.motion.speed_mps
        / control_hz
    )
    if actor.loop:
        distance %= distances[-1]
    elif distance <= 0.0:
        return positions[0].copy(), np.zeros(3)
    elif distance >= distances[-1]:
        return positions[-1].copy(), np.zeros(3)

    right = np.searchsorted(distances, distance, side="right")
    ds = distances[right] - distances[right - 1]
    du_ds = (parameters[right] - parameters[right - 1]) / ds
    parameter = parameters[right - 1] + (distance - distances[right - 1]) * du_ds
    position, tangent = _path_state(positions, parameter, actor.interpolation, actor.loop)
    return (
        position,
        tangent * du_ds * actor.motion.speed_mps * actor.motion.initial_direction,
    )


def motion_path(actor: Any) -> MotionPath:
    """Return a distance-timed actor's cached path and arc-length table."""

    positions_key = tuple(
        tuple(float(value) for value in keyframe.position_offset_m)
        for keyframe in actor.keyframes
    )
    return _arc_length_table(positions_key, actor.interpolation, actor.loop)


def motion_path_length(actor: Any) -> float:
    """Return the arc length of a distance-timed actor's path."""

    return float(motion_path(actor)[2][-1])


@functools.lru_cache(maxsize=256)
def _arc_length_table(
    positions_key: tuple[tuple[float, ...], ...],
    interpolation: str,
    loop: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache one path and its parameter-to-distance lookup table."""

    positions = np.asarray(positions_key)
    parameters = np.arange(len(positions), dtype=np.float64)
    if interpolation == "cubic":
        pieces = [
            _segment_parameters(positions, segment, loop)[:-1]
            for segment in range(len(positions) - 1)
        ]
        parameters = np.concatenate([*pieces, np.asarray([len(positions) - 1])])

    samples = np.asarray([_path_state(positions, parameter, interpolation, loop)[0] for parameter in parameters])
    chord_lengths = np.linalg.norm(np.diff(samples, axis=0), axis=1)
    distances = np.concatenate(([0.0], np.cumsum(chord_lengths)))
    if not math.isfinite(float(distances[-1])) or distances[-1] <= 0.0:
        raise ValueError("motion.speed_mps path must have positive finite arc length")
    keep = np.concatenate(([True], np.diff(distances) > 0.0))
    parameters, distances = parameters[keep], distances[keep]
    for array in (positions, parameters, distances):
        array.setflags(write=False)
    return positions, parameters, distances


def _segment_parameters(positions: np.ndarray, segment: int, loop: bool) -> np.ndarray:
    """Refine one cubic segment until linear arc-length inversion is accurate."""

    error = "cubic motion path has a stationary tangent or is ill-conditioned"
    critical = _tangent_extrema(positions, segment, loop)
    candidates = [float(segment), *critical, float(segment + 1)]
    tangent_norms = [
        np.linalg.norm(_path_state(positions, value, "cubic", loop)[1])
        for value in candidates
    ]
    if min(tangent_norms) <= max(tangent_norms) * _ARC_LENGTH_RELATIVE_TOLERANCE:
        raise ValueError(error)

    parameters = np.asarray([segment, segment + 1], dtype=np.float64)
    for _ in range(_ARC_LENGTH_MAX_REFINEMENTS):
        refined = [parameters[0]]
        for left, right in zip(parameters, parameters[1:]):
            endpoints = [_path_state(positions, value, "cubic", loop) for value in (left, right)]
            chord = np.linalg.norm(endpoints[1][0] - endpoints[0][0])
            extrema = [left, *(value for value in critical if left < value < right), right]
            ratios = [
                np.linalg.norm(_path_state(positions, value, "cubic", loop)[1])
                * (right - left) / chord
                for value in extrema
            ] if chord else [math.inf]
            if max(abs(ratio - 1.0) for ratio in ratios) > _ARC_LENGTH_RELATIVE_TOLERANCE:
                refined.append((left + right) / 2.0)
            refined.append(right)
        if len(refined) - 1 > _ARC_LENGTH_MAX_CELLS_PER_SEGMENT:
            raise ValueError(error)
        if len(refined) == len(parameters):
            return parameters
        parameters = np.asarray(refined)
    raise ValueError(error)


def _tangent_extrema(positions: np.ndarray, segment: int, loop: bool) -> list[float]:
    """Return interior extrema of tangent magnitude for one cubic segment."""

    a, midpoint, endpoint = (
        _path_state(positions, segment + fraction, "cubic", loop)[1]
        for fraction in (0.0, 0.5, 1.0)
    )
    c = 2.0 * (endpoint + a - 2.0 * midpoint)
    b = endpoint - a - c
    coefficients = np.stack((a, b, c), axis=1)
    squared = sum((np.convolve(axis, axis) for axis in coefficients), np.zeros(5))
    derivative = np.arange(1, 5) * squared[1:]
    roots = np.polynomial.polynomial.polyroots(derivative) if np.any(derivative) else []
    return [
        segment + root.real
        for root in roots
        if abs(root.imag) < 1e-9 and 0.0 < root.real < 1.0
    ]


def _path_state(
    positions: Sequence[np.ndarray], parameter: float, interpolation: str, loop: bool
) -> tuple[np.ndarray, np.ndarray]:
    segment = min(int(parameter), len(positions) - 2)
    fraction = parameter - segment
    if interpolation == "linear":
        left = positions[segment]
        derivative = positions[segment + 1] - left
        return left + fraction * derivative, derivative
    if interpolation == "cubic":
        return _catmull_rom(positions, segment, fraction, loop=loop)
    raise ValueError(f"unsupported interpolation {interpolation!r}")


def _catmull_rom(
    positions: Sequence[np.ndarray],
    left_index: int,
    fraction: float,
    *,
    loop: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a uniform Catmull-Rom segment and its derivative."""

    if loop:
        points = positions[:-1]
        count = len(points)
        p0 = points[(left_index - 1) % count]
        p1 = points[left_index % count]
        p2 = points[(left_index + 1) % count]
        p3 = points[(left_index + 2) % count]
    else:
        last = len(positions) - 1
        p0 = positions[max(0, left_index - 1)]
        p1 = positions[left_index]
        p2 = positions[min(last, left_index + 1)]
        p3 = positions[min(last, left_index + 2)]

    t = fraction
    t2 = t * t
    t3 = t2 * t
    position = 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )
    derivative = 0.5 * (
        (-p0 + p2)
        + 2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t
        + 3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t2
    )
    return position, derivative


def _stable_seed(*parts: int | str) -> int:
    payload = ":".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
