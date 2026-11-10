from pathlib import Path

import numpy as np
import pytest

from vlalagbench.config import (
    ClosedPathTrajectory,
    DynamicActor,
    Keyframe,
    Motion,
    ReflectedTrajectory,
    TrajectoryEpisode,
    load_experiment,
)
from vlalagbench.trajectory import (
    motion_path,
    motion_path_length,
    offset_and_velocity,
    reflected_kinematics,
    resolve_dynamic_phases,
    sample_trajectory,
)


def _trajectory(speed_mps=0.15):
    return ReflectedTrajectory(
        target="basket",
        path_start_offset_m=[-0.495, 0.0, 0.0],
        path_end_offset_m=[0.37, 0.0, 0.0],
        speed_mps=speed_mps,
        phase_bins=12,
    )


def test_random_path_phase_is_paired_by_initial_state():
    actor = DynamicActor(
        target="basket",
        start_step=0,
        end_step=None,
        loop=False,
        interpolation="linear",
        motion=Motion(speed_mps=0.1, initial_phase="random"),
        keyframes=[
            Keyframe(position_offset_m=[0.0, 0.0, 0.0]),
            Keyframe(position_offset_m=[1.0, 0.0, 0.0]),
        ],
    )
    faster = actor.model_copy(
        update={"motion": actor.motion.model_copy(update={"speed_mps": 0.2})}
    )

    phase = resolve_dynamic_phases(
        [actor], base_seed=7, task_id=1, initial_state_id="official:4"
    )[0].motion.initial_phase
    paired_phase = resolve_dynamic_phases(
        [faster], base_seed=7, task_id=1, initial_state_id="official:4"
    )[0].motion.initial_phase
    other_phase = resolve_dynamic_phases(
        [actor], base_seed=7, task_id=1, initial_state_id="official:5"
    )[0].motion.initial_phase

    assert phase == paired_phase
    assert phase != other_phase
    assert actor.motion.initial_phase == "random"


def test_sampling_balances_phase_bins_and_directions():
    samples = [
        sample_trajectory(_trajectory(), base_seed=1701, task_id=1, episode_id=i)
        for i in range(24)
    ]

    assert [sample.phase_bin for sample in samples] == [
        value for value in range(12) for _ in range(2)
    ]
    assert [sample.initial_direction for sample in samples] == [1, -1] * 12
    for left, right in zip(samples[::2], samples[1::2], strict=True):
        assert left.initial_phase == right.initial_phase


def test_sampling_pairs_initial_conditions_across_speeds():
    slow = sample_trajectory(
        _trajectory(0.1), base_seed=1701, task_id=1, episode_id=17
    )
    fast = sample_trajectory(
        _trajectory(0.2), base_seed=1701, task_id=1, episode_id=17
    )

    assert slow.phase_bin == fast.phase_bin
    assert slow.initial_phase == fast.initial_phase
    assert slow.initial_direction == fast.initial_direction
    assert slow.speed_mps == 0.1
    assert fast.speed_mps == 0.2


def test_sampling_can_seed_shuffle_one_balanced_replicate():
    trajectory = _trajectory().model_copy(update={"shuffle": True})
    samples = [
        sample_trajectory(trajectory, base_seed=1701, task_id=1, episode_id=i)
        for i in range(24)
    ]
    repeated = [
        sample_trajectory(trajectory, base_seed=1701, task_id=1, episode_id=i)
        for i in range(24)
    ]

    assert samples == repeated
    assert [sample.phase_bin for sample in samples] != [
        value for value in range(12) for _ in range(2)
    ]
    assert sorted(sample.phase_bin for sample in samples) == [
        value for value in range(12) for _ in range(2)
    ]
    assert sum(sample.initial_direction == 1 for sample in samples) == 12
    assert sum(sample.initial_direction == -1 for sample in samples) == 12
    for phase_bin in range(12):
        paired = [sample for sample in samples if sample.phase_bin == phase_bin]
        assert {sample.initial_direction for sample in paired} == {-1, 1}
        assert paired[0].initial_phase == paired[1].initial_phase


def test_sampling_resolves_a_closed_path_to_paired_dynamic_actors():
    trajectory = ClosedPathTrajectory(
        kind="closed_path",
        target="basket",
        interpolation="linear",
        speed_mps=0.1,
        phase_bins=2,
        shuffle=True,
        keyframes=[
            Keyframe(position_offset_m=[0.0, 0.0, 0.0]),
            Keyframe(position_offset_m=[1.0, 0.0, 0.0]),
            Keyframe(position_offset_m=[0.0, 0.0, 0.0]),
        ],
    )
    samples = [
        sample_trajectory(trajectory, base_seed=7, task_id=1, episode_id=index)
        for index in range(4)
    ]

    assert all(isinstance(sample, DynamicActor) for sample in samples)
    phase_pairs = {}
    for sample in samples:
        phase_pairs.setdefault(sample.motion.initial_phase, set()).add(
            sample.motion.initial_direction
        )
    assert len(phase_pairs) == 2
    assert all(directions == {-1, 1} for directions in phase_pairs.values())


def test_explicit_episode_schedule_preserves_exact_paired_starts():
    trajectory = ReflectedTrajectory(
        target="basket",
        path_start_offset_m=[0.0, 0.0, 0.0],
        path_end_offset_m=[1.0, 0.0, 0.0],
        speed_mps=0.1,
        episode_schedule=[
            TrajectoryEpisode(initial_phase=0.17, initial_direction=1),
            TrajectoryEpisode(initial_phase=0.17, initial_direction=-1),
            TrajectoryEpisode(initial_phase=0.73, initial_direction=-1),
            TrajectoryEpisode(initial_phase=0.73, initial_direction=1),
        ],
    )

    samples = [
        sample_trajectory(trajectory, base_seed=7, task_id=1, episode_id=index)
        for index in range(4)
    ]

    assert [sample.initial_phase for sample in samples] == [0.17, 0.17, 0.73, 0.73]
    assert [sample.initial_direction for sample in samples] == [1, -1, -1, 1]
    with np.testing.assert_raises_regex(IndexError, "outside its explicit schedule"):
        sample_trajectory(trajectory, base_seed=7, task_id=1, episode_id=4)


def test_reflected_kinematics_preserves_speed_and_reverses_at_edges():
    actor = sample_trajectory(
        ReflectedTrajectory(
            target="basket",
            path_start_offset_m=[0.0, 0.0, 0.0],
            path_end_offset_m=[1.0, 0.0, 0.0],
            speed_mps=0.25,
            phase_bins=1,
        ),
        base_seed=7,
        task_id=1,
        episode_id=0,
    ).model_copy(update={"initial_phase": 0.0})

    position, velocity = reflected_kinematics(actor, 2.0, control_hz=1.0)
    np.testing.assert_allclose(position, [0.5, 0.0, 0.0])
    np.testing.assert_allclose(velocity, [0.25, 0.0, 0.0])

    position, velocity = reflected_kinematics(actor, 5.0, control_hz=1.0)
    np.testing.assert_allclose(position, [0.75, 0.0, 0.0])
    np.testing.assert_allclose(velocity, [-0.25, 0.0, 0.0])


def _actor(**updates):
    values = {
        "target": "plate",
        "start_step": 0,
        "end_step": None,
        "loop": True,
        "interpolation": "linear",
        "keyframes": [
            Keyframe(step=0, position_offset_m=[0.0, 0.0, 0.0]),
            Keyframe(step=1, position_offset_m=[0.05, 0.0, 0.0]),
            Keyframe(step=2, position_offset_m=[0.0, 0.0, 0.0]),
        ],
    }
    values.update(updates)
    return DynamicActor(**values)


def _speed_actor(
    points,
    *,
    speed_mps=0.2,
    interpolation="linear",
    loop=False,
    **updates,
):
    values = {
        "target": "plate",
        "start_step": 0,
        "end_step": None,
        "loop": loop,
        "interpolation": interpolation,
        "motion": Motion(speed_mps=speed_mps),
        "keyframes": [Keyframe(position_offset_m=point) for point in points],
    }
    values.update(updates)
    return DynamicActor(**values)


def _circle_points(radius, count):
    points = [
        [
            radius * np.cos(-np.pi / 2 + 2 * np.pi * index / count),
            radius + radius * np.sin(-np.pi / 2 + 2 * np.pi * index / count),
            0.0,
        ]
        for index in range(count)
    ]
    points.append(points[0])
    return points


ROOT = Path(__file__).resolve().parents[1]


def test_piecewise_motion_returns_position_and_velocity_in_si_units():
    position, velocity = offset_and_velocity(_actor(), 0.5, control_hz=20.0)
    np.testing.assert_allclose(position, [0.025, 0.0, 0.0])
    np.testing.assert_allclose(velocity, [1.0, 0.0, 0.0])

    position, velocity = offset_and_velocity(_actor(), 1.5, control_hz=20.0)
    np.testing.assert_allclose(position, [0.025, 0.0, 0.0])
    np.testing.assert_allclose(velocity, [-1.0, 0.0, 0.0])


def test_constant_speed_linear_motion_honors_timing_bounds():
    actor = _speed_actor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], speed_mps=0.25)
    np.testing.assert_allclose(
        offset_and_velocity(actor, 20.0, 20.0), ([0.25, 0.0, 0.0], [0.25, 0.0, 0.0])
    )
    np.testing.assert_allclose(
        offset_and_velocity(actor, 80.0, 20.0), ([1.0, 0.0, 0.0], np.zeros(3))
    )
    delayed = actor.model_copy(update={"start_step": 10})
    np.testing.assert_array_equal(offset_and_velocity(delayed, 9.999, 20.0)[0], np.zeros(3))
    np.testing.assert_allclose(
        offset_and_velocity(delayed, 30.0, 20.0)[0], [0.25, 0.0, 0.0]
    )
    ended = actor.model_copy(update={"end_step": 10})
    np.testing.assert_allclose(
        offset_and_velocity(ended, 12.0, 20.0), ([0.125, 0.0, 0.0], np.zeros(3))
    )


def test_constant_speed_motion_starts_at_its_arc_length_phase():
    actor = _speed_actor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        motion=Motion(speed_mps=0.25, initial_phase=0.4),
    )

    np.testing.assert_allclose(
        offset_and_velocity(actor, 0.0, 20.0),
        ([0.4, 0.0, 0.0], [0.25, 0.0, 0.0]),
    )


def test_delayed_motion_waits_still_at_its_phase_point():
    actor = _speed_actor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        start_step=20,
        motion=Motion(speed_mps=0.25, initial_phase=0.4),
    )

    for step in (0.0, 10.0, 19.999):
        np.testing.assert_allclose(
            offset_and_velocity(actor, step, 20.0), ([0.4, 0.0, 0.0], np.zeros(3))
        )
    np.testing.assert_allclose(
        offset_and_velocity(actor, 20.0, 20.0), ([0.4, 0.0, 0.0], [0.25, 0.0, 0.0])
    )
    np.testing.assert_allclose(offset_and_velocity(actor, 24.0, 20.0)[0], [0.45, 0.0, 0.0])


def test_linear_speed_does_not_depend_on_waypoint_spacing():
    sparse = _speed_actor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    dense = _speed_actor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.8, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    for seconds in (0.02, 1.0, 4.9):
        np.testing.assert_allclose(
            offset_and_velocity(dense, seconds * 20.0, 20.0),
            offset_and_velocity(sparse, seconds * 20.0, 20.0),
        )


def test_cubic_arc_length_speed_is_density_independent_and_continuous():
    for waypoint_count in [8, 24]:
        actor = _speed_actor(
            _circle_points(radius=0.4, count=waypoint_count),
            interpolation="cubic",
            loop=True,
        )
        period = motion_path_length(actor) / actor.motion.speed_mps
        states = [offset_and_velocity(actor, phase * period * 20.0, 20.0) for phase in np.linspace(0, 1, 65, endpoint=False)]
        np.testing.assert_allclose(
            [np.linalg.norm(velocity) for _, velocity in states], 0.2, rtol=2e-3
        )
        np.testing.assert_allclose(
            offset_and_velocity(actor, period * 20.0, 20.0), states[0], atol=1e-12
        )

        epsilon = 1e-4
        seconds = 0.371 * period
        before = offset_and_velocity(actor, (seconds - epsilon) * 20.0, 20.0)[0]
        after = offset_and_velocity(actor, (seconds + epsilon) * 20.0, 20.0)[0]
        np.testing.assert_allclose(
            (after - before) / (2 * epsilon),
            offset_and_velocity(actor, seconds * 20.0, 20.0)[1],
            rtol=2e-3,
            atol=2e-5,
        )


def test_doubling_speed_halves_loop_period():
    slow = _speed_actor(_circle_points(0.4, 8), interpolation="cubic", loop=True)
    fast = slow.model_copy(update={"motion": Motion(speed_mps=0.4)})
    period = motion_path_length(slow) / slow.motion.speed_mps
    assert motion_path_length(fast) / fast.motion.speed_mps == period / 2
    np.testing.assert_allclose(
        offset_and_velocity(fast, period * 0.37 * 10.0, 20.0)[0],
        offset_and_velocity(slow, period * 0.37 * 20.0, 20.0)[0],
    )


def test_constant_speed_cubic_rejects_singular_paths():
    paths = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[-0.4166726666666667, 0.0, 0.0], [0.0, 0.0, 0.0],
         [0.08333933333333331, 0.0, 0.0], [0.5000119999999998, 0.0, 0.0]],
    ]
    for points in paths:
        with pytest.raises(ValueError, match="stationary tangent or is ill-conditioned"):
            offset_and_velocity(_speed_actor(points, interpolation="cubic"), 0.0, 20.0)


def test_cubic_position_and_velocity_agree_at_lookup_boundaries():
    actor = _speed_actor(
        [[0.789886, -0.613253, 0.0], [0.781178, 0.386709, 0.0],
         [0.494903, 1.344856, 0.0], [0.982598, 2.217870, 0.0],
         [-0.009678, 2.093816, 0.0]],
        interpolation="cubic",
    )
    distances = motion_path(actor)[2]
    for index, distance in enumerate(distances[1:-1], start=1):
        epsilon = min(distance - distances[index - 1], distances[index + 1] - distance)
        epsilon /= actor.motion.speed_mps * 1000.0
        seconds = distance / actor.motion.speed_mps
        before = offset_and_velocity(actor, (seconds - epsilon) * 20.0, 20.0)[0]
        after = offset_and_velocity(actor, (seconds + epsilon) * 20.0, 20.0)[0]
        velocity = offset_and_velocity(actor, seconds * 20.0, 20.0)[1]
        assert np.linalg.norm((after - before) / (2 * epsilon) - velocity) <= 2e-3 * actor.motion.speed_mps


def test_mvp_motion_cycles_remain_eight_seconds():
    directory = ROOT / "configs/experiments/libero/libero_object_task1_mvp_motion"
    for path in directory.glob("*.yaml"):
        actor = load_experiment(path).dynamics[0]
        assert actor.motion is not None
        assert motion_path_length(actor) / actor.motion.speed_mps == pytest.approx(8.0, abs=1e-4)


def test_reflected_trajectory_uses_the_existing_dynamic_driver():
    actor = sample_trajectory(
        ReflectedTrajectory(
            target="plate",
            path_start_offset_m=[-0.5, 0.0, 0.0],
            path_end_offset_m=[0.5, 0.0, 0.0],
            speed_mps=0.1,
            phase_bins=1,
        ),
        base_seed=7,
        task_id=1,
        episode_id=0,
    ).model_copy(update={"initial_phase": 0.5})

    position, velocity = offset_and_velocity(actor, 1.0, control_hz=20.0)

    np.testing.assert_allclose(position, [0.005, 0.0, 0.0])
    np.testing.assert_allclose(velocity, [0.1, 0.0, 0.0])


def test_cubic_motion_follows_a_smooth_closed_curve():
    actor = _actor(
        interpolation="cubic",
        keyframes=[
            Keyframe(step=0, position_offset_m=[0.0, 0.0, 0.0]),
            Keyframe(step=1, position_offset_m=[1.0, 0.0, 0.0]),
            Keyframe(step=2, position_offset_m=[1.0, 1.0, 0.0]),
            Keyframe(step=3, position_offset_m=[0.0, 1.0, 0.0]),
            Keyframe(step=4, position_offset_m=[0.0, 0.0, 0.0]),
        ],
    )

    position_before, velocity_before = offset_and_velocity(actor, 0.999, control_hz=1.0)
    position_at, velocity_at = offset_and_velocity(actor, 1.0, control_hz=1.0)

    np.testing.assert_allclose(position_at, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(position_before, position_at, atol=1e-3)
    np.testing.assert_allclose(velocity_before, velocity_at, atol=3e-3)


def test_motion_freezes_at_end_step():
    position, velocity = offset_and_velocity(_actor(loop=False, end_step=1), 1.5, 20.0)
    np.testing.assert_allclose(position, [0.05, 0.0, 0.0])
    np.testing.assert_array_equal(velocity, np.zeros(3))


def test_constant_speed_motion_can_traverse_a_loop_in_reverse():
    actor = _speed_actor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        loop=True,
        motion=Motion(speed_mps=0.2, initial_phase=0.4, initial_direction=-1),
    )

    position, velocity = offset_and_velocity(actor, 1.0, control_hz=20.0)

    np.testing.assert_allclose(position, [0.79, 0.0, 0.0])
    np.testing.assert_allclose(velocity, [-0.2, 0.0, 0.0])
