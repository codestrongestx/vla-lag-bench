import math
import types

import numpy as np
import pytest

from vlalagbench.config import DynamicActor, Keyframe, Motion
from vlalagbench.simulators.mujoco.dynamics import DynamicEnv


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


def test_dynamic_env_moves_only_xy_from_post_warmup_pose_every_two_milliseconds():
    raw = _RawEnv()
    env = DynamicEnv(
        raw,
        [
            _actor(
                loop=False,
                keyframes=[
                    Keyframe(step=0, position_offset_m=[0.0, 0.0, 0.0]),
                    Keyframe(step=1, position_offset_m=[0.05, 0.02, 0.4]),
                    Keyframe(step=2, position_offset_m=[0.0, 0.0, 0.0]),
                ],
            )
        ],
    )
    env.reset()
    env.set_init_state(None)
    raw.env.sim._sim.data.qpos = np.asarray(
        [0.2, -0.1, 0.9025, 0.9239, 0.0, 0.3827, 0.0]
    )
    raw.env.sim._sim.data.qvel = np.asarray([0.01, 0.02, -0.03, 0.4, 0.5, 0.6])
    env.set_dynamic_step(0)
    env.step([0.0] * 7)

    assert raw.env.sim is env._sim
    assert raw.env.sim._sim.step_count == 25
    assert raw.env.sim._sim.data.qpos_writes == 50
    assert raw.env.sim._sim.data.qvel_writes == 50
    np.testing.assert_allclose(
        raw.env.sim._sim.data.qpos,
        [0.25, -0.08, 0.9025, 0.9239, 0.0, 0.3827, 0.0],
    )
    np.testing.assert_allclose(
        raw.env.sim._sim.data.qvel,
        [-1.0, -0.4, -0.03, 0.4, 0.5, 0.6],
    )

    recorded_pose = np.arange(7, dtype=np.float64)
    raw.env.sim._sim.data.qpos = recorded_pose.copy()
    env.apply_dynamic_time(1)
    np.testing.assert_array_equal(raw.env.sim._sim.data.qpos, recorded_pose)
    env.apply_dynamic_time(1, free_joints=True)
    np.testing.assert_allclose(raw.env.sim._sim.data.qpos[:2], [0.25, -0.08])

    env.close()
    assert raw.env.sim is raw.sim


def test_dynamic_env_applies_initial_object_positions_before_attaching_motion():
    raw = _RawEnv()
    env = DynamicEnv(raw, [_actor()])
    env.reset()
    env.set_initial_object_positions({"plate": [0.2, -0.1, 0.9]})
    env.set_initial_object_yaws({"plate": math.pi / 2})

    env.set_init_state(None)

    np.testing.assert_allclose(raw.sim.data.qpos[:3], [0.2, -0.1, 0.9])
    np.testing.assert_allclose(
        raw.sim.data.qpos[3:], [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
    )
    np.testing.assert_allclose(env._sim._drivers[0].initial_position, [0.2, -0.1, 0.9])


@pytest.mark.parametrize("required", [1, 2, 3])
def test_dynamic_env_releases_a_stably_grasped_free_joint(required):
    raw = _RawEnv()
    env = DynamicEnv(
        raw,
        [
            _speed_actor(
                [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]],
                release_on_grasp_steps=required,
            )
        ],
    )
    env.reset()
    env.set_init_state(None)
    raw.grasping = True
    env.step([0.0] * 7)
    assert raw.env.sim._drivers[0].grasp_steps == 0

    contacts = [True] * (required - 1) + [False] + [True] * required
    for step, grasping in enumerate(contacts):
        raw.grasping = grasping
        env.set_dynamic_step(step)
        env.step([0.0] * 7)
        assert env._sim._drivers[0].released == (step == len(contacts) - 1)

    driver = raw.env.sim._drivers[0]
    assert driver.grasp_steps == required
    assert driver.released is True

    writes = raw.env.sim._sim.data.qpos_writes
    raw.env.sim._sim.data.qpos[:3] = [0.3, 0.4, 0.9]
    raw.grasping = False
    env.set_dynamic_step(len(contacts))
    env.step([0.0] * 7)
    np.testing.assert_allclose(raw.env.sim._sim.data.qpos[:3], [0.3, 0.4, 0.9])
    assert raw.env.sim._sim.data.qpos_writes == writes

    env.reset()
    env.set_init_state(None)
    assert env._sim._drivers[0].grasp_steps == 0
    assert env._sim._drivers[0].released is False


@pytest.mark.parametrize("required", [None, 1, 2, 3])
def test_driven_target_preserves_vertical_and_rotational_physics(required):
    raw = _RawEnv()
    env = DynamicEnv(
        raw,
        [
            _speed_actor(
                [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]],
                release_on_grasp_steps=required,
            )
        ],
    )
    env.reset()
    env.set_init_state(None)
    sim = raw.env.sim._sim
    sim.data.qpos = np.asarray([0.2, -0.1, 0.073, 1.0, 0.0, 0.0, 0.0])
    env.set_dynamic_step(0)

    def physics_step():
        sim.step_count += 1
        sim.data.qpos[2] += 0.001
        angle = sim.step_count * 0.01
        sim.data.qpos[3:] = [np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0]
        sim.data.qvel[2:] = [0.5, 5.0, 0.0, 0.0]

    sim.step = physics_step
    env.step([0.0] * 7)

    assert not env._sim._drivers[0].released
    np.testing.assert_allclose(sim.data.qpos[:3], [0.21, -0.1, 0.098])
    np.testing.assert_allclose(sim.data.qpos[3:], [np.cos(0.125), np.sin(0.125), 0.0, 0.0])
    np.testing.assert_allclose(sim.data.qvel, [0.2, 0.0, 0.5, 5.0, 0.0, 0.0])


def test_grasp_target_can_settle_vertically_during_phase_hold():
    raw = _RawEnv()
    env = DynamicEnv(raw, [_speed_actor(
        [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], release_on_grasp_steps=3,
    )])
    env.reset()
    env.set_init_state(None)
    env.prepare_dynamic_step(0)
    settled = np.asarray([0.073, 0.9239, 0.0, 0.0, 0.3827])
    raw.env.sim._sim.data.qpos[2:] = settled
    raw.env.sim._sim.data.qvel[2:] = [0.1, 0.2, 0.3, 0.4]

    env.hold_dynamic_time(0)

    np.testing.assert_allclose(raw.env.sim._sim.data.qpos[2:], settled)
    np.testing.assert_allclose(raw.env.sim._sim.data.qvel[2:], [0.1, 0.2, 0.3, 0.4])
    env.prepare_dynamic_step(0)
    raw.env.sim._sim.data.qpos[2] = 0.01
    env.apply_dynamic_time(1, free_joints=True)
    np.testing.assert_allclose(raw.env.sim._sim.data.qpos[2:], [0.01, *settled[1:]])
    np.testing.assert_allclose(raw.env.sim._sim.data.qvel[2:], [0.1, 0.2, 0.3, 0.4])


def test_dynamic_env_rejects_grasp_release_for_a_jointless_fixture():
    raw = _RawEnv(static=True)
    env = DynamicEnv(
        raw,
        [
            _speed_actor(
                [[0.0, 0.0, 0.0], [0.05, 0.04, 0.0]],
                target="caddy",
                release_on_grasp_steps=3,
            )
        ],
    )
    env.reset()

    with pytest.raises(ValueError, match="requires a free-joint target"):
        env.set_init_state(None)


def test_dynamic_env_restores_an_already_applied_initial_phase():
    raw = _RawEnv()
    env = DynamicEnv(
        raw,
        [
            _speed_actor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                motion=Motion(speed_mps=0.2, initial_phase=0.4),
            )
        ],
    )
    env.reset()
    env.set_init_state(None)
    raw.env.sim._sim.data.qpos[0] = 0.4

    assert env.restore_prepared_dynamic_step(0) is True
    env.step([0.0] * 7)

    np.testing.assert_allclose(raw.env.sim._sim.data.qpos[0], 0.41)


def test_repreparing_step_zero_preserves_the_settled_phase_position():
    raw = _RawEnv()
    env = DynamicEnv(
        raw,
        [
            _speed_actor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                motion=Motion(speed_mps=0.2, initial_phase=0.4),
            )
        ],
    )
    env.reset()
    env.set_init_state(None)

    env.prepare_dynamic_step(0)
    np.testing.assert_allclose(raw.env.sim._sim.data.qpos[0], 0.4)
    env.set_dynamic_step(None)
    raw.env.sim._sim.data.qpos[0] = 0.405

    env.prepare_dynamic_step(0)

    np.testing.assert_allclose(raw.env.sim._sim.data.qpos[0], 0.405)
    np.testing.assert_allclose(env._sim._drivers[0].initial_position[0], 0.005)


def test_dynamic_hold_keeps_the_initial_phase_still_during_warmup():
    raw = _RawEnv()
    env = DynamicEnv(
        raw,
        [
            _speed_actor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                motion=Motion(speed_mps=0.2, initial_phase=0.4),
            )
        ],
    )
    env.reset()
    env.set_init_state(None)
    env.prepare_dynamic_step(0)
    env.hold_dynamic_time(0)
    raw.env.sim._sim.data.qpos[0] = 0.9

    env.step([0.0] * 7)

    np.testing.assert_allclose(raw.env.sim._sim.data.qpos[0], 0.4)
    np.testing.assert_allclose(raw.env.sim._sim.data.qvel[0], 0.0)


def test_dynamic_env_moves_a_jointless_fixture_body_from_initial_pose():
    raw = _RawEnv(static=True)
    env = DynamicEnv(
        raw,
        [
            _speed_actor(
                [[0.0, 0.0, 0.0], [0.05, 0.04, 0.0]],
                target="caddy",
                speed_mps=np.hypot(0.05, 0.04) * 20.0,
            )
        ],
    )
    env.reset()
    env.set_init_state(None)
    assert env._sim._drivers[0].motion_path is not None
    raw.env.sim._sim.model.body_pos[0] = [0.4, 0.5, 0.6]
    env.set_dynamic_step(0)
    env.step([0.0] * 7)

    np.testing.assert_allclose(raw.env.sim._sim.model.body_pos[0], [0.45, 0.54, 0.6])

    raw.env.sim._sim.model.body_pos[0, :2] = 0.0
    env.apply_dynamic_time(1)
    np.testing.assert_allclose(raw.env.sim._sim.model.body_pos[0], [0.45, 0.54, 0.6])


class _Data:
    def __init__(self):
        self.qpos = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.qvel = np.zeros(6)
        self.qpos_writes = 0
        self.qvel_writes = 0

    def get_joint_qpos(self, _joint):
        return self.qpos

    def get_joint_qvel(self, _joint):
        return self.qvel

    def set_joint_qpos(self, _joint, value):
        self.qpos = np.asarray(value)
        self.qpos_writes += 1

    def set_joint_qvel(self, _joint, value):
        self.qvel = np.asarray(value)
        self.qvel_writes += 1


class _Sim:
    def __init__(self, *, static=False):
        self.data = _Data()
        expected = "caddy" if static else "plate"
        self.model = types.SimpleNamespace(
            body_name2id=lambda name: 0 if name == expected else (_ for _ in ()).throw(KeyError(name)),
            body_jntnum=[0 if static else 1],
            body_jntadr=[0],
            body_pos=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
            joint_id2name=lambda joint_id: "plate_joint" if joint_id == 0 else None,
        )
        self.step_count = 0

    def step(self):
        self.step_count += 1

    def forward(self):
        pass


class _RawEnv:
    def __init__(self, *, static=False):
        self.sim = _Sim(static=static)
        self.grasping = False
        grasp_object = types.SimpleNamespace(root_body="plate")
        self.env = types.SimpleNamespace(
            sim=self.sim,
            control_timestep=0.05,
            model_timestep=0.002,
            objects_dict={"plate": grasp_object},
            robots=[types.SimpleNamespace(gripper=object())],
            _check_grasp=lambda _gripper, obj: obj is grasp_object and self.grasping,
        )

    def reset(self):
        return {}

    def set_init_state(self, _state):
        return {}

    def get_sim_state(self):
        return self.sim.data.qpos.copy()

    def regenerate_obs_from_state(self, state):
        return {"state": state}

    def step(self, _action):
        for _ in range(25):
            self.env.sim.step()
        return {}, 0.0, False, {}

    def close(self):
        pass
