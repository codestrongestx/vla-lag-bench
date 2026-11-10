from __future__ import annotations

import dataclasses
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vlalagbench.benchmarks.dynamicvla.tracing import (
    DynamicVLATrace,
    DynamicVLATraceRecorder,
    TRACE_SCHEMA_VERSION,
    _migrate_meta,
)
from vlalagbench.simulators.isaaclab.tracing import (
    capture_env_state,
    max_state_error,
    restore_env_state,
    step_with_pre_reset_capture,
)


class _WrenchAsset:
    device = "cpu"

    def __init__(self):
        self._external_force_b = torch.tensor([[[1.0, 2.0, 3.0]]])
        self._external_torque_b = torch.tensor([[[4.0, 5.0, 6.0]]])
        self._external_wrench_positions_b = torch.tensor([[[0.1, 0.2, 0.3]]])
        self.has_external_wrench = True
        self.uses_external_wrench_positions = True
        self._use_global_wrench_frame = False

    def set_external_force_and_torque(
        self, forces, torques, *, positions, is_global
    ):
        self.has_external_wrench = bool(forces.numel())
        self._external_force_b.zero_()
        self._external_torque_b.zero_()
        if self.has_external_wrench:
            self._external_force_b.copy_(forces)
            self._external_torque_b.copy_(torques)
        self.uses_external_wrench_positions = positions is not None
        self._external_wrench_positions_b.zero_()
        if positions is not None:
            self._external_wrench_positions_b.copy_(positions)
        self._use_global_wrench_frame = is_global


class _ArticulationWrenchAsset(_WrenchAsset):
    def __init__(self):
        super().__init__()
        self._external_force_b = torch.zeros((1, 2, 3))
        self._external_torque_b = torch.zeros((1, 2, 3))
        self._external_wrench_positions_b = torch.zeros((1, 2, 3))
        self.has_external_wrench = False
        self.uses_external_wrench_positions = False

    def set_external_force_and_torque(
        self, forces, torques, *, positions, is_global
    ):
        indices = torch.arange(2)
        self._external_force_b.flatten(0, 1)[indices] = forces.flatten(0, 1)
        self._external_torque_b.flatten(0, 1)[indices] = torques.flatten(0, 1)
        self.has_external_wrench = bool(forces.any() or torques.any())
        self.uses_external_wrench_positions = positions is not None
        self._use_global_wrench_frame = is_global


class _Scene:
    device = "cpu"

    def __init__(self):
        self.root_pose = torch.arange(7, dtype=torch.float32).reshape(1, 7)
        self.joint_position = torch.arange(3, dtype=torch.float32).reshape(1, 3)
        self.articulations = {"robot": _ArticulationWrenchAsset()}
        self.rigid_objects = {"object": _WrenchAsset()}
        self.writes = 0
        self.updates = []

    def get_state(self, *, is_relative):
        assert is_relative is False
        return {
            "articulation": {
                "robot": {
                    "root_pose": self.root_pose,
                    "joint_position": self.joint_position,
                }
            }
        }

    def reset_to(self, state, *, is_relative):
        assert is_relative is False
        robot = state["articulation"]["robot"]
        self.root_pose.copy_(robot["root_pose"])
        self.joint_position.copy_(robot["joint_position"])

    def write_data_to_sim(self):
        self.writes += 1

    def update(self, dt):
        self.updates.append(dt)


def _env():
    scene = _Scene()
    action_manager = SimpleNamespace(
        action=torch.zeros((1, 8)),
        prev_action=torch.full((1, 8), -1.0),
    )
    sim = SimpleNamespace(forward=lambda: setattr(sim, "forwards", sim.forwards + 1))
    sim.forwards = 0
    return SimpleNamespace(
        scene=scene,
        sim=sim,
        episode_length_buf=torch.tensor([4]),
        common_step_counter=9,
        _sim_step_counter=36,
        action_manager=action_manager,
        event_manager=SimpleNamespace(
            _interval_term_time_left=[torch.tensor([0.25])]
        ),
    )


def test_isaaclab_state_round_trip_restores_scene_and_control_buffers():
    env = _env()
    state = capture_env_state(env)
    env.scene.root_pose.add_(100)
    env.scene.joint_position.mul_(0)
    env.episode_length_buf.fill_(99)
    env.common_step_counter = 77
    env._sim_step_counter = 308
    env.action_manager.action.fill_(3)
    env.event_manager._interval_term_time_left[0].fill_(0.9)
    env.scene.rigid_objects["object"].set_external_force_and_torque(
        torch.zeros((0, 3)),
        torch.zeros((0, 3)),
        positions=None,
        is_global=True,
    )

    restore_env_state(env, state)

    np.testing.assert_array_equal(env.scene.root_pose, np.arange(7).reshape(1, 7))
    np.testing.assert_array_equal(env.scene.joint_position, np.arange(3).reshape(1, 3))
    assert env.episode_length_buf.item() == 4
    assert env.common_step_counter == 9
    assert env._sim_step_counter == 36
    assert env.action_manager.action.eq(0).all()
    torch.testing.assert_close(
        env.event_manager._interval_term_time_left[0], torch.tensor([0.25])
    )
    wrench = env.scene.rigid_objects["object"]
    torch.testing.assert_close(
        wrench._external_force_b, torch.tensor([[[1.0, 2.0, 3.0]]])
    )
    torch.testing.assert_close(
        wrench._external_torque_b, torch.tensor([[[4.0, 5.0, 6.0]]])
    )
    assert wrench.has_external_wrench
    assert wrench.uses_external_wrench_positions
    assert wrench._use_global_wrench_frame is False
    assert env.scene.writes == 1
    assert env.sim.forwards == 1
    assert env.scene.updates == [0.0]


def test_isaaclab_state_round_trip_restores_random_generators():
    random.seed(10)
    np.random.seed(11)
    torch.manual_seed(12)
    state = capture_env_state(_env())
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.random()
    np.random.rand()
    torch.rand(1)

    env = _env()
    restore_env_state(env, state)

    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(1), expected[2])


def test_state_error_treats_matching_rng_sentinels_as_equal():
    path = ("runtime", "rng", "python_gauss")

    assert max_state_error({path: np.asarray([np.nan])}, {path: np.asarray([np.nan])}) == (
        0.0,
        path,
    )
    assert max_state_error({path: np.asarray([0.0])}, {path: np.asarray([np.nan])}) == (
        float("inf"),
        path,
    )


def test_state_error_rejects_shape_broadcasting():
    path = ("scene", "rigid_object", "object", "root_pose")

    with pytest.raises(ValueError, match="shapes differ"):
        max_state_error(
            {path: np.zeros((1, 7), dtype=np.float32)},
            {path: np.zeros((7,), dtype=np.float32)},
        )


def test_step_capture_runs_before_terminal_auto_reset_and_restores_hook():
    recorded = []
    manager = SimpleNamespace(record_pre_reset=lambda env_ids: recorded.append(env_ids))
    original_hook = manager.record_pre_reset

    class AutoResetEnv:
        def __init__(self):
            self.unwrapped = self
            self.recorder_manager = manager
            self.value = 0

        def step(self, _action):
            self.value = 10
            self.recorder_manager.record_pre_reset([0])
            self.value = 0

    env = AutoResetEnv()
    captured = []

    was_terminal = step_with_pre_reset_capture(
        env, object(), lambda: captured.append(env.value)
    )

    assert was_terminal is True
    assert captured == [10]
    assert recorded == [[0]]
    assert manager.record_pre_reset is original_hook


@pytest.mark.parametrize("images", [False, True])
def test_dynamicvla_trace_round_trip_uses_tensor_manifest_without_pickle(
    tmp_path, images
):
    env = _env()
    recorder = DynamicVLATraceRecorder(env, images=images)
    recorder.record_observation(
        {
            ("images", "front"): np.zeros((1, 2, 2, 3)),
            ("state", "end_effector", "pose"): np.zeros((1, 7)),
        }
    )
    env.scene.root_pose.add_(1)
    env.episode_length_buf.add_(1)
    recorder.record_observation(
        {
            ("images", "front"): np.ones((1, 2, 2, 3)),
            ("state", "end_effector", "pose"): np.ones((1, 7)),
        }
    )
    recorder.record_step(
        torch.ones((1, 8)),
        success=True,
        done=True,
        terminations={"success": True, "time_out": False},
        protocol={"policy": {"queue": [{"action": np.ones(8)}]}},
    )
    trace = recorder.finish(
        {
            "task_id": 2,
            "task_name": "example-task",
            "episode": 3,
            "environment_index": 1,
            "replica_id": 1,
            "environment_path": "/assets/example.json",
            "environment_sha256": "a" * 64,
            "environment": {"seed": 7, "instruction": "pick"},
            "asset_sha256": {"scene.usd": "b" * 64},
            "instruction": "pick the object",
            "seed": 7,
            "upstream_revision": "c" * 40,
            "checkpoint": "/checkpoint",
            "checkpoint_sha256": "d" * 64,
            "checkpoint_config_sha256": "e" * 64,
            "policy_noise_scheme": "sha256-seeded-cpu-torch-randn-v1",
            "policy_torch_version": "2.4.0",
            "policy_seed": 0,
            "policy_episode_seed": 41,
            "episode_length_s": 8.0,
            "control_hz": 25.0,
            "physics_time_step": 0.04,
            "termination_tolerance": 0.07,
            "disable_fabric": False,
            "path_tracing": False,
            "simulator_versions": {"isaaclab": "2.2.1", "isaacsim": "4.5.0"},
            "inference_mode": "continuous",
            "action_index_origin": "observation",
            "latency_steps": 6,
            "action_chunk_steps": 20,
            "execution_horizon_steps": 14,
            "success_term": "success",
            "timeout_terms": ["time_out"],
            "status": 0,
            "success": True,
        },
        final_policy={"pending_chunk": {"index": 0, "actions": np.zeros((20, 8))}},
    )

    path = trace.save(tmp_path / "episode.npz")
    loaded = DynamicVLATrace.load(path)

    assert loaded.meta.task_id == 2
    assert loaded.actions.shape == (1, 1, 8)
    assert loaded.num_steps == 1
    assert loaded.protocol[0]["policy"]["queue"][0]["action"] == [1.0] * 8
    assert loaded.final_policy["pending_chunk"]["index"] == 0
    assert loaded.meta.schema_version == TRACE_SCHEMA_VERSION
    assert loaded.has_images is images
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["observation_manifest"][()]))
    assert [item["path"] for item in manifest] == (
        [["images", "front"]] if images else []
    ) + [["state", "end_effector", "pose"]]
    if images:
        np.testing.assert_array_equal(
            loaded.observation(1)[("images", "front")], np.ones((1, 2, 2, 3))
        )
    else:
        assert ("images", "front") not in loaded.observations
    np.testing.assert_array_equal(
        loaded.observation(1)[("state", "end_effector", "pose")], np.ones((1, 7))
    )
    np.testing.assert_array_equal(
        loaded.state(1)[("scene", "articulation", "robot", "root_pose")],
        np.arange(7).reshape(1, 7) + 1,
    )

    legacy = dataclasses.replace(
        loaded, meta=loaded.meta.model_copy(update={"schema_version": 8})
    )
    legacy.save(tmp_path / "legacy.npz")
    assert DynamicVLATrace.load(tmp_path / "legacy.npz").has_images is images


def test_dynamicvla_trace_recorder_serializes_a_zero_step_policy_failure():
    recorder = DynamicVLATraceRecorder(_env())
    recorder.record_observation(
        {
            ("images", "front"): np.zeros((1, 2, 2, 3)),
            ("state", "end_effector", "pose"): np.zeros((1, 7)),
        }
    )

    trace = recorder.finish(
        {
            "task_id": 0,
            "task_name": "example-task",
            "episode": 0,
            "environment_index": 0,
            "replica_id": 0,
            "environment_path": "/assets/example.json",
            "environment_sha256": "a" * 64,
            "environment": {"seed": 7},
            "asset_sha256": {},
            "instruction": "pick",
            "seed": 7,
            "upstream_revision": "b" * 40,
            "checkpoint": "/checkpoint",
            "checkpoint_sha256": "c" * 64,
            "checkpoint_config_sha256": "d" * 64,
            "policy_noise_scheme": (
                "sha256-policy-task-environment-replica-request-cpu-torch-v2"
            ),
            "policy_torch_version": "2.4.0",
            "policy_seed": 11,
            "policy_episode_seed": 41,
            "episode_length_s": 8.0,
            "control_hz": 25.0,
            "physics_time_step": 0.04,
            "termination_tolerance": 0.07,
            "disable_fabric": False,
            "path_tracing": False,
            "simulator_versions": {"isaaclab": "2.2.1", "isaacsim": "4.5.0"},
            "inference_mode": "synchronous",
            "action_index_origin": "release",
            "latency_steps": 2,
            "action_chunk_steps": 20,
            "execution_horizon_steps": 4,
            "success_term": "success",
            "timeout_terms": ["time_out"],
            "status": 3,
            "success": False,
        },
        final_policy={},
    )

    assert trace.num_steps == 0
    assert trace.actions.shape == (0, 1, 8)
    assert not trace.has_images
    assert trace.observations[("state", "end_effector", "pose")].shape[0] == 1
    assert trace.protocol == []


def test_dynamicvla_v1_trace_identity_is_migrated_explicitly():
    migrated = _migrate_meta(
        {
            "schema_version": 1,
            "task_id": 3,
            "episode": 4,
            "environment_path": "/assets/legacy-example.json",
            "seed": 7,
            "instruction": "legacy",
            "latency_steps": 1,
            "action_chunk_steps": 20,
        }
    )

    assert migrated == {
        "schema_version": TRACE_SCHEMA_VERSION,
        "task_id": 3,
        "task_name": "legacy-example",
        "episode": 4,
        "environment_index": 3,
        "replica_id": 4,
        "environment_path": "/assets/legacy-example.json",
        "seed": 7,
        "instruction": "legacy",
        "latency_steps": 1,
        "action_chunk_steps": 20,
        "policy_episode_seed": 8185886803589303160,
        "policy_seed": 0,
        "inference_mode": "continuous",
            "action_index_origin": "observation",
        "execution_horizon_steps": 19,
    }


def test_dynamicvla_trace_rejects_an_unbounded_schema_version():
    with pytest.raises(ValueError, match="schema version must be an integer"):
        _migrate_meta({"schema_version": float("inf")})


def test_trace_recorder_keeps_terminal_state_captured_before_autoreset():
    env = _env()
    recorder = DynamicVLATraceRecorder(env)
    recorder.record_observation(
        {
            ("images", "front"): np.zeros((1, 1, 1, 3)),
            ("state", "end_effector", "pose"): np.zeros((1, 7)),
        }
    )
    env.scene.root_pose.add_(10)
    recorder.capture_post_step()
    recorder.record_observation(
        {
            ("images", "front"): np.ones((1, 1, 1, 3)),
            ("state", "end_effector", "pose"): np.ones((1, 7)),
        }
    )
    env.scene.root_pose.zero_()

    recorder.record_step(
        torch.ones((1, 8)),
        success=True,
        done=True,
        terminations={"success": True, "time_out": False},
        protocol={},
    )
    trace = recorder.finish(
        {
            "task_id": 0,
            "task_name": "example-task",
            "episode": 0,
            "environment_index": 0,
            "replica_id": 0,
            "environment_path": "/assets/example.json",
            "environment_sha256": "a" * 64,
            "environment": {"seed": 7},
            "asset_sha256": {},
            "instruction": "pick",
            "seed": 7,
            "upstream_revision": "b" * 40,
            "checkpoint": "/checkpoint",
            "checkpoint_sha256": "c" * 64,
            "checkpoint_config_sha256": "d" * 64,
            "policy_noise_scheme": "sha256-seeded-cpu-torch-randn-v1",
            "policy_torch_version": "2.4.0",
            "policy_seed": 0,
            "policy_episode_seed": 41,
            "episode_length_s": 8.0,
            "control_hz": 25.0,
            "physics_time_step": 0.04,
            "termination_tolerance": 0.07,
            "disable_fabric": False,
            "path_tracing": False,
            "simulator_versions": {"isaaclab": "2.2.1", "isaacsim": "4.5.0"},
            "inference_mode": "continuous",
            "action_index_origin": "observation",
            "latency_steps": 6,
            "action_chunk_steps": 20,
            "execution_horizon_steps": 14,
            "success_term": "success",
            "timeout_terms": ["time_out"],
            "status": 0,
            "success": True,
        },
        final_policy={},
    )

    np.testing.assert_array_equal(
        trace.state(1)[("scene", "articulation", "robot", "root_pose")],
        np.arange(7).reshape(1, 7) + 10,
    )
