"""Owned state snapshots preserve observation/trace fields and buffer lifetime."""
import numpy as np
import torch

from vlalagbench.benchmarks.dynamicvla.runtime import (
    flatten_policy_observation, owned_cpu_policy_state, policy_observation_message,
)
from vlalagbench.benchmarks.dynamicvla.tracing import DynamicVLATraceRecorder


def test_owned_state_has_identical_values_keys_dtypes_and_no_source_aliases():
    source = {"end_effector": {"pos": torch.tensor([[1., 2., 3.]], dtype=torch.float64),
                               "quat": torch.tensor([[1., 0., 0., 0.]])},
              "other": object()}
    owned = owned_cpu_policy_state(source)
    assert owned.keys() == source.keys()
    assert owned["other"] is source["other"]
    assert owned["end_effector"].keys() == source["end_effector"].keys()
    for name, value in source["end_effector"].items():
        snapshot = owned["end_effector"][name]
        assert snapshot.device.type == "cpu" and snapshot.dtype == value.dtype
        assert snapshot.data_ptr() != value.data_ptr()
        torch.testing.assert_close(snapshot, value, rtol=0, atol=0)
    source["end_effector"]["pos"].fill_(9)
    assert owned["end_effector"]["pos"].tolist() == [[1., 2., 3.]]


def test_trace_ipc_and_path_share_snapshot_without_synthetic_key_leakage():
    cameras = {"opst_cam": {"rgb": np.zeros((1, 2, 2, 3), dtype=np.uint8)}}
    source = {"end_effector": {"pos": torch.tensor([[1., 2., 3.]]),
                               "quat": torch.tensor([[1., 0., 0., 0.]])}}
    baseline = policy_observation_message(cameras, source, index=5, seed=7, latency_steps=3)
    owned = owned_cpu_policy_state(source)
    message = policy_observation_message(cameras, owned, index=5, seed=7, latency_steps=3)
    assert message.keys() == baseline.keys()
    assert message["observation.images.opst_cam"] is cameras["opst_cam"]["rgb"]
    for key, value in baseline["observation.state"]["end_effector"].items():
        np.testing.assert_array_equal(message["observation.state"]["end_effector"][key], value)
    recorder = object.__new__(DynamicVLATraceRecorder)
    recorder._images = False
    recorder._observations = None
    recorder.record_observation(flatten_policy_observation(cameras, owned))
    assert set(recorder._observations) == {("state", "end_effector", "pos"),
                                          ("state", "end_effector", "quat")}
    path_position = owned["end_effector"]["pos"].cpu().numpy()
    source["end_effector"]["pos"].fill_(10)
    next_frame = owned_cpu_policy_state(source)
    np.testing.assert_array_equal(path_position, [[1., 2., 3.]])
    np.testing.assert_array_equal(message["observation.state"]["end_effector"]["pos"], path_position)
    assert next_frame["end_effector"]["pos"].tolist() == [[10., 10., 10.]]
    # Recorder ownership remains independent even if a consumer modifies its snapshot.
    owned["end_effector"]["pos"].fill_(20)
    np.testing.assert_array_equal(recorder._observations[("state", "end_effector", "pos")][0], [[1., 2., 3.]])


def test_existing_gripper_and_noncontiguous_state_are_preserved():
    source = {"end_effector": {"pos": torch.arange(12.).reshape(2, 6)[:, ::2],
                               "gripper": torch.tensor([[0.], [1.]])}}
    owned = owned_cpu_policy_state(source)
    message = policy_observation_message({}, owned, index=0, seed=0, latency_steps=0)
    np.testing.assert_array_equal(message["observation.state"]["end_effector"]["pos"],
                                  source["end_effector"]["pos"].numpy())
    np.testing.assert_array_equal(message["observation.state"]["end_effector"]["gripper"], [[0.], [1.]])
