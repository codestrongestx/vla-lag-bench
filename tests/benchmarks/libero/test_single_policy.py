import numpy as np

from vlalagbench.benchmarks.libero.policy import REQUEST_KEY, RESPONSE_KEY, SingleRequestPolicy, deterministic_noise
from vlalagbench.benchmarks.libero.policy_server import METADATA_KEY


def test_wire_keys_remain_compatible_with_recorded_request_identities():
    assert METADATA_KEY == "mujoco_dynamic"
    assert REQUEST_KEY == "_mujoco_dynamic_request"
    assert RESPONSE_KEY == "mujoco_dynamic_request"


def test_noise_depends_on_request_identity_not_arrival_order():
    first = {"policy_seed": 7, "task_id": 3, "episode_id": 2, "request_id": 4}
    second = {**first, "request_id": 5}
    a = deterministic_noise(first, action_horizon=10, action_dim=32)
    b = deterministic_noise(second, action_horizon=10, action_dim=32)
    again = deterministic_noise(first, action_horizon=10, action_dim=32)
    assert a.shape == (10, 32)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, again)
    assert not np.array_equal(a, b)


def test_single_request_uses_public_api_and_keeps_episode_noise_independent():
    class Policy:
        def infer(self, observation, *, noise):
            assert REQUEST_KEY not in observation
            assert noise.shape == (10, 32)
            return {"actions": noise[:, :7], "state": observation["state"]}

    policy = SingleRequestPolicy(Policy(), action_horizon=10, action_dim=32)
    first = {"state": [1, 2], REQUEST_KEY: {"policy_seed": 7, "task_id": 2, "episode_id": 4, "request_id": 0}}
    other = {**first, REQUEST_KEY: {**first[REQUEST_KEY], "episode_id": 5}}
    a = policy.infer(first)
    b = policy.infer(other)
    again = policy.infer(first)
    np.testing.assert_array_equal(a["actions"], again["actions"])
    assert not np.array_equal(a["actions"], b["actions"])
    assert a[RESPONSE_KEY]["episode_id"] == 4
    assert REQUEST_KEY in first
