import hashlib

import numpy as np
import pytest

from vlalagbench.benchmarks.libero import _vendor, evaluator
from vlalagbench.simulators.mujoco.camera_gate import CameraGate


class _ImagePolicy:
    """Actions hash the exact request images, so any pixel change diverges."""

    def __init__(self, chunk):
        self.chunk = chunk
        self.digests = []

    def infer(self, element):
        digest = hashlib.sha256(
            element["observation/image"].tobytes() + element["observation/wrist_image"].tobytes()
        ).digest()
        self.digests.append(digest)
        actions = np.random.default_rng(int.from_bytes(digest[:8], "little")).uniform(-0.5, 0.5, (self.chunk, 7))
        actions[:, -1] = -1.0
        return {"actions": actions}


def _rollout(gate, mode, latency):
    libero = _vendor.libero()
    suite = libero.benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    config = evaluator.EvalConfig(
        policy="dummy", mode=mode, latency_steps=latency, action_chunk_steps=4,
        execution_horizon_steps=2, num_steps_wait=2, max_steps=12, env_resolution=64,
        render_policy_observations_only=gate,
    )
    env = evaluator._recording_env(task, config, 0)
    try:
        obs = evaluator._initialize_episode(env, suite.get_task_init_states(0), 0)
        policy = _ImagePolicy(config.action_chunk_steps)
        result = evaluator.run_episode(env, task.language, policy, config, obs=obs)
        return result, policy.digests, env.finish().states
    finally:
        env.close()


@pytest.mark.parametrize(("mode", "latency"), [("synchronous", 0), ("synchronous", 3), ("lazy", 1), ("continuous", 2)])
def test_gated_rollout_matches_ungated_bit_for_bit(mode, latency):
    ungated, gated = _rollout(False, mode, latency), _rollout(True, mode, latency)
    assert gated[0] == ungated[0] and gated[0]["error"] is None
    assert gated[1] == ungated[1]
    np.testing.assert_array_equal(gated[2], ungated[2])


def test_closed_gate_refuses_unrendered_policy_images():
    libero = _vendor.libero()
    suite = libero.benchmark.get_benchmark_dict()["libero_spatial"]()
    config = evaluator.EvalConfig(max_steps=1, env_resolution=64)
    env = evaluator._recording_env(suite.get_task(0), config, 0)
    try:
        evaluator._initialize_episode(env, suite.get_task_init_states(0), 0)
        gate = CameraGate(env)
        gate.open = False
        env.step([0.0] * 6 + [-1.0])
        with pytest.raises(RuntimeError, match="unrendered"):
            gate.require_fresh()
        gate.open = True
        env.step([0.0] * 6 + [-1.0])
        gate.require_fresh()
    finally:
        env.close()


def test_gate_keeps_observable_modality():
    sensor = lambda cache: np.zeros((2, 2, 3))
    sensor.__modality__ = "image"
    observable = type("Observable", (), {"modality": "image", "_sensor": sensor, "_current_observed_value": None})()
    host = type("Host", (), {"sim": None, "control_timestep": 0.05, "model_timestep": 0.002,
                             "_observables": {"agentview_image": observable}})()
    CameraGate(host)
    assert observable._sensor.__modality__ == "image"
