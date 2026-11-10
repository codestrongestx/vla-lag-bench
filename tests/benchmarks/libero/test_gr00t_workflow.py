import numpy as np
import pytest

from tests.benchmarks.libero.test_gr00t import model, observation


@pytest.mark.parametrize("mode", ["synchronous", "lazy", "continuous"])
@pytest.mark.parametrize("latency", [0, 2, 4])
def test_gr00t_evaluation_continuation_and_trace_accounting(tmp_path, monkeypatch, mode, latency):
    from vlalagbench.benchmarks.libero import continuation, evaluator
    from vlalagbench.benchmarks.libero.gr00t_adapter import AXES, Gr00tRequestPolicy, format_observation
    from vlalagbench.benchmarks.libero.policy_server import METADATA_KEY
    from vlalagbench.simulators.mujoco.tracing import EpisodeTrace, TraceRecordingEnv
    from tests.benchmarks.libero.test_continuation import _Env

    monkeypatch.setattr("vlalagbench.simulators.mujoco.tracing.collect_env_meta", lambda _: {"control_freq": 20.0})
    monkeypatch.setattr("vlalagbench.simulators.mujoco.tracing.collect_runtime_meta", lambda: {})
    monkeypatch.setattr(evaluator, "_policy_element", lambda *args: format_observation(observation(), "move"))
    monkeypatch.setattr(continuation, "_policy_element", evaluator._policy_element)

    class PublicPolicy:
        def get_action(self, obs):
            return {key: np.random.uniform(0, .1, (1, 16, 1)) for key in (*AXES, "gripper")}, {}

    policy = Gr00tRequestPolicy(PublicPolicy())
    identity = {**model().model_dump(), "files_sha256": {"processor_config.json": "a" * 64}}
    policy.get_server_metadata = lambda: {METADATA_KEY: identity}
    env = TraceRecordingEnv(_Env())
    env.reset()
    config = evaluator.EvalConfig(model=model(), mode=mode, latency_steps=latency,
                                  action_chunk_steps=16, execution_horizon_steps=5,
                                  num_steps_wait=0, max_steps=13)
    result = evaluator.run_episode(env, "move", policy, config, obs={}, task_id=0, episode_id=0)
    assert result["error"] is None
    assert result["steps"] == 13
    assert result["control_steps"] == result["steps"] + result["wait_steps"]
    source = env.finish(extra_meta={"task_id": 0, "task_description": "move", "task_suite": "libero_spatial",
                                    "num_steps_wait": 0, "action_chunk_steps": 16, "execution_horizon_steps": 5,
                                    "latency_steps": latency, "model": model().model_dump(), "policy_identity": identity})
    path = tmp_path / "task00_ep000.npz"
    source.save(path)
    monkeypatch.setattr(continuation, "load_libero_trace", lambda _: EpisodeTrace.load(path))
    connection = continuation.PolicyConnection.from_trace(source)
    assert connection.model == model() and connection.action_chunk_steps == 16
    out = tmp_path / "extended.npz"
    row = continuation.continue_trace(continuation.ContinuationSpec(path, mode, 23),
                                     TraceRecordingEnv(_Env()), policy, connection, out)
    extended = EpisodeTrace.load(out)
    assert row["total_active_steps"] == 23
    np.testing.assert_array_equal(extended.actions[:source.num_steps], source.actions)
    np.testing.assert_array_equal(extended.states[:source.num_steps+1], source.states)
    assert extended.meta.request_audits[:len(source.chunks)] == source.meta.request_audits
    assert len(extended.meta.request_audits) == len(extended.chunks)
    assert extended.chunks.shape[1:] == (16, 7)
    assert all(audit["request_id"] == i for i, audit in enumerate(extended.meta.request_audits))
    for field in ("files_sha256", "backbone_processors_sha256"):
        policy.get_server_metadata = lambda: {METADATA_KEY: {**identity, field: {"processor_config.json": "b" * 64}}}
        with pytest.raises(ValueError, match="policy identity"):
            continuation.continue_trace(continuation.ContinuationSpec(path, mode, 23),
                                        TraceRecordingEnv(_Env()), policy, connection, out)
