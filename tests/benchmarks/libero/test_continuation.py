from __future__ import annotations

import numpy as np
import pytest

from vlalagbench.benchmarks.libero import continuation
from vlalagbench.benchmarks.libero.continuation import _restore_dynamic_actor_origins, replay_prefix
from vlalagbench.benchmarks.libero.policy import REQUEST_KEY, RESPONSE_KEY
from vlalagbench.timing import ChunkScheduler, TimingConfig
from vlalagbench.simulators.mujoco.tracing import EpisodeTrace, TraceMeta
from tests.benchmarks.libero.robosuite_fakes import fake_robots


def test_connection_from_trace_preserves_backend_inference_settings():
    from vlalagbench.config import Gr00tModelSpec

    source, _, _ = _trace("synchronous")
    source.meta = source.meta.model_copy(update={"policy_identity": {"pytorch_compile_mode": "reduce-overhead"}})
    assert continuation.PolicyConnection.from_trace(source).compile_mode == "reduce-overhead"

    model = Gr00tModelSpec(backend="gr00t", checkpoint="nvidia/GR00T-N1.7-LIBERO",
                          revision="2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21",
                          checkpoint_subdir="libero_spatial")
    source.meta = source.meta.model_copy(update={"model": model.model_dump(), "action_chunk_steps": 16, "policy_seed": 19,
                                                "policy_identity": {"inference": "eager"}})
    connection = continuation.PolicyConnection.from_trace(source, port=8100)
    assert connection.compile_mode == "eager"
    assert connection.model == model
    assert connection.action_chunk_steps == 16
    assert connection.policy_seed == source.meta.policy_seed
    assert connection.port == 8100


@pytest.mark.parametrize("scheme", ["synchronous", "lazy", "continuous"])
def test_single_request_extension_preserves_prefix_and_continues_request_ids(tmp_path, monkeypatch, scheme):
    from vlalagbench.simulators.mujoco.tracing import TraceRecordingEnv

    source, _, _ = _trace(scheme)
    source.actions = np.pad(source.actions, ((0, 0), (0, 5)))
    source.chunks = np.pad(source.chunks, ((0, 0), (0, 0), (0, 5)))
    source.meta = source.meta.model_copy(update={"task_id": 1, "task_suite": "libero_object", "task_description": "move the object"})
    monkeypatch.setattr(continuation, "load_libero_trace", lambda _: source)
    monkeypatch.setattr(continuation, "_policy_element", lambda obs, *args: dict(obs))
    monkeypatch.setattr("vlalagbench.simulators.mujoco.tracing.collect_env_meta", lambda _: {"control_freq": 20.0})
    monkeypatch.setattr("vlalagbench.simulators.mujoco.tracing.collect_runtime_meta", lambda: {})
    requests = []

    class Policy:
        def infer(self, observation):
            identity = observation[REQUEST_KEY]
            requests.append(identity)
            return {"actions": np.pad(_chunk(identity["request_id"]), ((0, 0), (0, 5))), RESPONSE_KEY: {**identity, "noise_sha256": "a" * 64}}

    original_steps = int(source.acting.sum())
    spec = continuation.ContinuationSpec(tmp_path / "task01_ep003.npz", scheme, original_steps + 8)
    connection = continuation.PolicyConnection(policy_seed=19, action_chunk_steps=source.meta.action_chunk_steps)
    output = tmp_path / "extended.npz"
    row = continuation.continue_trace(spec, TraceRecordingEnv(_Env()), Policy(), connection, output)
    extended = EpisodeTrace.load(output)

    assert requests
    assert [r["request_id"] for r in requests] == list(range(len(source.chunks), len(source.chunks) + len(requests)))
    assert all(r["policy_seed"] == 19 and r["episode_id"] == 3 for r in requests)
    assert row["total_active_steps"] == original_steps + 8
    assert row["new_policy_requests"] == len(requests)
    for name, length in (("states", source.num_steps + 1), ("actions", source.num_steps), ("chunks", len(source.chunks))):
        np.testing.assert_array_equal(getattr(extended, name)[:length], getattr(source, name))
    assert extended.meta.deadline_extension["policy_implementation"] == "single"
    assert extended.meta.deadline_extension["policy_seed"] == 19


def test_continuation_rejects_a_response_for_another_request():
    identity = {"policy_seed": 7, "task_id": 1, "episode_id": 3, "request_id": 8}
    for response in ({}, {RESPONSE_KEY: {**identity, "request_id": 7, "noise_sha256": "a" * 64}}):
        with pytest.raises(RuntimeError, match="continuation request"):
            continuation._validate_response_identity(response, identity)


def test_continuation_worker_closes_each_trace_environment_after_success_or_failure(tmp_path, monkeypatch):
    source, _, _ = _trace("synchronous")
    environments = []

    class Env:
        closed = False

        def close(self):
            assert not self.closed
            self.closed = True

    def make_env(*args, **kwargs):
        assert all(env.closed for env in environments)
        assert kwargs["hard_reset"] is True
        env = Env()
        environments.append(env)
        return env

    def extend(spec, *args):
        if spec.trace.name == "task01_ep001.npz":
            raise RuntimeError("interrupted episode")
        return {"source_trace": str(spec.trace), "success": True, "error": None}

    monkeypatch.setattr(continuation, "load_libero_trace", lambda _: source)
    monkeypatch.setattr(continuation, "make_env_from_trace", make_env)
    monkeypatch.setattr(continuation, "TraceRecordingEnv", lambda env: env)
    monkeypatch.setattr(continuation, "continue_trace", extend)
    monkeypatch.setattr(continuation, "_WORKER_POLICY", object())
    monkeypatch.setattr(continuation, "_WORKER_CONNECTION", continuation.PolicyConnection())
    monkeypatch.setattr(continuation, "_WORKER_OUT", tmp_path)
    monkeypatch.setattr(continuation, "_WORKER_TAG", "test")
    (tmp_path / "progress").mkdir()
    specs = [continuation.ContinuationSpec(tmp_path / f"task01_ep{i:03d}.npz", "synchronous", 600) for i in range(3)]

    result = continuation._run_continuation_shard(specs)

    assert result == {"attempted": 3, "successes": 2, "errors": 1}
    assert len(environments) == 3
    assert all(env.closed for env in environments)


class _State:
    def __init__(self, value: float):
        self.value = value

    def flatten(self):
        return np.asarray([self.value], dtype=np.float64)


class _Sim:
    def __init__(self):
        self.value = 0.0

    def get_state(self):
        return _State(self.value)

    def set_state_from_flattened(self, state):
        self.value = float(state[0])

    def forward(self):
        pass


class _Env:
    control_freq = 20.0

    def __init__(self):
        self.sim = _Sim()
        self.robots = fake_robots()

    def reset(self):
        self.sim.value = 0.0

    def set_init_state(self, state):
        self.sim.value = float(state[0])
        return {"value": self.sim.value}

    def step(self, action):
        self.sim.value += float(np.sum(action))
        return {"value": self.sim.value}, 0.0, False, {}

    def set_dynamic_step(self, step):
        pass

    def record_step(self, **kwargs):
        pass

    def record_request(self, *args):
        pass


@pytest.mark.parametrize("scheme", ["synchronous", "lazy", "continuous"])
def test_replay_prefix_restores_scheduler_for_exact_next_step(scheme):
    trace, uninterrupted, next_step = _trace(scheme)
    restored = replay_prefix(_Env(), trace, scheme, state_atol=0.0)

    expected_request = uninterrupted.request(next_step)
    actual_request = restored.scheduler.request(next_step)
    assert actual_request == expected_request
    if expected_request is not None:
        chunk = _chunk(expected_request.id)
        uninterrupted.deliver(expected_request, chunk)
        restored.scheduler.deliver(actual_request, chunk)
    actual = restored.scheduler.action(next_step)
    expected = uninterrupted.action(next_step)
    assert (
        actual.active,
        actual.reason,
        actual.chunk_id,
        actual.index_in_chunk,
        actual.observation_step,
    ) == (
        expected.active,
        expected.reason,
        expected.chunk_id,
        expected.index_in_chunk,
        expected.observation_step,
    )
    np.testing.assert_array_equal(actual.action, expected.action)


def test_restore_dynamic_actor_origin_from_recorded_model_xml():
    trace, _, _ = _trace("synchronous")
    trace.meta = TraceMeta.model_validate(
        {**trace.meta.model_dump(mode="json"), "dynamics": [{"target": "moving_body"}]}
    )
    trace.model_xml = '<mujoco><worldbody><body name="moving_body" pos="1 2 3" quat="0 0 0 1"/></worldbody></mujoco>'
    env = _Env()
    env.sim.model = _Model()

    _restore_dynamic_actor_origins(env, trace)

    np.testing.assert_array_equal(env.sim.model.body_pos[0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(env.sim.model.body_quat[0], [0.0, 0.0, 0.0, 1.0])


def test_replay_prefix_audits_transient_done_without_aborting():
    trace, _, _ = _trace("synchronous")
    env = _Env()
    original_step = env.step
    calls = 0

    def step(action):
        nonlocal calls
        observation, reward, _, info = original_step(action)
        calls += 1
        return observation, reward, calls == 1, info

    env.step = step
    restored = replay_prefix(env, trace, "synchronous")

    assert restored.done_mismatches == 1


def test_replay_prefix_does_not_prepare_dynamic_step_zero_twice():
    class Env(_Env):
        def __init__(self):
            super().__init__()
            self.dynamic_steps = []

        def set_dynamic_step(self, step):
            self.dynamic_steps.append(step)

        def prepare_dynamic_step(self, step):
            assert step == 0
            self.sim.value += 0.5
            return True

    trace, _, _ = _trace("synchronous")
    trace.states[2:] += 0.5
    env = Env()

    replay_prefix(env, trace, "synchronous", state_atol=0.0)

    assert env.dynamic_steps[:3] == [None, None, 1]


def test_replay_prefix_restores_a_prepared_step_zero_without_warmup():
    class Env(_Env):
        def __init__(self):
            super().__init__()
            self.dynamic_steps = []
            self.restored = False

        def set_dynamic_step(self, step):
            self.dynamic_steps.append(step)

        def restore_prepared_dynamic_step(self, step):
            assert step == 0
            self.restored = True
            return True

    trace, _, _ = _trace("synchronous", wait_steps=0)
    trace.states += 0.5
    env = Env()

    replay_prefix(env, trace, "synchronous", state_atol=0.0)

    assert env.restored is True
    assert env.dynamic_steps[0] == 1


def test_replay_prefix_restores_a_prepared_step_zero_before_warmup():
    class Env(_Env):
        def __init__(self):
            super().__init__()
            self.restored = False
            self.held_times = []

        def restore_prepared_dynamic_step(self, step):
            assert step == 0
            self.restored = True
            return True

        def hold_dynamic_time(self, time_steps):
            self.held_times.append(time_steps)
            return True

    trace, _, _ = _trace("synchronous", wait_steps=2)
    trace.meta = trace.meta.model_copy(
        update={"dynamic_step_zero_prepared_before_warmup": True}
    )
    env = Env()

    replay_prefix(env, trace, "synchronous", state_atol=0.0)

    assert env.restored is True
    assert env.held_times == [0.0, 0.0]


class _Model:
    def __init__(self):
        self.body_pos = np.zeros((1, 3))
        self.body_quat = np.zeros((1, 4))

    def body_name2id(self, name):
        assert name == "moving_body"
        return 0


def _trace(scheme, wait_steps=2):
    config = TimingConfig(
        scheme=scheme,
        action_chunk_steps=6,
        execution_horizon_steps=3,
        latency_steps=2,
    )
    scheduler = ChunkScheduler(config)
    actions = [np.zeros(2) for _ in range(wait_steps)]
    states = [np.zeros(1) for _ in range(wait_steps + 1)]
    acting = [False] * wait_steps
    chunk_ids = [-1] * wait_steps
    action_indices = [-1] * wait_steps
    observation_steps = [-1] * wait_steps
    request_steps = []
    arrival_steps = []
    chunks = []
    value = 0.0
    control_steps = 17
    for step in range(control_steps):
        request = scheduler.request(step)
        if request is not None:
            chunk = _chunk(request.id)
            scheduler.deliver(request, chunk)
            request_steps.append(request.sent_step)
            arrival_steps.append(request.available_step)
            chunks.append(chunk)
        scheduled = scheduler.action(step)
        action = np.asarray(scheduled.action) if scheduled.active else np.zeros(2)
        value += float(action.sum())
        actions.append(action)
        states.append(np.asarray([value]))
        acting.append(scheduled.active)
        chunk_ids.append(-1 if scheduled.chunk_id is None else scheduled.chunk_id)
        action_indices.append(-1 if scheduled.index_in_chunk is None else scheduled.index_in_chunk)
        observation_steps.append(-1 if scheduled.observation_step is None else scheduled.observation_step)
    trace = EpisodeTrace(
        meta=TraceMeta.model_validate(
            {
                "schema_version": 3,
                "environment": {"kind": "generic"},
                "control_freq": 20.0,
                "num_steps": len(actions),
                "success": False,
                "num_steps_wait": wait_steps,
                "action_chunk_steps": config.action_chunk_steps,
                "execution_horizon_steps": config.execution_horizon_steps,
                "latency_steps": config.latency_steps,
            }
        ),
        states=np.asarray(states),
        actions=np.asarray(actions),
        rewards=np.zeros(len(actions)),
        dones=np.zeros(len(actions), dtype=bool),
        acting=np.asarray(acting),
        chunk_ids=np.asarray(chunk_ids),
        action_indices=np.asarray(action_indices),
        observation_steps=np.asarray(observation_steps),
        request_steps=np.asarray(request_steps),
        arrival_steps=np.asarray(arrival_steps),
        chunks=np.asarray(chunks),
    )
    return trace, scheduler, control_steps


def _chunk(request_id):
    return np.asarray([[request_id + index, -(request_id + index)] for index in range(6)])
