import types

import numpy as np

from vlalagbench.simulators.mujoco.tracing import EpisodeTrace, TraceRecordingEnv


class _FakeState:
    def __init__(self, vec):
        self._vec = vec

    def flatten(self):
        return np.asarray(self._vec)


class _FakeSim:
    def __init__(self):
        self.state = np.zeros(5)  # [time, qpos(2), qvel(2)]
        self.data = types.SimpleNamespace(qpos=np.zeros(2), qvel=np.zeros(2))
        self.model = types.SimpleNamespace(get_xml=lambda: "<mujoco><fake/></mujoco>")

    def get_state(self):
        return _FakeState(self.state.copy())


class _FakeEnv:
    def __init__(self):
        self.sim = _FakeSim()
        self.control_freq = 20

    def reset(self):
        self.sim.state[:] = 0.0
        return {}

    def set_init_state(self, state):
        self.sim.state[:] = state
        return {}

    def step(self, action):
        self.sim.state[0] += 0.05
        self.sim.state[1:3] += np.asarray(action)[:2]
        return {}, 0.5, False, {}


def test_trace_records_states_and_actions(tmp_path):
    env = TraceRecordingEnv(_FakeEnv(), meta={"task_description": "fake task"})
    env.reset()
    env.set_init_state(np.arange(5.0))
    for i in range(3):
        env.record_step(
            acting=True,
            chunk_id=0,
            action_index=i,
            observation_step=0,
        )
        env.step([0.1 * i, 1.0])
    env.record_request(0, 0, 2, np.ones((3, 2)))
    trace = env.finish(extra_meta={"init_state_index": 4})

    assert trace.num_steps == 3
    assert trace.states.shape == (4, 5)
    assert trace.actions.shape == (3, 2)
    np.testing.assert_array_equal(trace.states[0], np.arange(5.0))
    np.testing.assert_allclose(trace.states[1][1:3], np.arange(5.0)[1:3] + [0.0, 1.0])

    path = trace.save(tmp_path / "ep.npz")
    loaded = EpisodeTrace.load(path)
    np.testing.assert_array_equal(loaded.states, trace.states)
    np.testing.assert_array_equal(loaded.actions, trace.actions)
    np.testing.assert_array_equal(loaded.rewards, np.full(3, 0.5))
    np.testing.assert_array_equal(loaded.acting, np.ones(3, dtype=bool))
    np.testing.assert_array_equal(loaded.chunk_ids, np.zeros(3, dtype=np.int64))
    np.testing.assert_array_equal(loaded.action_indices, np.arange(3))
    np.testing.assert_array_equal(loaded.observation_steps, np.zeros(3, dtype=np.int64))
    np.testing.assert_array_equal(loaded.request_steps, [0])
    np.testing.assert_array_equal(loaded.arrival_steps, [2])
    np.testing.assert_array_equal(loaded.chunks, np.ones((1, 3, 2)))
    assert loaded.meta.task_description == "fake task"
    assert loaded.meta.init_state_index == 4
    assert loaded.meta.num_steps == 3
    assert loaded.meta.success is False
    assert loaded.meta.nq == 2
    assert loaded.model_xml == "<mujoco><fake/></mujoco>"


def test_request_timing_arrays_align_with_requests(tmp_path):
    env = TraceRecordingEnv(_FakeEnv())
    env.reset()
    env.record_request(0, 0, 2, np.ones((3, 2)), {"queue_ms": 1.5, "infer_ms": 40.0, "batch_size": 3, "max_batch_size": 8})
    env.record_request(1, 5, 7, np.ones((3, 2)))
    trace = env.finish()
    trace.save(tmp_path / "trace.npz")
    loaded = EpisodeTrace.load(tmp_path / "trace.npz")

    assert loaded.queue_ms.tolist()[0] == 1.5 and np.isnan(loaded.queue_ms[1])
    assert loaded.infer_ms.tolist()[0] == 40.0
    assert loaded.batch_size.tolist() == [3, -1]
    assert loaded.max_batch_size.tolist() == [8, -1]
    assert len(loaded.queue_ms) == len(loaded.request_steps) == 2


def test_reset_starts_a_fresh_episode():
    env = TraceRecordingEnv(_FakeEnv())
    env.reset()
    env.step([1.0, 1.0])
    env.reset()
    env.step([0.5, 0.5])
    trace = env.finish()
    assert trace.num_steps == 1
    np.testing.assert_array_equal(trace.actions[0], [0.5, 0.5])
    np.testing.assert_array_equal(trace.acting, [False])
