import numpy as np
import pytest

from vlalagbench.benchmarks.libero import evaluator
from vlalagbench.benchmarks.libero import policy_server
from tests.benchmarks.libero.robosuite_fakes import fake_robots


class _Env:
    def __init__(self):
        self.steps = 0
        self.actions = []
        self.robots = fake_robots()

    def step(self, action):
        self.steps += 1
        self.actions.append(np.asarray(action))
        return {"step": self.steps}, 0.0, False, {}

    def set_dynamic_step(self, step):
        pass


class _Policy:
    def __init__(self):
        self.requests = 0

    def infer(self, element):
        self.requests += 1
        return {"actions": np.zeros((4, 7))}


def test_episode_reports_evaluation_steps_and_policy_requests(monkeypatch):
    monkeypatch.setattr(evaluator, "_policy_element", lambda *args: {})
    env = _Env()
    policy = _Policy()
    config = evaluator.EvalConfig(
        policy="dummy",
        action_chunk_steps=4,
        execution_horizon_steps=2,
        num_steps_wait=2,
        max_steps=3,
    )

    result = evaluator.run_episode(env, "task", policy, config, obs={})

    assert result == {
        "success": False,
        "steps": 3,
        "control_steps": 3,
        "wait_steps": 0,
        "policy_requests": 2,
        "error": None,
    }
    assert env.steps == 5
    assert policy.requests == 2


def test_sync_waits_do_not_drift_and_do_not_consume_acting_budget(monkeypatch):
    monkeypatch.setattr(evaluator, "_policy_element", lambda *args: {})
    env = _Env()

    class Policy:
        def infer(self, _element):
            actions = np.ones((10, 7))
            actions[:, -1] = 0.75
            return {"actions": actions}

    result = evaluator.run_episode(
        env,
        "task",
        Policy(),
        evaluator.EvalConfig(
            policy="dummy",
            action_chunk_steps=10,
            execution_horizon_steps=5,
            latency_steps=2,
            num_steps_wait=0,
            max_steps=6,
        ),
        obs={},
    )

    assert result["steps"] == 6
    assert result["control_steps"] == 10
    assert result["wait_steps"] == 4
    np.testing.assert_array_equal(env.actions[0], evaluator.LIBERO_DUMMY_ACTION)
    np.testing.assert_array_equal(env.actions[1], evaluator.LIBERO_DUMMY_ACTION)
    np.testing.assert_array_equal(env.actions[7][:-1], np.zeros(6))
    assert env.actions[7][-1] == 0.75


def test_first_success_contact_ends_episode_without_hold_window(monkeypatch):
    monkeypatch.setattr(evaluator, "_policy_element", lambda *args: {})

    class ContactEnv(_Env):
        def step(self, action):
            self.steps += 1
            self.actions.append(np.asarray(action))
            return {"step": self.steps}, 0.0, self.steps == 1, {}

    class Policy:
        def infer(self, _element):
            return {"actions": np.zeros((10, 7))}

    env = ContactEnv()
    result = evaluator.run_episode(
        env,
        "task",
        Policy(),
        evaluator.EvalConfig(
            policy="dummy",
            action_chunk_steps=10,
            execution_horizon_steps=5,
            num_steps_wait=0,
            max_steps=300,
        ),
        obs={},
    )

    assert result["success"] is True
    assert result["steps"] == 1
    assert result["control_steps"] == 1
    assert env.steps == 1


def test_trajectory_is_visible_to_the_first_policy_request(monkeypatch):
    monkeypatch.setattr(evaluator, "_policy_element", lambda obs, *_args: obs)
    monkeypatch.setattr(evaluator, "get_sim_state", lambda _env: "moved-state")

    class Env(_Env):
        def __init__(self):
            super().__init__()
            self.dynamic_steps = []
            self.prepared_at_steps = []
            self.refreshed = False

        def prepare_dynamic_step(self, step):
            assert step == 0
            self.prepared_at_steps.append(self.steps)
            return True

        def refresh_current_state(self):
            self.refreshed = True

        def regenerate_obs_from_state(self, state):
            return {"state": state}

        def set_dynamic_step(self, step):
            self.dynamic_steps.append(step)

    class Policy:
        def __init__(self):
            self.observations = []

        def infer(self, element):
            self.observations.append(element)
            return {"actions": np.zeros((4, 7))}

    env = Env()
    policy = Policy()
    config = evaluator.EvalConfig(
        policy="dummy",
        trajectory=object(),
        action_chunk_steps=4,
        execution_horizon_steps=2,
        num_steps_wait=2,
        max_steps=1,
    )

    evaluator.run_episode(env, "task", policy, config, obs={"state": "old"})

    assert policy.observations[0]["state"] == "moved-state"
    assert env.refreshed is True
    assert env.prepared_at_steps == [0, 2]
    assert env.dynamic_steps == [None, None]


def test_episode_shards_keep_each_worker_on_one_task():
    shards = evaluator._episode_shards([5, 9], list(range(10)), workers=4)

    assert shards == [
        (5, (0, 2, 4, 6, 8)),
        (5, (1, 3, 5, 7, 9)),
        (9, (0, 2, 4, 6, 8)),
        (9, (1, 3, 5, 7, 9)),
    ]


def test_dynamic_episode_queue_hands_out_single_episodes_in_order():
    assert evaluator._episode_queue([5, 9], [3, 40]) == [(5, (3,)), (5, (40,)), (9, (3,)), (9, (40,))]


def test_episode_subset_keeps_its_ids():
    shards = evaluator._episode_shards([1], [3, 40, 41, 199], workers=2)

    assert shards == [(1, (3, 41)), (1, (40, 199))]


def test_episode_initialization_can_repeat_selected_official_state():
    class Env:
        def set_initial_object_positions(self, positions):
            self.positions = dict(positions)

        def set_initial_object_yaws(self, yaws):
            self.yaws = dict(yaws)

        def reset(self):
            return None

        def set_init_state(self, state):
            return {"state": state}

    config = evaluator.EvalConfig(seed=7, initial_state_indices=[1])

    selection = evaluator._select_initial_state(config, 3, state_count=2)
    result = evaluator._initialize_episode(
        Env(), ["official-0", "official-1"], selection[0]
    )

    assert selection == (1, 7)
    assert result == {"state": "official-1"}


def test_episode_initialization_applies_object_position_overrides():
    class Env:
        def set_initial_object_positions(self, positions):
            self.positions = dict(positions)

        def set_initial_object_yaws(self, yaws):
            self.yaws = dict(yaws)

        def reset(self):
            return None

        def set_init_state(self, state):
            return {"state": state}

    env = Env()
    result = evaluator._initialize_episode(
        env,
        ["official-0"],
        0,
        {"basket_1_main": [0.1, 0.2, 0.035]},
        {"basket_1_main": 0.4},
    )

    assert result == {"state": "official-0"}
    assert env.positions == {"basket_1_main": [0.1, 0.2, 0.035]}
    assert env.yaws == {"basket_1_main": 0.4}


def test_recording_env_ignores_robosuite_horizon(monkeypatch):
    observed = {}

    class Task:
        problem_folder = "problem"
        bddl_file = "task.bddl"
        name = "task"
        language = "do the task"

    def fake_make_env(task, resolution, seed, **kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(evaluator, "make_env", fake_make_env)
    monkeypatch.setattr(evaluator, "DynamicEnv", lambda env, dynamics: env)
    monkeypatch.setattr(evaluator, "TraceRecordingEnv", lambda env, meta: (env, meta))
    monkeypatch.setattr(evaluator, "git_commit", lambda root: "a" * 40)

    evaluator._recording_env(Task(), evaluator.EvalConfig(), task_id=5)

    assert observed["ignore_done"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("openpi_config", "pi0_libero"),
        ("checkpoint", "gs://other-checkpoint"),
        ("action_chunk_steps", 5),
    ],
)
def test_policy_metadata_handshake_rejects_mismatches(field, value):
    metadata = policy_server.build_metadata("pi05_libero", "gs://checkpoint", 10)
    policy_server.validate_metadata(
        metadata,
        openpi_config="pi05_libero",
        checkpoint="gs://checkpoint",
        action_chunk_steps=10,
    )

    expected = {
        "openpi_config": "pi05_libero",
        "checkpoint": "gs://checkpoint",
        "action_chunk_steps": 10,
    }
    expected[field] = value
    with pytest.raises(RuntimeError, match=field):
        policy_server.validate_metadata(
            metadata,
            **expected,
        )


def test_episode_deadline_cannot_be_defaulted():
    # A run whose deadline nobody stated would carry a horizon nothing records
    # as a choice, so the evaluator refuses it rather than picking one.
    with pytest.raises(ValueError, match="max_steps"):
        evaluator.EvalConfig().resolved_max_steps()
    assert evaluator.EvalConfig(max_steps=600).resolved_max_steps() == 600


def test_fresh_episodes_cannot_inherit_environment_or_rng_history(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import random

    created = []

    class Env:
        def __init__(self):
            self.seeded_draw = (np.random.random(), random.random())
            self.used = False
            self.closed = False
            created.append(self)

        def set_dynamic_actors(self, actors):
            pass

        def finish(self, extra_meta):
            assert extra_meta["episode_reset"] == "fresh"
            assert extra_meta["episode_id"] in {0, 1}
            return SimpleNamespace(save=lambda path: path)

        def close(self):
            self.closed = True

    suite = SimpleNamespace(get_task=lambda _: SimpleNamespace(language="task"), get_task_init_states=lambda _: [0, 1])
    monkeypatch.setattr(evaluator, "_recording_env", lambda *args: Env())
    monkeypatch.setattr(evaluator, "_initialize_episode", lambda *args: {})

    def episode(env, *args, **kwargs):
        assert not env.used
        env.used = True
        np.random.random(19)
        random.random()
        return {"success": False, "steps": 1, "control_steps": 1, "error": None}

    monkeypatch.setattr(evaluator, "run_episode", episode)
    evaluator._run_shard(suite, object(), evaluator.EvalConfig(max_steps=1), tmp_path, 2, [0, 1])
    assert len(created) == 2 and all(env.closed for env in created)
    assert created[0].seeded_draw == created[1].seeded_draw
