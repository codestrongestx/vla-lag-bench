"""Scene reuse changes execution order only, retaining original episode identities."""
import copy

from vlalagbench.benchmarks.dynamicvla.simulator import ordered_episode_replicas


def test_grouping_uses_first_occurrence_and_keeps_episode_metadata():
    rows = {
        (0, 0, 12): {"initial_state_index": 2, "environment_sha256": "z", "instruction": "third", "seed": 12},
        (0, 0, 10): {"initial_state_index": 0, "environment_sha256": "z", "instruction": "first", "seed": 10},
        (0, 0, 14): {"initial_state_index": 4, "environment_sha256": "a", "instruction": "fifth", "seed": 14},
        (0, 0, 11): {"initial_state_index": 1, "environment_sha256": "a", "instruction": "second", "seed": 11},
        (0, 0, 13): {"initial_state_index": 3, "environment_sha256": "m", "instruction": "fourth", "seed": 13},
    }
    before = copy.deepcopy(rows)
    objects = {key: id(value) for key, value in rows.items()}
    assert ordered_episode_replicas(rows, 0, 0) == [10, 11, 12, 13, 14]
    assert ordered_episode_replicas(rows, 0, 0, group_identical_scenes=True) == [10, 12, 11, 14, 13]
    assert rows == before
    assert {key: id(value) for key, value in rows.items()} == objects


def test_unique_scenes_and_other_task_environments_keep_original_order():
    rows = {
        (0, 0, 8): {"initial_state_index": 3, "environment_sha256": "b"},
        (0, 0, 7): {"initial_state_index": 1, "environment_sha256": "z"},
        (1, 0, 0): {"initial_state_index": 0, "environment_sha256": "z"},
        (0, 1, 0): {"initial_state_index": 0, "environment_sha256": "z"},
    }
    assert ordered_episode_replicas(rows, 0, 0) == [7, 8]
    assert ordered_episode_replicas(rows, 0, 0, group_identical_scenes=True) == [7, 8]
    assert ordered_episode_replicas(rows, 2, 0, group_identical_scenes=True) == []


def test_mixed_replica_order_and_absent_state_index_are_stable():
    rows = {
        (0, 0, 5): {"initial_state_index": 1, "environment_sha256": "first"},
        (0, 0, 4): {"initial_state_index": 1, "environment_sha256": "second"},
        (0, 0, 3): {"initial_state_index": 0, "environment_sha256": "second"},
        (0, 0, 2): {"environment_sha256": "first"},
        (0, 0, 1): {"initial_state_index": 2, "environment_sha256": "first"},
    }
    assert ordered_episode_replicas(rows, 0, 0) == [2, 3, 4, 5, 1]
    assert ordered_episode_replicas(rows, 0, 0, group_identical_scenes=True) == [2, 5, 1, 3, 4]
