"""Pin what a refactor could silently break: expansion, hashing, order, select, invariants."""

from pathlib import Path

import pytest
import yaml

from vlalagbench.config import Placement, StudySpec
from vlalagbench.study.spec import (
    STUDY_RECORD,
    check_placement,
    expand,
    ordered,
    resolve,
    selected,
    study_hash,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = {
    "schema_version": 2,
    "name": "probe",
    "base_experiment": "configs/experiments/libero/libero_object_task1_native_basket_ellipse/basket_ellipse_paired_200.yaml",
    "sweep": {
        "trajectory.speed_mps": [0.001, 0.125],
        "inference.mode": ["synchronous", "continuous"],
        "inference.latency_ms": [0, 200],
    },
    "override": {"benchmark.max_episode_steps": 600},
    "policy": {
        "implementation": "single",
        "max_batch_size": 1,
        "batch_padding": "none",
        "max_wait_ms": 0.0,
        "deterministic_ops": False,
        "pytorch_compile_mode": "max-autotune-no-cudagraphs",
    },
}
PLACEMENT = {
    "schema_version": 1,
    "host": "test",
    "gpu_budget": 3,
    "shards": [
        {"policy_gpu": 0, "port": 8100, "cpus": "0-13"},
        {"policy_gpu": 1, "port": 8101, "cpus": "14-27"},
    ],
    "render_gpu": 2,
    "workers_per_shard": 12,
    "policy_python": "/usr/bin/python3",
    "output_root": "/nonexistent",
}


def _study(**changes) -> StudySpec:
    return StudySpec.model_validate({**STUDY, **changes})


def _placement(**changes) -> Placement:
    return Placement.model_validate({**PLACEMENT, **changes})


def test_expansion_applies_sweep_then_override_and_names_each_point():
    points = expand(_study(), ROOT)

    assert len(points) == 8
    point = next(point for point in points if point.name == "0.125_continuous_200")
    assert point.experiment.trajectory.speed_mps == 0.125
    assert point.experiment.inference.mode == "continuous"
    assert point.experiment.inference.latency_ms == 200
    assert point.experiment.benchmark.max_episode_steps == 600
    assert point.experiment.name == "probe/0.125_continuous_200"
    assert point.experiment.benchmark.episodes_per_task == 200


def test_hash_ignores_order_and_name_but_not_policy_or_override():
    points = expand(_study(), ROOT)
    base = study_hash(_study(), points)

    assert study_hash(_study(name="renamed", order=[{"inference.latency_ms": "descending"}]), points) == base
    assert study_hash(_study(policy={**STUDY["policy"], "deterministic_ops": True}), points) != base
    assert study_hash(_study(override={"benchmark.max_episode_steps": 300}), expand(_study(override={"benchmark.max_episode_steps": 300}), ROOT)) != base


def test_order_sorts_by_rank_list_then_numeric_key():
    spec = _study(order=[{"inference.mode": ["continuous", "synchronous"]}, {"inference.latency_ms": "descending"}, {"trajectory.speed_mps": "ascending"}])

    names = [point.name for point in ordered(spec, expand(spec, ROOT))]

    assert names[:4] == ["0.001_continuous_200", "0.125_continuous_200", "0.001_continuous_0", "0.125_continuous_0"]
    assert names[4].endswith("synchronous_200")


def test_select_narrows_the_queue_without_changing_the_hash():
    points = expand(_study(), ROOT)

    subset = selected(points, ["inference.mode=synchronous", "trajectory.speed_mps=0.125"])

    assert [point.name for point in subset] == ["0.125_synchronous_0", "0.125_synchronous_200"]
    assert study_hash(_study(), points) == resolve(_study(), _placement(), ROOT, selects=["inference.mode=synchronous"]).study_hash


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"gpu_budget": 2}, "gpu_budget"),
        ({"render_gpu": 0}, "policy_memory_fraction"),
        ({"shards": [{"policy_gpu": 0, "port": 8100, "cpus": "0-3"}, {"policy_gpu": 0, "port": 8101, "cpus": "4-7"}]}, "same policy_gpu"),
        ({"shards": [{"policy_gpu": 0, "port": 8100, "cpus": "0-3"}, {"policy_gpu": 1, "port": 8100, "cpus": "4-7"}]}, "same port"),
    ],
)
def test_each_placement_invariant_fires(changes, message):
    with pytest.raises(ValueError, match=message):
        resolve(_study(policy={**STUDY["policy"], "batch_padding": "none"}), _placement(**changes), ROOT)


def test_resume_refuses_a_study_directory_with_another_hash(tmp_path):
    (tmp_path / "probe").mkdir()
    (tmp_path / "probe" / STUDY_RECORD).write_text(yaml.safe_dump({"study_hash": "0" * 64}))

    with pytest.raises(ValueError, match="different study_hash"):
        resolve(_study(), _placement(output_root=str(tmp_path)), ROOT)

    plan = resolve(_study(), _placement(output_root=str(tmp_path / "fresh")), ROOT)
    check_placement(plan)


def test_gr00t_shared_gpu_preserves_study_and_request_protocol():
    from vlalagbench.study.spec import load_study, shard_runtime

    spec = load_study(ROOT / "configs/studies/gr00t_speed_sweep_20260914/basket_ellipse_600.yaml")
    shards = [{"policy_gpu": 0, "port": 8100 + i, "cpus": f"{4 * i}-{4 * i + 3}"} for i in range(3)]
    shared = resolve(spec, _placement(gpu_budget=1, render_gpu=0, shards=shards,
                                     policy_memory_fraction=0.3, workers_per_shard=4), ROOT)
    single = resolve(spec, _placement(gpu_budget=1, render_gpu=0, shards=shards[:1],
                                     policy_memory_fraction=0.6, workers_per_shard=4), ROOT)

    assert shared.study_hash == single.study_hash
    assert [p.experiment for p in shared.points] == [p.experiment for p in single.points]
    for index in range(3):
        server = shard_runtime(shared, index).policy_server
        assert server.implementation == "single"
        assert server.max_batch_size == 1


@pytest.mark.parametrize(("fraction", "cpus", "message"), [
    (0.5, "4-7", "memory fractions"),
    (0.3, "3-6", "disjoint CPU"),
])
def test_gr00t_shared_gpu_rejects_overcommitted_placement(fraction, cpus, message):
    from vlalagbench.study.spec import load_study

    spec = load_study(ROOT / "configs/studies/gr00t_speed_sweep_20260914/basket_ellipse_600.yaml")
    shards = [{"policy_gpu": 0, "port": 8100, "cpus": "0-3"},
              {"policy_gpu": 0, "port": 8101, "cpus": cpus}]
    with pytest.raises(ValueError, match=message):
        resolve(spec, _placement(gpu_budget=1, render_gpu=0, shards=shards,
                                policy_memory_fraction=fraction), ROOT)


def test_selection_parses_numbers_and_rejects_silent_empty_queues():
    points = expand(_study(), ROOT)
    assert selected(points, ["trajectory.speed_mps=0.0010"]) == selected(points, ["trajectory.speed_mps=0.001"])
    for select in ("trajectory.speed_mps=99", "unknown=1", "inference.mode", "trajectory.speed_mps=fast"):
        with pytest.raises(ValueError, match="select"):
            selected(points, [select])


def test_new_protocol_is_single_request_fresh_and_has_a_distinct_hash():
    from vlalagbench.study.spec import shard_runtime

    legacy = _study(schema_version=1, policy={**STUDY["policy"], "implementation": "dynamic_batching", "max_batch_size": 2, "batch_padding": "fixed"})
    spec = _study(schema_version=2, policy={})
    plan = resolve(spec, _placement(workers_per_shard=4), ROOT)
    runtime = shard_runtime(plan, 0)
    assert runtime.policy_server.implementation == "single"
    assert runtime.policy_server.max_batch_size == 1
    assert runtime.policy_server.deterministic_ops is False
    assert runtime.environment.reuse_task_envs is False
    assert plan.study_hash != study_hash(legacy, expand(legacy, ROOT))
    assert resolve(spec, _placement(workers_per_shard=1), ROOT).study_hash == plan.study_hash
    with pytest.raises(ValueError, match="historical and read-only"):
        shard_runtime(resolve(legacy, _placement(), ROOT), 0)


def test_overrides_cannot_mislabel_sweep_conditions():
    with pytest.raises(ValueError, match="overlaps"):
        _study(override={"trajectory.speed_mps": 0.125})


def test_recorded_study_opens_without_original_base_experiment(tmp_path, monkeypatch):
    from vlalagbench.study import runner
    from vlalagbench.study.spec import open_study

    spec = _study(schema_version=2, policy={})
    plan = resolve(spec, _placement(output_root=str(tmp_path)), ROOT)
    runner.prepare(plan)
    monkeypatch.setattr("vlalagbench.study.spec.expand", lambda *args: (_ for _ in ()).throw(AssertionError("must read snapshots")))
    reopened = open_study(plan.study_dir)
    assert reopened.study_hash == plan.study_hash
    assert reopened.points == plan.points
    config = plan.study_dir / "configs" / f"{plan.points[0].name}.yaml"
    config.write_text(config.read_text().replace("max_episode_steps: 600", "max_episode_steps: 500"))
    with pytest.raises(ValueError, match="study_hash"):
        open_study(plan.study_dir)


def test_batched_study_carries_request_shaping_into_shard_runtimes_and_hash():
    from vlalagbench.study.spec import shard_runtime

    single = resolve(_study(schema_version=2, policy={}), _placement(), ROOT)
    batched_policy = {"implementation": "dynamic_batching", "max_batch_size": 4, "max_wait_ms": 5.0, "batch_padding": "fixed"}
    batched = resolve(_study(schema_version=2, policy=batched_policy), _placement(), ROOT)
    server = shard_runtime(batched, 0).policy_server
    assert (server.implementation, server.max_batch_size, server.max_wait_ms, server.batch_padding) == ("dynamic_batching", 4, 5.0, "fixed")
    assert batched.study_hash != single.study_hash


def test_openpi_replicas_may_share_a_gpu_within_the_memory_budget():
    shards = [{"policy_gpu": 0, "port": 8100, "cpus": "0-3"}, {"policy_gpu": 0, "port": 8101, "cpus": "4-7"}]
    check_placement(resolve(_study(), _placement(shards=shards, render_gpu=0, policy_memory_fraction=0.45), ROOT))
    with pytest.raises(ValueError, match="at most 0.9"):
        resolve(_study(), _placement(shards=shards, render_gpu=0, policy_memory_fraction=0.5), ROOT)
