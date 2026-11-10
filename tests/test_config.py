import datetime as dt
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vlalagbench.config import (
    Benchmark,
    ClosedPathTrajectory,
    CodeVersion,
    COMPARISON_RESULT_SCHEMA_VERSION,
    ComparisonMetrics,
    ComparisonResult,
    ComparisonRow,
    DynamicActor,
    DynamicVLABenchmark,
    DynamicVLATask,
    DynamicVLASeeds,
    DynamicVLAExperiment,
    DynamicVLAInference,
    DynamicVLAModelSpec,
    DynamicVLARuntime,
    DynamicVLAPlacement,
    DynamicVLAShard,
    DynamicVLAInstallation,
    DynamicVLAStudyPolicy,
    EnvironmentRuntime,
    EpisodeArtifact,
    EpisodeResult,
    EXPERIMENT_SCHEMA_VERSION,
    Experiment,
    GPUAllocation,
    Hardware,
    Inference,
    Keyframe,
    ModelSpec,
    Motion,
    Placement,
    PolicyServer,
    Shard,
    StudyPolicy,
    StudySpec,
    Videos,
    ReflectedTrajectory,
    RunLatency,
    RunResult,
    RUN_RESULT_SCHEMA_VERSION,
    RunSummary,
    Runtime,
    RUNTIME_SCHEMA_VERSION,
    Seeds,
    SourceConfig,
    TaskSummary,
    load_comparison_result,
    load_experiment,
    load_run_result,
    load_runtime,
    write_yaml,
)


ROOT = Path(__file__).resolve().parents[1]


def test_yaml_contract_round_trips_and_rejects_ambiguity(tmp_path):
    experiment_paths = sorted((ROOT / "configs/experiments").rglob("*.yaml"))
    experiments = [
        load_experiment(path)
        for path in experiment_paths
        if "deprecated" not in path.relative_to(ROOT / "configs/experiments").parts
    ]
    deprecated_experiments = [
        load_experiment(path)
        for path in experiment_paths
        if "deprecated" in path.relative_to(ROOT / "configs/experiments").parts
    ]
    runtimes = [load_runtime(path) for path in sorted((ROOT / "configs/runtime").glob("*.yaml"))]
    assert experiments and runtimes
    assert all(experiment.name.startswith("deprecated-") for experiment in deprecated_experiments)

    for index, document in enumerate([*experiments, *deprecated_experiments, *runtimes]):
        path = write_yaml(document, tmp_path / f"config-{index}.yaml")
        loader = (
            load_experiment
            if isinstance(document, (Experiment, DynamicVLAExperiment))
            else load_runtime
        )
        assert loader(path) == document

    experiment = load_experiment(ROOT / "configs/experiments/libero/pi05_libero_spatial.yaml")
    assert experiment.effective_latency_steps == 0
    invalid_inference = experiment.inference.model_copy(
        update={"execution_horizon_steps": experiment.inference.action_chunk_steps + 1}
    )
    with pytest.raises(ValidationError, match="execution_horizon_steps must not exceed"):
        Inference.model_validate(invalid_inference.model_dump())
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Experiment.model_validate({**experiment.model_dump(), "surprise": True})
    missing_kind = experiment.model_dump()
    del missing_kind["benchmark"]["kind"]
    assert Experiment.model_validate(missing_kind).benchmark.kind == "libero"
    scoped = experiment.model_dump()
    scoped["benchmark"]["task_ids"] = [0]
    scoped["dynamics_by_task"] = {0: experiment.dynamics}
    assert Experiment.model_validate(scoped).dynamics_by_task[0] == experiment.dynamics
    scoped["dynamics_by_task"] = {1: experiment.dynamics}
    with pytest.raises(ValidationError, match="outside benchmark.task_ids"):
        Experiment.model_validate(scoped)
    fixed_state = experiment.model_dump()
    fixed_state["benchmark"]["initial_state_indices"] = [0]
    assert Experiment.model_validate(fixed_state).benchmark.initial_state_indices == [0]
    fixed_state["benchmark"]["initial_state_indices"] = []
    with pytest.raises(ValidationError, match="initial_state_indices must not be empty"):
        Experiment.model_validate(fixed_state)
    overridden = experiment.model_dump()
    overridden["benchmark"]["task_ids"] = [1]
    overridden["initial_state_overrides"] = [
        {
            "task_id": 1,
            "initial_state_index": 0,
            "object_positions_m": {"basket_1_main": [0.1, 0.2, 0.035]},
            "object_yaws_rad": {"basket_1_main": 0.4},
        }
    ]
    parsed_override = Experiment.model_validate(overridden)
    assert parsed_override.initial_state_overrides[0].object_positions_m == {
        "basket_1_main": [0.1, 0.2, 0.035]
    }
    assert parsed_override.initial_state_overrides[0].object_yaws_rad == {
        "basket_1_main": 0.4
    }
    yaw_only = overridden.copy()
    yaw_only["initial_state_overrides"] = [
        {
            "task_id": 1,
            "initial_state_index": 0,
            "object_yaws_rad": {"basket_1_main": 0.4},
        }
    ]
    assert Experiment.model_validate(yaw_only).initial_state_overrides[0].object_yaws_rad
    overridden["initial_state_overrides"] *= 2
    with pytest.raises(ValidationError, match="one entry per task and state"):
        Experiment.model_validate(overridden)
    now = dt.datetime(2026, 8, 20, 12, 0, tzinfo=dt.timezone.utc)
    source = SourceConfig(
        source="configs/experiments/libero/example.yaml",
        sha256="a" * 64,
        resolved="experiment.yaml",
    )
    episode = EpisodeResult(
        task_id=0,
        episode_id=0,
        seed=7,
        success=True,
        steps=42,
        control_steps=50,
        wait_steps=8,
        policy_requests=9,
        trace="artifacts/run-1/episode-0.npz",
        error=None,
    )
    run_result = RunResult(
        schema_version=RUN_RESULT_SCHEMA_VERSION,
        run_id="run-1",
        status="completed",
        error=None,
        experiment=source,
        runtime=source.model_copy(update={"source": "configs/runtime/libero.yaml", "resolved": "runtime.yaml"}),
        code=CodeVersion(commit="b" * 40, dirty=False),
        hardware=Hardware(
            hostname="gpu-node-1",
            cpu_count=64,
            gpus=[GPUAllocation(type="NVIDIA GeForce RTX 4090", count=2)],
        ),
        started_at=now,
        finished_at=now,
        latency=RunLatency(control_hz=20, requested_ms=51, effective_steps=2),
        episode_data=EpisodeArtifact(
            format="jsonl",
            path="artifacts/run-1/episodes.jsonl",
            sha256="c" * 64,
            records=1,
        ),
        summary=RunSummary(
            episodes=1,
            successes=1,
            success_rate=1.0,
            episode_errors=0,
            policy_requests=9,
            mean_active_steps=42.0,
            mean_control_steps=50.0,
            mean_wait_steps=8.0,
            tasks=[
                TaskSummary(
                    task_id=0,
                    episodes=1,
                    successes=1,
                    success_rate=1.0,
                    mean_active_steps=42.0,
                    mean_control_steps=50.0,
                    mean_wait_steps=8.0,
                )
            ],
        ),
    )
    assert episode.trace == "artifacts/run-1/episode-0.npz"
    run_path = write_yaml(run_result, tmp_path / "result.yaml")
    assert load_run_result(run_path) == run_result

    comparison = ComparisonResult(
        schema_version=COMPARISON_RESULT_SCHEMA_VERSION,
        name="sync-vs-async",
        created_at=now,
        run_ids=["run-1"],
        rows=[
            ComparisonRow(
                label="sync",
                run_ids=["run-1"],
                metrics=ComparisonMetrics(success_rate=1.0, mean_episode_steps=42.0),
            )
        ],
    )
    comparison_path = write_yaml(comparison, tmp_path / "comparison.yaml")
    assert load_comparison_result(comparison_path) == comparison

    schema = yaml.safe_load((ROOT / "configs/schema.yaml").read_text())
    assert set(schema) == {
        "experiment",
        "dynamicvla_experiment",
        "runtime",
        "dynamicvla_runtime",
        "dynamicvla_study_policy",
        "dynamicvla_placement",
        "study",
        "placement",
        "videos",
        "run_result",
        "comparison_result",
    }
    assert schema["experiment"]["schema_version"] == EXPERIMENT_SCHEMA_VERSION
    assert schema["runtime"]["schema_version"] == RUNTIME_SCHEMA_VERSION
    assert schema["run_result"]["schema_version"] == RUN_RESULT_SCHEMA_VERSION
    assert schema["comparison_result"]["schema_version"] == COMPARISON_RESULT_SCHEMA_VERSION
    _assert_keys(schema["experiment"], Experiment)
    _assert_keys(schema["experiment"]["model"], ModelSpec)
    _assert_keys(schema["experiment"]["benchmark"], Benchmark)
    _assert_keys(schema["experiment"]["dynamics"][0], DynamicActor)
    _assert_keys(schema["experiment"]["dynamics"][0]["motion"], Motion)
    _assert_keys(schema["experiment"]["dynamics"][0]["keyframes"][0], Keyframe)
    assert set(schema["experiment"]["trajectory"]) == (
        set(ReflectedTrajectory.model_fields) | set(ClosedPathTrajectory.model_fields)
    )
    _assert_keys(schema["experiment"]["inference"], Inference)
    _assert_keys(schema["experiment"]["seeds"], Seeds)
    _assert_keys(schema["dynamicvla_experiment"], DynamicVLAExperiment)
    _assert_keys(schema["dynamicvla_experiment"]["model"], DynamicVLAModelSpec)
    _assert_keys(schema["dynamicvla_experiment"]["benchmark"], DynamicVLABenchmark)
    _assert_keys(schema["dynamicvla_experiment"]["inference"], DynamicVLAInference)
    _assert_keys(schema["runtime"], Runtime)
    _assert_keys(schema["runtime"]["policy_server"], PolicyServer)
    _assert_keys(schema["runtime"]["environment"], EnvironmentRuntime)
    _assert_keys(schema["dynamicvla_runtime"], DynamicVLARuntime)
    _assert_keys(schema["dynamicvla_experiment"]["benchmark"]["tasks"][0], DynamicVLATask)
    _assert_keys(schema["dynamicvla_experiment"]["seeds"], DynamicVLASeeds)
    _assert_keys(schema["dynamicvla_study_policy"], DynamicVLAStudyPolicy)
    _assert_keys(schema["dynamicvla_placement"], DynamicVLAPlacement)
    _assert_keys(schema["dynamicvla_placement"]["shards"][0], DynamicVLAShard)
    _assert_keys(schema["dynamicvla_placement"]["installation"], DynamicVLAInstallation)
    _assert_keys(schema["run_result"], RunResult)
    _assert_keys(schema["run_result"]["experiment"], SourceConfig)
    _assert_keys(schema["run_result"]["code"], CodeVersion)
    _assert_keys(schema["run_result"]["hardware"], Hardware)
    _assert_keys(schema["run_result"]["hardware"]["gpus"][0], GPUAllocation)
    _assert_keys(schema["run_result"]["latency"], RunLatency)
    _assert_keys(schema["run_result"]["episode_data"], EpisodeArtifact)
    _assert_keys(schema["run_result"]["summary"], RunSummary)
    _assert_keys(schema["run_result"]["summary"]["tasks"][0], TaskSummary)
    _assert_keys(schema["study"], StudySpec)
    _assert_keys(schema["study"]["policy"], StudyPolicy)
    _assert_keys(schema["placement"], Placement)
    _assert_keys(schema["placement"]["shards"][0], Shard)
    _assert_keys(schema["videos"], Videos)
    _assert_keys(schema["comparison_result"], ComparisonResult)
    _assert_keys(schema["comparison_result"]["rows"][0], ComparisonRow)
    _assert_keys(schema["comparison_result"]["rows"][0]["metrics"], ComparisonMetrics)


@pytest.mark.parametrize("mode", ["synchronous", "lazy", "continuous"])
@pytest.mark.parametrize("missing", [True, False])
def test_dynamicvla_requires_an_explicit_horizon_for_every_mode(mode, missing):
    data = load_experiment(
        ROOT / "configs/experiments/dynamicvla/dom_screen_240ms.yaml"
    ).model_dump()
    data["inference"]["mode"] = mode
    if missing:
        data["inference"].pop("execution_horizon_steps")
    else:
        data["inference"]["execution_horizon_steps"] = None
    with pytest.raises(ValidationError, match="execution_horizon_steps"):
        DynamicVLAExperiment.model_validate(data)


def test_dynamicvla_contract_rejects_protocol_ambiguity():
    experiment = load_experiment(ROOT / "configs/experiments/dynamicvla/dom_screen_240ms.yaml")
    runtime = load_runtime(ROOT / "configs/runtime/dynamicvla.yaml")

    assert isinstance(experiment, DynamicVLAExperiment)
    assert isinstance(runtime, DynamicVLARuntime)
    assert runtime.replica_batch_size == 1
    assert runtime.trace_writer_workers == 1
    assert runtime.antialiasing_mode == "DLSS"
    assert runtime.dlss_mode == "Balanced"
    assert runtime.enable_dl_denoiser is True
    assert runtime.enable_dlssg is False
    assert runtime.render_mode == "RaytracedLighting"
    assert experiment.effective_latency_steps == 6
    assert len(experiment.benchmark.tasks) == 1
    assert len(experiment.benchmark.tasks[0].environments) == 9
    assert experiment.benchmark.resolved_episode_counts == {
        "dom-nine-dimension-screen": 18
    }
    assert experiment.seeds.policy == 0

    fxaa_runtime = DynamicVLARuntime.model_validate(
        {**runtime.model_dump(), "antialiasing_mode": "FXAA"}
    )
    assert fxaa_runtime.antialiasing_mode == "FXAA"

    data = experiment.model_dump()
    data["inference"]["latency_ms"] = 250
    with pytest.raises(ValidationError, match="align to the 25 Hz control grid"):
        DynamicVLAExperiment.model_validate(data)

    data["inference"]["latency_ms"] = 800
    with pytest.raises(ValidationError, match="latency plus execution horizon"):
        DynamicVLAExperiment.model_validate(data)

    data = experiment.model_dump()
    data["inference"]["mode"] = "synchronous"
    data["inference"]["execution_horizon_steps"] = None
    with pytest.raises(ValidationError, match="execution_horizon_steps"):
        DynamicVLAExperiment.model_validate(data)
    data["inference"]["execution_horizon_steps"] = 20
    synchronous = DynamicVLAExperiment.model_validate(data)
    assert synchronous.resolved_execution_horizon_steps == 20
    data["inference"]["execution_horizon_steps"] = 21
    with pytest.raises(ValidationError, match="must not exceed action_chunk_steps"):
        DynamicVLAExperiment.model_validate(data)

    data = experiment.model_dump()
    data["inference"]["mode"] = "lazy"
    data["inference"]["execution_horizon_steps"] = None
    with pytest.raises(ValidationError, match="execution_horizon_steps"):
        DynamicVLAExperiment.model_validate(data)
    data["inference"]["execution_horizon_steps"] = 14
    for mode in ("lazy", "continuous"):
        data["inference"]["mode"] = mode
        data["inference"]["latency_ms"] = 240
        valid = DynamicVLAExperiment.model_validate(data)
        assert valid.effective_latency_steps == 6
        assert valid.resolved_execution_horizon_steps == 14
        for latency_ms in (280, 360, 480, 560):
            data["inference"]["latency_ms"] = latency_ms
            with pytest.raises(ValidationError, match="latency plus execution horizon"):
                DynamicVLAExperiment.model_validate(data)

    data = experiment.model_dump()
    data["benchmark"]["tasks"][0]["environments"] = ["../outside.json"]
    with pytest.raises(ValidationError, match="relative to the runtime assets_root"):
        DynamicVLAExperiment.model_validate(data)

    data = experiment.model_dump()
    data["benchmark"]["tasks"].append(data["benchmark"]["tasks"][0])
    with pytest.raises(ValidationError, match="task names must be globally unique"):
        DynamicVLAExperiment.model_validate(data)

    data = experiment.model_dump()
    data["benchmark"]["tasks"].append(
        {
            "name": "another-task",
            "environments": [data["benchmark"]["tasks"][0]["environments"][0]],
        }
    )
    with pytest.raises(ValidationError, match="environment paths must be globally unique"):
        DynamicVLAExperiment.model_validate(data)

    data = experiment.model_dump()
    del data["seeds"]
    with pytest.raises(ValidationError, match="Field required"):
        DynamicVLAExperiment.model_validate(data)

    data = experiment.model_dump()
    environment = data["benchmark"]["tasks"][0]["environments"][0]
    parent, name = environment.rsplit("/", 1)
    data["benchmark"]["tasks"][0]["environments"].append(f"{parent}/./{name}")
    with pytest.raises(ValidationError, match="environment paths must be globally unique"):
        DynamicVLAExperiment.model_validate(data)


def test_reflected_trajectory_is_one_balanced_speed_condition():
    experiment = load_experiment(
        ROOT
        / "tests/fixtures/configs/experiments/libero/libero_object_task1_motion"
        / "pi05_libero_object_task1_max_runway_015mps.yaml"
    )

    assert experiment.trajectory.speed_mps == 0.15
    assert experiment.benchmark.episodes_per_task == 2 * 12 * 2

    data = experiment.model_dump()
    data["benchmark"]["episodes_per_task"] = 25
    with pytest.raises(ValidationError, match=r"divisible by phase_bins \* 2"):
        Experiment.model_validate(data)


def test_explicit_trajectory_schedule_must_match_episode_count():
    experiment = load_experiment(
        ROOT
        / "tests/fixtures/configs/experiments/libero/libero_object_task1_motion"
        / "pi05_libero_object_task1_max_runway_015mps.yaml"
    )
    data = experiment.model_dump()
    data["trajectory"]["phase_bins"] = None
    data["trajectory"]["episode_schedule"] = [
        {"initial_phase": 0.2, "initial_direction": 1},
        {"initial_phase": 0.2, "initial_direction": -1},
    ]

    with pytest.raises(ValidationError, match="schedule length must equal"):
        Experiment.model_validate(data)

    data["benchmark"]["episodes_per_task"] = 2
    scheduled = Experiment.model_validate(data)
    assert scheduled.trajectory.phase_bins is None
    assert len(scheduled.trajectory.episode_schedule) == 2

    data["trajectory"]["phase_bins"] = 1
    with pytest.raises(ValidationError, match="exactly one"):
        Experiment.model_validate(data)


def test_dynamic_actor_timing_modes_are_strict_and_unambiguous():
    actor = {
        "target": "plate",
        "start_step": 0,
        "end_step": None,
        "loop": False,
        "interpolation": "linear",
        "keyframes": [
            {"position_offset_m": [0.0, 0.0, 0.0]},
            {"position_offset_m": [1.0, 0.0, 0.0]},
        ],
    }
    legacy_keyframes = [
        {**keyframe, "step": index * 10}
        for index, keyframe in enumerate(actor["keyframes"])
    ]
    legacy = DynamicActor.model_validate({**actor, "keyframes": legacy_keyframes})
    assert legacy.motion is None
    distance_timed = DynamicActor.model_validate({**actor, "motion": {"speed_mps": 0.25}})
    assert distance_timed.motion == Motion(speed_mps=0.25)
    randomized = DynamicActor.model_validate(
        {**actor, "motion": {"speed_mps": 0.25, "initial_phase": "random"}}
    )
    assert randomized.motion.initial_phase == "random"

    invalid = [
        (
            {
                **actor,
                "motion": {"speed_mps": 0.25},
                "keyframes": [{**keyframe, "step": None} for keyframe in actor["keyframes"]],
            },
            "requires spatial waypoints without step",
        ),
        (actor, "legacy timing requires step on every keyframe"),
        ({**actor, "motion": {"speed_mps": 0.0}}, "greater than 0"),
        (
            {
                **actor,
                "motion": {"speed_mps": 0.25},
                "keyframes": [actor["keyframes"][0], {"position_offset_m": [1.0, 0.0, 1.0]}],
            },
            "planar XY",
        ),
    ]
    for data, message in invalid:
        with pytest.raises(ValidationError, match=message):
            DynamicActor.model_validate(data)


def _assert_keys(template, model):
    assert set(template) == set(model.model_fields), model.__name__
