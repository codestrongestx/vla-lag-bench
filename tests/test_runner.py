import hashlib
import json

import pytest

from vlalagbench.config import (
    CodeVersion,
    GPUAllocation,
    Hardware,
    load_experiment,
    load_run_result,
    load_runtime,
    write_yaml,
)
from vlalagbench import runner
from vlalagbench.benchmarks.dynamicvla import adapter as dynamicvla_adapter
from vlalagbench.benchmarks.libero import adapter as libero_adapter
from vlalagbench.benchmarks.libero.envs import require_supported_control_hz


def test_runner_writes_result_through_model(tmp_path, monkeypatch):
    experiment = load_experiment(runner.REPO_ROOT / "configs/experiments/libero/pi05_libero_spatial.yaml")
    runtime = load_runtime(runner.REPO_ROOT / "configs/runtime/libero.yaml")
    experiment_path = write_yaml(experiment, tmp_path / "configs/experiments/libero/experiment.yaml")
    runtime_path = write_yaml(runtime, tmp_path / "configs/runtime/runtime.yaml")

    def fake_eval(config):
        assert config.openpi_config == "pi05_libero"
        assert config.checkpoint == "gs://openpi-assets/checkpoints/pi05_libero"
        assert config.action_chunk_steps == 10
        assert config.execution_horizon_steps == 5
        assert config.workers == 1
        assert config.dynamics_by_task == experiment.dynamics_by_task
        config.out_dir.mkdir(parents=True, exist_ok=True)
        (config.out_dir / "episode.npz").write_bytes(b"trace")
        return {
            "episodes": [
                {
                    "task_id": 0,
                    "episode": 0,
                    "success": True,
                    "steps": 42,
                    "control_steps": 50,
                    "wait_steps": 8,
                    "policy_requests": 9,
                    "trace": "episode.npz",
                    "error": None,
                }
            ]
        }

    monkeypatch.setattr(libero_adapter, "run_eval", fake_eval)
    monkeypatch.setattr(
        runner, "_git_state", lambda root: CodeVersion(commit="a" * 40, dirty=False)
    )
    monkeypatch.setattr(runner, "collect_hardware", _hardware)

    result = runner.run_experiment(
        experiment_path,
        runtime_path,
        repo_root=tmp_path,
        run_id="run-1",
    )

    result_path = tmp_path / "runs/run-1/result.yaml"
    assert load_run_result(result_path) == result
    assert result.summary.success_rate == 1.0
    assert result.summary.tasks[0].task_id == 0
    assert result.experiment.source == "configs/experiments/libero/experiment.yaml"
    episode_path = tmp_path / "artifacts/run-1/episodes.jsonl"
    episode_content = episode_path.read_bytes()
    assert result.episode_data.path == "artifacts/run-1/episodes.jsonl"
    assert result.episode_data.sha256 == hashlib.sha256(episode_content).hexdigest()
    assert json.loads(episode_content)["episode_id"] == 0
    assert len(result_path.read_text().splitlines()) < 100
    assert "episode_id:" not in result_path.read_text()
    assert (tmp_path / "runs/run-1/experiment.yaml").read_bytes() == experiment_path.read_bytes()
    assert (tmp_path / "runs/run-1/runtime.yaml").read_bytes() == runtime_path.read_bytes()


@pytest.mark.parametrize(
    "artifact_root_variable",
    ["VLALAGBENCH_ARTIFACT_ROOT", "MUJOCO_DYNAMIC_ARTIFACT_ROOT"],
)
def test_runner_can_write_artifacts_outside_repo(
    tmp_path,
    monkeypatch,
    artifact_root_variable,
):
    experiment = load_experiment(runner.REPO_ROOT / "configs/experiments/libero/pi05_libero_spatial.yaml")
    runtime = load_runtime(runner.REPO_ROOT / "configs/runtime/libero.yaml")
    experiment_path = write_yaml(experiment, tmp_path / "configs/experiments/libero/experiment.yaml")
    runtime_path = write_yaml(runtime, tmp_path / "configs/runtime/runtime.yaml")
    external = tmp_path.parent / f"{tmp_path.name}-artifacts"

    def fake_eval(config):
        assert config.out_dir == external / "run-external"
        config.out_dir.mkdir(parents=True, exist_ok=True)
        (config.out_dir / "episode.npz").write_bytes(b"trace")
        return {
            "episodes": [
                {
                    "task_id": 0,
                    "episode": 0,
                    "success": False,
                    "steps": 1,
                    "control_steps": 2,
                    "wait_steps": 1,
                    "policy_requests": 1,
                    "trace": "episode.npz",
                    "error": None,
                }
            ]
        }

    monkeypatch.setenv(artifact_root_variable, str(external))
    monkeypatch.setattr(libero_adapter, "run_eval", fake_eval)
    monkeypatch.setattr(runner, "_git_state", lambda root: CodeVersion(commit="a" * 40, dirty=False))
    monkeypatch.setattr(runner, "collect_hardware", _hardware)

    result = runner.run_experiment(
        experiment_path,
        runtime_path,
        repo_root=tmp_path,
        run_id="run-external",
    )

    episode_path = external / "run-external/episodes.jsonl"
    episode = json.loads(episode_path.read_text())
    assert result.episode_data.path == episode_path.as_posix()
    assert episode["trace"] == (external / "run-external/episode.npz").as_posix()


def test_libero_preflight_has_no_run_side_effects(tmp_path):
    experiment = load_experiment(
        runner.REPO_ROOT / "configs/experiments/libero/pi05_libero_spatial.yaml"
    )
    experiment = experiment.model_copy(
        update={
            "benchmark": experiment.benchmark.model_copy(update={"control_hz": 30})
        }
    )
    runtime = load_runtime(runner.REPO_ROOT / "configs/runtime/libero.yaml")
    experiment_path = write_yaml(experiment, tmp_path / "experiment.yaml")
    runtime_path = write_yaml(runtime, tmp_path / "runtime.yaml")

    with pytest.raises(NotImplementedError, match="20 control_hz"):
        runner.run_experiment(
            experiment_path,
            runtime_path,
            repo_root=tmp_path,
            run_id="invalid-control-rate",
        )

    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    ("section", "changes"),
    [
        ("policy_server", {"implementation": "vanilla"}),
        ("environment", {"reuse_task_envs": True}),
        ("environment", {"initial_state_strategy": "official_then_seeded"}),
    ],
)
def test_historical_runtime_is_readable_but_cannot_start_a_run(tmp_path, section, changes):
    from vlalagbench.config import Runtime
    from vlalagbench.benchmarks.libero.policy_server import serve

    document = load_runtime(runner.REPO_ROOT / "configs/runtime/libero.yaml").model_dump()
    document[section].update(changes)
    runtime = Runtime.model_validate(document)
    runtime_path = write_yaml(runtime, tmp_path / "runtime.yaml")
    assert load_runtime(runtime_path) == runtime

    experiment_path = runner.REPO_ROOT / "configs/experiments/libero/pi05_libero_spatial.yaml"
    with pytest.raises(ValueError, match="historical LIBERO runtimes are read-only"):
        runner.run_experiment(experiment_path, runtime_path, repo_root=tmp_path)
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "artifacts").exists()
    # Rejection must precede importing OpenPI or loading the checkpoint.
    with pytest.raises(ValueError, match="historical LIBERO runtimes are read-only"):
        serve("pi05_libero", "gs://checkpoint", runtime)


def test_libero_adapter_accepts_dynamic_actors():
    experiment = load_experiment(
        runner.REPO_ROOT / "tests/fixtures/configs/experiments/libero/pi05_libero_spatial_sync_0ms.yaml"
    )

    assert experiment.dynamics
    require_supported_control_hz(experiment.benchmark.control_hz)


def test_runner_normalizes_dynamicvla_without_libero_fields(tmp_path, monkeypatch):
    experiment = load_experiment(
        runner.REPO_ROOT / "configs/experiments/dynamicvla/dom_screen_240ms.yaml"
    )
    runtime = load_runtime(runner.REPO_ROOT / "configs/runtime/dynamicvla.yaml")
    experiment_path = write_yaml(experiment, tmp_path / "configs/experiments/dynamicvla/run.yaml")
    runtime_path = write_yaml(runtime, tmp_path / "configs/runtime/dynamicvla.yaml")

    def fake_evaluate(_self, _experiment, _runtime, _artifact_dir):
        _artifact_dir.mkdir(parents=True, exist_ok=True)
        (_artifact_dir / "dynamicvla_task00_ep000.json").write_bytes(b"trace")
        return [
            {
                "task_id": 0,
                "episode": 0,
                "seed": 11,
                "success": True,
                "steps": 80,
                "control_steps": 83,
                "wait_steps": 3,
                "policy_requests": 14,
                "trace": "dynamicvla_task00_ep000.json",
                "error": None,
            }
        ]

    monkeypatch.setattr(dynamicvla_adapter.DynamicVLAAdapter, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        runner, "_git_state", lambda root: CodeVersion(commit="a" * 40, dirty=False)
    )
    monkeypatch.setattr(runner, "collect_hardware", _hardware)

    result = runner.run_experiment(
        experiment_path, runtime_path, repo_root=tmp_path, run_id="dynamicvla-run"
    )

    assert result.latency.requested_ms == 240
    assert result.latency.effective_steps == 6
    assert result.summary.tasks[0].task_id == 0
    episode = json.loads(
        (tmp_path / "artifacts/dynamicvla-run/episodes.jsonl").read_text()
    )
    assert episode["seed"] == 11


def _hardware():
    return Hardware(
        hostname="gpu-node-1",
        cpu_count=64,
        gpus=[GPUAllocation(type="NVIDIA GeForce RTX 4090", count=1)],
    )
