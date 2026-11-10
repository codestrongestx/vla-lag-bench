import pytest


def test_suite_presets_match_checkpoints_and_native_chunk_length():
    from pathlib import Path
    from vlalagbench.config import Experiment, load_experiment
    root = Path(__file__).resolve().parents[3]
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        experiment = load_experiment(root / f"configs/experiments/libero/gr00t_{suite}.yaml")
        assert experiment.benchmark.task_ids == "all"
        assert experiment.benchmark.initial_state_indices == list(range(5))
        assert experiment.model.checkpoint_subdir == suite
        invalid = experiment.model_dump()
        invalid["inference"]["action_chunk_steps"] = 10
        with pytest.raises(ValueError, match="action_chunk_steps"):
            Experiment.model_validate(invalid)


def test_gr00t_server_uses_experiment_and_separate_environment(tmp_path, monkeypatch):
    from vlalagbench.study import runner
    from vlalagbench.study.spec import resolve
    from tests.test_study_spec import ROOT, _placement, _study
    from tests.benchmarks.libero.test_gr00t import model
    spec = _study(
        name="gr00t-server-test",
        sweep={"inference.latency_ms": [0]},
        override={
            "model": model().model_copy(update={"checkpoint_subdir": "libero_object"}).model_dump(),
            "inference.action_chunk_steps": 16,
        },
        policy={"pytorch_compile_mode": "eager"},
    )
    placement = _placement(output_root=str(tmp_path), policy_python="/test/gr00t/bin/python")
    plan = resolve(spec, placement, ROOT)
    (plan.study_dir / "logs").mkdir(parents=True)
    captured = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda args, **kwargs: captured.append((args, kwargs["env"])))
    runtime = plan.study_dir / "runtimes/shard01.yaml"
    runner._start_server(plan, 1, runtime)
    args, env = captured[0]
    assert args[0] == placement.policy_python
    assert "--experiment" in args and "--openpi-config" not in args
    assert args[args.index("--experiment") + 1].endswith(f"configs/{plan.points[0].name}.yaml")
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "Isaac-GR00T" in env["PYTHONPATH"]
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
