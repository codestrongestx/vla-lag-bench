from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from vlalagbench.benchmarks.dynamicvla import adapter
from vlalagbench.benchmarks.dynamicvla.adapter import (
    ADAPTER,
    _build_manifest,
    _load_episodes,
    _preflight,
    _stop_process,
    _wait_for_processes,
)
from vlalagbench.benchmarks.dynamicvla.tracing import (
    DynamicVLATrace,
    DynamicVLATraceMeta,
)
from vlalagbench.benchmarks.dynamicvla.protocol import (
    POLICY_NOISE_SCHEME,
    derive_episode_seed,
    derive_request_seed,
)
from vlalagbench.config import load_experiment, load_runtime
from vlalagbench.files import local_path
from vlalagbench.runner import REPO_ROOT


def test_adapter_rejects_a_libero_runtime_before_creating_artifacts():
    experiment = load_experiment(
        REPO_ROOT / "configs/experiments/dynamicvla/dom_screen_240ms.yaml"
    )
    runtime = load_runtime(REPO_ROOT / "configs/runtime/libero.yaml")

    with pytest.raises(TypeError, match="require a DynamicVLA runtime"):
        ADAPTER.validate(experiment, runtime)


@pytest.mark.parametrize("batch_size", [2, 3, 10, 20])
def test_adapter_rejects_parallel_environments_before_creating_artifacts(tmp_path, batch_size):
    experiment = load_experiment(
        REPO_ROOT / "configs/experiments/dynamicvla/dom_screen_240ms.yaml"
    )
    runtime = load_runtime(REPO_ROOT / "configs/runtime/dynamicvla.yaml")
    ADAPTER.validate(experiment, runtime)
    output = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="requires replica_batch_size=1"):
        ADAPTER.evaluate(
            experiment, runtime.model_copy(update={"replica_batch_size": batch_size}), output
        )
    assert not output.exists()


@pytest.mark.parametrize("trace_images", [None, False, True])
def test_simulator_stores_camera_frames_only_on_explicit_opt_in(
    tmp_path, monkeypatch, trace_images
):
    experiment = load_experiment(
        REPO_ROOT / "configs/experiments/dynamicvla/dom_screen_240ms.yaml"
    )
    runtime = load_runtime(REPO_ROOT / "configs/runtime/dynamicvla.yaml")
    assert runtime.trace_images is False
    assert "trace_images" not in runtime.model_dump()
    if trace_images is not None:
        runtime = runtime.model_copy(update={"trace_images": trace_images})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    paths = {
        "root": tmp_path,
        "assets": tmp_path,
        "checkpoint": checkpoint,
        "simulator_python": tmp_path / "python",
        "policy_python": tmp_path / "python",
    }
    commands = []

    def popen(command, **_kwargs):
        commands.append(command)
        raise OSError("stop before launching")

    monkeypatch.setattr(adapter, "_discover_isaac_library_paths", lambda _: [])
    monkeypatch.setattr(adapter.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="stop before launching"):
        adapter._run_processes(
            experiment, runtime, paths, tmp_path, tmp_path / "environments.txt",
            tmp_path / "manifest.json", tmp_path / "receipts.jsonl",
        )

    assert ("--trace-images" in commands[0]) is bool(trace_images)
    assert "--save" not in commands[0]


@pytest.mark.parametrize("num_envs", [0, 2, 10, 20])
def test_simulator_cli_rejects_parallel_environments_before_vendor_import(monkeypatch, num_envs):
    from vlalagbench.benchmarks.dynamicvla.simulator import main

    monkeypatch.delenv("VLALAGBENCH_DYNAMICVLA_ROOT", raising=False)
    with pytest.raises(SystemExit) as error:
        main(["--num_envs", str(num_envs)])
    assert error.value.code == 2


def test_preflight_rejects_untracked_upstream_files(tmp_path, monkeypatch):
    experiment = load_experiment(
        REPO_ROOT / "configs/experiments/dynamicvla/dom_screen_240ms.yaml"
    )
    runtime = load_runtime(REPO_ROOT / "configs/runtime/dynamicvla.yaml")
    root = tmp_path / "DynamicVLA"
    assets = tmp_path / "assets"
    checkpoint = tmp_path / "checkpoint"
    environment = assets / "environment.json"
    for path in [
        root / "simulations/evaluate.py",
        root / "scripts/inference.py",
        root / "policies/dynamicvla/modeling_dynamicvla.py",
        environment,
        checkpoint / "config.json",
        checkpoint / "model.safetensors",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (assets / "scenes").mkdir()
    (assets / "objects").mkdir()
    simulator_python = tmp_path / "simulator-python"
    policy_python = tmp_path / "policy-python"
    for executable in [simulator_python, policy_python]:
        executable.touch(mode=0o755)

    experiment = experiment.model_copy(
        update={
            "model": experiment.model.model_copy(update={"checkpoint": str(checkpoint)}),
            "benchmark": experiment.benchmark.model_copy(
                update={
                    "tasks": [
                        experiment.benchmark.tasks[0].model_copy(
                            update={"environments": [environment.name]}
                        )
                    ]
                }
            ),
        }
    )
    runtime = runtime.model_copy(
        update={
            "dynamicvla_root": str(root),
            "assets_root": str(assets),
            "simulator_python": str(simulator_python),
            "policy_python": str(policy_python),
        }
    )
    responses = iter([experiment.benchmark.upstream_revision, "?? sitecustomize.py"])
    monkeypatch.setattr(adapter, "_command", lambda *_args: next(responses))

    with pytest.raises(ValueError, match="tracked or untracked modifications"):
        _preflight(experiment, runtime)


def test_checkout_changes_ignore_stray_python_temp_directories():
    porcelain = "\n".join([
        "?? tmpkwthut_u/_remote_module_non_scriptable.py",
        "?? sitecustomize.py",
        "?? tmp/notes.py",
        " M simulations/evaluate.py",
    ])
    assert adapter.checkout_changes(porcelain) == [
        "?? sitecustomize.py", "?? tmp/notes.py", " M simulations/evaluate.py",
    ]
    assert adapter.checkout_changes("?? tmpkwthut_u/_remote_module_non_scriptable.py") == []


def test_preflight_rejects_environment_symlink_aliases(tmp_path):
    experiment = load_experiment(
        REPO_ROOT / "configs/experiments/dynamicvla/dom_screen_240ms.yaml"
    )
    runtime = load_runtime(REPO_ROOT / "configs/runtime/dynamicvla.yaml")
    environment = tmp_path / "environment.json"
    environment.write_text("{}")
    alias = tmp_path / "alias.json"
    alias.symlink_to(environment)
    experiment = experiment.model_copy(
        update={
            "benchmark": experiment.benchmark.model_copy(
                update={
                    "tasks": [
                        experiment.benchmark.tasks[0].model_copy(
                            update={"environments": [environment.name, alias.name]}
                        )
                    ]
                }
            )
        }
    )
    runtime = runtime.model_copy(update={"assets_root": str(tmp_path)})

    with pytest.raises(ValueError, match="resolve to globally unique files"):
        _preflight(experiment, runtime)


def test_runtime_executable_preserves_virtualenv_symlink(tmp_path):
    system_python = tmp_path / "python3"
    system_python.touch(mode=0o755)
    virtualenv_python = tmp_path / "venv/bin/python"
    virtualenv_python.parent.mkdir(parents=True)
    virtualenv_python.symlink_to(system_python)

    executable = adapter._runtime_executable(virtualenv_python)

    assert executable == virtualenv_python
    assert executable.resolve() == system_python


def test_stop_process_reaps_a_child_that_exits_before_the_signal(monkeypatch):
    waits = []
    process = SimpleNamespace(
        pid=123,
        poll=lambda: None,
        wait=lambda timeout: waits.append(timeout),
    )

    def process_already_exited(*_args):
        raise ProcessLookupError

    monkeypatch.setattr(adapter.os, "killpg", process_already_exited)

    _stop_process(process)

    assert waits == [5]


def test_stop_process_does_not_mask_failure_when_killed_child_cannot_reap(
    monkeypatch,
):
    signals = []
    process = SimpleNamespace(
        pid=123,
        poll=lambda: None,
        wait=lambda timeout: (_ for _ in ()).throw(
            adapter.subprocess.TimeoutExpired("simulator", timeout)
        ),
    )
    monkeypatch.setattr(
        adapter.os,
        "killpg",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
    )

    _stop_process(process)

    assert signals == [(123, adapter.signal.SIGTERM), (123, adapter.signal.SIGKILL)]


def test_policy_failure_waits_for_simulator_failure_evidence(monkeypatch):
    now = [0.0]
    simulator_statuses = iter([None, None, 3])
    simulator = SimpleNamespace(poll=lambda: next(simulator_statuses))
    policy = SimpleNamespace(poll=lambda: 1)
    monkeypatch.setattr(adapter.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        adapter.time,
        "sleep",
        lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    with pytest.raises(RuntimeError, match="simulator exited with status 3"):
        _wait_for_processes(
            simulator,
            policy,
            timeout_s=10.0,
            failure_flush_timeout_s=5.0,
        )

    assert now[0] == 1.0


def test_episode_receipts_become_shared_runner_outputs(tmp_path):
    experiment = load_experiment(
        REPO_ROOT / "configs/experiments/dynamicvla/dom_screen_240ms.yaml"
    )
    experiment = experiment.model_copy(
        update={
            "benchmark": experiment.benchmark.model_copy(
                update={
                    "tasks": [
                        experiment.benchmark.tasks[0].model_copy(
                            update={"environments": ["example.json", "second.json"]}
                        )
                    ],
                    "episodes_per_environment": 1,
                }
            )
        }
    )
    environment = tmp_path / "example.json"
    environment.write_text(
        json.dumps({"seed": 7, "instruction": "test", "episode_length_s": 8.0})
    )
    second_environment = tmp_path / "second.json"
    second_environment.write_text(
        json.dumps({"seed": 8, "instruction": "second", "episode_length_s": 8.0})
    )
    manifest = _build_manifest(experiment, tmp_path)
    episode_seed = derive_episode_seed(
        experiment.seeds.policy,
        experiment.benchmark.tasks[0].name,
        manifest["episodes"][0]["environment_sha256"],
        0,
    )
    request_events = [
        {
            "step": source,
            "request_id": request_id,
            "observation_step": source,
            "release_step": source + 6,
            "request_seed": derive_request_seed(episode_seed, request_id),
            "noise_sha256": f"{request_id:064x}",
        }
        for request_id, source in enumerate([5, 11, 17])
    ]
    release_events = [
        {**event, "step": event["release_step"], "chunk_index": event["request_id"]}
        for event in request_events[:2]
    ]
    applied_events = [
        {
            "step": step,
            "chunk_index": (step - 11) // 6,
            "observation_step": 5 + 6 * ((step - 11) // 6),
            "action_index": 6 + (step - 11) % 6,
        }
        for step in range(11, 20)
    ]
    control_events = [
        {
            "step": step,
            "action": [[0.0] * 8],
            "wait_reason": (
                "policy_warmup" if step < 5 else "inference_wait" if step < 11 else "model"
            ),
            "chunk_index": (
                applied_events[step - 11]["chunk_index"] if step >= 11 else None
            ),
            "observation_step": (
                applied_events[step - 11]["observation_step"] if step >= 11 else None
            ),
            "action_index": (
                applied_events[step - 11]["action_index"] if step >= 11 else None
            ),
        }
        for step in range(20)
    ]
    receipt = {
        "task_id": 0,
        "task_name": "dom-nine-dimension-screen",
        "episode": 0,
        "environment_index": 0,
        "replica_id": 0,
        "environment": str(environment),
        "seed": 7,
        "instruction": "test",
        "episode_seed": episode_seed,
        "success": True,
        "status": 0,
        "mode": "continuous",
        "action_index_origin": "observation",
        "execution_horizon_steps": 14,
        "execution_horizon_basis": "post_latency_action_count",
        "injected_steps": 6,
        "n_steps": 20,
        "applied": 9,
        "coverage": 0.45,
        "startup_gap": 11,
        "steady_gaps": 0,
        "gap_steps": list(range(11)),
        "requests": 3,
        "request_events": request_events,
        "releases": 2,
        "release_events": release_events,
        "applied_events": applied_events,
        "control_events": control_events,
        "staleness_hist": dict(
            Counter(
                event["step"] - event["observation_step"]
                for event in applied_events
            )
        ),
        "blocked_s": 4.0,
        "violations": 0,
        "timeouts": 0,
        "errors": [],
        "exact": True,
        "trace": "dynamicvla_task00_ep000.npz",
    }
    trace_path = DynamicVLATrace(
        meta=DynamicVLATraceMeta.model_validate(
            {
                "schema_version": 7,
                "task_id": 0,
                "task_name": "dom-nine-dimension-screen",
                "episode": 0,
                "environment_index": 0,
                "replica_id": 0,
                "environment_path": str(environment),
                "environment_sha256": adapter._sha256(environment),
                "environment": {
                    "seed": 7,
                    "instruction": "test",
                    "episode_length_s": 8.0,
                },
                "asset_sha256": {},
                "instruction": "test",
                "seed": 7,
                "upstream_revision": experiment.benchmark.upstream_revision,
                "checkpoint": str(local_path(experiment.model.checkpoint).resolve()),
                "checkpoint_sha256": experiment.model.checkpoint_sha256,
                "checkpoint_config_sha256": "f" * 64,
                "policy_noise_scheme": POLICY_NOISE_SCHEME,
                "policy_torch_version": "2.4.0",
                "policy_seed": experiment.seeds.policy,
                "policy_episode_seed": episode_seed,
                "episode_length_s": 8.0,
                "control_hz": 25.0,
                "physics_time_step": 0.04,
                "termination_tolerance": 0.07,
                "disable_fabric": False,
                "path_tracing": False,
                "simulator_versions": {"isaaclab": "2.2.1", "isaacsim": "4.5.0"},
                "inference_mode": "continuous",
                "action_index_origin": "observation",
                "latency_steps": 6,
                "action_chunk_steps": 20,
                "execution_horizon_steps": 14,
                "execution_horizon_basis": "post_latency_action_count",
                "success_term": "success",
                "timeout_terms": ["time_out"],
                "num_steps": 20,
                "status": 0,
                "success": True,
                "recorded_at": "2026-08-28T00:00:00+00:00",
                "python_version": "3.11",
                "numpy_version": np.__version__,
            }
        ),
        states={("scene", "rigid_object", "object", "root_pose"): np.zeros((21, 1, 7))},
        observations={
            ("images", "front"): np.zeros((21, 1, 1, 1, 3), dtype=np.uint8),
            ("state", "end_effector", "pos"): np.zeros((21, 1, 3)),
        },
        actions=np.zeros((20, 1, 8)),
        success=np.asarray([False] * 19 + [True]),
        done=np.asarray([False] * 19 + [True]),
        terminations=[
            {"success": False, "time_out": False} for _ in range(19)
        ]
        + [{"success": True, "time_out": False}],
        protocol=[
            {
                "request": next(
                    (
                        {key: value for key, value in event.items() if key != "step"}
                        for event in request_events
                        if event["step"] == step
                    ),
                    None,
                )
            }
            for step in range(20)
        ],
        final_policy={},
    ).save(tmp_path / receipt["trace"])
    receipt["trace_sha256"] = adapter._sha256(trace_path)
    second_trace_name = "dynamicvla_task00_ep001.npz"
    second_episode_seed = derive_episode_seed(
        experiment.seeds.policy,
        experiment.benchmark.tasks[0].name,
        manifest["episodes"][1]["environment_sha256"],
        0,
    )
    second_trace = DynamicVLATrace.load(trace_path)
    second_trace.meta = second_trace.meta.model_copy(
        update={
            "episode": 1,
            "environment_index": 1,
            "environment_path": str(second_environment),
            "environment_sha256": adapter._sha256(second_environment),
            "environment": {
                "seed": 8,
                "instruction": "second",
                "episode_length_s": 8.0,
            },
            "instruction": "second",
            "seed": 8,
            "policy_episode_seed": second_episode_seed,
            "num_steps": 4,
            "status": 2,
            "success": False,
        }
    )
    second_trace.states = {
        path: values[:5] for path, values in second_trace.states.items()
    }
    second_trace.observations = {
        path: values[:5] for path, values in second_trace.observations.items()
    }
    second_trace.actions = np.zeros((4, 1, 8))
    second_trace.success = np.zeros(4, dtype=np.bool_)
    second_trace.done = np.asarray([False, False, False, True])
    second_trace.terminations = [
        {"success": False, "time_out": False} for _ in range(3)
    ] + [{"success": False, "time_out": True}]
    second_trace.protocol = [
        {
            "request": None,
            "release": None,
            "wait_reason": "policy_warmup",
            "chunk_index": None,
            "observation_step": None,
            "action_index": None,
            "blocked_s": 0.0,
        }
        for _ in range(4)
    ]
    second_trace.final_policy = {}
    second_trace_path = second_trace.save(tmp_path / second_trace_name)
    second_receipt = {
        **receipt,
        "episode": 1,
        "environment_index": 1,
        "environment": str(second_environment),
        "seed": 8,
        "instruction": "second",
        "episode_seed": second_episode_seed,
        "success": False,
        "status": 2,
        "n_steps": 4,
        "applied": 0,
        "coverage": 0.0,
        "startup_gap": 4,
        "steady_gaps": 0,
        "gap_steps": list(range(4)),
        "requests": 0,
        "request_events": [],
        "releases": 0,
        "release_events": [],
        "applied_events": [],
        "control_events": [
            {
                "step": step,
                "action": [[0.0] * 8],
                "wait_reason": "policy_warmup",
                "chunk_index": None,
                "observation_step": None,
                "action_index": None,
            }
            for step in range(4)
        ],
        "staleness_hist": {},
        "blocked_s": 0.0,
        "trace": second_trace_name,
        "trace_sha256": adapter._sha256(second_trace_path),
    }
    receipts = tmp_path / "receipts.jsonl"
    receipts.write_text(
        json.dumps(receipt) + "\n" + json.dumps(second_receipt) + "\n"
    )

    episodes = _load_episodes(
        experiment, receipts, tmp_path, manifest=manifest
    )

    assert episodes == [
        {
            "task_id": 0,
            "episode": 0,
            "seed": 7,
            "success": True,
            "status": 0,
            "steps": 9,
            "control_steps": 20,
            "wait_steps": 11,
            "policy_requests": 3,
            "trace": "dynamicvla_task00_ep000.npz",
            "error": None,
        },
        {
            "task_id": 0,
            "episode": 1,
            "seed": 8,
            "success": False,
            "status": 2,
            "steps": 0,
            "control_steps": 4,
            "wait_steps": 4,
            "policy_requests": 0,
            "trace": "dynamicvla_task00_ep001.npz",
            "error": None,
        },
    ]
    assert DynamicVLATrace.load(tmp_path / episodes[0]["trace"]).meta.success is True

    receipt["instruction"] = "different prompt"
    receipts.write_text(
        json.dumps(receipt) + "\n" + json.dumps(second_receipt) + "\n"
    )
    with pytest.raises(RuntimeError, match="wrong literal prompt"):
        _load_episodes(experiment, receipts, tmp_path, manifest=manifest)


def test_episode_receipts_must_match_the_configured_environment(tmp_path):
    experiment = load_experiment(
        REPO_ROOT / "configs/experiments/dynamicvla/dom_screen_240ms.yaml"
    )
    experiment = experiment.model_copy(
        update={
            "benchmark": experiment.benchmark.model_copy(
                update={
                    "tasks": [
                        experiment.benchmark.tasks[0].model_copy(
                            update={"environments": ["expected.json"]}
                        )
                    ],
                    "episodes_per_environment": 1,
                }
            )
        }
    )
    expected_environment = tmp_path / "expected.json"
    expected_environment.write_text(json.dumps({"seed": 7, "instruction": "test"}))
    receipt = {
        "task_id": 0,
        "task_name": "dom-nine-dimension-screen",
        "episode": 0,
        "environment_index": 0,
        "replica_id": 0,
        "environment": str(tmp_path / "other.json"),
    }
    receipts = tmp_path / "receipts.jsonl"
    receipts.write_text(json.dumps(receipt) + "\n")

    with pytest.raises(RuntimeError, match="used .*other.json, expected .*expected.json"):
        _load_episodes(
            experiment,
            receipts,
            tmp_path,
            manifest=_build_manifest(experiment, tmp_path),
        )


def test_episode_validation_rejects_environment_changed_after_manifest(tmp_path):
    experiment = load_experiment(
        REPO_ROOT / "configs/experiments/dynamicvla/dom_screen_240ms.yaml"
    )
    experiment = experiment.model_copy(
        update={
            "benchmark": experiment.benchmark.model_copy(
                update={
                    "tasks": [
                        experiment.benchmark.tasks[0].model_copy(
                            update={"environments": ["example.json"]}
                        )
                    ],
                    "episodes_per_environment": 1,
                }
            )
        }
    )
    environment = tmp_path / "example.json"
    environment.write_text(json.dumps({"seed": 7, "instruction": "before"}))
    manifest = _build_manifest(experiment, tmp_path)
    environment.write_text(json.dumps({"seed": 7, "instruction": "after"}))
    receipts = tmp_path / "receipts.jsonl"
    receipts.write_text(
        json.dumps(
            {
                **manifest["episodes"][0],
                "episode": 0,
                "environment": str(environment),
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="changed after manifest creation"):
        _load_episodes(experiment, receipts, tmp_path, manifest=manifest)


@pytest.mark.parametrize(("task_count", "expected"), [(1, 200), (9, 1800)])
def test_manifest_resolves_complete_environment_replica_cells(
    tmp_path, task_count, expected
):
    experiment = load_experiment(
        REPO_ROOT / "configs/experiments/dynamicvla/dom_screen_240ms.yaml"
    )
    task_template = experiment.benchmark.tasks[0]
    tasks = []
    for task_id in range(task_count):
        environments = []
        for environment_index in range(10):
            relative = f"task-{task_id}/environment-{environment_index}.json"
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "task": task_id,
                        "environment": environment_index,
                        "seed": environment_index,
                        "instruction": f"task {task_id} environment {environment_index}",
                    }
                )
            )
            environments.append(relative)
        tasks.append(
            task_template.model_copy(
                update={"name": f"task-{task_id}", "environments": environments}
            )
        )
    experiment = experiment.model_copy(
        update={
            "benchmark": experiment.benchmark.model_copy(
                update={"tasks": tasks, "episodes_per_environment": 20}
            )
        }
    )

    manifest = _build_manifest(experiment, tmp_path)
    cells = {
        (
            episode["task_id"],
            episode["environment_index"],
            episode["replica_id"],
            episode["episode_id"],
        )
        for episode in manifest["episodes"]
    }

    assert len(manifest["episodes"]) == expected
    assert len(cells) == expected
    assert [task["resolved_episode_count"] for task in manifest["tasks"]] == [
        200
    ] * task_count
    assert all(
        episode["episode_id"]
        == episode["environment_index"] * 20 + episode["replica_id"]
        for episode in manifest["episodes"]
    )
    assert all(
        episode["seed"] == episode["environment_index"]
        and episode["instruction"]
        == f"task {episode['task_id']} environment {episode['environment_index']}"
        for episode in manifest["episodes"]
    )


def test_active_budget_preserves_environment_prompt_and_noise_identity(tmp_path):
    experiment = load_experiment(
        REPO_ROOT / "configs/experiments/dynamicvla/dom_screen_240ms.yaml"
    )
    for task in experiment.benchmark.tasks:
        for relative in task.environments:
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"seed": 7, "instruction": "paired prompt", "episode_length_s": 12.0}))
    original = _build_manifest(experiment, tmp_path)
    for budget in (300, 750):
        changed = experiment.model_copy(update={"benchmark": experiment.benchmark.model_copy(
            update={"max_episode_steps": budget})})
        manifest = _build_manifest(changed, tmp_path)
        for episode in manifest["episodes"]:
            assert episode.pop("max_episode_steps") == budget
        assert manifest == original
    assert "max_episode_steps" not in experiment.benchmark.model_dump()
