"""Exercise the unified study lifecycle without starting Isaac or a policy process."""

import json
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import yaml

from vlalagbench.cli import main
from vlalagbench.config import Videos, DynamicVLAPlacement, DynamicVLAStudyPolicy, StudySpec, load_experiment, load_runtime, write_yaml
from vlalagbench.study import runner, videos
from vlalagbench.study.point import read_point, verify_files, verify_point
from vlalagbench.study.spec import load_placement, load_study, open_study, resolve, shard_runtime
from vlalagbench.benchmarks.dynamicvla import study as backend
from vlalagbench.files import sha256
from tests.benchmarks.dynamicvla.test_protocol import _scheduled_receipt

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "configs/studies/dynamicvla_screen.yaml"
PLACEMENT = ROOT / "configs/placements/dynamicvla.yaml"


def test_fcan04_pilot_and_remainder_cover_existing_200_conditions():
    pilot = ROOT / "configs/studies/fcan04_native_pilot20_20260928"
    remainder = ROOT / "configs/studies/fcan04_native_remaining180_20260928"
    mapping = json.loads((remainder / "selection.json").read_text())
    source = load_experiment(ROOT / mapping["source"])
    pilot_ids = mapping["pilot_source_episode_ids"]
    remaining_ids = mapping["remaining_source_episode_ids"]
    assert len(pilot_ids) == 20 and len(remaining_ids) == 180
    assert sorted(pilot_ids + remaining_ids) == list(range(200))
    for directory, ids in ((pilot, pilot_ids), (remainder, remaining_ids)):
        experiment = load_experiment(directory / "base.yaml")
        assert experiment.benchmark.episodes_per_environment == len(ids)
        assert experiment.benchmark.initial_states_per_environment == len(ids)
        assert len(experiment.initial_state_overrides) == len(ids)
        assert len(experiment.trajectory.episode_schedule) == len(ids)
        assert experiment.trajectory.keyframes == source.trajectory.keyframes
        for local, original in enumerate(ids):
            placement = experiment.initial_state_overrides[local]
            expected = source.initial_state_overrides[original % 50]
            assert placement.initial_state_index == local
            assert placement.environment == experiment.benchmark.tasks[0].environments[0]
            assert placement.object_positions_m == expected.object_positions_m
            assert placement.object_yaws_rad == expected.object_yaws_rad
            assert experiment.trajectory.episode_schedule[local] == source.trajectory.episode_schedule[original]

    total = 0
    for mode, cells in (("synchronous", 21), ("continuous", 14), ("lazy", 14)):
        old = resolve(load_study(pilot / f"{mode}.yaml"), load_placement(PLACEMENT), ROOT)
        new = resolve(load_study(remainder / f"{mode}.yaml"), load_placement(PLACEMENT), ROOT)
        assert len(old.points) == len(new.points) == cells
        assert old.spec.policy == new.spec.policy
        assert old.study_dir != new.study_dir
        by_name = lambda plan: sorted(plan.points, key=lambda point: point.name)
        for before, after in zip(by_name(old), by_name(new), strict=True):
            assert before.values == after.values
            configs = [point.experiment.model_dump() for point in (before, after)]
            for config in configs:
                config.pop("name")
                config.pop("initial_state_overrides")
                config["trajectory"].pop("episode_schedule")
                config["benchmark"].pop("episodes_per_environment")
                config["benchmark"].pop("initial_states_per_environment")
            assert configs[0] == configs[1]
        runtime = shard_runtime(new, 0)
        assert not runtime.trace_images and not runtime.save_videos
        total += cells * 180
    assert total == 8820


def plan_at(tmp_path, **changes):
    spec = load_study(STUDY).model_dump()
    spec.update(sweep={"inference.latency_ms": [0, 120]}, **changes)
    spec["override"] = {"benchmark.episodes_per_environment": 1,
        "benchmark.tasks": [{"name": "test", "environments": ["tests/one.json"]}]}
    placement = load_placement(PLACEMENT).model_copy(update={"output_root": str(tmp_path)})
    return resolve(StudySpec.model_validate(spec), placement, ROOT)


def test_cli_plan_needs_no_gpu_or_vendor_checkout(capsys, tmp_path):
    spec = load_study(STUDY).model_dump()
    spec["base_experiment"] = str(ROOT / spec["base_experiment"])
    spec["sweep"] = {"inference.latency_ms": [0, 120, 240]}
    valid_study = tmp_path / "study.yaml"
    write_yaml(StudySpec.model_validate(spec), valid_study)
    assert main(["study", "plan", str(valid_study), "--placement", str(PLACEMENT),
                 "--select", "inference.latency_ms=240"]) == 0
    output = capsys.readouterr().out
    assert "1 of 3 queued" in output
    assert "simulator gpu 0" in output


def test_study_identity_freezes_camera_and_batch_semantics(tmp_path):
    plan = plan_at(tmp_path)
    assert isinstance(plan.spec.policy, DynamicVLAStudyPolicy)
    runtime = shard_runtime(plan, 0)
    assert not runtime.save_videos
    assert not runtime.trace_images
    assert runtime.antialiasing_mode == "DLSS"
    assert (runtime.policy_gpu, runtime.simulator_gpu) == (0, 0)
    changed = plan_at(tmp_path, policy={"kind": "dynamicvla", "antialiasing_mode": "FXAA"})
    assert changed.study_hash != plan.study_hash
    placement = plan.placement.model_copy(update={"output_root": str(tmp_path / "elsewhere")})
    assert resolve(plan.spec, placement, ROOT).study_hash == plan.study_hash


def test_committed_dynamicvla_configs_keep_camera_frames_out_of_runs():
    runtimes = sorted((ROOT / "configs/runtime").glob("dynamicvla*.yaml"))
    assert runtimes
    for path in runtimes:
        runtime = load_runtime(path)
        assert not runtime.trace_images and not runtime.save_videos, path
    for path in sorted((ROOT / "configs/placements").glob("*.yaml")):
        if yaml.safe_load(path.read_text()).get("kind") == "dynamicvla":
            assert not load_placement(path).installation.trace_images, path


@pytest.mark.parametrize("change,match", [
    ({"policy": {}}, "policy.kind"),
    ({"policy": {"kind": "dynamicvla", "replica_batch_size": 2}}, "replica_batch_size"),
])
def test_plan_rejects_incompatible_policy(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        plan_at(tmp_path, **change)


def test_plan_rejects_overlapping_gpu_and_port_ownership(tmp_path):
    plan = plan_at(tmp_path)
    data = plan.placement.model_dump()
    data["shards"].append(dict(data["shards"][0]))
    with pytest.raises(ValueError, match="ports"):
        resolve(plan.spec, DynamicVLAPlacement.model_validate(data), ROOT)
    data["shards"][1].update(image_port=4100, action_port=4101)
    with pytest.raises(ValueError, match="share GPUs"):
        resolve(plan.spec, DynamicVLAPlacement.model_validate(data), ROOT)


def test_plan_allows_explicit_gpu_sharing_without_changing_study_identity(tmp_path):
    plan = plan_at(tmp_path)
    data = plan.placement.model_dump()
    data.update(allow_gpu_sharing=True, gpu_budget=1)
    data["shards"][0]["cpus"] = "0-3"
    data["shards"].append({**data["shards"][0], "image_port": 4100,
                           "action_port": 4101, "cpus": "4-7"})
    shared = resolve(plan.spec, DynamicVLAPlacement.model_validate(data), ROOT)
    assert shared.study_hash == plan.study_hash
    assert len(shared.placement.shards) == 2
    first, second = (shard_runtime(shared, index) for index in range(2))
    assert first.policy_gpu == second.policy_gpu == 0
    assert first.simulator_gpu == second.simulator_gpu == 0
    assert first.image_port != second.image_port
    assert first.action_port != second.action_port


@pytest.mark.parametrize("change,match", [
    ({"cpus": "3-7"}, "disjoint CPU"),
    ({"action_port": 4100}, "ports"),
    ({"simulator_gpu": 1}, "gpu_budget"),
])
def test_shared_gpu_placement_retains_resource_guards(tmp_path, change, match):
    plan = plan_at(tmp_path)
    data = plan.placement.model_dump()
    data.update(allow_gpu_sharing=True, gpu_budget=1)
    data["shards"][0]["cpus"] = "0-3"
    data["shards"].append({**data["shards"][0], "image_port": 4100,
                           "action_port": 4101, "cpus": "4-7", **change})
    with pytest.raises(ValueError, match=match):
        resolve(plan.spec, DynamicVLAPlacement.model_validate(data), ROOT)


def shared_backend_placement(plan, **second):
    data = plan.placement.model_dump()
    data.update(allow_gpu_sharing=True, gpu_budget=2)
    data["shards"][0].update(cpus="0-3", shared_endpoint="tcp://127.0.0.1:4200")
    data["shards"].append({**data["shards"][0], "image_port": 4100, "action_port": 4101,
                           "cpus": "4-7", **second})
    return data


def test_shared_backend_placement_keeps_study_identity(tmp_path):
    plan = plan_at(tmp_path)
    shared = resolve(plan.spec, DynamicVLAPlacement.model_validate(shared_backend_placement(plan)), ROOT)
    assert shared.study_hash == plan.study_hash
    assert {slot.shared_endpoint for slot in shared.placement.shards} == {"tcp://127.0.0.1:4200"}
    assert "shared_endpoint" not in plan.placement.model_dump(exclude_unset=False, mode="json")["shards"][0]


@pytest.mark.parametrize("change,match", [
    ({"policy_gpu": 1, "simulator_gpu": 1}, "same policy_gpu"),
    ({"shared_endpoint": "tcp://127.0.0.1:4101"}, "reuse shard ports"),
    ({"shared_endpoint": "tcp://0.0.0.0:4200"}, "pattern"),
])
def test_shared_backend_placement_guards(tmp_path, change, match):
    plan = plan_at(tmp_path)
    with pytest.raises(ValueError, match=match):
        resolve(plan.spec, DynamicVLAPlacement.model_validate(shared_backend_placement(plan, **change)), ROOT)


def fake_evaluate(experiment, runtime, artifact_dir):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    receipt = _scheduled_receipt(experiment.effective_latency_steps, 20, 12, execution_horizon_steps=experiment.resolved_execution_horizon_steps)
    receipt.update(task_id=0, episode=0, exact=True, errors=[], trace="episode.npz")
    meta = {"task_id": 0, "episode": 0, "task_name": "test", "environment_index": 0,
        "replica_id": 0, "success": False, "status": 1, "num_steps": receipt["n_steps"],
        "upstream_revision": experiment.benchmark.upstream_revision,
        "checkpoint_sha256": experiment.model.checkpoint_sha256,
        "policy_seed": experiment.seeds.policy, "inference_mode": "continuous",
        "action_index_origin": "observation", "latency_steps": experiment.effective_latency_steps,
        "action_chunk_steps": 20, "execution_horizon_steps": experiment.resolved_execution_horizon_steps,
        "execution_horizon_basis": "post_latency_action_count",
        "control_hz": 25.0}
    np.savez_compressed(artifact_dir / "episode.npz", meta=json.dumps(meta),
        observation_manifest=json.dumps([{"path": ["images", "opst_cam"], "key": "rgb"}]),
        rgb=np.zeros((receipt["n_steps"] + 1, 8, 8, 3), dtype=np.uint8))
    receipt["trace_sha256"] = sha256(artifact_dir / "episode.npz")
    (artifact_dir / "dynamicvla-manifest.json").write_text(json.dumps({"episodes": [{"task_id": 0, "episode_id": 0}]}))
    (artifact_dir / "dynamicvla-receipts.jsonl").write_text(json.dumps(receipt) + "\n")
    return [{"task_id": 0, "episode": 0, "seed": 7, "success": False, "status": 1,
        "steps": receipt["applied"], "control_steps": receipt["n_steps"],
        "wait_steps": receipt["n_steps"] - receipt["applied"], "policy_requests": receipt["requests"],
        "trace": "episode.npz", "error": None}]


@pytest.fixture
def completed(tmp_path, monkeypatch):
    from vlalagbench.benchmarks.dynamicvla.adapter import DynamicVLAAdapter
    from vlalagbench.config import Hardware, GPUAllocation
    import vlalagbench.runner as evaluation

    plan = plan_at(tmp_path)
    monkeypatch.setattr(runner, "plan_execution_identity", lambda plan: {"test": "fixed"})
    monkeypatch.setattr(runner.os, "sched_setaffinity", lambda *args: None, raising=False)
    monkeypatch.setattr(evaluation, "collect_hardware", lambda: Hardware(
        hostname="test", cpu_count=8, gpus=[GPUAllocation(type="test", count=1)]))
    monkeypatch.setattr(DynamicVLAAdapter, "evaluate", lambda self, *args: fake_evaluate(*args))
    monkeypatch.setattr(runner, "_start_server", lambda *args: pytest.fail("LIBERO server used for DynamicVLA"))
    runner.prepare(plan)
    assert runner.run_shard(plan, 0) == 0
    assert all(runner.state(plan.study_dir, point.name) == ("done", None) for point in plan.points)
    return plan


def test_run_verify_resume_report_and_state_replay_video(completed, monkeypatch, capsys):
    plan = completed
    reopened = open_study(plan.study_dir)
    assert reopened.study_hash == plan.study_hash
    assert main(["study", "verify", str(plan.study_dir)]) == 0
    assert main(["study", "status", str(plan.study_dir)]) == 0
    assert main(["study", "report", str(plan.study_dir)]) == 0
    assert (plan.study_dir / "success_rates.png").is_file()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("completed run started a process"))
    assert runner.run(plan, STUDY, PLACEMENT, []) == 0
    point = plan.points[0]
    directory = plan.study_dir / "points" / point.name
    record = read_point(directory)
    assert record.deadline_steps is None
    assert any(path.endswith("dynamicvla-receipts.jsonl") for path in record.files)
    verify_files(directory, record)
    # Videos replay saved states in Isaac with the point's recorded runtime, even when a trace holds RGB.
    from vlalagbench.benchmarks.dynamicvla import trace_runner
    from vlalagbench.config import Videos
    calls = []
    rendering_options = []
    def replay(trace, runtime, *, video, camera, simulator_gpu, **kwargs):
        calls.append((trace, runtime, camera, simulator_gpu))
        rendering_options.append(kwargs)
        video.write_bytes(b"video")
        video.with_name(video.stem + "-report.json").write_text("{}")
        if kwargs.get("separate_cameras"):
            outputs = {}
            for name in ("opst_cam", "side_cam", "wrist_cam"):
                output = video.with_name(f"video_{name}.mp4")
                output.write_bytes(name.encode())
                outputs[name] = str(output)
            return {"videos": outputs}
    monkeypatch.setattr(trace_runner, "run_trace_operations", replay)
    monkeypatch.setattr(videos, "ProcessPoolExecutor", lambda **kwargs: ThreadPoolExecutor(max_workers=kwargs["max_workers"]))
    manifest = videos.film_point(directory, point.name, Videos(clips_per_outcome=1, cameras=["opst_cam"], resolution=256), 1, gpu=3)
    assert manifest.complete and manifest.source == "simulator_replay" and manifest.resolution is None
    assert calls == [(directory / "artifacts" / record.run_id / "episode.npz", directory / "runs" / record.run_id / "runtime.yaml", "opst_cam", 3)]
    assert sorted(path.name for path in (directory / "videos").iterdir() if path.suffix != ".yaml") == [manifest.clips[0].path.split("/")[-1]]
    quality = Videos(clips_per_outcome=1, cameras=["opst_cam", "side_cam", "wrist_cam"], resolution=256,
                     dynamicvla_path_tracing_spp=64)
    updated = videos.film_point(directory, point.name, quality, 1, gpu=3)
    assert updated.settings_hash != manifest.settings_hash
    assert len(calls) == 2 and calls[0][:2] == calls[1][:2]
    assert calls[1][2] is None and len(updated.clips) == 3
    assert rendering_options[-1]["video_path_tracing_spp"] == 64
    videos.film_point(directory, point.name, quality, 1, gpu=3)
    assert len(calls) == 2


@pytest.mark.parametrize("override_gpu", [None, 3])
def test_copied_study_replay_uses_local_runtime_and_locks_local_gpu(completed, monkeypatch, tmp_path, override_gpu):
    from vlalagbench.benchmarks.dynamicvla import trace_runner

    plan = completed
    point = plan.queue[0]
    directory = plan.study_dir / "points" / point.name
    record = read_point(directory)
    recorded = directory / "runs" / record.run_id / "runtime.yaml"
    original_bytes = recorded.read_bytes()
    local = tmp_path / "local-runtime.yaml"
    runtime = load_runtime(recorded).model_copy(update={
        "dynamicvla_root": str(tmp_path / "local-source"),
        "assets_root": str(tmp_path / "local-assets"),
        "simulator_gpu": 7, "policy_gpu": 6,
    })
    write_yaml(runtime, local)
    locked = []
    calls = []
    expected_gpu = 0 if override_gpu is None else override_gpu

    @contextmanager
    def lock(gpus):
        assert gpus == [expected_gpu]
        assert not locked
        locked.extend(gpus)
        try:
            yield
        finally:
            locked.clear()

    def replay(trace, runtime_path, *, video, simulator_gpu, **kwargs):
        assert locked == [expected_gpu]
        assert simulator_gpu == expected_gpu
        assert runtime_path == local
        assert load_runtime(runtime_path).assets_root == str(tmp_path / "local-assets")
        calls.append(trace)
        video.write_bytes(b"video")

    monkeypatch.setattr(runner, "gpu_locks", lock)
    monkeypatch.setattr(trace_runner, "run_trace_operations", replay)
    monkeypatch.setattr(videos, "ProcessPoolExecutor", lambda **kwargs: ThreadPoolExecutor(max_workers=kwargs["max_workers"]))
    args = ["study", "videos", str(plan.study_dir), "--runtime", str(local), "--workers", "1"]
    if override_gpu is not None:
        args.extend(["--gpu", str(override_gpu)])
    assert main(args) == 0
    assert len(calls) == len(plan.queue)
    assert recorded.read_bytes() == original_bytes
    verify_files(directory, record)
    assert not locked
    # Placement changes reuse clips, but a renderer change must regenerate them.
    write_yaml(runtime.model_copy(update={"simulator_gpu": 5}), local)
    assert main(args) == 0
    assert len(calls) == len(plan.queue)
    write_yaml(runtime.model_copy(update={"antialiasing_mode": "FXAA"}), local)
    assert main(args) == 0
    assert len(calls) == 2 * len(plan.queue)


def test_verification_rejects_receipt_and_runtime_drift(completed):
    plan = completed
    point = plan.points[0]
    directory = plan.study_dir / "points" / point.name
    record = read_point(directory)
    receipt = directory / "artifacts" / record.run_id / "dynamicvla-receipts.jsonl"
    row = json.loads(receipt.read_text())
    row["status"] = 0
    receipt.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="digest"):
        verify_files(directory, record)
    with pytest.raises(ValueError, match="receipt"):
        verify_point(directory, name=point.name, values=point.values, experiment=point.experiment,
                     study_hash=plan.study_hash, policy=plan.spec.policy)
    runtime = shard_runtime(plan, 0).model_copy(update={"antialiasing_mode": "FXAA"})
    with pytest.raises(ValueError, match="semantics"):
        backend.verify_runtime(runtime, plan.spec.policy)


def test_identity_does_not_probe_another_running_shards_ports(tmp_path, monkeypatch):
    from types import SimpleNamespace

    plan = plan_at(tmp_path)
    calls = []
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    def preflight(experiment, runtime, *, check_ports=True):
        calls.append(check_ports)
        if check_ports:
            raise ValueError("another shard already owns these ports")
        return {"checkpoint": checkpoint, "policy_python": "/policy/python",
                "simulator_python": "/simulator/python", "manifest": {"environments": []}}
    monkeypatch.setattr(backend, "_preflight", preflight)
    monkeypatch.setattr(backend.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout='{"packages": {}}'))
    identity = backend.execution_identity(plan.points[0].experiment, shard_runtime(plan, 0), ROOT)
    assert calls == [False]
    assert set(identity["stacks"]) == {"policy_python", "simulator_python"}


def test_adapter_assigns_each_child_its_own_gpu_and_cleans_up_on_failure(tmp_path, monkeypatch):
    from vlalagbench.benchmarks.dynamicvla import adapter

    plan = plan_at(tmp_path)
    runtime = shard_runtime(plan, 0).model_copy(update={"policy_gpu": 2, "simulator_gpu": 5, "enable_dl_denoiser": False})
    paths = {key: tmp_path for key in ("root", "assets", "checkpoint", "policy_python", "simulator_python")}
    (tmp_path / "config.json").write_text("{}")
    started, stopped = [], []
    def start(command, **kwargs):
        started.append((command, kwargs))
        return len(started)
    monkeypatch.setattr(adapter.subprocess, "Popen", start)
    monkeypatch.setattr(adapter, "_discover_isaac_library_paths", lambda path: [])
    monkeypatch.setattr(adapter, "_wait_for_ports", lambda *args: None)
    def fail(*args, **kwargs):
        raise RuntimeError("simulator failed")
    monkeypatch.setattr(adapter, "_wait_for_processes", fail)
    monkeypatch.setattr(adapter, "_stop_process", stopped.append)
    with pytest.raises(RuntimeError, match="simulator failed"):
        adapter._run_processes(plan.points[0].experiment, runtime, paths, tmp_path,
            tmp_path / "environments.txt", tmp_path / "manifest.json", tmp_path / "receipts.jsonl")
    assert [kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _, kwargs in started] == ["5", "2"]
    kit_args = started[0][0][started[0][0].index("--kit_args") + 1]
    assert "--/renderer/multiGpu/enabled=false" in kit_args
    assert "--/renderer/activeGpu=5" in kit_args
    assert "--no-enable-dl-denoiser" in started[0][0]
    assert stopped == [2, 1]


def test_failed_point_is_retained_and_retried(completed):
    plan = completed
    point = plan.points[0]
    directory = plan.study_dir / "points" / point.name
    record = read_point(directory)
    result = directory / "runs" / record.run_id / "result.yaml"
    data = yaml.safe_load(result.read_text())
    data.update(status="failed", error="interrupted")
    result.write_text(yaml.safe_dump(data))
    marker = plan.study_dir / "queue" / f"{point.name}.done"
    marker.rename(marker.with_suffix(".todo"))
    assert runner.run_shard(plan, 0) == 0
    assert runner.state(plan.study_dir, point.name) == ("done", None)
    saved = list((plan.study_dir / "failed").glob(f"{point.name}-*/runs/*/result.yaml"))
    assert len(saved) == 1
    assert yaml.safe_load(saved[0].read_text())["error"] == "interrupted"
    inventory = read_point(directory).files
    assert {"dynamicvla-manifest.json", "dynamicvla-receipts.jsonl", "episode.npz"} <= {Path(path).name for path in inventory}


def test_verify_recovers_completed_native_receipts_without_rerunning(completed, monkeypatch):
    plan = completed
    point = plan.points[0]
    directory = plan.study_dir / "points" / point.name
    inventory = read_point(directory).files
    (directory / "point.yaml").unlink()
    (directory / "error.txt").write_text("old verifier expected episode_id")
    marker = plan.study_dir / "queue" / f"{point.name}.done"
    marker.rename(marker.with_suffix(".failed"))
    monkeypatch.setattr(runner, "run_experiment", lambda *a, **k: pytest.fail("verification reran evaluation"))
    assert main(["study", "verify", str(plan.study_dir)]) == 0
    assert read_point(directory).files == inventory
    assert runner.state(plan.study_dir, point.name) == ("done", None)
    assert not (directory / "error.txt").exists()


def test_denoiser_control_changes_study_identity_without_changing_legacy_defaults(tmp_path):
    from vlalagbench.config import DynamicVLAStudyPolicy
    assert "enable_dl_denoiser" not in DynamicVLAStudyPolicy().model_dump(mode="json")
    plan = plan_at(tmp_path)
    changed = plan_at(tmp_path, policy={"kind": "dynamicvla", "enable_dl_denoiser": False})
    assert changed.study_hash != plan.study_hash
    assert shard_runtime(changed, 0).enable_dl_denoiser is False


def test_example_study_has_only_supported_latency_points():
    plan = resolve(load_study(STUDY), load_placement(PLACEMENT), ROOT)
    assert [p.experiment.inference.latency_ms for p in plan.points] == [0, 120, 240]



@pytest.mark.parametrize("change", [
    {"antialiasing_mode": "FXAA"},
    {"enable_dl_denoiser": False},
    {"disable_fabric": True},
])
def test_observation_settings_are_hashed_and_verified(tmp_path, change):
    plan = plan_at(tmp_path)
    policy = {**plan.spec.policy.model_dump(), **change}
    changed = plan_at(tmp_path, policy=policy)
    assert changed.study_hash != plan.study_hash
    runtime = shard_runtime(changed, 0)
    backend.verify_runtime(runtime, changed.spec.policy)
    with pytest.raises(ValueError, match="semantics"):
        backend.verify_runtime(runtime, plan.spec.policy)


@pytest.mark.parametrize("field,value", [("dlss_mode", "Quality"), ("enable_dlssg", True), ("render_mode", None)])
def test_renderer_pins_change_study_identity_and_reach_runtime(tmp_path, field, value):
    plan = plan_at(tmp_path)
    policy = plan.spec.policy.model_dump()
    policy[field] = value
    changed = plan_at(tmp_path, policy=policy)
    assert changed.study_hash != plan.study_hash
    assert getattr(shard_runtime(changed, 0), field) == value


def test_live_video_placement_preserves_scientific_identity(tmp_path):
    plan = plan_at(tmp_path)
    installation = plan.placement.installation.model_copy(update={"save_videos": True, "video_episode_limit": 10})
    changed = resolve(plan.spec, plan.placement.model_copy(update={"installation": installation}), ROOT)
    assert changed.study_hash == plan.study_hash
    runtime = shard_runtime(changed, 0)
    assert runtime.save_videos and runtime.video_episode_limit == 10
    assert not runtime.trace_images
    backend.verify_runtime(runtime, plan.spec.policy)


@pytest.mark.parametrize("damage", [None, "hash", "frames", "escape"])
def test_live_video_is_verified_and_inventoried(completed, damage):
    point = completed.points[0]
    directory = completed.study_dir / "points" / point.name
    record = read_point(directory)
    artifacts = directory / "artifacts" / record.run_id
    receipt_path = artifacts / "dynamicvla-receipts.jsonl"
    receipt = json.loads(receipt_path.read_text())
    video = artifacts / "recorded.mp4"
    video.write_bytes(b"recorded camera video")
    receipt.update(video=video.name, video_sha256=sha256(video),
                   video_frames=receipt["n_steps"] + 1, video_cameras=["opst_cam"])
    if damage == "hash":
        video.write_bytes(b"corrupted")
    elif damage == "frames":
        receipt["video_frames"] -= 1
    elif damage == "escape":
        receipt["video"] = "../outside.mp4"
    receipt_path.write_text(json.dumps(receipt) + "\n")
    episodes = [json.loads(line) for line in (artifacts / "episodes.jsonl").read_text().splitlines()]
    if damage:
        with pytest.raises(ValueError, match="recorded video"):
            backend.verify_evidence(artifacts, episodes, point.experiment)
    else:
        assert backend.verify_evidence(artifacts, episodes, point.experiment)[video.name] == sha256(video)


def test_live_video_selection_uses_recorded_clips_without_gpu(completed, monkeypatch):
    point = completed.points[0]
    directory = completed.study_dir / "points" / point.name
    record = read_point(directory)
    artifacts = directory / "artifacts" / record.run_id
    runtime_path = directory / "runs" / record.run_id / "runtime.yaml"
    runtime = load_runtime(runtime_path).model_copy(update={"save_videos": True})
    write_yaml(runtime, runtime_path)
    receipt_path = artifacts / "dynamicvla-receipts.jsonl"
    receipt = json.loads(receipt_path.read_text())
    video = artifacts / "live.mp4"
    video.write_bytes(b"video")
    receipt.update(video=video.name, video_sha256=sha256(video), video_cameras=["opst_cam"])
    receipt_path.write_text(json.dumps(receipt) + "\n")
    # This fixture was edited above; receipt integrity is independently tested.
    monkeypatch.setattr(videos, "verify_files", lambda *_: None)
    monkeypatch.setattr(runner, "gpu_locks", lambda *_: pytest.fail("recorded video extraction requested GPU"))
    monkeypatch.setattr(videos, "ProcessPoolExecutor", lambda **kw: ThreadPoolExecutor(max_workers=kw["max_workers"]))
    extracted = []
    def extract(job):
        extracted.append(job)
        selection, camera, path, _ = job
        assert selection.recorded_video == video and camera == "opst_cam"
        path.write_bytes(b"extracted")
    monkeypatch.setattr(videos, "_render_one", extract)
    manifest = videos.film_point(directory, point.name, Videos(clips_per_outcome=3, cameras=["opst_cam"], resolution=256), 1)
    assert manifest.source == "recorded_policy_rgb"
    assert len(manifest.clips) == 1
    assert manifest.shortfall == {"success": 3, "failure": 2}
    options = Videos(clips_per_outcome=3, cameras=["opst_cam"], resolution=256)
    videos.film_point(directory, point.name, options, 1)
    assert len(extracted) == 1
    video.write_bytes(b"replacement recording")
    videos.film_point(directory, point.name, options, 1)
    assert len(extracted) == 2


def test_kit_tasking_threads_join_renderer_kit_args(tmp_path):
    from vlalagbench.benchmarks.dynamicvla import adapter

    plan = plan_at(tmp_path)
    runtime = shard_runtime(plan, 0)
    assert runtime.kit_tasking_threads == 4
    assert "--/plugins/carb.tasking.plugin/threadCount=4" in " ".join(adapter._renderer_gpu_args(runtime))
    capped = runtime.model_copy(update={"simulator_gpu": 5, "kit_tasking_threads": 4})
    flag, settings = adapter._renderer_gpu_args(capped)
    assert flag == "--kit_args"
    assert settings.split() == ["--/renderer/multiGpu/enabled=false", "--/renderer/activeGpu=5",
                                "--/plugins/carb.tasking.plugin/threadCount=4"]
    assert adapter._renderer_gpu_args(capped.model_copy(update={"simulator_gpu": None})) == [
        "--kit_args", "--/plugins/carb.tasking.plugin/threadCount=4"]
    data = plan.placement.model_dump()
    data["installation"]["kit_tasking_threads"] = 4
    placed = resolve(plan.spec, DynamicVLAPlacement.model_validate(data), ROOT)
    assert placed.study_hash == plan.study_hash
    assert shard_runtime(placed, 0).kit_tasking_threads == 4


@pytest.mark.parametrize("threads", [None, 2, 4])
def test_kit_tasking_override_survives_serialization(tmp_path, threads):
    from vlalagbench.benchmarks.dynamicvla import adapter
    from vlalagbench.config import DynamicVLARuntime

    plan = plan_at(tmp_path)
    data = plan.placement.model_dump()
    data["installation"]["kit_tasking_threads"] = threads
    placement = DynamicVLAPlacement.model_validate(data)
    restored = DynamicVLAPlacement.model_validate_json(placement.model_dump_json())
    changed = resolve(plan.spec, restored, ROOT)
    assert changed.study_hash == plan.study_hash
    runtime = shard_runtime(changed, 0)
    runtime = DynamicVLARuntime.model_validate_json(runtime.model_dump_json())
    assert runtime.kit_tasking_threads == threads
    settings = " ".join(adapter._renderer_gpu_args(runtime))
    if threads is None:
        assert "tasking" not in settings
    else:
        assert f"--/plugins/carb.tasking.plugin/threadCount={threads}" in settings


def test_dynamicvla_runtime_defaults_to_four_kit_threads(tmp_path):
    from vlalagbench.config import DynamicVLARuntime

    data = shard_runtime(plan_at(tmp_path), 0).model_dump()
    data.pop("kit_tasking_threads")
    assert DynamicVLARuntime.model_validate(data).kit_tasking_threads == 4
