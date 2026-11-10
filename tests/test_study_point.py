"""Verify holds a point to its study; video selection is stable."""

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from vlalagbench.config import (
    CodeVersion,
    EpisodeArtifact,
    GPUAllocation,
    Hardware,
    RunLatency,
    RunResult,
    RunSummary,
    SourceConfig,
    StudySpec,
    TaskSummary,
    write_yaml,
    load_runtime,
)
from vlalagbench.simulators.mujoco.tracing import EpisodeTrace, TraceMeta
from vlalagbench.study import point as stage
from vlalagbench.study import videos
from vlalagbench.files import sha256
from vlalagbench.study.spec import expand, study_hash
from tests.test_study_spec import ROOT, STUDY

HASH = "a" * 64
EXPERIMENT = expand(
    StudySpec.model_validate(
        {
            **STUDY,
            "sweep": {"inference.latency_ms": [0]},
            "override": {"benchmark.max_episode_steps": 600, "benchmark.episode_ids": [0, 1]},
        }
    ),
    ROOT,
)[0].experiment


def _trace(path: Path, success: bool, episode_id: int = 0) -> None:
    EpisodeTrace(
        meta=TraceMeta.model_validate(
            {"schema_version": 3, "environment": {"kind": "libero"}, "control_freq": 20.0, "num_steps": 1, "success": success, "error": None, "task_id": 1, "episode_id": episode_id}
        ),
        states=np.zeros((2, 3)),
        actions=np.zeros((1, 7)),
        rewards=np.zeros(1),
        dones=np.asarray([success]),
        acting=np.ones(1, dtype=bool),
        chunk_ids=np.zeros(1, dtype=np.int64),
        action_indices=np.zeros(1, dtype=np.int64),
        observation_steps=np.zeros(1, dtype=np.int64),
        request_steps=np.zeros(1, dtype=np.int64),
        arrival_steps=np.zeros(1, dtype=np.int64),
        chunks=np.zeros((1, 1, 7)),
    ).save(path)


@pytest.fixture
def point_dir(tmp_path):
    run_id = "run-1"
    artifacts = tmp_path / "artifacts" / run_id
    artifacts.mkdir(parents=True)
    rows = []
    for episode_id, success in ((0, True), (1, False)):
        trace = artifacts / f"libero_object_task01_ep{episode_id:03d}.npz"
        _trace(trace, success, episode_id)
        rows.append({"task_id": 1, "episode_id": episode_id, "seed": 7, "success": success, "steps": 1, "control_steps": 1, "wait_steps": 0, "policy_requests": 1, "trace": str(trace), "trace_sha256": sha256(trace), "error": None})
    content = "".join(json.dumps(row) + "\n" for row in rows)
    (artifacts / "episodes.jsonl").write_text(content)
    write_yaml(EXPERIMENT, tmp_path / "runs" / run_id / "experiment.yaml")
    write_yaml(load_runtime(ROOT / "configs/runtime/libero.yaml"), tmp_path / "runs" / run_id / "runtime.yaml")
    now = dt.datetime.now(dt.timezone.utc)
    write_yaml(
        RunResult(
            schema_version=4,
            run_id=run_id,
            status="completed",
            error=None,
            experiment=SourceConfig(source="x", sha256=sha256(tmp_path / "runs" / run_id / "experiment.yaml"), resolved="experiment.yaml"),
            runtime=SourceConfig(source="y", sha256=sha256(tmp_path / "runs" / run_id / "runtime.yaml"), resolved="runtime.yaml"),
            code=CodeVersion(commit="b" * 40, dirty=False),
            hardware=Hardware(hostname="h", cpu_count=1, gpus=[GPUAllocation(type="g", count=1)]),
            started_at=now,
            finished_at=now,
            latency=RunLatency(control_hz=20, requested_ms=0, effective_steps=0),
            episode_data=EpisodeArtifact(format="jsonl", path="episodes.jsonl", sha256=hashlib.sha256(content.encode()).hexdigest(), records=2),
            summary=RunSummary(
                episodes=2, successes=1, success_rate=0.5, episode_errors=0, policy_requests=2,
                mean_active_steps=1.0, mean_control_steps=1.0, mean_wait_steps=0.0,
                tasks=[TaskSummary(task_id=1, episodes=2, successes=1, success_rate=0.5, mean_active_steps=1.0, mean_control_steps=1.0, mean_wait_steps=0.0)],
            ),
            study_name="probe",
            study_hash=HASH,
        ),
        tmp_path / "runs" / run_id / "result.yaml",
    )
    return tmp_path


def _verify(point_dir, **changes):
    return stage.verify_point(point_dir, **{"name": "0", "values": {"inference.latency_ms": 0}, "experiment": EXPERIMENT, "study_hash": HASH, **changes})


def test_verify_passes_and_records_the_wilson_interval(point_dir):
    record = _verify(point_dir)

    assert (record.successes, record.episodes, record.deadline_steps) == (1, 2, 600)
    assert record.ci95_low < 0.5 < record.ci95_high
    assert yaml.safe_load((point_dir / stage.POINT_RECORD).read_text())["run_id"] == "run-1"


def test_verify_fails_on_a_missing_trace(point_dir):
    (point_dir / "artifacts/run-1/libero_object_task01_ep001.npz").unlink()
    with pytest.raises(FileNotFoundError):
        _verify(point_dir)


def test_verify_fails_on_an_episode_digest_mismatch(point_dir):
    with (point_dir / "artifacts/run-1/episodes.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="digest"):
        _verify(point_dir)


def test_verify_fails_when_the_run_is_not_the_study_condition(point_dir):
    other = EXPERIMENT.model_copy(update={"benchmark": EXPERIMENT.benchmark.model_copy(update={"max_episode_steps": 300})})
    with pytest.raises(ValueError, match="differs from the study"):
        _verify(point_dir, experiment=other)
    with pytest.raises(ValueError, match="study_hash"):
        _verify(point_dir, study_hash="c" * 64)


def test_selection_is_stable_and_reports_shortfall(tmp_path):
    episodes = [{"task_id": 1, "episode_id": index, "success": index % 3 == 0, "trace": f"ep{index}.npz"} for index in range(12)]

    first = videos.select_clips(episodes, "a_point", 3, tmp_path)
    again = videos.select_clips(episodes, "a_point", 3, tmp_path)
    other = videos.select_clips(episodes, "b_point", 3, tmp_path)

    assert first == again and first != other
    assert [selection.outcome for selection in first] == ["success"] * 3 + ["failure"] * 3
    assert len(videos.select_clips(episodes[:2], "a_point", 3, tmp_path)) == 2


def test_verify_rejects_modified_trace_arrays_with_unchanged_metadata(point_dir):
    path = point_dir / "artifacts/run-1/libero_object_task01_ep001.npz"
    with np.load(path, allow_pickle=False) as trace:
        data = {key: trace[key] for key in trace.files}
    data["actions"] = data["actions"] + 1
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="trace digest"):
        _verify(point_dir)


def test_verify_rejects_duplicate_episode_ids_even_when_counts_and_manifest_hash_agree(point_dir):
    episodes_path = point_dir / "artifacts/run-1/episodes.jsonl"
    rows = stage.load_episodes(episodes_path)
    rows[1]["episode_id"] = rows[0]["episode_id"]
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result_path = point_dir / "runs/run-1/result.yaml"
    data = yaml.safe_load(result_path.read_text())
    data["episode_data"]["sha256"] = sha256(episodes_path)
    result_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="identities"):
        _verify(point_dir)


def test_per_task_selection_picks_each_outcome_within_every_task(tmp_path):
    rows = [{"task_id": task, "episode_id": index, "success": index % 2 == 0, "trace": f"t{task}_{index}.npz"}
            for task in (0, 3) for index in range(10)]
    clips = videos.select_clips(rows, "point", 3, tmp_path, per_task=True)
    assert [(clip.task_id, clip.outcome) for clip in clips] == (
        [(0, "success")] * 3 + [(0, "failure")] * 3 + [(3, "success")] * 3 + [(3, "failure")] * 3)
    assert clips == videos.select_clips(rows, "point", 3, tmp_path, per_task=True)
    assert videos.select_clips(rows, "point", 3, tmp_path) == videos.select_clips(rows, "point", 3, tmp_path, per_task=False)


def test_multitask_clips_keep_task_and_outcome_identity(tmp_path):
    rows = [{"task_id": 0, "episode_id": 0, "success": True, "trace": "task0.npz"},
            {"task_id": 1, "episode_id": 0, "success": False, "trace": "task1.npz"}]
    clips = videos.select_clips(rows, "point", 3, tmp_path)
    assert [(clip.task_id, clip.outcome, clip.trace.name) for clip in clips] == [(0, "success", "task0.npz"), (1, "failure", "task1.npz")]


def test_video_failure_resumes_only_missing_clips(point_dir, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from vlalagbench.config import Videos

    _verify(point_dir)
    monkeypatch.setattr(videos, "ProcessPoolExecutor", lambda **kwargs: ThreadPoolExecutor(max_workers=kwargs["max_workers"]))
    rendered = []
    fail = {1}

    def render(job):
        selection, camera, path, resolution = job
        if selection.episode_id in fail:
            raise RuntimeError("renderer stopped")
        path.write_bytes(f"{camera}:{resolution}".encode())
        rendered.append(path.name)
        return path

    monkeypatch.setattr(videos, "_render_one", render)
    settings = Videos(clips_per_outcome=3, cameras=["agentview", "robot0_eye_in_hand"], resolution=256)
    with pytest.raises(RuntimeError, match="resume"):
        videos.film_point(point_dir, "point", settings, 2)
    assert len(rendered) == 2
    fail.clear()
    manifest = videos.film_point(point_dir, "point", settings, 2)
    assert manifest.complete and len(rendered) == 4
    assert manifest.shortfall == {"success": 2, "failure": 2}
    videos.film_point(point_dir, "point", settings, 2)
    assert len(rendered) == 4
