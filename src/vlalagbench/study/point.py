"""Verify immutable evaluation output before rendering, transfer, or reporting."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from vlalagbench.config import (
    DynamicVLAExperiment, DynamicVLAStudyPolicy, ExperimentDocument,
    NonEmptyStr, NonNegativeInt, PositiveInt, Sha256, SweepValue,
    UnitFloat, StudyPolicy, load_run_result, load_runtime,
)
from vlalagbench.files import atomic_text, sha256
from vlalagbench.statistics import wilson_interval

POINT_RECORD = "point.yaml"
VIDEO_MANIFEST = "videos/manifest.yaml"


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PointRecord(Record):
    name: NonEmptyStr
    values: dict[NonEmptyStr, SweepValue]
    run_id: NonEmptyStr
    study_hash: Sha256
    commit: NonEmptyStr
    code_dirty: bool = False
    deadline_steps: PositiveInt | None
    episodes: PositiveInt
    successes: NonNegativeInt
    success_rate: UnitFloat
    ci95_low: UnitFloat
    ci95_high: UnitFloat
    verified_at: dt.datetime
    # Empty only for historical points verified before trace checksums existed.
    files: dict[NonEmptyStr, Sha256] = Field(default_factory=dict)


def sole_run_id(point_dir: Path) -> str:
    (run_dir,) = (point_dir / "runs").iterdir()
    return run_dir.name


def verify_point(
    point_dir: Path, *, name: str, values: dict[str, SweepValue],
    experiment: ExperimentDocument, study_hash: str, require_trace_hash: bool = True,
    policy: StudyPolicy | DynamicVLAStudyPolicy | None = None,
) -> PointRecord:
    dynamicvla = isinstance(experiment, DynamicVLAExperiment)
    run_id = sole_run_id(point_dir)
    run_dir = point_dir / "runs" / run_id
    result = load_run_result(run_dir / "result.yaml")
    expected = yaml.safe_dump(experiment.model_dump(mode="python"), sort_keys=False)
    if (run_dir / "experiment.yaml").read_text() != expected:
        raise ValueError(f"{name}: the run's experiment differs from the study's expansion")
    if result.study_hash != study_hash:
        raise ValueError(f"{name}: run carries a different study_hash")
    if result.status != "completed" or result.summary.episode_errors:
        raise ValueError(f"{name}: run is not complete and error-free")
    for source in (result.experiment, result.runtime):
        if sha256(run_dir / source.resolved) != source.sha256:
            raise ValueError(f"{name}: {source.resolved} digest differs from result.yaml")
    if dynamicvla and policy is not None:
        from vlalagbench.benchmarks.dynamicvla.study import verify_runtime

        verify_runtime(load_runtime(run_dir / "runtime.yaml"), policy)
    elif policy is not None:
        runtime = load_runtime(run_dir / "runtime.yaml")
        if runtime.policy_server.model_dump(exclude={"host", "port"}) != policy.model_dump():
            raise ValueError(f"{name}: runtime policy differs from the study protocol")
        if runtime.environment.reuse_task_envs or runtime.environment.initial_state_strategy != "cycle_official":
            raise ValueError(f"{name}: runtime does not use fresh official-state episodes")
    episodes_path = point_dir / "artifacts" / run_id / "episodes.jsonl"
    if sha256(episodes_path) != result.episode_data.sha256:
        raise ValueError(f"{name}: episodes.jsonl digest differs from result.yaml")
    episodes = load_episodes(episodes_path)
    if dynamicvla:
        from vlalagbench.benchmarks.dynamicvla.study import expected_ids as dynamicvla_ids

        expected_ids = dynamicvla_ids(experiment)
        task_ids = list(range(len(experiment.benchmark.tasks)))
    else:
        task_ids = experiment.benchmark.task_ids
        if task_ids == "all":
            from vlalagbench.benchmarks.libero import _vendor

            suite = _vendor.libero().benchmark.get_benchmark_dict()[experiment.benchmark.suite]()
            task_ids = list(range(suite.n_tasks))
        episode_ids = experiment.benchmark.episode_ids or range(experiment.benchmark.episodes_per_task)
        expected_ids = {(task, episode) for task in task_ids for episode in episode_ids}
    actual_ids = [(episode["task_id"], episode["episode_id"]) for episode in episodes]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
        raise ValueError(f"{name}: missing, duplicate, or unexpected task/episode identities")
    if len(episodes) != result.summary.episodes or {task.task_id for task in result.summary.tasks} != set(task_ids):
        raise ValueError(f"{name}: result summary does not cover the expected episodes")
    files = {}
    traces = set()
    for episode in episodes:
        trace = point_dir / "artifacts" / run_id / Path(episode["trace"]).name
        if trace in traces:
            raise ValueError(f"{name}: two episodes reference {trace.name}")
        traces.add(trace)
        digest = sha256(trace)
        if require_trace_hash and not episode.get("trace_sha256"):
            raise ValueError(f"{name}: {trace.name} has no original trace digest")
        if episode.get("trace_sha256") and digest != episode["trace_sha256"]:
            raise ValueError(f"{name}: {trace.name} trace digest mismatch")
        with np.load(trace, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"][()]))
        if dynamicvla:
            from vlalagbench.benchmarks.dynamicvla.study import verify_trace

            verify_trace(meta, episode, experiment)
        elif episode["error"] is not None or meta["error"] is not None or meta["success"] is not episode["success"]:
            raise ValueError(f"{name}: trace {trace.name} disagrees with its episode row")
        if not dynamicvla and require_trace_hash and (meta.get("task_id"), meta.get("episode_id")) != (episode["task_id"], episode["episode_id"]):
            raise ValueError(f"{name}: trace {trace.name} has a different episode identity")
        files[trace.relative_to(point_dir).as_posix()] = digest
    if dynamicvla:
        from vlalagbench.benchmarks.dynamicvla.study import verify_evidence

        trace_dir = episodes_path.parent
        for filename, digest in verify_evidence(trace_dir, episodes, experiment).items():
            files[(trace_dir / filename).relative_to(point_dir).as_posix()] = digest
    successes = sum(episode["success"] for episode in episodes)
    if successes != result.summary.successes:
        raise ValueError(f"{name}: episode successes differ from the result summary")
    for task in result.summary.tasks:
        rows = [row for row in episodes if row["task_id"] == task.task_id]
        if task.episodes != len(rows) or task.successes != sum(row["success"] for row in rows):
            raise ValueError(f"{name}: task {task.task_id} summary differs from its episodes")
    for path in [episodes_path, *sorted(run_dir.glob("*.yaml"))]:
        files[path.relative_to(point_dir).as_posix()] = sha256(path)
    low, high = wilson_interval(successes, len(episodes))
    record = PointRecord(
        name=name, values=values, run_id=run_id, study_hash=study_hash,
        commit=result.code.commit, code_dirty=result.code.dirty,
        deadline_steps=experiment.benchmark.max_episode_steps,
        episodes=len(episodes), successes=successes, success_rate=successes / len(episodes),
        ci95_low=low, ci95_high=high, verified_at=dt.datetime.now(dt.timezone.utc), files=files,
    )
    write_record(record, point_dir / POINT_RECORD)
    return record


def read_point(point_dir: Path, study_hash: str | None = None) -> PointRecord:
    record = PointRecord.model_validate(yaml.safe_load((point_dir / POINT_RECORD).read_text()))
    if study_hash is not None and record.study_hash != study_hash:
        raise ValueError(f"{point_dir}: point belongs to a different study_hash")
    return record


def verify_files(point_dir: Path, record: PointRecord) -> None:
    for relative, digest in record.files.items():
        path = point_dir / relative
        if not path.resolve().is_relative_to(point_dir.resolve()):
            raise ValueError(f"{relative}: artifact path escapes point directory")
        if sha256(path) != digest:
            raise ValueError(f"{path}: verified artifact digest mismatch")


def load_episodes(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_record(record: BaseModel, path: Path) -> None:
    atomic_text(path, yaml.safe_dump(record.model_dump(mode="python"), sort_keys=False))
