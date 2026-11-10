"""Run one concrete experiment/runtime pair and emit its typed result."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from vlalagbench.benchmarks import get_adapter
from vlalagbench.config import (
    CodeVersion,
    EpisodeArtifact,
    EpisodeResult,
    Experiment,
    Hardware,
    RunLatency,
    RunResult,
    RUN_RESULT_SCHEMA_VERSION,
    RunSummary,
    Runtime,
    SourceConfig,
    TaskSummary,
    load_experiment,
    load_runtime,
    write_yaml,
)
from vlalagbench.hardware import collect_hardware
from vlalagbench.files import atomic_text, sha256


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_experiment(
    experiment_path: str | Path,
    runtime_path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    run_id: str | None = None,
    runs_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    study: tuple[str, str] | None = None,
) -> RunResult:
    """Run validated YAML configs and write ``<runs_root>/<run-id>/result.yaml``.

    ``runs_root`` and ``artifact_root`` default to the repository's ``runs/``
    and artifact directories; a study places them under its own point
    directory and names itself through ``study`` as ``(name, hash)``.
    """

    repo_root = Path(repo_root).resolve()
    experiment_path = Path(experiment_path).resolve()
    runtime_path = Path(runtime_path).resolve()
    experiment = load_experiment(experiment_path)
    runtime = load_runtime(runtime_path)
    adapter = get_adapter(experiment.benchmark.kind)
    adapter.validate(experiment, runtime)

    code = _git_state(repo_root)
    hardware = collect_hardware()
    started_at = dt.datetime.now(dt.timezone.utc)
    run_id = run_id or f"{started_at:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}-{experiment.name}"
    run_dir = Path(runs_root or repo_root / "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir = Path(artifact_root or _artifact_root(repo_root)) / run_id

    shutil.copyfile(experiment_path, run_dir / "experiment.yaml")
    shutil.copyfile(runtime_path, run_dir / "runtime.yaml")
    experiment_source = _source_config(experiment_path, repo_root, "experiment.yaml")
    runtime_source = _source_config(runtime_path, repo_root, "runtime.yaml")

    try:
        raw_episodes = adapter.evaluate(experiment, runtime, artifact_dir)
        episodes = [
            EpisodeResult(
                task_id=episode["task_id"],
                episode_id=episode["episode"],
                seed=episode["seed"],
                success=episode["success"],
                status=episode.get("status"),
                steps=episode["steps"],
                control_steps=episode["control_steps"],
                wait_steps=episode["wait_steps"],
                policy_requests=episode["policy_requests"],
                trace=_display_path(artifact_dir / episode["trace"], repo_root),
                trace_sha256=sha256(artifact_dir / episode["trace"]),
                error=episode["error"],
            )
            for episode in raw_episodes
        ]
        episode_data = _write_episode_data(episodes, artifact_dir, repo_root)
        result = _result(
            run_id=run_id,
            status="completed",
            error=None,
            experiment_source=experiment_source,
            runtime_source=runtime_source,
            code=code,
            hardware=hardware,
            started_at=started_at,
            experiment=experiment,
            episodes=episodes,
            episode_data=episode_data,
            study=study,
        )
    except Exception as exc:
        episodes = []
        episode_data = _write_episode_data(episodes, artifact_dir, repo_root)
        result = _result(
            run_id=run_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            experiment_source=experiment_source,
            runtime_source=runtime_source,
            code=code,
            hardware=hardware,
            started_at=started_at,
            experiment=experiment,
            episodes=episodes,
            episode_data=episode_data,
            study=study,
        )
        write_yaml(result, run_dir / "result.yaml")
        raise

    write_yaml(result, run_dir / "result.yaml")
    return result


def _result(
    *,
    run_id: str,
    status: str,
    error: str | None,
    experiment_source: SourceConfig,
    runtime_source: SourceConfig,
    code: CodeVersion,
    hardware: Hardware,
    started_at: dt.datetime,
    experiment: Experiment,
    episodes: list[EpisodeResult],
    episode_data: EpisodeArtifact,
    study: tuple[str, str] | None,
) -> RunResult:
    successes = sum(episode.success for episode in episodes)
    study_name, study_hash = study or (None, None)
    return RunResult(
        study_name=study_name,
        study_hash=study_hash,
        schema_version=RUN_RESULT_SCHEMA_VERSION,
        run_id=run_id,
        status=status,
        error=error,
        experiment=experiment_source,
        runtime=runtime_source,
        code=code,
        hardware=hardware,
        started_at=started_at,
        finished_at=dt.datetime.now(dt.timezone.utc),
        latency=RunLatency(
            control_hz=experiment.benchmark.control_hz,
            requested_ms=experiment.inference.latency_ms,
            effective_steps=experiment.effective_latency_steps,
        ),
        episode_data=episode_data,
        summary=RunSummary(
            episodes=len(episodes),
            successes=successes,
            success_rate=successes / len(episodes) if episodes else 0.0,
            episode_errors=sum(episode.error is not None for episode in episodes),
            policy_requests=sum(episode.policy_requests for episode in episodes),
            mean_active_steps=_mean(episode.steps for episode in episodes),
            mean_control_steps=_mean(episode.control_steps for episode in episodes),
            mean_wait_steps=_mean(episode.wait_steps for episode in episodes),
            tasks=_task_summaries(episodes),
        ),
    )


def _write_episode_data(
    episodes: list[EpisodeResult],
    artifact_dir: Path,
    repo_root: Path,
) -> EpisodeArtifact:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "episodes.jsonl"
    content = "".join(
        json.dumps(episode.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
        for episode in episodes
    )
    atomic_text(path, content)
    return EpisodeArtifact(
        format="jsonl",
        path=_display_path(path, repo_root),
        sha256=hashlib.sha256(content.encode()).hexdigest(),
        records=len(episodes),
    )


def _task_summaries(episodes: list[EpisodeResult]) -> list[TaskSummary]:
    grouped = {
        task_id: [episode for episode in episodes if episode.task_id == task_id]
        for task_id in sorted({episode.task_id for episode in episodes})
    }
    return [
        TaskSummary(
            task_id=task_id,
            episodes=len(task_episodes),
            successes=sum(episode.success for episode in task_episodes),
            success_rate=sum(episode.success for episode in task_episodes) / len(task_episodes),
            mean_active_steps=_mean(episode.steps for episode in task_episodes),
            mean_control_steps=_mean(episode.control_steps for episode in task_episodes),
            mean_wait_steps=_mean(episode.wait_steps for episode in task_episodes),
        )
        for task_id, task_episodes in grouped.items()
    ]


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _source_config(path: Path, repo_root: Path, resolved: str) -> SourceConfig:
    return SourceConfig(
        source=_display_path(path, repo_root),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        resolved=resolved,
    )


def _git_state(repo_root: Path) -> CodeVersion:
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return CodeVersion(commit=commit, dirty=dirty)


def _repo_path(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root).as_posix()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return _repo_path(path, repo_root)
    except ValueError:
        return path.resolve().as_posix()


def _artifact_root(repo_root: Path) -> Path:
    configured = os.getenv("VLALAGBENCH_ARTIFACT_ROOT") or os.getenv(
        "MUJOCO_DYNAMIC_ARTIFACT_ROOT"
    )
    return Path(configured).resolve() if configured else repo_root / "artifacts"
