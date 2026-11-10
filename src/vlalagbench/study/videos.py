"""Render selected traces independently of evaluation, resuming completed clips."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import multiprocessing
import random
import tempfile
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from vlalagbench.config import DynamicVLAExperiment, DynamicVLARuntime, load_experiment, load_runtime, NonEmptyStr, NonNegativeInt, PositiveInt, Sha256, Videos
from vlalagbench.files import sha256
from vlalagbench.study.point import (
    Record, VIDEO_MANIFEST, load_episodes, read_point, verify_files, write_record,
)


class Clip(Record):
    outcome: Literal["success", "failure"]
    task_id: NonNegativeInt
    episode_id: NonNegativeInt
    camera: NonEmptyStr
    path: NonEmptyStr
    sha256: Sha256


class VideoManifest(Record):
    settings_hash: Sha256
    clips_per_outcome: PositiveInt
    resolution: PositiveInt | None
    source: Literal["simulator_replay", "recorded_policy_rgb"] = "simulator_replay"
    shortfall: dict[Literal["success", "failure"], NonNegativeInt]
    complete: bool = False
    clips: list[Clip] = Field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class Selection:
    outcome: Literal["success", "failure"]
    task_id: int
    episode_id: int
    trace: Path
    kind: Literal["libero", "dynamicvla"] = "libero"
    runtime: Path | None = None
    gpu: int | None = None
    path_tracing_spp: int | None = None
    recorded_video: Path | None = None
    recorded_cameras: tuple[str, ...] = ()


def select_clips(episodes: list[dict], name: str, per_outcome: int, trace_dir: Path,
                 per_task: bool = False) -> list[Selection]:
    if per_task:
        return [selection for task_id in sorted({row["task_id"] for row in episodes})
                for selection in select_clips([row for row in episodes if row["task_id"] == task_id],
                                              f"{name}/task{task_id}", per_outcome, trace_dir)]
    selections = []
    for outcome, success in (("success", True), ("failure", False)):
        rows = sorted((row for row in episodes if row["success"] is success), key=lambda row: (row["task_id"], row["episode_id"]))
        random.Random(f"{name}/{outcome}").shuffle(rows)
        selections.extend(
            Selection(outcome, row["task_id"], row["episode_id"], trace_dir / Path(row["trace"]).name)
            for row in rows[:per_outcome]
        )
    return selections


def film_point(point_dir: Path, name: str, videos: Videos, workers: int, gpu: int | None = None,
               runtime_path: Path | None = None) -> VideoManifest:
    from vlalagbench.study.runner import gpu_locks

    record = read_point(point_dir)
    verify_files(point_dir, record)
    trace_dir = point_dir / "artifacts" / record.run_id
    episodes = load_episodes(trace_dir / "episodes.jsonl")
    per_task = bool(videos.per_task)
    selections = select_clips(episodes, name, videos.clips_per_outcome, trace_dir, per_task)
    groups = len({row["task_id"] for row in episodes}) if per_task else 1
    experiment = load_experiment(point_dir / "runs" / record.run_id / "experiment.yaml")
    dynamicvla = isinstance(experiment, DynamicVLAExperiment)
    if videos.dynamicvla_path_tracing_spp is not None and not dynamicvla:
        raise ValueError("path-tracing video settings require DynamicVLA")
    if runtime_path is not None and not dynamicvla:
        raise ValueError("--runtime is only supported for DynamicVLA study videos")
    replay_runtime_hash = None
    recorded = False
    if dynamicvla:
        runtime = runtime_path or point_dir / "runs" / record.run_id / "runtime.yaml"
        local_runtime = load_runtime(runtime)
        if not isinstance(local_runtime, DynamicVLARuntime):
            raise ValueError("study replay requires a DynamicVLA runtime")
        if gpu is None:
            gpu = 0
        # GPU placement does not determine the saved trajectory or video settings.
        replay_runtime_hash = hashlib.sha256(yaml.safe_dump(local_runtime.model_dump(
            exclude={"policy_gpu", "simulator_gpu"}), sort_keys=True).encode()).hexdigest()
        recorded = load_runtime(point_dir / "runs" / record.run_id / "runtime.yaml").save_videos
        if recorded:
            if videos.dynamicvla_path_tracing_spp is not None:
                raise ValueError("recorded videos cannot change renderer settings")
            receipts = [json.loads(line) for line in
                        (trace_dir / "dynamicvla-receipts.jsonl").read_text().splitlines() if line]
            available = {(row["task_id"], row["episode"]): row for row in receipts if row.get("video")}
            episodes = [row for row in load_episodes(trace_dir / "episodes.jsonl")
                        if (row["task_id"], row["episode_id"]) in available]
            selections = select_clips(episodes, name, videos.clips_per_outcome, trace_dir, per_task)
            selections = [dataclasses.replace(selection,
                recorded_video=trace_dir / available[selection.task_id, selection.episode_id]["video"],
                recorded_cameras=tuple(available[selection.task_id, selection.episode_id]["video_cameras"]))
                for selection in selections]
        selections = [dataclasses.replace(selection, kind="dynamicvla", runtime=runtime, gpu=gpu,
                                         path_tracing_spp=videos.dynamicvla_path_tracing_spp)
                      for selection in selections]
    settings_hash = hashlib.sha256(yaml.safe_dump({
        "videos": videos.model_dump(exclude_none=True),
        "source": "recorded_policy_rgb" if recorded else "simulator_replay",
        "sources": {selection.trace.name: sha256(selection.trace) for selection in selections},
        **({"recorded_sources": {selection.trace.name: {
            "sha256": sha256(selection.recorded_video), "cameras": list(selection.recorded_cameras)}
            for selection in selections}} if recorded else {}),
        **({"runtime": replay_runtime_hash} if dynamicvla else {}),
        **({"renderer_revision": "dynamicvla-multicamera-owned-buffers-v1"} if dynamicvla else {}),
    }, sort_keys=True).encode()).hexdigest()
    manifest = VideoManifest(
        settings_hash=settings_hash, clips_per_outcome=videos.clips_per_outcome, resolution=None if dynamicvla else videos.resolution,
        source="recorded_policy_rgb" if recorded else "simulator_replay",
        shortfall={outcome: groups * videos.clips_per_outcome - sum(selection.outcome == outcome for selection in selections)
                   for outcome in ("success", "failure")},
    )
    manifest_path = point_dir / VIDEO_MANIFEST
    if manifest_path.exists():
        previous = yaml.safe_load(manifest_path.read_text())
        if previous.get("settings_hash") == settings_hash:
            old = VideoManifest.model_validate(previous)
            manifest = manifest.model_copy(update={"clips": [clip for clip in old.clips if
                (point_dir / clip.path).is_file() and sha256(point_dir / clip.path) == clip.sha256]})
    completed = {clip.path for clip in manifest.clips}
    jobs = [
        (selection, camera, point_dir / "videos" / f"task{selection.task_id:02d}_ep{selection.episode_id:03d}_{selection.outcome}_{camera}.mp4", videos.resolution)
        for selection in selections for camera in videos.cameras
    ]
    pending = [job for job in jobs if job[2].relative_to(point_dir).as_posix() not in completed]
    if not pending:
        manifest = manifest.model_copy(update={"complete": True})
        write_record(manifest, manifest_path)
        return manifest
    # A partial manifest is a per-clip receipt, never a claim that filming finished.
    write_record(manifest, manifest_path)
    errors = []
    grouped = dynamicvla and not recorded and len(videos.cameras) > 1
    batches = []
    if grouped:
        for selection in selections:
            jobs_for_episode = [job for job in pending if job[0] == selection]
            if jobs_for_episode:
                batches.append(jobs_for_episode)
    else:
        batches = [[job] for job in pending]
    with (gpu_locks([gpu]) if dynamicvla and not recorded else nullcontext()), ProcessPoolExecutor(max_workers=min(workers, len(pending)), mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as executor:
        futures = {executor.submit(_render_episode if grouped else _render_one,
                                   batch if grouped else batch[0]): batch for batch in batches}
        for future in as_completed(futures):
            try:
                future.result()
                for selection, camera, path, _ in futures[future]:
                    clip = Clip(outcome=selection.outcome, task_id=selection.task_id, episode_id=selection.episode_id,
                                camera=camera, path=path.relative_to(point_dir).as_posix(), sha256=sha256(path))
                    manifest.clips.append(clip)
                write_record(manifest, manifest_path)
            except Exception as error:
                errors.append(error)
    if errors:
        raise RuntimeError(f"{len(errors)} clips failed; rerun study videos to resume: {errors[0]}") from errors[0]
    manifest = manifest.model_copy(update={"complete": True, "clips": sorted(manifest.clips, key=lambda clip: clip.path)})
    write_record(manifest, manifest_path)
    return manifest


def _render_episode(jobs) -> None:
    """Capture the selected episode once, then publish its requested cameras."""
    from vlalagbench.benchmarks.dynamicvla.trace_runner import run_trace_operations

    selection, _, first, _ = jobs[0]
    first.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".render-", dir=first.parent) as directory:
        report = run_trace_operations(
            selection.trace, selection.runtime, video=Path(directory)/"video.mp4",
            camera=None, verify=False, rejudge=False, atol=0.0,
            simulator_gpu=selection.gpu, video_path_tracing_spp=selection.path_tracing_spp,
            separate_cameras=True,
        )
        rendered = {camera: Path(report["videos"][camera]) for _, camera, _, _ in jobs}
        if not all(path.is_file() for path in rendered.values()):
            raise RuntimeError("renderer did not produce every requested camera")
        for _, camera, output, _ in jobs:
            rendered[camera].replace(output)


def _render_one(job: tuple[Selection, str, Path, int]) -> Path:
    selection, camera, output, resolution = job
    temporary = output.with_name(output.stem + ".partial.mp4")
    try:
        if selection.recorded_video is not None:
            if camera not in selection.recorded_cameras:
                raise ValueError(f"recorded video has no camera {camera!r}")
            count = len(selection.recorded_cameras)
            index = selection.recorded_cameras.index(camera)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(selection.recorded_video),
                "-vf", f"crop=iw/{count}:ih:iw*{index}/{count}:0", "-an", "-c:v", "libx264",
                "-threads", "1", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(temporary)], check=True)
        elif selection.kind == "dynamicvla":
            from vlalagbench.benchmarks.dynamicvla.trace_runner import run_trace_operations

            run_trace_operations(selection.trace, selection.runtime, video=temporary, camera=camera,
                                 verify=False, rejudge=False, atol=0.0, simulator_gpu=selection.gpu,
                                 **({"video_path_tracing_spp": selection.path_tracing_spp}
                                    if selection.path_tracing_spp is not None else {}))
        else:
            from vlalagbench.benchmarks.libero.envs import make_env_from_trace
            from vlalagbench.benchmarks.libero.tracing import load_libero_trace
            from vlalagbench.simulators.mujoco.replay import render_trace_video

            render_trace_video(load_libero_trace(selection.trace), temporary, make_env_from_trace, camera=camera, resolution=resolution)
        temporary.replace(output)
        for suffix in ("-report.json", "-trace-tool.log"):
            temporary.with_name(temporary.stem + suffix).unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return output
