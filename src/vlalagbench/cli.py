"""Command-line entry points.

- ``vlalagbench eval``: run one validated experiment/runtime YAML pair.
- ``vlalagbench replay``: render and/or verify recorded LIBERO MuJoCo traces.
- ``vlalagbench dynamicvla-trace``: render or audit Isaac Lab state traces.
- ``vlalagbench dynamicvla-continue``: resume a trace at a longer deadline.
- ``vlalagbench study``: plan and run a whole sweep from a study and a placement.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib


def _add_eval_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("eval", help="run a LIBERO or DynamicVLA experiment and record traces")
    p.add_argument("experiment", help="path to an experiment YAML")
    p.add_argument("runtime", help="path to configs/runtime/*.yaml")


def _add_replay_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("replay", help="render and/or verify LIBERO MuJoCo traces")
    p.add_argument("traces", nargs="+", help="paths to .npz trace files")
    p.add_argument("--camera", default="agentview", help="e.g. agentview, robot0_eye_in_hand")
    p.add_argument("--resolution", type=int, default=512, help="video resolution (square)")
    p.add_argument("--fps", type=float, default=None, help="default: recorded control frequency")
    p.add_argument("--out-dir", default=None, help="default: next to each trace")
    p.add_argument("--verify", action="store_true", help="also re-simulate actions and report drift")
    p.add_argument("--no-video", action="store_true", help="skip video rendering")


def _add_dynamicvla_trace_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "dynamicvla-trace", help="replay or independently verify a DynamicVLA trace"
    )
    p.add_argument("trace")
    p.add_argument("runtime")
    p.add_argument("--video", help="re-render camera frames from saved states into this MP4")
    p.add_argument("--camera", help="default: every camera")
    p.add_argument("--separate-cameras", action="store_true", help="write one MP4 per camera from one replay")
    p.add_argument("--video-path-tracing-spp", type=int,
                   help="path-tracing samples per pixel for selected video replay only")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--rejudge", action="store_true")
    p.add_argument("--atol", type=float, default=1e-5)


def _add_dynamicvla_continue_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "dynamicvla-continue", help="extend a failed DynamicVLA trace deadline"
    )
    p.add_argument("trace")
    p.add_argument("runtime")
    p.add_argument("out_trace")
    p.add_argument("--additional-steps", type=int, required=True)
    p.add_argument("--resume-step", type=int)
    p.add_argument("--checkpoint")


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _add_study_parser(sub: argparse._SubParsersAction) -> None:
    study = sub.add_parser("study", help="evaluate, verify, render, or report a study")
    actions = study.add_subparsers(dest="action", required=True)
    for action in ("plan", "run"):
        parser = actions.add_parser(action)
        parser.add_argument("study", help="study specification YAML")
        parser.add_argument("--placement", required=True, help="compute placement YAML")
        parser.add_argument("--select", action="append", default=[], metavar="KEY=VALUE")
    for action in ("status", "verify", "videos", "report"):
        parser = actions.add_parser(action)
        parser.add_argument("directory", type=pathlib.Path, help="recorded study directory (local or copied)")
        parser.add_argument("--select", action="append", default=[], metavar="KEY=VALUE")
        if action == "videos":
            parser.add_argument("--runtime", type=pathlib.Path, help="local DynamicVLA replay runtime; leaves the recorded runtime unchanged")
            parser.add_argument("--gpu", type=int, help="local rendering GPU (LIBERO: EGL default; DynamicVLA: GPU 0, independent of the recording GPU)")
            parser.add_argument("--cpus", help="Linux CPU set, e.g. 0-13")
            parser.add_argument("--workers", type=_positive_int, default=4)
            parser.add_argument("--camera", action="append", help="repeat for multiple cameras")
            parser.add_argument("--resolution", type=_positive_int, default=256)
            parser.add_argument("--clips-per-outcome", type=_positive_int, default=3)
            parser.add_argument("--per-task", action="store_true", help="select --clips-per-outcome clips per task instead of per point")
        elif action == "report":
            parser.add_argument("--publish", type=pathlib.Path, metavar="DIR")


def _run_study(args: argparse.Namespace) -> int:
    import contextlib
    import os

    from vlalagbench.runner import REPO_ROOT
    from vlalagbench.config import Shard, Videos, DynamicVLAPlacement, DynamicVLAExperiment
    from vlalagbench.study import runner
    from vlalagbench.study.spec import load_placement, load_study, open_study, resolve
    from vlalagbench.study.point import POINT_RECORD, read_point, verify_files, verify_point

    if args.action in {"plan", "run"}:
        plan = resolve(load_study(args.study), load_placement(args.placement), REPO_ROOT, selects=args.select)
        if args.action == "run":
            return runner.run(plan, pathlib.Path(args.study).resolve(), pathlib.Path(args.placement).resolve(), args.select)
        print(f"study: {plan.spec.name}  hash: {plan.study_hash}")
        print(f"output: {plan.study_dir}\npoints: {len(plan.queue)} of {len(plan.points)} queued")
        for index, point in enumerate(plan.queue, start=1):
            print(f"  {index:3d}. {point.name}")
        for index, shard in enumerate(plan.placement.shards):
            if isinstance(plan.placement, DynamicVLAPlacement):
                print(f"shard {index:02d}: policy gpu {shard.policy_gpu}  simulator gpu {shard.simulator_gpu}  ports {shard.image_port}/{shard.action_port}  cpus {shard.cpus}")
            else:
                print(f"shard {index:02d}: policy gpu {shard.policy_gpu}  port {shard.port}  cpus {shard.cpus}  workers {plan.placement.workers_per_shard}")
        if not isinstance(plan.placement, DynamicVLAPlacement):
            print(f"environment rendering gpu: {plan.placement.render_gpu}")
        print("stages: evaluate, verify. Videos are a separate command.")
        print("invariants: ok" if plan.spec.schema_version == 2 else "historical schema 1: read-only")
        return 0
    plan = open_study(args.directory, selects=args.select)
    if args.action == "status":
        runner.status(plan)
        return 0
    if args.action == "report":
        from vlalagbench.study.report import report

        with runner.study_lock(plan.study_dir, "report.lock"):
            outputs = report(plan, args.publish)
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
        return 0
    with contextlib.ExitStack() as stack:
        stack.enter_context(runner.study_lock(plan.study_dir))
        if args.action == "videos":
            from vlalagbench.study.videos import film_point

            dynamicvla = isinstance(plan.points[0].experiment, DynamicVLAExperiment)
            if args.runtime is not None and not dynamicvla:
                raise ValueError("--runtime is only supported for DynamicVLA study videos")
            settings = Videos(clips_per_outcome=args.clips_per_outcome, cameras=args.camera or (["opst_cam"] if dynamicvla else ["agentview", "robot0_eye_in_hand"]), resolution=args.resolution, per_task=True if args.per_task else None)
            if args.gpu is not None:
                if args.gpu < 0:
                    raise ValueError("--gpu must be nonnegative")
                if not dynamicvla:
                    stack.enter_context(runner.gpu_locks([args.gpu]))
                    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(args.gpu), CUDA_VISIBLE_DEVICES=str(args.gpu))
            if args.cpus:
                cpus = Shard(policy_gpu=0, port=1, cpus=args.cpus).cpu_set
                if not cpus or len(cpus) > 32 or args.workers > len(cpus):
                    raise ValueError("--cpus must contain 1 to 32 cores and cover --workers")
                if not hasattr(os, "sched_setaffinity"):
                    raise ValueError("--cpus requires Linux")
                os.sched_setaffinity(0, cpus)
        failures = 0
        for point in plan.queue:
            point_dir = plan.study_dir / "points" / point.name
            has_record = (point_dir / POINT_RECORD).exists()
            if not has_record and not (args.action == "verify" and runner._completed_result(point_dir)):
                continue
            try:
                record = read_point(point_dir, plan.study_hash) if has_record else None
                if record is not None:
                    verify_files(point_dir, record)
                if args.action == "videos":
                    film_point(point_dir, point.name, settings, args.workers, gpu=args.gpu, runtime_path=args.runtime)
                else:
                    if record is not None and not record.files:
                        raise ValueError("historical point has no original trace inventory; its integrity cannot be retroactively established")
                    if has_record:
                        verify_point(point_dir, name=point.name, values=point.values, experiment=point.experiment,
                                     study_hash=plan.study_hash, policy=plan.spec.policy if plan.spec.schema_version == 2 else None)
                    else:
                        if plan.spec.schema_version != 2:
                            raise ValueError("only schema 2 completed evaluations can receive a new verification record")
                        runner._verify(plan, point)
                    if not has_record:
                        previous = runner.state(plan.study_dir, point.name)
                        if previous is not None and previous[0] != "done":
                            suffix = f"claimed.{previous[1]}" if previous[0] == "claimed" else previous[0]
                            marker = plan.study_dir / "queue" / f"{point.name}.{suffix}"
                            marker.rename(marker.with_name(f"{point.name}.done"))
                        (point_dir / "error.txt").unlink(missing_ok=True)
                print(f"{point.name}: {args.action} complete")
            except Exception:
                logging.exception("%s: %s failed", point.name, args.action)
                failures += 1
        return int(failures > 0)


def _run_eval(args: argparse.Namespace) -> int:
    from vlalagbench.runner import REPO_ROOT, run_experiment

    result = run_experiment(args.experiment, args.runtime)
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    print(f"result: {REPO_ROOT / 'runs' / result.run_id / 'result.yaml'}")
    return 0


def _run_replay(args: argparse.Namespace) -> int:
    from vlalagbench.benchmarks.libero.envs import make_env_from_trace
    from vlalagbench.benchmarks.libero.evaluator import idle_step
    from vlalagbench.benchmarks.libero.tracing import load_libero_trace
    from vlalagbench.simulators.mujoco.replay import render_trace_video, verify_trace

    status = 0
    for trace_path in map(pathlib.Path, args.traces):
        trace = load_libero_trace(trace_path)
        print(f"{trace_path.name}: task={trace.meta.task_description!r} "
              f"steps={trace.num_steps} success={trace.meta.success}")
        if not args.no_video:
            out_dir = pathlib.Path(args.out_dir) if args.out_dir else trace_path.parent
            video = render_trace_video(
                trace,
                out_dir / f"{trace_path.stem}_{args.camera}.mp4",
                env_factory=make_env_from_trace,
                camera=args.camera,
                resolution=args.resolution,
                fps=args.fps,
            )
            print(f"  video: {video}")
        if args.verify:
            report = verify_trace(trace, env_factory=make_env_from_trace, idle_step=idle_step)
            print(f"  verify: {json.dumps(report)}")
            if not report["within_atol"]:
                status = 1
    return status


def _run_dynamicvla_trace(args: argparse.Namespace) -> int:
    from vlalagbench.benchmarks.dynamicvla.trace_runner import run_trace_operations

    report = run_trace_operations(
        pathlib.Path(args.trace),
        pathlib.Path(args.runtime),
        video=pathlib.Path(args.video) if args.video else None,
        camera=args.camera,
        verify=args.verify,
        rejudge=args.rejudge,
        atol=args.atol,
        video_path_tracing_spp=args.video_path_tracing_spp,
        separate_cameras=args.separate_cameras,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    verification = report.get("verification")
    rejudgement = report.get("rejudgement")
    return int(
        bool(verification and not verification["within_atol"])
        or bool(rejudgement and not rejudgement["matches_recorded"])
    )


def _run_dynamicvla_continue(args: argparse.Namespace) -> int:
    from vlalagbench.benchmarks.dynamicvla.trace_runner import continue_trace

    report = continue_trace(
        pathlib.Path(args.trace),
        pathlib.Path(args.runtime),
        pathlib.Path(args.out_trace),
        additional_steps=args.additional_steps,
        resume_step=args.resume_step,
        checkpoint=pathlib.Path(args.checkpoint) if args.checkpoint else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="vlalagbench")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_eval_parser(sub)
    _add_replay_parser(sub)
    _add_dynamicvla_trace_parser(sub)
    _add_dynamicvla_continue_parser(sub)
    _add_study_parser(sub)
    args = parser.parse_args(argv)
    if args.command == "eval":
        return _run_eval(args)
    if args.command == "replay":
        return _run_replay(args)
    if args.command == "dynamicvla-trace":
        return _run_dynamicvla_trace(args)
    if args.command == "dynamicvla-continue":
        return _run_dynamicvla_continue(args)
    if args.command == "study":
        return _run_study(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
