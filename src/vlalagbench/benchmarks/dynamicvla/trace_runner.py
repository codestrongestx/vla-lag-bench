"""Launch DynamicVLA trace operations in the pinned simulator environments."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from vlalagbench.benchmarks.dynamicvla import adapter
from vlalagbench.benchmarks.dynamicvla.continuation import preflight_continuation
from vlalagbench.benchmarks.dynamicvla.tracing import DynamicVLATrace
from vlalagbench.config import DynamicVLARuntime, load_runtime


def run_trace_operations(
    trace_path: Path,
    runtime_path: Path,
    *,
    video: Path | None,
    camera: str | None,
    verify: bool,
    rejudge: bool,
    atol: float,
    simulator_gpu: int | None = None,
    video_path_tracing_spp: int | None = None,
    separate_cameras: bool = False,
) -> dict[str, Any]:
    if separate_cameras and video is None:
        raise ValueError("separate cameras requires a video output")
    if video_path_tracing_spp is not None and (
        video is None or verify or rejudge or not 1 <= video_path_tracing_spp <= 128
    ):
        raise ValueError("path-tracing samples must be 1..128 and require video-only replay")
    trace_path = trace_path.expanduser().resolve()
    trace, runtime = _inputs(trace_path, runtime_path)
    if simulator_gpu is not None:
        runtime = runtime.model_copy(update={"simulator_gpu": simulator_gpu})
    if video is None:
        report_path = trace_path.with_name(f"{trace_path.stem}-trace-report.json")
        log_path = trace_path.with_name(f"{trace_path.stem}-trace-tool.log")
    else:
        video = video.expanduser().resolve()
        video.parent.mkdir(parents=True, exist_ok=True)
        report_path = video.with_name(f"{video.stem}-report.json")
        log_path = video.with_name(f"{video.stem}-trace-tool.log")
    report_path.unlink(missing_ok=True)
    command, env = _simulator_command(
        trace_path, runtime, report_path, video=video, camera=camera
    )
    if video_path_tracing_spp is not None:
        command.extend(["--video-path-tracing-spp", str(video_path_tracing_spp)])
    if separate_cameras:
        command.append("--separate-cameras")
    if verify:
        command.append("--verify")
    if rejudge:
        command.append("--rejudge")
    command.extend(["--atol", str(atol)])
    _run_logged(
        command,
        env,
        Path(runtime.dynamicvla_root),
        log_path,
        timeout_s=runtime.suite_timeout_s,
    )
    if not report_path.is_file():
        raise RuntimeError(
            "DynamicVLA trace tool produced no report\n"
            f"trace tool log:\n{adapter._tail(log_path)}"
        )
    return json.loads(report_path.read_text())


def continue_trace(
    trace_path: Path,
    runtime_path: Path,
    out_trace: Path,
    *,
    additional_steps: int,
    resume_step: int | None,
    checkpoint: Path | None,
) -> dict[str, Any]:
    trace_path = trace_path.expanduser().resolve()
    out_trace = out_trace.expanduser().resolve()
    trace, runtime = _inputs(trace_path, runtime_path)
    if additional_steps <= 0:
        raise ValueError("additional_steps must be positive")
    if out_trace == trace_path:
        raise ValueError("continued trace must not overwrite its source")
    checkpoint = (checkpoint or Path(trace.meta.checkpoint)).expanduser().resolve()
    model_path = checkpoint / "model.safetensors"
    if adapter._sha256(model_path) != trace.meta.checkpoint_sha256:
        raise ValueError(f"checkpoint checksum does not match trace: {model_path}")
    config_path = checkpoint / "config.json"
    if adapter._sha256(config_path) != trace.meta.checkpoint_config_sha256:
        raise ValueError(f"checkpoint config checksum does not match trace: {config_path}")
    checkpoint_config = json.loads(config_path.read_text())
    resolved_resume_step = trace.num_steps if resume_step is None else resume_step
    preflight_continuation(trace, resolved_resume_step, checkpoint_config)
    report_path = out_trace.with_name(f"{out_trace.stem}-report.json")
    report_path.unlink(missing_ok=True)
    simulator_command, simulator_env = _simulator_command(
        trace_path, runtime, report_path, video=None, camera=None
    )
    simulator_command.extend(
        [
            "--continue-steps",
            str(additional_steps),
            "--out-trace",
            str(out_trace),
            "--checkpoint",
            str(checkpoint),
            "--host",
            runtime.host,
            "--img-port",
            str(runtime.image_port),
            "--act-port",
            str(runtime.action_port),
            "--startup-timeout-s",
            str(runtime.startup_timeout_s),
            "--reply-timeout-s",
            str(runtime.reply_timeout_s),
        ]
    )
    if resume_step is not None:
        simulator_command.extend(["--resume-step", str(resume_step)])

    common_env = _common_env(runtime)
    policy_command = [
        str(adapter._runtime_executable(runtime.policy_python)),
        "-m",
        "vlalagbench.benchmarks.dynamicvla.policy",
        "--dynamicvla-root",
        str(Path(runtime.dynamicvla_root).expanduser().resolve()),
        "--weights",
        str(checkpoint),
        "--host",
        runtime.host,
        "--img-port",
        str(runtime.image_port),
        "--act-port",
        str(runtime.action_port),
        "--mode",
        trace.meta.inference_mode,
        "--execution-horizon-steps",
        str(trace.meta.execution_horizon_steps),
        "--latency-steps",
        str(trace.meta.latency_steps),
        "--reply-timeout-s",
        str(runtime.reply_timeout_s),
    ]
    out_trace.parent.mkdir(parents=True, exist_ok=True)
    simulator_log = out_trace.with_name(f"{out_trace.stem}-simulator.log")
    policy_log = out_trace.with_name(f"{out_trace.stem}-policy.log")
    simulator = policy = None
    with simulator_log.open("w", encoding="utf-8") as sim_stream, policy_log.open(
        "w", encoding="utf-8"
    ) as policy_stream:
        try:
            simulator = subprocess.Popen(
                simulator_command,
                cwd=runtime.dynamicvla_root,
                env=simulator_env,
                stdout=sim_stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            adapter._wait_for_ports(
                simulator,
                runtime.host,
                (runtime.image_port, runtime.action_port),
                runtime.startup_timeout_s,
            )
            policy = subprocess.Popen(
                policy_command,
                cwd=runtime.dynamicvla_root,
                env=common_env,
                stdout=policy_stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            adapter._wait_for_processes(
                simulator,
                policy,
                runtime.suite_timeout_s,
                failure_flush_timeout_s=2 * runtime.reply_timeout_s + 30.0,
            )
        except Exception as exc:
            raise RuntimeError(
                f"{exc}\nsimulator log:\n{adapter._tail(simulator_log)}"
                f"\npolicy log:\n{adapter._tail(policy_log)}"
            ) from exc
        finally:
            adapter._stop_process(policy)
            adapter._stop_process(simulator)
    if not report_path.is_file():
        raise RuntimeError(
            "DynamicVLA continuation produced no report\n"
            f"simulator log:\n{adapter._tail(simulator_log)}"
        )
    return json.loads(report_path.read_text())["continuation"]


def _inputs(
    trace_path: Path, runtime_path: Path
) -> tuple[DynamicVLATrace, DynamicVLARuntime]:
    trace_path = trace_path.expanduser().resolve()
    runtime = load_runtime(runtime_path)
    if not isinstance(runtime, DynamicVLARuntime):
        raise TypeError("DynamicVLA trace operations require a DynamicVLA runtime")
    trace = DynamicVLATrace.load(trace_path)
    root = Path(runtime.dynamicvla_root).expanduser().resolve()
    revision = adapter._command(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        "read DynamicVLA revision",
    )
    if revision != trace.meta.upstream_revision:
        raise ValueError(
            f"DynamicVLA checkout is {revision}, trace requires "
            f"{trace.meta.upstream_revision}"
        )
    return trace, runtime


def _common_env(runtime: DynamicVLARuntime) -> dict[str, str]:
    src_root = Path(__file__).resolve().parents[3]
    root = Path(runtime.dynamicvla_root).expanduser().resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src_root), str(root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    if runtime.policy_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(runtime.policy_gpu)
    return env


def _simulator_command(
    trace_path: Path,
    runtime: DynamicVLARuntime,
    report_path: Path,
    *,
    video: Path | None,
    camera: str | None,
) -> tuple[list[str], dict[str, str]]:
    root = Path(runtime.dynamicvla_root).expanduser().resolve()
    assets = Path(runtime.assets_root).expanduser().resolve()
    simulator_python = adapter._runtime_executable(runtime.simulator_python)
    env = _common_env(runtime)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    if runtime.simulator_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(runtime.simulator_gpu)
    library_paths = [
        *runtime.simulator_library_paths,
        *adapter._discover_isaac_library_paths(simulator_python),
    ]
    if library_paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*library_paths, env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)
    command = [
        str(simulator_python),
        "-m",
        "vlalagbench.benchmarks.dynamicvla.trace_tool",
        "--dynamicvla-root",
        str(root),
        "--assets-root",
        str(assets),
        "--trace",
        str(trace_path.expanduser().resolve()),
        "--report",
        str(report_path.expanduser().resolve()),
        "--headless",
        "--enable_cameras",
    ]
    command.extend(adapter._renderer_gpu_args(runtime))
    command.extend(adapter._renderer_setting_args(runtime))
    if video is not None:
        command.extend(["--video", str(video.expanduser().resolve())])
    if camera is not None:
        command.extend(["--camera", camera])
    return command, env


def _run_logged(
    command: list[str],
    env: dict[str, str],
    cwd: Path,
    log_path: Path,
    *,
    timeout_s: float,
) -> None:
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"DynamicVLA trace tool exited with {completed.returncode}\n"
            f"trace tool log:\n{adapter._tail(log_path)}"
        )
