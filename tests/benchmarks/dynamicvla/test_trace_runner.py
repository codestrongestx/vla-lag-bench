from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from vlalagbench.benchmarks.dynamicvla import trace_runner


def test_trace_operations_resolve_the_input_and_return_the_report(tmp_path, monkeypatch):
    trace_path = tmp_path / "episode.npz"
    trace_path.touch()
    report_path = tmp_path / "episode-trace-report.json"
    report_path.write_text(json.dumps({"summary": {"steps": 999}}))
    seen = {}

    def inputs(path, runtime_path):
        seen["trace"] = path
        seen["runtime"] = runtime_path
        return object(), SimpleNamespace(dynamicvla_root=tmp_path, suite_timeout_s=30.0)

    monkeypatch.setattr(trace_runner, "_inputs", inputs)
    monkeypatch.setattr(
        trace_runner,
        "_simulator_command",
        lambda *_args, **_kwargs: (["trace-tool"], {}),
    )
    monkeypatch.setattr(
        trace_runner,
        "_run_logged",
        lambda *_args, **_kwargs: report_path.write_text(
            json.dumps({"summary": {"steps": 4}})
        ),
    )

    report = trace_runner.run_trace_operations(
        trace_path,
        tmp_path / "runtime.yaml",
        video=None,
        camera=None,
        verify=False,
        rejudge=False,
        atol=1e-5,
    )

    assert seen["trace"] == trace_path.resolve()
    assert report == {"summary": {"steps": 4}}


def test_video_outputs_do_not_replace_the_trace_audit(tmp_path, monkeypatch):
    trace_path = tmp_path / "episode.npz"
    trace_path.touch()
    trace_report = tmp_path / "episode-trace-report.json"
    trace_report.write_text(json.dumps({"verification": {"exact": True}}))
    video_path = tmp_path / "replays" / "episode-front.mp4"
    seen = {}

    monkeypatch.setattr(
        trace_runner,
        "_inputs",
        lambda *_args: (
            object(),
            SimpleNamespace(dynamicvla_root=tmp_path, suite_timeout_s=30.0),
        ),
    )

    def simulator_command(_trace, _runtime, report_path, *, video, camera):
        seen.update(report=report_path, video=video, camera=camera)
        return ["trace-tool"], {}

    def run_logged(_command, _env, _cwd, log_path, *, timeout_s):
        seen.update(log=log_path, timeout=timeout_s, command=_command)
        seen["report"].write_text(json.dumps({"video": str(video_path)}))

    monkeypatch.setattr(trace_runner, "_simulator_command", simulator_command)
    monkeypatch.setattr(trace_runner, "_run_logged", run_logged)

    report = trace_runner.run_trace_operations(
        trace_path,
        tmp_path / "runtime.yaml",
        video=video_path,
        camera="front",
        verify=False,
        rejudge=False,
        atol=1e-5,
        video_path_tracing_spp=64,
    )

    assert report == {"video": str(video_path)}
    assert seen["report"] == video_path.with_name("episode-front-report.json")
    assert seen["log"] == video_path.with_name("episode-front-trace-tool.log")
    assert seen["video"] == video_path.resolve()
    assert seen["camera"] == "front"
    assert seen["command"][-4:] == ["--video-path-tracing-spp", "64", "--atol", "1e-05"]
    assert json.loads(trace_report.read_text()) == {"verification": {"exact": True}}


def test_continuation_preflight_runs_before_artifacts_or_processes(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model_path = checkpoint / "model.safetensors"
    config_path = checkpoint / "config.json"
    model_path.write_bytes(b"weights")
    config_path.write_text(
        json.dumps(
            {
                "type": "dynamicvla",
                "n_action_steps": 20,
                "delta_timestamps": {"observation": [-2, 0]},
            }
        )
    )
    trace = SimpleNamespace(
        num_steps=3,
        meta=SimpleNamespace(
            checkpoint=str(checkpoint),
            checkpoint_sha256=trace_runner.adapter._sha256(model_path),
            checkpoint_config_sha256=trace_runner.adapter._sha256(config_path),
        ),
    )
    runtime = SimpleNamespace()
    monkeypatch.setattr(trace_runner, "_inputs", lambda *_args: (trace, runtime))
    monkeypatch.setattr(
        trace_runner,
        "preflight_continuation",
        lambda *_args: (_ for _ in ()).throw(ValueError("not restorable")),
    )
    monkeypatch.setattr(
        trace_runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("continuation started a process"),
    )
    out_trace = tmp_path / "continued.npz"
    report_path = tmp_path / "continued-report.json"
    report_path.write_text("preserve")

    with pytest.raises(ValueError, match="not restorable"):
        trace_runner.continue_trace(
            tmp_path / "source.npz",
            tmp_path / "runtime.yaml",
            out_trace,
            additional_steps=2,
            resume_step=None,
            checkpoint=None,
        )

    assert report_path.read_text() == "preserve"


def test_replay_command_pins_vulkan_gpu_and_forwards_renderer_override(tmp_path, monkeypatch):
    from vlalagbench.config import DynamicVLARuntime

    runtime = DynamicVLARuntime(
        schema_version=3, kind="dynamicvla", name="replay-test",
        dynamicvla_root=str(tmp_path), assets_root=str(tmp_path),
        simulator_python=sys.executable, policy_python=sys.executable,
        policy_gpu=0, simulator_gpu=3, antialiasing_mode="FXAA", enable_dl_denoiser=False,
        dlss_mode="Balanced", enable_dlssg=False, render_mode="RaytracedLighting",
    )
    monkeypatch.setattr(trace_runner.adapter, "_discover_isaac_library_paths", lambda _: [])
    command, env = trace_runner._simulator_command(
        tmp_path / "trace.npz", runtime, tmp_path / "report.json",
        video=tmp_path / "video.mp4", camera="wrist_cam",
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert command[command.index("--antialiasing-mode") + 1] == "FXAA"
    assert "--no-enable-dl-denoiser" in command
    assert "--no-enable-dlssg" in command
    assert command[command.index("--dlss-mode") + 1] == "Balanced"
    assert command[command.index("--render-mode") + 1] == "RaytracedLighting"
    kit_args = command[command.index("--kit_args") + 1]
    assert "--/renderer/multiGpu/enabled=false" in kit_args
    assert "--/renderer/activeGpu=3" in kit_args
