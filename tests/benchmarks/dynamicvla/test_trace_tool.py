import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from vlalagbench.benchmarks.dynamicvla import trace_tool


def test_main_writes_report_before_isaac_fast_shutdown(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    trace = SimpleNamespace(
        meta=SimpleNamespace(
            task_id=0,
            episode=0,
            success=False,
            episode_length_s=1.0,
        ),
        num_steps=1,
        states={},
        actions=np.zeros((1, 1, 8)),
        protocol=[{}],
        final_policy={},
    )

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--headless", action="store_true")

        def __init__(self, _args):
            self.app = SimpleNamespace(close=self._fast_shutdown)

        @staticmethod
        def _fast_shutdown(*, wait_for_replicator):
            assert wait_for_replicator is False
            raise SystemExit(0)

    simulations = types.ModuleType("simulations")
    evaluate = types.ModuleType("simulations.evaluate")
    evaluate.AppLauncher = AppLauncher
    simulations.evaluate = evaluate
    monkeypatch.setitem(sys.modules, "simulations", simulations)
    monkeypatch.setitem(sys.modules, "simulations.evaluate", evaluate)
    monkeypatch.setattr(trace_tool.DynamicVLATrace, "load", lambda _path: trace)
    monkeypatch.setattr(trace_tool, "_validate_inputs", lambda *_args: None)

    with pytest.raises(SystemExit, match="0"):
        trace_tool.main(
            [
                "--dynamicvla-root",
                str(tmp_path),
                "--assets-root",
                str(tmp_path),
                "--trace",
                str(tmp_path / "trace.npz"),
                "--report",
                str(report_path),
                "--headless",
            ]
        )

    report = json.loads(report_path.read_text())
    assert report["summary"]["steps"] == 1


def test_continuation_is_rejected_before_app_launcher(tmp_path, monkeypatch):
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
        num_steps=1,
        meta=SimpleNamespace(
            checkpoint=str(checkpoint),
            checkpoint_sha256=trace_tool._sha256(model_path),
            checkpoint_config_sha256=trace_tool._sha256(config_path),
        ),
    )
    launched = []

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--headless", action="store_true")

        def __init__(self, _args):
            launched.append(True)

    simulations = types.ModuleType("simulations")
    evaluate = types.ModuleType("simulations.evaluate")
    evaluate.AppLauncher = AppLauncher
    simulations.evaluate = evaluate
    monkeypatch.setitem(sys.modules, "simulations", simulations)
    monkeypatch.setitem(sys.modules, "simulations.evaluate", evaluate)
    monkeypatch.setattr(trace_tool.DynamicVLATrace, "load", lambda _path: trace)
    monkeypatch.setattr(trace_tool, "_validate_inputs", lambda *_args: None)
    monkeypatch.setattr(
        trace_tool,
        "preflight_continuation",
        lambda *_args: (_ for _ in ()).throw(ValueError("not restorable")),
    )

    with pytest.raises(ValueError, match="not restorable"):
        trace_tool.main(
            [
                "--dynamicvla-root",
                str(tmp_path),
                "--assets-root",
                str(tmp_path),
                "--trace",
                str(tmp_path / "trace.npz"),
                "--continue-steps",
                "2",
                "--out-trace",
                str(tmp_path / "continued.npz"),
                "--headless",
            ]
        )

    assert launched == []


@pytest.mark.parametrize("recorded", ["2:FXAA", "3:DLSS"])
@pytest.mark.parametrize("override", [None, "FXAA", "DLSS"])
def test_replay_preserves_recorded_aa_unless_explicitly_overridden(recorded, override):
    trace = SimpleNamespace(meta=SimpleNamespace(simulator_versions={
        "isaaclab": "0.45.9", "renderer_antialiasing": recorded,
    }))
    args = SimpleNamespace(antialiasing_mode=override, continue_steps=None)
    assert trace_tool._replay_antialiasing(trace, args) == (override or recorded.split(":")[1])
    args.continue_steps = 1
    if override and override != recorded.split(":")[1]:
        with pytest.raises(ValueError, match="cannot override"):
            trace_tool._replay_antialiasing(trace, args)


def test_package_validation_separates_renderer_provenance(tmp_path, monkeypatch):
    trace = SimpleNamespace(meta=SimpleNamespace(
        upstream_revision="abc", asset_sha256={}, simulator_versions={
            "isaaclab": "0.45.9", "isaacsim": "4.5.0.0",
            "renderer_antialiasing": "2:FXAA", "renderer_active_gpu": "3",
        },
    ))
    monkeypatch.setattr(trace_tool.subprocess, "run", lambda command, **kwargs:
                        SimpleNamespace(stdout="abc" if "rev-parse" in command else ""))
    versions = {"isaaclab": "0.45.9", "isaacsim": "4.5.0.0"}
    monkeypatch.setattr(trace_tool.importlib.metadata, "version", versions.__getitem__)
    trace_tool._validate_inputs(trace, tmp_path, tmp_path)
    versions["isaacsim"] = "different"
    with pytest.raises(ValueError, match="simulator versions"):
        trace_tool._validate_inputs(trace, tmp_path, tmp_path)


def test_asset_validation_follows_symlinked_shared_assets_but_rejects_escapes(tmp_path, monkeypatch):
    shared = tmp_path / "shared" / "objects"
    shared.mkdir(parents=True)
    (shared / "can.usd").write_bytes(b"can")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "objects").symlink_to(shared)
    (tmp_path / "outside.usd").write_bytes(b"can")
    trace = SimpleNamespace(meta=SimpleNamespace(upstream_revision="abc", simulator_versions={},
        asset_sha256={"objects/can.usd": trace_tool._sha256(shared / "can.usd")}))
    monkeypatch.setattr(trace_tool.subprocess, "run", lambda command, **kwargs:
                        SimpleNamespace(stdout="abc" if "rev-parse" in command else ""))
    trace_tool._validate_inputs(trace, tmp_path, assets)
    trace.meta.asset_sha256 = {"../outside.usd": trace_tool._sha256(tmp_path / "outside.usd")}
    with pytest.raises(ValueError, match="escapes"):
        trace_tool._validate_inputs(trace, tmp_path, assets)


def test_replay_restores_recorded_dlss_execution_mode(monkeypatch):
    values = {"/rtx/post/dlss/execMode": 0}
    settings = SimpleNamespace(set=values.__setitem__, get=values.get)
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(
        settings=SimpleNamespace(get_settings=lambda: settings)))
    trace = SimpleNamespace(meta=SimpleNamespace(simulator_versions={
        "renderer_dlss_mode": "1:Balanced",
    }))
    trace_tool._restore_recorded_dlss_mode(trace)
    assert values["/rtx/post/dlss/execMode"] == 1
    trace_tool._restore_recorded_dlss_mode(trace, override="Quality")
    assert values["/rtx/post/dlss/execMode"] == 2


def test_replay_preserves_dl_denoiser_and_rejects_changed_continuation():
    trace = SimpleNamespace(meta=SimpleNamespace(simulator_versions={"renderer_dl_denoiser": "True"}))
    args = SimpleNamespace(enable_dl_denoiser=None, continue_steps=None)
    assert trace_tool._replay_dl_denoiser(trace, args) is True
    args.enable_dl_denoiser = False
    assert trace_tool._replay_dl_denoiser(trace, args) is False
    args.continue_steps = 1
    with pytest.raises(ValueError, match="cannot override"):
        trace_tool._replay_dl_denoiser(trace, args)


@pytest.mark.parametrize("field,value", [("dlss_mode", "Quality"), ("enable_dlssg", True), ("render_mode", "RaytracedLighting")])
def test_replay_renderer_overrides_rejected_for_continuation(field, value):
    trace = SimpleNamespace(meta=SimpleNamespace(simulator_versions={
        "renderer_dlss_mode": "1:Balanced", "renderer_dlssg": "False", "renderer_mode": "PathTracing",
    }))
    args = SimpleNamespace(continue_steps=None, **{field: value})
    assert trace_tool._replay_extra_renderer(trace, args)[field] == value
    args.continue_steps = 1
    with pytest.raises(ValueError, match="cannot override"):
        trace_tool._replay_extra_renderer(trace, args)


def test_path_tracing_is_explicit_video_only_and_does_not_mutate_provenance(monkeypatch):
    trace = SimpleNamespace(meta=SimpleNamespace(simulator_versions={"renderer_mode": "RaytracedLighting"}))
    args = SimpleNamespace(continue_steps=None, video="out.mp4", verify=False,
                           rejudge=False, video_path_tracing_spp=64)
    assert trace_tool._replay_extra_renderer(trace, args)["render_mode"] == "PathTracing"
    assert trace.meta.simulator_versions["renderer_mode"] == "RaytracedLighting"
    values = {}
    settings = SimpleNamespace(set=values.__setitem__, get=values.get)
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(
        settings=SimpleNamespace(get_settings=lambda: settings)))
    assert trace_tool._configure_video_samples(64) == {
        "/rtx/pathtracing/spp": 64, "/rtx/pathtracing/totalSpp": 64}
    for field, value in [("video", None), ("verify", True), ("rejudge", True), ("continue_steps", 1)]:
        changed = SimpleNamespace(**{**vars(args), field: value})
        with pytest.raises(ValueError, match="video-only"):
            trace_tool._replay_extra_renderer(trace, changed)
    args.video_path_tracing_spp = 0
    with pytest.raises(ValueError, match="between"):
        trace_tool._replay_extra_renderer(trace, args)
