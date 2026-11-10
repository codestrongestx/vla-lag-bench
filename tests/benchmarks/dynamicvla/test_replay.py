from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from vlalagbench.benchmarks.dynamicvla.replay import (
    rejudge_trace,
    render_trace_video,
    render_diagnostic_video,
    state_summary,
    verify_trace_actions,
)
from vlalagbench.benchmarks.dynamicvla.tracing import (
    DynamicVLATrace,
    DynamicVLATraceMeta,
)
from vlalagbench.simulators.isaaclab.tracing import capture_env_state


class _Sensor:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class _Scene:
    device = "cpu"

    def __init__(self):
        self.position = torch.zeros((1, 1), dtype=torch.float32)
        self.sensors = {"front": _Sensor()}
        self.updates = []

    def get_state(self, *, is_relative):
        assert not is_relative
        return {"rigid_object": {"object": {"root_pose": self.position}}}

    def reset_to(self, state, *, is_relative):
        assert not is_relative
        self.position.copy_(state["rigid_object"]["object"]["root_pose"])

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        self.updates.append(dt)


class _Env:
    def __init__(self):
        self.unwrapped = self
        self.scene = _Scene()
        self.device = "cpu"
        self.step_dt = 0.04
        self.sim = SimpleNamespace(forward=lambda: None, render=lambda: None)
        self.recorder_manager = SimpleNamespace(record_pre_reset=lambda _ids: None)
        term = SimpleNamespace(
            func=lambda env, **_params: env.scene.position[:, 0] >= 2,
            params={},
            time_out=False,
        )
        time_out = SimpleNamespace(
            func=lambda env, **_params: env.scene.position[:, 0] > 100,
            params={},
            time_out=True,
        )
        terms = {"success": term, "time_out": time_out}
        self.termination_manager = SimpleNamespace(
            active_terms=list(terms), get_term_cfg=terms.__getitem__
        )

    def step(self, action):
        self.scene.position[:, 0] += action[:, 0]


def _trace() -> DynamicVLATrace:
    rng_states = {
        path: np.stack([value, value, value])
        for path, value in capture_env_state(_Env()).items()
        if path[:2] == ("runtime", "rng")
    }
    return DynamicVLATrace(
        meta=DynamicVLATraceMeta.model_validate(
            {
                "schema_version": 5,
                "task_id": 0,
                "task_name": "example-task",
                "episode": 1,
                "environment_index": 0,
                "replica_id": 1,
                "environment_path": "/assets/example.json",
                "environment_sha256": "a" * 64,
                "environment": {"seed": 7},
                "asset_sha256": {},
                "instruction": "move",
                "seed": 7,
                "upstream_revision": "b" * 40,
                "checkpoint": "/checkpoint",
                "checkpoint_sha256": "c" * 64,
                "checkpoint_config_sha256": "d" * 64,
                "policy_noise_scheme": "sha256-seeded-cpu-torch-randn-v1",
                "policy_torch_version": "2.4.0",
                "policy_seed": 0,
                "policy_episode_seed": 41,
                "episode_length_s": 8.0,
                "control_hz": 25.0,
                "physics_time_step": 0.04,
                "termination_tolerance": 0.07,
                "disable_fabric": False,
                "path_tracing": False,
                "simulator_versions": {
                    "isaaclab": "2.2.1",
                    "isaacsim": "4.5.0",
                },
                "inference_mode": "continuous",
                "latency_steps": 1,
                "action_chunk_steps": 4,
                "execution_horizon_steps": 3,
                "success_term": "success",
                "timeout_terms": ["time_out"],
                "num_steps": 2,
                "status": 0,
                "success": True,
                "recorded_at": "2026-08-28T00:00:00+00:00",
                "python_version": "3.11",
                "numpy_version": np.__version__,
            }
        ),
        states={
            ("scene", "rigid_object", "object", "root_pose"): np.asarray(
                [[[0.0]], [[1.0]], [[2.0]]], dtype=np.float32
            ),
            **rng_states,
        },
        observations={
            ("images", "front"): np.asarray(
                [[[[[0.0]]]], [[[[1.0]]]], [[[[2.0]]]]], dtype=np.float32
            ),
            ("state", "end_effector", "pos"): np.asarray(
                [[[0.0]], [[1.0]], [[2.0]]], dtype=np.float32
            ),
        },
        actions=np.asarray(
            [[[1.0] + [0.0] * 7], [[1.0] + [0.0] * 7]], dtype=np.float32
        ),
        success=np.asarray([False, True]),
        done=np.asarray([False, True]),
        terminations=[
            {"success": False, "time_out": False},
            {"success": True, "time_out": False},
        ],
        protocol=[{}, {}],
        final_policy={},
    )


def test_action_replay_verifies_saved_isaac_states():
    report = verify_trace_actions(_trace(), _Env(), atol=0.0)

    assert report == {
        "steps": 2,
        "initial_state_error": 0.0,
        "max_state_error": 0.0,
        "max_error_path": ["scene", "rigid_object", "object", "root_pose"],
        "exact": True,
        "within_atol": True,
        "first_step_exceeding_atol": None,
    }


def test_render_diagnostics_isolate_state_writes_and_copy_camera_buffers(tmp_path, monkeypatch):
    from vlalagbench.benchmarks.dynamicvla import replay

    for mode, expected_restores in (("frozen", 1), ("frozen-restore", 4), ("states", 4)):
        env = _Env()
        env.physics_dt = .01
        env.cfg = SimpleNamespace(decimation=4, sim=SimpleNamespace(render_interval=1))
        restores, renders, saved = [], [], {}
        monkeypatch.setattr(replay, "restore_env_state", lambda _env, state: restores.append(state))
        env.sim.render = lambda: renders.append(1)
        buffer = np.zeros((1, 2, 2, 3), dtype=np.uint8)

        def camera_views(*_args):
            buffer[:] = len(renders)
            return {"front": {"rgb": buffer}, "side": {"rgb": buffer}}

        upstream = SimpleNamespace(
            get_camera_views=camera_views,
            get_frames=lambda frames, **_: next(iter(frames.values())),
            dump_video=lambda frames, path, **_: saved.update({path.name: frames}),
        )
        report = render_diagnostic_video(_trace(), env, upstream, tmp_path/'probe.mp4',
                                         mode=mode, frames=3, samples=2, png_frames=(0, 1, 2))
        assert len(restores) == expected_restores
        assert len(renders) == 6
        assert report['frames'] == 3
        assert len(report['raw_frames']) == 6
        assert (tmp_path/'raw/front_00000.png').is_file()
        assert set(saved) == {'probe_front.mp4', 'probe_side.mp4'}
        assert [int(frame[0,0,0]) for frame in saved['probe_front.mp4']] == [2, 4, 6]


def test_action_diagnostic_reports_drift_and_omits_auto_reset_frame(tmp_path):
    env = _Env()
    env.physics_dt = .01
    env.cfg = SimpleNamespace(decimation=4, sim=SimpleNamespace(render_interval=1))
    upstream = SimpleNamespace(
        get_camera_views=lambda *_: {'front': {'rgb': np.zeros((1,2,2,3))}},
        get_frames=lambda frames, **_: next(iter(frames.values())),
        dump_video=lambda *_args, **_kwargs: None,
    )
    report = render_diagnostic_video(_trace(), env, upstream, tmp_path/'probe.mp4',
                                     mode='actions', frames=3)
    assert report['max_state_error'] == 0
    assert report['max_scene_state_error'] == 0
    assert report['frames'] == 2


def test_rejudgement_calls_success_term_on_saved_states():
    report = rejudge_trace(_trace(), _Env(), success_term="success")

    assert report["success"] is True
    assert report["first_success_step"] == 1
    assert report["matches_recorded"] is True
    assert report["termination_mismatches"] == []
    assert report["done_mismatches"] == []
    assert report["success_mismatches"] == []


def test_rejudgement_ignores_timeout_term_serialization_order():
    trace = _trace()
    trace.meta = trace.meta.model_copy(
        update={"timeout_terms": ["time_out", "container_dropping"]}
    )
    trace.terminations = [
        {"container_dropping": False, **terms} for terms in trace.terminations
    ]
    env = _Env()
    existing = {
        name: env.termination_manager.get_term_cfg(name)
        for name in env.termination_manager.active_terms
    }
    existing["container_dropping"] = SimpleNamespace(
        func=lambda env, **_params: env.scene.position[:, 0] < -100,
        params={},
        time_out=True,
    )
    env.termination_manager = SimpleNamespace(
        active_terms=list(existing), get_term_cfg=existing.__getitem__
    )

    report = rejudge_trace(trace, env, success_term="success")

    assert report["matches_recorded"] is True


def test_state_summary_reads_current_and_legacy_pending_chunk_layouts():
    trace = _trace()
    trace.final_policy = {
        "scheduler": {
            "pending": {
                "request": {"id": 1, "sent_step": 1, "available_step": 2},
                "chunk": [[1.0]],
            }
        }
    }
    assert state_summary(trace)["pending_chunk"] is True

    trace.final_policy = {"scheduler": {"pending": None}}
    assert state_summary(trace)["pending_chunk"] is False

    trace.final_policy = {"pending_chunk": {"actions": [[1.0]]}}
    assert state_summary(trace)["pending_chunk"] is True


def test_separate_camera_videos_share_one_replay_and_own_frame_buffers(tmp_path):
    env = _Env()
    pixels = np.zeros((1, 1, 1, 3))
    reads, saved = [], {}

    def camera_views(_sensors, _outputs):
        pixels[:] = env.scene.position.item()
        reads.append(env.scene.position.item())
        return {name: {"rgb": pixels} for name in ("front", "side", "wrist")}

    upstream = SimpleNamespace(
        get_camera_views=camera_views,
        get_frames=lambda frames, **kwargs: next(iter(frames.values())),
        dump_video=lambda frames, path, **kwargs: saved.update(
            {path.name: [frame.item(0) for frame in frames]}),
    )
    paths = render_trace_video(_trace(), env, upstream, tmp_path/"episode.mp4",
                               separate_cameras=True)
    assert reads == [0, 1, 2]
    assert set(paths) == {"front", "side", "wrist"}
    assert saved == {f"episode_{name}.mp4": [0, 1, 2] for name in paths}


def test_state_playback_renders_after_the_episode(tmp_path):
    env = _Env()
    dumped = {}

    def camera_views(_sensors, _outputs):
        pixel = env.scene.position.item()
        return {"front": {"rgb": np.full((1, 1, 1, 3), pixel)}}

    def video_frames(camera_frames, *, state_keys):
        dumped["camera_frames"] = camera_frames
        dumped["state_keys"] = state_keys
        return camera_frames["front_rgb"]

    upstream_sim = SimpleNamespace(
        get_camera_views=camera_views,
        get_frames=video_frames,
        dump_video=lambda frames, path, *, fps: dumped.update(
            frames=frames, path=path, fps=fps
        ),
    )

    path = render_trace_video(
        _trace(), env, upstream_sim, tmp_path / "replay.mp4", camera="front"
    )

    assert path == tmp_path / "replay.mp4"
    assert [frame.item(0) for frame in dumped["frames"]] == [0, 1, 2]
    assert list(dumped["camera_frames"]) == ["front_rgb"]
    assert dumped["state_keys"] == []
    assert dumped["fps"] == 25.0
    # Cameras advance one control step per frame; a reset would discard the
    # renderer's denoising history.
    assert env.scene.sensors["front"].resets == 0
    assert env.scene.updates.count(0.04) == 3


def test_rejudge_handles_failure_before_the_first_control_step():
    trace = _trace()
    trace.meta = trace.meta.model_copy(update={"num_steps": 0, "status": 3, "success": False})
    trace.terminations = []
    trace.success = np.empty(0, dtype=bool)
    trace.done = np.empty(0, dtype=bool)
    trace.actions = trace.actions[:0]
    report = rejudge_trace(trace, _Env(), success_term="success")
    assert report["steps"] == 0
    assert report["success"] is False
    assert report["matches_recorded"] is True
