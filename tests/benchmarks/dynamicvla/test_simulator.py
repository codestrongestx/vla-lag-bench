from __future__ import annotations

import json
import math
import sys
import threading
import types
from types import SimpleNamespace

import pytest
import torch

from vlalagbench.benchmarks.dynamicvla import simulator
from vlalagbench.benchmarks.dynamicvla.simulator import (
    _TraceReceiptWriter,
    _asset_hashes,
    _episode_length_s_for_steps,
    _reset_active_action_deadline,
    _sha256,
)
from vlalagbench.benchmarks.dynamicvla.runtime import policy_observation_message


def test_policy_observation_uses_batch_sized_empty_gripper_feature():
    message = policy_observation_message(
        {"front": {"rgb": torch.zeros((2, 1, 1, 3))}},
        {
            "end_effector": {
                "pos": torch.zeros((2, 3)),
                "quat": torch.zeros((2, 4)),
            }
        },
        index=3,
        seed=7,
        latency_steps=2,
    )

    end_effector = message["observation.state"]["end_effector"]
    assert end_effector["pos"].shape == (2, 3)
    assert end_effector["quat"].shape == (2, 4)
    assert end_effector["gripper"].shape == (2, 0)


def test_episode_length_seconds_round_trips_to_exact_isaac_control_steps():
    step_dt = 0.04
    for steps in (7, 14, 56, 385, 402, 411, 496, 510, 5000):
        seconds = _episode_length_s_for_steps(steps, step_dt)

        assert math.ceil(seconds / step_dt) == steps
        assert seconds > (steps - 1) * step_dt


def test_active_action_deadline_resets_between_episodes():
    class Env:
        def __init__(self):
            self.unwrapped = self
            self.step_dt = 0.04
            self.cfg = SimpleNamespace(episode_length_s=8.12)

        @property
        def max_episode_length(self):
            return math.ceil(self.cfg.episode_length_s / self.step_dt)

    env = Env()
    upstream = SimpleNamespace(
        get_sim_results=SimpleNamespace(cfg={"episode_length_s": 8.0})
    )

    assert env.max_episode_length == 203
    assert _reset_active_action_deadline(upstream, env) == 200
    assert env.max_episode_length == 200

    env.cfg.episode_length_s = _episode_length_s_for_steps(219, env.step_dt)
    assert _reset_active_action_deadline(upstream, env) == 200
    assert env.max_episode_length == 200


def test_explicit_active_budget_replaces_native_and_resets_without_override():
    class Env:
        step_dt = .04
        def __init__(self):
            self.unwrapped = self
            self.cfg = SimpleNamespace(episode_length_s=12.)
        @property
        def max_episode_length(self):
            return math.ceil(self.cfg.episode_length_s / self.step_dt)
    env = Env()
    upstream = SimpleNamespace(get_sim_results=SimpleNamespace(
        cfg={'episode_length_s':12.}, vlalagbench_max_episode_steps=750))
    assert _reset_active_action_deadline(upstream, env) == 750
    env.cfg.episode_length_s = _episode_length_s_for_steps(756, env.step_dt)
    assert _reset_active_action_deadline(upstream, env) == 750
    upstream.get_sim_results.vlalagbench_max_episode_steps = None
    assert _reset_active_action_deadline(upstream, env) == 300


@pytest.mark.parametrize("mode,op", [("FXAA", 2), ("DLSS", 3)])
def test_renderer_is_rechecked_after_environment_creation(monkeypatch, mode, op):
    values = {
        "/rtx/post/aa/op": 3,
        "/rtx/post/dlss/execMode": 1,
    }
    settings = SimpleNamespace(get=values.get)
    carb = types.ModuleType("carb")
    carb.settings = SimpleNamespace(get_settings=lambda: settings)
    core = types.ModuleType("omni.replicator.core")
    core.settings = SimpleNamespace(
        set_render_rtx_realtime=lambda *, antialiasing: values.__setitem__(
            "/rtx/post/aa/op",
            {"Off": 0, "TAA": 1, "FXAA": 2, "DLSS": 3, "DLAA": 4}[
                antialiasing
            ],
        )
    )
    replicator = types.ModuleType("omni.replicator")
    replicator.core = core
    omni = types.ModuleType("omni")
    omni.replicator = replicator
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", core)
    monkeypatch.setattr(
        simulator,
        "_simulator_versions",
        lambda: {"renderer_antialiasing": f"{values['/rtx/post/aa/op']}:{mode}"},
    )
    env = object()

    def get_test_env(*_args, **_kwargs):
        values["/rtx/post/aa/op"] = 0
        return env

    upstream = SimpleNamespace(get_test_env=get_test_env)

    simulator._configure_renderer(upstream, antialiasing=mode)

    assert values["/rtx/post/aa/op"] == op
    assert upstream.get_test_env() is env
    assert values["/rtx/post/aa/op"] == op


def test_trace_writer_compresses_concurrently_and_commits_receipts_in_order(
    tmp_path,
):
    second_saved = threading.Event()

    class Trace:
        def __init__(self, payload: bytes, *, wait_for_second: bool = False) -> None:
            self.payload = payload
            self.wait_for_second = wait_for_second

        def save(self, path) -> None:
            if self.wait_for_second:
                assert second_saved.wait(timeout=2)
            else:
                second_saved.set()
            path.write_bytes(self.payload)

    receipts = tmp_path / "receipts.jsonl"
    writer = _TraceReceiptWriter(max_workers=2)
    for episode, trace in enumerate(
        [Trace(b"first", wait_for_second=True), Trace(b"second")]
    ):
        writer.submit(
            trace,
            tmp_path / f"episode-{episode}.npz",
            receipts,
            {
                "task_id": 0,
                "task_name": "MG",
                "episode": episode,
                "environment_index": 0,
                "replica_id": episode,
                "environment": "/assets/environment.json",
                "seed": 7,
                "instruction": f"prompt {episode}",
                "trace": f"episode-{episode}.npz",
            },
            {"status": 0, "latency": {}},
        )
    writer.finish()

    rows = [json.loads(line) for line in receipts.read_text().splitlines()]
    assert [row["episode"] for row in rows] == [0, 1]
    assert all(
        row["trace_sha256"] == _sha256(tmp_path / row["trace"]) for row in rows
    )


@pytest.mark.parametrize("mode", ["synchronous", "lazy", "continuous"])
@pytest.mark.parametrize("reuse_cpu_observation", [False, True])
def test_simulator_validates_noise_identity_and_holds_control_target(monkeypatch, mode, reuse_cpu_observation):
    if reuse_cpu_observation:
        monkeypatch.setenv(simulator.ENV_REUSE_CPU_OBSERVATION, "1")
    else:
        monkeypatch.delenv(simulator.ENV_REUSE_CPU_OBSERVATION, raising=False)
    action_index = 0 if mode == "synchronous" else 2
    termination = types.ModuleType("configs.termination_cfg")
    termination.get_done_term = lambda _: "success"
    robot = types.ModuleType("configs.robot_cfg")
    robot.get_robot_name = lambda _path: "robot"
    configs = types.ModuleType("configs")
    configs.__path__ = []
    configs.termination_cfg = termination
    configs.robot_cfg = robot
    monkeypatch.setitem(sys.modules, "configs", configs)
    monkeypatch.setitem(sys.modules, "configs.termination_cfg", termination)
    monkeypatch.setitem(sys.modules, "configs.robot_cfg", robot)

    replies = iter(
        [
            {
                "task_ack": True,
                "execution_horizon_basis": "post_latency_action_count",
                "instruction": "test",
                "policy_noise_scheme": "scheme",
                "policy_torch_version": "version",
                "episode_seed": 41,
                "inference_mode": mode,
                "action_index_origin": "release" if mode == "synchronous" else "observation",
                "execution_horizon_steps": 18,
                "action_chunk_steps": 20,
            },
            {
                "action": None,
                "meta": {
                    "reply_to": 0,
                    "request": {"request_id": 0},
                    "release": None,
                    "blocked_s": 0.0,
                    "wait_reason": "inference_wait",
                    "chunk_index": None,
                    "observation_step": None,
                    "action_index": None,
                },
            },
            {
                "action": None,
                "meta": {
                    "reply_to": 1,
                    "request": None,
                    "release": None,
                    "blocked_s": 0.0,
                    "wait_reason": "inference_wait",
                    "chunk_index": None,
                    "observation_step": None,
                    "action_index": None,
                },
            },
            {
                "action": torch.ones((1, 8)),
                "meta": {
                    "reply_to": 2,
                    "request": None,
                    "release": {"request_id": 0},
                    "blocked_s": 0.0,
                    "wait_reason": "model",
                    "chunk_index": 0,
                    "observation_step": 0,
                    "action_index": action_index,
                },
            },
            *(
                {
                    "action": None,
                    "meta": {
                        "reply_to": step,
                        "request": None,
                        "release": None,
                        "blocked_s": 0.0,
                        "wait_reason": "inference_wait",
                        "chunk_index": None,
                        "observation_step": None,
                        "action_index": None,
                    },
                }
                for step in (3, 4)
            ),
            None,
        ]
    )
    monkeypatch.setattr(simulator, "receive_reply", lambda *args: next(replies))
    monkeypatch.setattr(
        simulator,
        "read_policy_observation",
        lambda *args: (
            {"front": {"rgb": torch.zeros((1, 1, 1, 3))}},
            {"end_effector": {"pos": torch.zeros((1, 3))}},
        ),
    )

    recorder_images = []

    class Recorder:
        def __init__(self, env, *, images=False):
            recorder_images.append(images)

        def record_observation(self, observation):
            pass

        def record_step(self, *args, **kwargs):
            pass

    monkeypatch.setattr(simulator, "DynamicVLATraceRecorder", Recorder)
    monkeypatch.setattr(simulator, "action_tensor", lambda _up, value, _env: value)
    control_targets = []
    monkeypatch.setattr(
        simulator,
        "step_and_capture_terminal",
        lambda _env, action, *_args, **_kwargs: control_targets.append(action.clone())
        or False,
    )
    monkeypatch.setattr(
        simulator,
        "evaluate_termination_terms",
        lambda *_args: {"success": False},
    )
    validated = []

    def validate(receipt, **kwargs):
        validated.append(receipt)
        assert receipt["violations"] == 0
        assert receipt["status"] == 3
        assert receipt["success"] is False
        assert receipt["staleness_hist"] == {2: 1}
        assert kwargs == {
            "mode": mode,
            "latency_steps": 2,
            "action_chunk_steps": 20,
            "execution_horizon_steps": 18,
        }
        return []

    monkeypatch.setattr(simulator, "receipt_errors", validate)
    term_mgr = SimpleNamespace(
        active_terms=["success"],
        get_term_cfg=lambda name: SimpleNamespace(time_out=False),
    )
    unwrapped = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                cfg=SimpleNamespace(spawn=SimpleNamespace(usd_path="robot.usd"))
            )
        },
        num_envs=1,
        max_episode_length=200,
        step_dt=0.04,
        cfg=SimpleNamespace(episode_length_s=8.0),
        action_manager=SimpleNamespace(
            get_term=lambda _name: SimpleNamespace(
                _compute_frame_pose=lambda: pose_reads.append(1)
                or (torch.full((1, 3), 0.5 * len(pose_reads)), torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
            )
        ),
    )
    pose_reads = []
    env = SimpleNamespace(
        env=SimpleNamespace(termination_manager=term_mgr),
        unwrapped=unwrapped,
    )
    upstream = SimpleNamespace(
        torch=torch,
        get_sim_results=SimpleNamespace(cfg={"seed": 7, "instruction": "test"}),
        zmq=SimpleNamespace(
            Poller=lambda: SimpleNamespace(register=lambda *args: None), POLLIN=1
        ),
    )
    obs_socket = SimpleNamespace(send_pyobj=lambda message: None)

    result = simulator.simulate_injected(
        upstream,
        env,
        obs_socket,
        object(),
        {"robot": torch.zeros(8)},
        mode=mode,
        execution_horizon_steps=18,
        latency_steps=2,
        reply_timeout_s=1.0,
        action_chunk_steps=20,
    )

    assert recorder_images == [False]
    assert validated[0]["episode_seed"] == 41
    assert result["latency"]["exact"] is True
    assert result["latency"]["deadline_basis"] == "active_action_steps"
    assert result["latency"]["active_action_deadline_steps"] == 200
    assert result["latency"]["deadline_paused_steps"] == 4
    assert unwrapped.cfg.episode_length_s == pytest.approx(8.16)
    torch.testing.assert_close(control_targets[0], control_targets[1])
    torch.testing.assert_close(control_targets[3], control_targets[4])
    # The hand pose is read once, on the first wait step after the model action,
    # and sent with that action's gripper command.
    torch.testing.assert_close(control_targets[3], torch.tensor([[0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0]]))
    assert pose_reads == [1]


@pytest.mark.parametrize("mode", ["synchronous", "lazy", "continuous"])
@pytest.mark.parametrize("latency", [0, 3, 6, 9, 12])
@pytest.mark.parametrize("chunk_steps", [14, 20, 32])
def test_fixed_horizon_policy_replies_pass_real_simulator_validation(
    monkeypatch, mode, latency, chunk_steps
):
    from vlalagbench.benchmarks.dynamicvla.policy import _streaming_action
    from tests.benchmarks.dynamicvla.test_protocol import _model

    horizon = 14
    if mode != "synchronous" and latency + horizon > chunk_steps:
        with pytest.raises(ValueError, match="latency plus execution horizon"):
            _model(latency, chunk_steps, mode=mode, execution_horizon_steps=horizon)
        return
    model = _model(
        latency, chunk_steps, mode=mode, execution_horizon_steps=horizon
    )
    termination = types.ModuleType("configs.termination_cfg")
    termination.get_done_term = lambda _: "success"
    robot = types.ModuleType("configs.robot_cfg")
    robot.get_robot_name = lambda _: "robot"
    configs = types.ModuleType("configs")
    configs.__path__ = []
    configs.termination_cfg, configs.robot_cfg = termination, robot
    for module in (configs, termination, robot):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    observations = []
    protocol = []
    ack_sent = False

    def receive(*_args):
        nonlocal ack_sent
        if not ack_sent:
            ack_sent = True
            return {
                "task_ack": True, "instruction": "test",
                "policy_noise_scheme": "test", "policy_torch_version": torch.__version__,
                "episode_seed": 17, "inference_mode": mode,
                "action_index_origin": "release" if mode == "synchronous" else "observation",
                "execution_horizon_steps": horizon,
                "execution_horizon_basis": "post_latency_action_count", "action_chunk_steps": chunk_steps,
            }
        index = observations[-1]["index"]
        if index < 5:
            return {"action": None, "meta": {"reply_to": index, "wait_reason": "policy_warmup"}}
        selected = _streaming_action(model, {"index": index})
        action = None
        if selected is not None:
            _source, position = map(int, selected.split(":"))
            action = torch.full((1, 8), float(position))
        return {"action": action, "meta": model.last_action_meta}

    steps = 0

    def advance(_env, _action, _recorder, *, terminal_observation):
        nonlocal steps
        steps += 1
        if steps == 80:
            terminal_observation()
            return True
        return False

    monkeypatch.setattr(simulator, "receive_reply", receive)
    monkeypatch.setattr(simulator, "action_tensor", lambda _up, action, _env: action)
    monkeypatch.setattr(simulator, "read_policy_observation", lambda *_: (
        {"front": {"rgb": torch.zeros((1, 1, 1, 3))}},
        {"end_effector": {"pos": torch.zeros((1, 3))}},
    ))
    monkeypatch.setattr(simulator, "step_and_capture_terminal", advance)
    monkeypatch.setattr(simulator, "evaluate_termination_terms", lambda *_: {"success": steps == 80})
    monkeypatch.setattr(simulator, "DynamicVLATraceRecorder", lambda _, **_kwargs: SimpleNamespace(
        record_observation=lambda _: None,
        record_step=lambda *_, **values: protocol.append(values["protocol"]),
    ))
    manager = SimpleNamespace(
        active_terms=["success"], get_term_cfg=lambda _: SimpleNamespace(time_out=False)
    )
    env = SimpleNamespace(
        env=SimpleNamespace(termination_manager=manager),
        unwrapped=SimpleNamespace(
            scene={"robot": SimpleNamespace(cfg=SimpleNamespace(spawn=SimpleNamespace(usd_path="robot")))},
            num_envs=1, max_episode_length=200, step_dt=0.04,
            cfg=SimpleNamespace(episode_length_s=8.0),
            action_manager=SimpleNamespace(get_term=lambda _name: SimpleNamespace(
                _compute_frame_pose=lambda: (torch.zeros((1, 3)), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
            )),
        ),
    )
    upstream = SimpleNamespace(
        torch=torch,
        get_sim_results=SimpleNamespace(cfg={"seed": 7, "instruction": "test"}),
        zmq=SimpleNamespace(Poller=lambda: SimpleNamespace(register=lambda *_: None), POLLIN=1),
    )
    result = simulator.simulate_injected(
        upstream, env, SimpleNamespace(send_pyobj=observations.append), object(),
        {"robot": torch.zeros(8)}, mode=mode, execution_horizon_steps=horizon,
        latency_steps=latency, reply_timeout_s=1.0, action_chunk_steps=chunk_steps,
    )
    receipt = result["latency"]
    assert receipt["exact"], receipt["errors"]
    assert receipt["violations"] == 0
    assert receipt["applied_events"]
    for event in receipt["applied_events"]:
        assert 0 <= event["action_index"] < (horizon if mode == "synchronous" else latency + horizon)
        assert protocol[event["step"]]["policy"]["action_index"] == event["step"]
        if mode != "synchronous":
            assert event["action_index"] == event["step"] - event["observation_step"]


def test_suite_persists_first_reply_timeout_without_retrying(tmp_path, monkeypatch):
    environment = tmp_path / "environment.json"
    environment.write_text(
        json.dumps({"seed": 7, "instruction": "test", "episode_length_s": 8.0})
    )
    environment_sha256 = _sha256(environment)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "episodes_per_environment": 1,
                "environments": [
                    {
                        "task_id": 0,
                        "task_name": "task",
                        "environment_index": 0,
                        "path": str(environment),
                        "sha256": environment_sha256,
                    }
                ],
                "episodes": [
                    {
                        "task_id": 0,
                        "environment_index": 0,
                        "replica_id": 0,
                        "environment": str(environment),
                        "environment_sha256": environment_sha256,
                        "instruction": "test",
                    }
                ],
            }
        )
    )
    environment_list = tmp_path / "environments.txt"
    environment_list.write_text(f"{environment}\n")
    sim_config = tmp_path / "sim.yaml"
    sim_config.write_text("robots: {}\n")
    receipts = tmp_path / "receipts.jsonl"
    trace_dir = tmp_path / "traces"
    episode_seed = simulator.derive_episode_seed(
        11, "task", environment_sha256, 0
    )

    class Trace:
        def save(self, path):
            path.write_bytes(b"failed trace")
            return path

    class Recorder:
        def finish(self, metadata, *, final_policy):
            assert metadata["status"] == 3
            assert final_policy == {}
            return Trace()

    result = {
        "status": 3,
        "cam_views": [{}],
        "ee_path": [],
        "instruction": "test",
        "policy_noise_scheme": "scheme",
        "policy_torch_version": "version",
        "policy_episode_seed": episode_seed,
        "success_term": "success",
        "timeout_terms": ["time_out"],
        "trace_recorder": Recorder(),
        "latency": {
            "episode_seed": episode_seed,
            "mode": "synchronous",
            "execution_horizon_steps": 4,
            "injected_steps": 2,
            "errors": ["policy reply timed out"],
            "exact": False,
        },
    }
    sent = []
    obs_socket = SimpleNamespace(send_pyobj=sent.append)
    act_socket = object()
    suite_attempts = []

    def get_sim_results(*_args):
        suite_attempts.append(True)
        return result

    upstream = SimpleNamespace(
        yaml=SimpleNamespace(
            load=lambda *_args, **_kwargs: {"robots": {}}, FullLoader=object()
        ),
        torch=torch,
        get_zmq_sockets=lambda *_args: (obs_socket, act_socket),
        get_sim_results=get_sim_results,
        get_episode_name=lambda *_args: "failure.mp4",
        np=SimpleNamespace(array=lambda value: value),
    )
    replies = iter(
        [
            {"vla": "model", "epoch": 0},
            {"identity_ack": True, "episode_seed": episode_seed},
        ]
    )
    monkeypatch.setattr(simulator, "wait_for_action", lambda *_args: next(replies))
    monkeypatch.setattr(simulator, "_asset_hashes", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(simulator, "_simulator_versions", lambda: {})
    args = SimpleNamespace(
        sim_cfg_file=str(sim_config),
        device="cpu",
        host="127.0.0.1",
        img_port=1,
        act_port=2,
        startup_timeout_s=1.0,
        reply_timeout_s=1.0,
        output_dir=str(tmp_path / "output"),
        env_cfg=str(environment_list),
        manifest=str(manifest),
        num_envs=1,
        scene_dir=str(tmp_path),
        object_dir=str(tmp_path),
        physics_time_step=0.04,
        tolerance=0.07,
        disable_fabric=False,
        path_tracing=False,
        receipts=str(receipts),
        trace_dir=str(trace_dir),
        policy_seed=11,
        save=False,
        upstream_revision="a" * 40,
        checkpoint="/checkpoint",
        checkpoint_sha256="b" * 64,
        checkpoint_config_sha256="c" * 64,
        mode="synchronous",
        latency_steps=2,
        action_chunk_steps=20,
        execution_horizon_steps=4,
    )

    with pytest.raises(TimeoutError, match="policy reply timed out"):
        simulator._run_suite(
            upstream, SimpleNamespace(is_running=lambda: True), args
        )

    failure = json.loads(receipts.read_text())
    assert failure["status"] == 3
    assert failure["exact"] is False
    assert failure["errors"] == ["policy reply timed out"]
    assert (trace_dir / failure["trace"]).is_file()
    assert suite_attempts == [True]
    assert sent == [
        {
            "episode_identity": {
                "policy_seed": 11,
                "task_name": "task",
                "environment_sha256": environment_sha256,
                "replica_id": 0,
            }
        }
    ]


def test_suite_retains_first_step_policy_failure_without_resampling(
    tmp_path, monkeypatch
):
    environment = tmp_path / "environment.json"
    environment.write_text(
        json.dumps(
            {
                "seed": 7,
                "instruction": {
                    "task": "place",
                    "objects": ["object"],
                    "containers": ["box"],
                },
                "episode_length_s": 8.0,
            }
        )
    )
    environment_sha256 = _sha256(environment)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "episodes_per_environment": 1,
                "environments": [
                    {
                        "task_id": 0,
                        "task_name": "task",
                        "environment_index": 0,
                        "path": str(environment),
                        "sha256": environment_sha256,
                    }
                ],
                "episodes": [
                    {
                        "task_id": 0,
                        "environment_index": 0,
                        "replica_id": 0,
                        "environment": str(environment),
                        "environment_sha256": environment_sha256,
                        "instruction": "literal prompt",
                    }
                ],
            }
        )
    )
    environment_list = tmp_path / "environments.txt"
    environment_list.write_text(f"{environment}\n")
    sim_config = tmp_path / "sim.yaml"
    sim_config.write_text("robots: {}\n")
    receipts = tmp_path / "receipts.jsonl"
    trace_dir = tmp_path / "traces"
    episode_seed = simulator.derive_episode_seed(11, "task", environment_sha256, 0)

    class Trace:
        def save(self, path):
            path.write_bytes(b"first-step policy failure")
            return path

    class Recorder:
        def finish(self, metadata, *, final_policy):
            assert metadata["status"] == 2
            assert metadata["instruction"] == "literal prompt"
            assert final_policy == {"request_id": 0}
            return Trace()

    result = {
        "status": 2,
        "cam_views": [{}],
        "ee_path": [],
        "instruction": "literal prompt",
        "policy_noise_scheme": "scheme",
        "policy_torch_version": "version",
        "policy_episode_seed": episode_seed,
        "success_term": "success",
        "timeout_terms": ["time_out"],
        "trace_recorder": Recorder(),
        "latency": {
            "episode_seed": episode_seed,
            "mode": "synchronous",
            "execution_horizon_steps": 14,
            "injected_steps": 3,
            "n_steps": 1,
            "applied": 0,
            "requests": 0,
            "errors": [],
            "exact": True,
        },
    }
    attempts = []

    class InstructionGenerator:
        @staticmethod
        def generate_instruction(_metadata):
            return "unpaired upstream prompt"

    def get_sim_results(*_args):
        attempts.append(True)
        result["instruction"] = InstructionGenerator.generate_instruction(None)
        return result

    sent = []
    obs_socket = SimpleNamespace(send_pyobj=sent.append, close=lambda: None)
    act_socket = SimpleNamespace(close=lambda: None)
    upstream = SimpleNamespace(
        yaml=SimpleNamespace(
            load=lambda *_args, **_kwargs: {"robots": {}}, FullLoader=object()
        ),
        torch=torch,
        get_zmq_sockets=lambda *_args: (obs_socket, act_socket),
        get_sim_results=get_sim_results,
        InstructionGenerator=InstructionGenerator,
        get_episode_name=lambda *_args: "failure.mp4",
        np=SimpleNamespace(array=lambda value: value),
    )
    replies = iter(
        [
            {"vla": "model", "epoch": 0},
            {"identity_ack": True, "episode_seed": episode_seed},
            {"ack": 1, "final_policy": {"request_id": 0}},
            {"suite_ack": True},
        ]
    )
    monkeypatch.setattr(simulator, "wait_for_action", lambda *_args: next(replies))
    monkeypatch.setattr(simulator, "_asset_hashes", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(simulator, "_simulator_versions", lambda: {})
    args = SimpleNamespace(
        sim_cfg_file=str(sim_config),
        device="cpu",
        host="127.0.0.1",
        img_port=1,
        act_port=2,
        startup_timeout_s=1.0,
        reply_timeout_s=1.0,
        output_dir=str(tmp_path / "output"),
        env_cfg=str(environment_list),
        manifest=str(manifest),
        num_envs=1,
        scene_dir=str(tmp_path),
        object_dir=str(tmp_path),
        physics_time_step=0.04,
        tolerance=0.07,
        disable_fabric=False,
        path_tracing=False,
        receipts=str(receipts),
        trace_dir=str(trace_dir),
        policy_seed=11,
        save=False,
        upstream_revision="a" * 40,
        checkpoint="/checkpoint",
        checkpoint_sha256="b" * 64,
        checkpoint_config_sha256="c" * 64,
        mode="synchronous",
        latency_steps=3,
        action_chunk_steps=20,
        execution_horizon_steps=14,
    )

    simulator._run_suite(upstream, SimpleNamespace(is_running=lambda: True), args)

    failure = json.loads(receipts.read_text())
    assert attempts == [True]
    assert failure["status"] == 2
    assert failure["success"] is False
    assert failure["instruction"] == "literal prompt"
    assert failure["replica_id"] == 0
    assert failure["trace_codec"] == "npz-deflate-level-1-v1"
    assert failure["trace_bytes"] > 0
    assert failure["trace_persist_s"] >= 0.0
    assert (
        InstructionGenerator.generate_instruction(None)
        == "unpaired upstream prompt"
    )
    assert (trace_dir / failure["trace"]).is_file()


def test_asset_manifest_hashes_dom_bundles_and_skips_remote_robot(tmp_path):
    scenes = tmp_path / "scenes"
    objects = tmp_path / "objects"
    textures = scenes / "textures"
    object_bundle = objects / "bottle"
    textures.mkdir(parents=True)
    object_bundle.mkdir(parents=True)
    scene = scenes / "kitchen.usd"
    texture = textures / "wall.png"
    object_usd = object_bundle / "bottle.usd"
    material = object_bundle / "material.mdl"
    scene.write_bytes(b"scene")
    texture.write_bytes(b"texture")
    object_usd.write_bytes(b"object")
    material.write_bytes(b"material")
    environment = {
        "house": {"spawn": {"usd_path": "/source/scenes/kitchen.usd"}},
        "object": {"spawn": {"usd_path": "/source/objects/bottle/bottle.usd"}},
        "robot": {"spawn": {"usd_path": "https://example.test/franka.usd"}},
    }

    hashes = _asset_hashes(environment, scene_dir=scenes, object_dir=objects)

    assert hashes["scenes/kitchen.usd"] == _sha256(scene)
    assert set(hashes) == {
        "scenes/kitchen.usd",
        "scenes/textures",
        "objects/bottle/bottle.usd",
        "objects/bottle",
    }


def test_asset_manifest_rejects_unresolved_local_usd(tmp_path):
    scenes = tmp_path / "scenes"
    objects = tmp_path / "objects"
    scenes.mkdir()
    objects.mkdir()

    with pytest.raises(FileNotFoundError, match="cannot uniquely resolve"):
        _asset_hashes(
            {"house": {"spawn": {"usd_path": "/missing/kitchen.usd"}}},
            scene_dir=scenes,
            object_dir=objects,
        )


def test_dl_denoiser_setting_is_reapplied_after_environment_preset(monkeypatch):
    values = {"/rtx-transient/dldenoiser/enabled": True}
    settings = SimpleNamespace(get=values.get, set=values.__setitem__)
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: settings)))
    monkeypatch.setattr(simulator, "_simulator_versions", lambda: dict(values))
    def get_test_env():
        values["/rtx-transient/dldenoiser/enabled"] = True
        return "env"
    upstream = SimpleNamespace(get_test_env=get_test_env)
    simulator._configure_renderer(upstream, dl_denoiser=False)
    assert values["/rtx-transient/dldenoiser/enabled"] is False
    assert upstream.get_test_env() == "env"
    assert values["/rtx-transient/dldenoiser/enabled"] is False


@pytest.mark.parametrize("refuse_setting", [False, True])
def test_renderer_contract_survives_environment_preset_or_fails(monkeypatch, refuse_setting):
    expected = {
        "/rtx/post/dlss/execMode": 1,
        "/rtx-transient/dldenoiser/enabled": True,
        "/rtx-transient/dlssg/enabled": False,
        "/rtx/rendermode": "RaytracedLighting",
    }
    values = {}
    def set_value(key, value):
        if not (refuse_setting and key == "/rtx/post/dlss/execMode"):
            values[key] = value
    settings = SimpleNamespace(get=values.get, set=set_value)
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: settings)))
    monkeypatch.setattr(simulator, "_simulator_versions", lambda: dict(values))
    def get_test_env():
        values.update({"/rtx/post/dlss/execMode": 0, "/rtx-transient/dldenoiser/enabled": False,
                       "/rtx-transient/dlssg/enabled": True, "/rtx/rendermode": "PathTracing"})
        return "env"
    upstream = SimpleNamespace(get_test_env=get_test_env)
    if refuse_setting:
        with pytest.raises(RuntimeError, match="did not retain requested"):
            simulator._configure_renderer(upstream, dlss_mode="Balanced", dl_denoiser=True,
                                          enable_dlssg=False, render_mode="RaytracedLighting")
    else:
        simulator._configure_renderer(upstream, dlss_mode="Balanced", dl_denoiser=True,
                                      enable_dlssg=False, render_mode="RaytracedLighting")
        assert upstream.get_test_env() == "env"
        assert values == expected
