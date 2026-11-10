"""Run the pinned DynamicVLA worker through the shared chunk scheduler."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import logging
import os
import queue
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

from vlalagbench.benchmarks.dynamicvla.protocol import (
    POLICY_NOISE_SCHEME,
    POLICY_STARTUP_STEPS,
    derive_episode_seed,
    derive_request_seed,
)
from vlalagbench.timing import ChunkScheduler, Request, TimingConfig


def _load_upstream_inference(root: Path) -> Any:
    path = root / "scripts/inference.py"
    spec = importlib.util.spec_from_file_location(
        "vlalagbench_dynamicvla_upstream_inference", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DynamicVLA inference module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _streaming_action(model: Any, batch: dict[str, Any], noise: Any = None) -> Any:
    """Transport worker chunks through the shared control-step scheduler."""
    index = int(batch["index"])
    scheduler: ChunkScheduler[Any] = model._vlalagbench_scheduler
    blocked_s = 0.0
    released_chunk = None
    request_event = None
    worker_request = model._vlalagbench_worker_request
    if worker_request is not None and index >= worker_request.available_step:
        blocked_s = _receive_worker_chunk(model, index)

    request = scheduler.request(index)
    if request is not None:
        request_noise = noise if noise is not None else _request_noise(model, request.id)
        request_event = {
            **_request_event(request),
            "request_seed": derive_request_seed(model._vlalagbench_episode_seed, request.id),
            "noise_sha256": _noise_checksum(request_noise),
        }
        transport = getattr(model, "_vlalagbench_transport", None)
        if transport is None:
            model.q_in.update({"obs": (batch, request_noise)})
        else:
            transport.submit(batch, request_noise, request_id=request.id,
                             episode_seed=model._vlalagbench_episode_seed)
        model._vlalagbench_worker_request = request
        if request.available_step == index:
            blocked_s = _receive_worker_chunk(model, index)

    scheduled = scheduler.action(index)
    released_request = scheduler.released_request
    if released_request is not None:
        released_chunk = _clone_value(
            {
                "index": released_request.sent_step,
                "actions": scheduler.released_chunk,
            }
        )
    if scheduled.active:
        model._action_index = (
            scheduled.observation_step
            + scheduled.index_in_chunk
            + (
                model._vlalagbench_timing.latency_steps
                if model._vlalagbench_timing.scheme == "synchronous"
                else 0
            )
        )
    model.last_action_meta = {
        "reply_to": index,
        "request": request_event,
        "release": (
            None
            if released_request is None
            else {
                **_request_event(released_request),
                "chunk_index": released_request.id,
            }
        ),
        "chunk_index": scheduled.chunk_id,
        "observation_step": scheduled.observation_step,
        "action_index": scheduled.index_in_chunk,
        "wait_reason": scheduled.reason,
        "blocked_s": round(blocked_s, 4),
        "released_chunk": released_chunk,
        "policy": _policy_snapshot(model),
    }
    return scheduled.action


def _receive_worker_chunk(model: Any, step: int) -> float:
    request = model._vlalagbench_worker_request
    if request is None:
        raise RuntimeError("DynamicVLA has no worker request to receive")
    started = time.perf_counter()
    try:
        transport = getattr(model, "_vlalagbench_transport", None)
        chunk = (model.q_out.get(timeout=model._vlalagbench_reply_timeout_s)
                 if transport is None else transport.receive(model._vlalagbench_reply_timeout_s))
    except queue.Empty:
        raise RuntimeError(
            "DynamicVLA inference timed out after "
            f"{model._vlalagbench_reply_timeout_s:.1f}s at control step {step}"
        ) from None
    blocked_s = time.perf_counter() - started
    if int(chunk["index"]) != request.sent_step:
        raise RuntimeError(
            f"DynamicVLA worker returned observation {chunk['index']} "
            f"for request {request.sent_step}"
        )
    model._vlalagbench_scheduler.deliver(request, chunk["actions"])
    model._vlalagbench_worker_request = None
    return blocked_s


def _request_event(request: Request) -> dict[str, int]:
    return {
        "request_id": request.id,
        "observation_step": request.sent_step,
        "release_step": request.available_step,
    }


def _install_scheduler(
    model: Any,
    mode: str,
    execution_horizon_steps: int | None,
    latency: int,
    timeout: float,
) -> None:
    horizon = execution_horizon_steps
    if horizon is None:
        raise ValueError(f"{mode} inference requires an execution horizon")
    # Upstream _get_streaming_action skips current_index - observation_index
    # actions before checking whether the queue is empty, including the first
    # chunk. Fixed-latency async therefore skips L actions even at startup;
    # unlike LIBERO's cold-start rule, this offset must not be capped by prior
    # executed actions. Synchronous execution starts at action 0.
    model._vlalagbench_timing = TimingConfig(
        scheme=mode,
        action_chunk_steps=model.config.n_action_steps,
        execution_horizon_steps=horizon,
        latency_steps=latency,
        promotion_offset_steps=0 if mode == "synchronous" else latency,
        execution_horizon_basis="post_latency_action_count",
    )
    model._vlalagbench_reply_timeout_s = timeout
    _reset_scheduler(model)
    model._get_streaming_action = MethodType(_streaming_action, model)
    model._vlalagbench_episode_seed = None
    model.last_action_meta = None


def _reset_scheduler(model: Any) -> None:
    model._vlalagbench_scheduler = ChunkScheduler(
        model._vlalagbench_timing, start_step=POLICY_STARTUP_STEPS
    )
    model._vlalagbench_worker_request = None


def _pop_meta(model: Any, observation: dict[str, Any]) -> dict[str, Any]:
    meta = model.last_action_meta
    model.last_action_meta = None
    if meta is not None:
        return meta
    return {
        "reply_to": observation["index"],
        "request": None,
        "release": None,
        "chunk_index": None,
        "observation_step": None,
        "action_index": None,
        "wait_reason": "policy_warmup",
        "blocked_s": 0.0,
        "released_chunk": None,
        "policy": _policy_snapshot(model),
    }


def _policy_snapshot(model: Any) -> dict[str, Any]:
    return {
        "scheduler": _clone_value(model._vlalagbench_scheduler.snapshot()),
        "action_index": int(model._action_index),
        "episode_seed": model._vlalagbench_episode_seed,
    }


def _request_noise(model: Any, request_id: int) -> Any:
    import torch

    episode_seed = model._vlalagbench_episode_seed
    if episode_seed is None:
        raise RuntimeError("DynamicVLA policy noise seed is not initialized")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_request_seed(episode_seed, request_id))
    return torch.randn(
        (1, model.config.chunk_size, model.config.max_action_dim),
        dtype=torch.float32,
        generator=generator,
    )


def _noise_checksum(noise: Any) -> str:
    values = noise.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _clone_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clone_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clone_value(child) for child in value]
    detach = getattr(value, "detach", None)
    if detach is not None:
        return detach().cpu().numpy().copy()
    copy = getattr(value, "copy", None)
    return copy() if copy is not None else value


def _final_policy_snapshot(model: Any, get_action: Any) -> dict[str, Any]:
    if model._vlalagbench_worker_request is not None:
        _receive_worker_chunk(
            model, model._vlalagbench_worker_request.available_step
        )
    snapshot = _policy_snapshot(model)
    snapshot["get_action_count"] = int(getattr(get_action, "count", -3))
    snapshot["base_state"] = _clone_value(getattr(get_action, "state", None))
    return snapshot


def _restore_policy_snapshot(model: Any, get_action: Any, snapshot: dict[str, Any]) -> None:
    import torch

    model._vlalagbench_episode_seed = int(
        snapshot.get("episode_seed", snapshot.get("noise_seed"))
    )
    model._action_index = int(snapshot["action_index"])
    model._vlalagbench_scheduler = ChunkScheduler.from_snapshot(
        model._vlalagbench_timing,
        snapshot["scheduler"],
        torch.as_tensor,
    )
    model._vlalagbench_worker_request = None
    setattr(get_action, "count", int(snapshot["get_action_count"]))
    base_state = snapshot.get("base_state")
    setattr(
        get_action,
        "state",
        None if base_state is None else model._vlalagbench_numpy.asarray(base_state),
    )


def _run_tests(upstream: Any, latency_steps: int, *args: Any) -> tuple[int, dict[str, Any]]:
    (
        obs_socket,
        act_socket,
        vla_cfg,
        model,
        weights,
        epoch,
        rotation,
        use_delta_action,
        output_dir,
    ) = args
    n_tests = 0
    test_results: dict[str, Any] = {}
    instruction = None
    episode_identity: dict[str, Any] | None = None
    actions: list[Any] = []
    observations: list[dict[str, Any]] = []
    while True:
        observation = upstream.get_latest_observation(obs_socket)
        if observation is None:
            time.sleep(0.0005)
            continue
        if "episode_identity" in observation:
            episode_identity = observation["episode_identity"]
            episode_seed = _identity_episode_seed(episode_identity)
            act_socket.send_pyobj(
                {"identity_ack": True, "episode_seed": episode_seed}
            )
            continue
        if "success_rates" in observation:
            logging.info("Test suite done with success rates: %s", observation["success_rates"])
            act_socket.send_pyobj({"suite_ack": True})
            break
        if "success" in observation:
            final_policy = _final_policy_snapshot(model, upstream._get_action)
            upstream._handle_test_case_finished(
                observation, test_results, model, weights, epoch, actions, output_dir
            )
            n_tests += 1
            logging.info(
                "[Test%02d] Test done with %s in %d steps and %d actions.",
                n_tests,
                "SUCCESS" if observation["success"] else "FAILURE",
                len(observation["ee_path"]),
                len(observation["actions"]),
            )
            act_socket.send_pyobj({"ack": n_tests, "final_policy": final_policy})
            continue
        if "task" in observation:
            if episode_identity is None:
                raise RuntimeError("DynamicVLA task arrived without an episode identity")
            instruction = observation["task"]
            if not isinstance(instruction, str) or not instruction:
                raise RuntimeError("DynamicVLA task instruction is invalid")
            logging.info("[Test%02d] Received task: %s", n_tests, instruction)
            actions = []
            observations = []
            model.reset()
            _reset_scheduler(model)
            model.last_action_meta = None
            model._vlalagbench_episode_seed = None
            setattr(upstream._get_action, "states", [])
            resume = observation.get("resume")
            if resume is None:
                if hasattr(upstream._get_action, "count"):
                    delattr(upstream._get_action, "count")
                if hasattr(upstream._get_action, "state"):
                    delattr(upstream._get_action, "state")
            else:
                observations = list(resume["history"])
                _restore_policy_snapshot(
                    model, upstream._get_action, resume["policy"]
                )
            expected_episode_seed = _identity_episode_seed(episode_identity)
            if model._vlalagbench_episode_seed is None:
                model._vlalagbench_episode_seed = expected_episode_seed
            elif model._vlalagbench_episode_seed != expected_episode_seed:
                raise RuntimeError("DynamicVLA restored episode seed does not match identity")
            schedule_ack = {
                "task_ack": True,
                "policy_noise_scheme": POLICY_NOISE_SCHEME,
                "policy_torch_version": upstream.torch.__version__,
                "inference_mode": model._vlalagbench_timing.scheme,
                "action_index_origin": (
                    "release"
                    if model._vlalagbench_timing.scheme == "synchronous"
                    else "observation"
                ),
                "execution_horizon_steps": (
                    model._vlalagbench_timing.execution_horizon_steps
                ),
                "execution_horizon_basis": "post_latency_action_count",
                "action_chunk_steps": model._vlalagbench_timing.action_chunk_steps,
            }
            schedule_ack.update(
                instruction=instruction, episode_seed=expected_episode_seed,
            )
            act_socket.send_pyobj(schedule_ack)
            if resume is not None:
                continue
        elif instruction is not None:
            observation["task"] = instruction
        else:
            continue

        if "index" in observation:
            observed_latency = observation.get("latency_steps")
            if observed_latency != latency_steps:
                raise RuntimeError(
                    "DynamicVLA latency mismatch: simulator sent "
                    f"{observed_latency!r}, policy expects {latency_steps}"
                )
        observations.append(observation)
        action = upstream._get_action(
            model,
            upstream._get_observations(observations, vla_cfg["delta_timestamps"]),
            rotation,
            use_delta_action,
            output_dir is not None,
        )
        if "index" not in observation:
            continue
        meta = _pop_meta(model, observation)
        meta["get_action_count"] = int(getattr(upstream._get_action, "count", -3))
        meta["base_state"] = _clone_value(getattr(upstream._get_action, "state", None))
        act_socket.send_pyobj({"action": action, "meta": meta})
        if action is not None:
            logging.debug("Sending action: %s", upstream.np.round(action, 2))
            actions.append(action)
    return n_tests, test_results


def _identity_episode_seed(identity: dict[str, Any]) -> int:
    return derive_episode_seed(
        int(identity["policy_seed"]),
        str(identity["task_name"]),
        str(identity["environment_sha256"]),
        int(identity["replica_id"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamicvla-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--img-port", type=int, default=3186)
    parser.add_argument("--act-port", type=int, default=3188)
    parser.add_argument(
        "--mode", choices=("synchronous", "lazy", "continuous"), required=True
    )
    parser.add_argument("--execution-horizon-steps", type=int, required=True)
    parser.add_argument("--latency-steps", type=int, required=True)
    parser.add_argument("--reply-timeout-s", type=float, required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    from vlalagbench.benchmarks.dynamicvla.idle_backoff import ENV_BACKOFF_MS, backoff_seconds
    from vlalagbench.benchmarks.dynamicvla.shared_backend import ENV_ENDPOINT

    if os.environ.get(ENV_BACKOFF_MS):
        if os.environ.get(ENV_ENDPOINT):
            parser.error("shared inference and idle backoff are separate, mutually exclusive experiments")
        backoff_seconds(os.environ[ENV_BACKOFF_MS])

    root = Path(args.dynamicvla_root).resolve()
    sys.path.insert(0, str(root))
    upstream = _load_upstream_inference(root)

    original_get_model = upstream.get_vla_model
    shared_frontends = []

    def get_model(*model_args: Any, **model_kwargs: Any) -> Any:
        from vlalagbench.benchmarks.dynamicvla.shared_backend import ENV_ENDPOINT, make_frontend

        endpoint = os.environ.get(ENV_ENDPOINT)
        if endpoint:
            model, config = make_frontend(upstream, original_get_model, *model_args,
                **model_kwargs, endpoint=endpoint, timeout=args.reply_timeout_s)
            shared_frontends.append(model)
        elif os.environ.get(ENV_BACKOFF_MS):
            from vlalagbench.benchmarks.dynamicvla.idle_backoff import with_idle_backoff
            model, config = with_idle_backoff(upstream, original_get_model, *model_args, **model_kwargs)
        else:
            model, config = original_get_model(*model_args, **model_kwargs)
        _install_scheduler(
            model,
            args.mode,
            args.execution_horizon_steps,
            args.latency_steps,
            args.reply_timeout_s,
        )
        model._vlalagbench_numpy = upstream.np
        return model, config

    upstream.get_vla_model = get_model
    upstream.run_tests = lambda *run_args: _run_tests(
        upstream, args.latency_steps, *run_args
    )
    upstream.args = args
    try:
        upstream.main(
            args.weights,
            "dynamic-vla-DOM",
            0,
            "euler",
            True,
            True,
            args.host,
            args.img_port,
            args.act_port,
            args.output_dir,
        )
    finally:
        for frontend in shared_frontends:
            frontend._vlalagbench_transport.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s", level=logging.INFO
    )
    raise SystemExit(main())
