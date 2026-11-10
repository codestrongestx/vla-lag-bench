"""Opt-in, single-model chunk inference for isolated DynamicVLA frontends.

Run only on trusted local sockets: the existing DynamicVLA transport uses pickle.
The GPU backend handles one batch-size-one request at a time. Scheduling and
observation history remain in the individual policy frontend processes.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import pickle
import queue
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

ENV_ENDPOINT = "VLALAGBENCH_DYNAMICVLA_SHARED_ENDPOINT"


def _check_endpoint(endpoint: str) -> None:
    prefix = "tcp://127.0.0.1:"
    if not endpoint.startswith(prefix):
        raise ValueError("shared inference requires a loopback TCP endpoint")
    try:
        port = int(endpoint[len(prefix):])
    except ValueError:
        raise ValueError("shared inference endpoint needs a numeric port") from None
    if not 1 <= port <= 65535:
        raise ValueError("shared inference port is out of range")


def model_identity(checkpoint: str | Path, config: Any) -> dict[str, str]:
    checkpoint = Path(checkpoint).resolve()
    values = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else vars(config)
    payload = json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))
    return {"checkpoint": str(checkpoint),
            "checkpoint_config_sha256": hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
            "resolved_config_sha256": hashlib.sha256(payload.encode()).hexdigest()}


def with_streaming_factory(upstream: Any, get_model: Callable, checkpoint: str | Path,
                           use_delta_action: bool, streaming: bool, factory: Callable):
    """Reuse upstream config resolution without constructing its neural wrapper."""
    if not streaming:
        raise ValueError("shared chunk inference requires streaming mode")
    checkpoint = Path(checkpoint).resolve()
    model_type = json.loads((checkpoint / "config.json").read_text())["type"]
    model_class = upstream.utils.helpers.get_policy_class(model_type)
    original = model_class.__dict__.get("get_streaming_model")
    model_class.get_streaming_model = staticmethod(factory)
    try:
        return get_model(checkpoint, use_delta_action, streaming)
    finally:
        if original is None:
            delattr(model_class, "get_streaming_model")
        else:
            model_class.get_streaming_model = original


class ChunkTransport:
    """One frontend owns its socket, outstanding request, and reset generation."""

    def __init__(self, endpoint: str, identity: dict, timeout: float):
        import zmq

        _check_endpoint(endpoint)
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self.client_id = uuid.uuid4().hex
        self._socket.setsockopt(zmq.IDENTITY, self.client_id.encode())
        self._socket.connect(endpoint)
        self.identity = identity
        self.generation = 0
        self.pending: dict | None = None
        hello = uuid.uuid4().hex
        try:
            self._socket.send_pyobj({"kind": "hello", "client_id": self.client_id,
                                     "hello": hello, "identity": identity})
            response = self._receive(timeout)
            if (response.get("kind") != "hello" or response.get("hello") != hello
                    or response.get("identity") != identity or response.get("error")):
                raise RuntimeError(f"shared inference handshake failed: {response}")
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout: float) -> dict:
        if not self._socket.poll(max(1, int(timeout * 1000)), self._zmq.POLLIN):
            raise queue.Empty("shared inference response timed out")
        value = self._socket.recv_pyobj()
        if not isinstance(value, dict):
            raise RuntimeError("shared inference returned an invalid message")
        return value

    def reset(self) -> None:
        self.generation += 1
        self.pending = None

    def submit(self, batch: dict, noise: Any, *, request_id: int, episode_seed: int) -> None:
        if self.pending is not None:
            raise RuntimeError("shared frontend already has an outstanding chunk")
        tags = {"client_id": self.client_id, "generation": self.generation,
                "request_id": int(request_id), "observation_step": int(batch["index"]),
                "episode_seed": int(episode_seed)}
        self._socket.send_pyobj({"kind": "infer", "identity": self.identity,
                                "tags": tags, "batch": batch, "noise": noise})
        self.pending = tags

    def receive(self, timeout: float) -> dict:
        if self.pending is None:
            raise RuntimeError("shared frontend has no outstanding chunk")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty("shared inference response timed out")
            response = self._receive(remaining)
            tags = response.get("tags", {})
            if tags.get("client_id") != self.client_id:
                raise RuntimeError("shared inference response belongs to another client")
            if isinstance(tags.get("generation"), int) and tags["generation"] < self.generation:
                continue
            if tags != self.pending:
                raise RuntimeError("shared inference response has different request tags")
            if response.get("kind") != "chunk" or response.get("error"):
                raise RuntimeError(f"shared inference failed: {response.get('error', 'invalid response')}")
            self.pending = None
            return {"index": tags["observation_step"], "actions": response["actions"]}

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()


class SharedFrontend:
    """The neural weights and preprocessing modules live exclusively in the backend."""

    def __init__(self, config: Any, transport: ChunkTransport):
        self.config = config
        self._vlalagbench_transport = transport
        self._action_index = 0

    def reset(self) -> None:
        self._action_index = 0
        self._vlalagbench_transport.reset()

    def select_action(self, batch: dict, noise: Any = None):
        return self._get_streaming_action(batch, noise)


def make_frontend(upstream: Any, get_model: Callable, checkpoint: str | Path,
                  use_delta_action: bool, streaming: bool, *, endpoint: str, timeout: float):
    def factory(path, config):
        return SharedFrontend(config, ChunkTransport(endpoint, model_identity(path, config), timeout))
    return with_streaming_factory(upstream, get_model, checkpoint, use_delta_action, streaming, factory)


def infer_chunk(model: Any, batch: dict, noise: Any, *, device: str = "cuda"):
    """Match the upstream streaming worker, including its delta-action convention."""
    import torch

    if noise is None:
        raise ValueError("shared inference requires explicit request noise")
    with torch.no_grad():
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        if batch["observation.state"].shape[0] != 1:
            raise ValueError("shared inference accepts batch size one only")
        if batch.get("dt_scale", 1.0) != 1.0:
            raise ValueError("shared inference requires fixed simulation-step timing")
        # sample_actions updates its noise in place; each request owns this buffer.
        noise = noise.to(device).clone()
        latest_state = batch["observation.state"][:, -1:, :]
        prepared = model._prepare_batch(batch)
        actions = model._get_action_chunk(prepared, noise)
        if model.config.use_delta_action:
            action_dim = actions.shape[-1] - 1
            actions[..., :action_dim] += latest_state[..., :action_dim]
        return actions.transpose(0, 1).cpu()


def handle_request(message: dict, route: bytes, identity: dict, infer: Callable) -> dict:
    """Pure routing/validation boundary used by the blocking server and tests."""
    tags = message.get("tags", {})
    response = {"kind": "chunk", "tags": tags}
    if message.get("kind") == "hello":
        response = {"kind": "hello", "hello": message.get("hello"), "identity": identity}
    try:
        if message.get("identity") != identity:
            raise ValueError("checkpoint/config identity mismatch")
        if message.get("kind") == "hello":
            if message.get("client_id", "").encode() != route:
                raise ValueError("hello client identity differs from socket route")
            return response
        if message.get("kind") != "infer":
            raise ValueError("unknown shared inference request")
        if set(tags) != {"client_id", "generation", "request_id", "observation_step", "episode_seed"}:
            raise ValueError("incomplete request tags")
        if tags["client_id"].encode() != route:
            raise ValueError("request client identity differs from socket route")
        if any(type(tags[key]) is not int or tags[key] < 0
               for key in ("generation", "request_id", "observation_step", "episode_seed")):
            raise ValueError("request tags must be nonnegative integers")
        if tags["observation_step"] != message["batch"]["index"]:
            raise ValueError("observation step differs from request tags")
        response["actions"] = infer(message["batch"], message["noise"])
    except Exception as exc:
        response["error"] = f"{type(exc).__name__}: {exc}"
        logging.exception("shared inference request failed")
    return response


def load_backend_model(root: Path, weights: Path, *, warmup: bool = True):
    """Load exactly the original streaming worker model, without a wrapper model."""
    import sys
    import torch
    from vlalagbench.benchmarks.dynamicvla.policy import _load_upstream_inference

    sys.path.insert(0, str(root.resolve()))
    upstream = _load_upstream_inference(root.resolve())
    captured = {}
    def capture(path, config):
        captured.update(config=config)
        return SimpleNamespace(config=config)
    with_streaming_factory(upstream, upstream.get_vla_model, weights, True, True, capture)
    config = captured["config"]
    model_type = json.loads((weights / "config.json").read_text())["type"]
    model_class = upstream.utils.helpers.get_policy_class(model_type)
    model = model_class.from_pretrained(weights, config=config)
    model.eval()
    model = model.cuda()
    identity = model_identity(weights, config)
    if warmup:
        with torch.no_grad():
            dummy = {key: torch.zeros(1, config.n_obs_steps, *feature.shape,
                                     dtype=torch.float32, device="cuda")
                     for key, feature in config.input_features.items()}
            dummy["task"] = ["dummy text input"]
            model._get_action_chunk(dummy)
    return model, identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamicvla-root", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args(argv)
    _check_endpoint(args.endpoint)
    if args.ready_file is not None and args.ready_file.exists():
        parser.error("--ready-file already exists")
    import zmq

    model, identity = load_backend_model(args.dynamicvla_root, args.weights)
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.bind(args.endpoint)
        if args.ready_file is not None:
            from vlalagbench.files import atomic_text
            atomic_text(args.ready_file, json.dumps({"endpoint": args.endpoint, "identity": identity}) + "\n")
        logging.info("Shared DynamicVLA backend ready: %s", json.dumps(identity, sort_keys=True))
        while True:
            route, payload = socket.recv_multipart()
            message = pickle.loads(payload)
            response = handle_request(message, route, identity,
                                      lambda batch, noise: infer_chunk(model, batch, noise))
            socket.send_multipart([route, pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)])
    finally:
        socket.close(linger=0)
        context.term()
        if args.ready_file is not None:
            args.ready_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")
    raise SystemExit(main())
