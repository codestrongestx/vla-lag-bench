"""Opt-in batched policy serving for LIBERO studies.

A batching server holds a request for up to ``max_wait_ms`` while others
arrive and runs up to ``max_batch_size`` of them through the model in one
pass. Each request keeps its own seeded diffusion noise, so batching changes
only floating-point evaluation order, never which noise a request sees. With
``batch_padding: fixed`` every pass has ``max_batch_size`` rows, so a
request's numerics do not depend on how many others shared its pass.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import http
import logging
import time
import traceback
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from vlalagbench.benchmarks.libero.policy import (
    REQUEST_KEY, RESPONSE_KEY, RNG_SCHEME, deterministic_noise, request_seed,
)


logger = logging.getLogger(__name__)

BatchInfer = Callable[[Sequence[dict[str, Any]]], Sequence[dict[str, Any]]]


@dataclasses.dataclass
class _Request:
    observation: dict[str, Any]
    future: asyncio.Future[dict[str, Any]]
    enqueued_at: float


class DynamicBatcher:
    """Combine concurrent requests without allowing more than one GPU call at a time."""

    def __init__(self, infer_batch: BatchInfer, *, max_batch_size: int, max_wait_ms: float) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be nonnegative")
        self._infer_batch = infer_batch
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_ms / 1000.0
        self._queue: asyncio.Queue[_Request] = asyncio.Queue()

    async def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        await self._queue.put(_Request(observation, future, loop.time()))
        return await future

    async def run(self) -> None:
        """Serve requests until cancelled; a failed batch does not stop later batches."""
        try:
            while True:
                await self._execute(await self._next_batch())
        except asyncio.CancelledError:
            error = RuntimeError("dynamic batcher stopped")
            while not self._queue.empty():
                request = self._queue.get_nowait()
                if not request.future.done():
                    request.future.set_exception(error)
            raise

    async def _next_batch(self) -> list[_Request]:
        batch = [await self._queue.get()]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.max_wait_seconds
        while len(batch) < self.max_batch_size:
            try:
                if self.max_wait_seconds == 0:
                    batch.append(self._queue.get_nowait())
                    continue
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except (asyncio.QueueEmpty, TimeoutError):
                break
        return batch

    async def _execute(self, batch: list[_Request]) -> None:
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        try:
            outputs = list(await asyncio.to_thread(self._infer_batch, [request.observation for request in batch]))
            if len(outputs) != len(batch):
                raise RuntimeError(f"batch policy returned {len(outputs)} outputs for {len(batch)} requests")
        except Exception as error:
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(error)
            return
        infer_ms = (loop.time() - started_at) * 1000.0
        for request, output in zip(batch, outputs, strict=True):
            result = dict(output)
            result["server_timing"] = {
                **result.get("server_timing", {}),
                "queue_ms": (started_at - request.enqueued_at) * 1000.0,
                "infer_ms": infer_ms,
                "batch_size": len(batch),
                "max_batch_size": self.max_batch_size,
            }
            if not request.future.done():
                request.future.set_result(result)


class PipelinedBatcher(DynamicBatcher):
    """Prepare the next batch on the CPU while the GPU runs the current one.

    ``prepare`` turns observations into model inputs; ``execute`` runs them.
    At most one ``execute`` runs at a time. The next batch is cut once enough
    requests wait to fill it, or when the GPU becomes free, so overlapping
    never shrinks batches below what the queue holds. Otherwise the cut waits
    until the running pass is about one preparation time from finishing, so
    the batch collects as many requests as it can and is ready as the GPU
    frees. Keep clients' EGL rendering off this GPU: back-to-back passes
    starve their contexts.
    """

    def __init__(self, prepare: Callable[[Sequence[dict[str, Any]]], Any],
                 execute: Callable[[Any], Sequence[dict[str, Any]]], *, max_batch_size: int, max_wait_ms: float) -> None:
        super().__init__(lambda observations: execute(prepare(observations)),
                         max_batch_size=max_batch_size, max_wait_ms=max_wait_ms)
        self._prepare = prepare
        self._execute_prepared = execute
        self._served = self._passes = 0
        self._busy = 0.0
        self._prepare_s = self._execute_s = self._running_since = 0.0

    async def run(self) -> None:
        running: asyncio.Task[None] | None = None
        loop = asyncio.get_running_loop()
        reported = loop.time()
        try:
            while True:
                if loop.time() - reported >= 60.0:
                    elapsed, reported = loop.time() - reported, loop.time()
                    logger.info("served %.1f req/s, mean batch %.1f, GPU busy %.0f%%, queue %d",
                                self._served / elapsed, self._served / max(self._passes, 1),
                                100.0 * self._busy / elapsed, self._queue.qsize())
                    self._served = self._passes = 0
                    self._busy = 0.0
                while (running is not None and not running.done() and self._queue.qsize() < self.max_batch_size
                       and loop.time() - self._running_since < self._execute_s - self._prepare_s):
                    await asyncio.sleep(0.001)
                batch = await self._next_batch()
                started_at = loop.time()
                try:
                    prepared = await asyncio.to_thread(self._prepare, [request.observation for request in batch])
                    self._prepare_s = 0.8 * self._prepare_s + 0.2 * (loop.time() - started_at)
                except Exception as error:
                    for request in batch:
                        if not request.future.done():
                            request.future.set_exception(error)
                    continue
                if running is not None:
                    await running
                self._running_since = loop.time()
                running = asyncio.create_task(self._finish(batch, prepared, started_at))
        except asyncio.CancelledError:
            if running is not None:
                running.cancel()
            error = RuntimeError("dynamic batcher stopped")
            while not self._queue.empty():
                request = self._queue.get_nowait()
                if not request.future.done():
                    request.future.set_exception(error)
            raise

    async def _finish(self, batch: list[_Request], prepared: Any, started_at: float) -> None:
        loop = asyncio.get_running_loop()
        executing = loop.time()
        try:
            outputs = list(await asyncio.to_thread(self._execute_prepared, prepared))
            self._execute_s = 0.8 * self._execute_s + 0.2 * (loop.time() - executing)
            self._served += len(batch)
            self._passes += 1
            self._busy += loop.time() - executing
            if len(outputs) != len(batch):
                raise RuntimeError(f"batch policy returned {len(outputs)} outputs for {len(batch)} requests")
        except Exception as error:
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(error)
            return
        infer_ms = (loop.time() - started_at) * 1000.0
        for request, output in zip(batch, outputs, strict=True):
            result = dict(output)
            result["server_timing"] = {
                **result.get("server_timing", {}),
                "queue_ms": (started_at - request.enqueued_at) * 1000.0,
                "infer_ms": infer_ms,
                "batch_size": len(batch),
                "max_batch_size": self.max_batch_size,
            }
            if not request.future.done():
                request.future.set_result(result)


class BatchedWebsocketPolicyServer:
    """OpenPI-compatible websocket transport backed by :class:`DynamicBatcher`."""

    def __init__(self, policy: Any, *, host: str, port: int, metadata: dict[str, Any],
                 max_batch_size: int, max_wait_ms: float, batcher: DynamicBatcher | None = None,
                 ping_interval: float | None = 20.0) -> None:
        self._host = host
        self._ping_interval = ping_interval
        self._port = port
        self._metadata = metadata
        self._batcher = batcher or DynamicBatcher(policy.infer_batch, max_batch_size=max_batch_size,
                                                  max_wait_ms=max_wait_ms)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        import websockets.asyncio.server as websocket_server

        batch_task = asyncio.create_task(self._batcher.run())
        try:
            async with websocket_server.serve(
                self._handler, self._host, self._port, compression=None, max_size=None,
                process_request=_health_check, ping_interval=self._ping_interval,
            ) as server:
                await server.serve_forever()
        finally:
            batch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await batch_task

    async def _handler(self, websocket: Any) -> None:
        import websockets
        from openpi_client import msgpack_numpy

        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                observation = msgpack_numpy.unpackb(await websocket.recv())
                action = await self._batcher.infer(observation)
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                return
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="Internal server error")
                raise


def _health_check(connection: Any, request: Any) -> Any | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def _identities(observations: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    identities, cleaned = [], []
    for observation in observations:
        clean = dict(observation)
        identity = clean.pop(REQUEST_KEY, None)
        if not isinstance(identity, dict):
            raise ValueError(f"{REQUEST_KEY} must be a request identity")
        identities.append(identity)
        cleaned.append(clean)
    return identities, cleaned


def _pad_rows(value: Any, rows: int) -> np.ndarray:
    """Extend the batch axis to ``rows`` by repeating the first row."""
    value = np.asarray(value)
    return np.concatenate([value, np.repeat(value[:1], rows - len(value), axis=0)])


class OpenPIBatchPolicy:
    """Run several OpenPI requests through one ``sample_actions`` call.

    Uses the same input and output transforms as ``Policy.infer``; a batch of
    one has the same shapes as the single-request path.
    """

    def __init__(self, policy: Any, *, action_horizon: int, action_dim: int,
                 max_batch_size: int, batch_padding: str) -> None:
        if getattr(policy, "_is_pytorch_model", False):
            raise TypeError("batched OpenPI serving supports the JAX model only")
        self._policy = policy
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self._rows = max_batch_size if batch_padding == "fixed" else 0
        self._max_size = max_batch_size

    def infer_batch(self, observations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_types

        identities, cleaned = _identities(observations)
        policy = self._policy
        transformed = [policy._input_transform(jax.tree.map(lambda value: value, clean)) for clean in cleaned]
        inputs = jax.tree.map(lambda *values: np.stack([np.asarray(value) for value in values]), *transformed)
        noise = np.stack([
            deterministic_noise(identity, action_horizon=self.action_horizon, action_dim=self.action_dim)
            for identity in identities
        ])
        count = len(identities)
        if self._rows > count:
            inputs = jax.tree.map(lambda value: _pad_rows(value, self._rows), inputs)
            noise = _pad_rows(noise, self._rows)
        inputs = jax.tree.map(jnp.asarray, inputs)
        policy._rng, sample_rng = jax.random.split(policy._rng)
        started = time.monotonic()
        actions = policy._sample_actions(
            sample_rng, model_types.Observation.from_dict(inputs),
            **{**policy._sample_kwargs, "noise": jnp.asarray(noise)},
        )
        state, actions = np.asarray(inputs["state"]), np.asarray(actions)
        model_ms = (time.monotonic() - started) * 1000.0
        outputs = []
        for index, identity in enumerate(identities):
            output = policy._output_transform({"state": state[index], "actions": actions[index]})
            output["policy_timing"] = {"infer_ms": model_ms}
            output[RESPONSE_KEY] = {**identity, "noise_sha256": hashlib.sha256(noise[index].tobytes()).hexdigest()}
            outputs.append(output)
        return outputs

    def warm_up(self, example: dict[str, Any]) -> None:
        """Compile every batch shape this server can run before clients connect."""
        sizes = [self._rows] if self._rows else range(1, 1 + self._max_size)
        for size in sizes:
            self.infer_batch([{**example, REQUEST_KEY: {"warm_up": index}} for index in range(size)])


class _NoiseTorch:
    """Stand-in for the ``torch`` module inside GR00T's action head.

    Its single ``torch.randn`` call draws the initial flow-matching noise for
    the whole batch. This draws each row exactly as a lone request would
    after ``torch.manual_seed(seed)``: a fresh generator at that seed.
    """

    def __init__(self, torch: Any, seeds: Sequence[int]) -> None:
        self._torch = torch
        self._seeds = list(seeds)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._torch, name)

    def randn(self, *args: Any, size: Any = None, dtype: Any = None, device: Any = None, **kwargs: Any) -> Any:
        if not self._seeds or args or kwargs or size is None or size[0] != len(self._seeds):
            raise RuntimeError(f"unexpected GR00T noise draw: size={size} args={args} kwargs={kwargs}")
        torch, seeds, self._seeds = self._torch, self._seeds, []
        rows = []
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            rows.append(torch.randn((1, *size[1:]), generator=generator, dtype=dtype, device=device))
        return torch.cat(rows)


class Gr00tBatchPolicy:
    """Run several GR00T requests through one ``get_action`` call."""

    def __init__(self, policy: Any, *, max_batch_size: int, batch_padding: str) -> None:
        import torch
        from gr00t.model.gr00t_n1d7 import gr00t_n1d7

        self.policy = policy
        self._module = gr00t_n1d7
        self._torch = torch
        self._rows = max_batch_size if batch_padding == "fixed" else 0
        self._max_size = max_batch_size

    def infer_batch(self, observations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        from vlalagbench.benchmarks.libero.gr00t_adapter import decode_actions, isolated_rng

        identities, cleaned = _identities(observations)
        seeds = [request_seed(identity) for identity in identities]
        count = len(cleaned)
        rows = cleaned + [cleaned[0]] * max(0, self._rows - count)
        row_seeds = seeds + [seeds[0]] * (len(rows) - count)
        batch = {
            "video": {key: np.concatenate([row["video"][key] for row in rows]) for key in rows[0]["video"]},
            "state": {key: np.concatenate([row["state"][key] for row in rows]) for key in rows[0]["state"]},
            "language": {key: [item for row in rows for item in row["language"][key]] for key in rows[0]["language"]},
        }
        original = self._module.torch
        with isolated_rng(seeds[0]):
            self._module.torch = _NoiseTorch(original, row_seeds)
            try:
                actions, _ = self.policy.get_action(batch)
            finally:
                self._module.torch = original
        outputs = []
        for index, (identity, seed) in enumerate(zip(identities, seeds, strict=True)):
            row = {key: np.asarray(value)[index:index + 1] for key, value in actions.items()}
            outputs.append({"actions": decode_actions(row),
                            RESPONSE_KEY: {**identity, "seed": seed, "rng_scheme": RNG_SCHEME}})
        return outputs

    def warm_up(self, example: dict[str, Any]) -> None:
        sizes = [self._rows] if self._rows else range(1, 1 + self._max_size)
        for size in sizes:
            self.infer_batch([{**example, REQUEST_KEY: {"warm_up": index}} for index in range(size)])
