import asyncio

import numpy as np
import pytest

from vlalagbench.benchmarks.libero import policy_server
from vlalagbench.benchmarks.libero.batching import DynamicBatcher, _NoiseTorch
from vlalagbench.config import PolicyServer, StudySpec


def _server(**changes):
    return PolicyServer.model_validate({
        "host": "127.0.0.1", "port": 8000, "implementation": "dynamic_batching",
        "pytorch_compile_mode": "max-autotune-no-cudagraphs", "max_batch_size": 4,
        "max_wait_ms": 5.0, "batch_padding": "none", **changes,
    })


def test_batcher_groups_concurrent_requests_and_reports_shapes():
    calls = []

    def infer_batch(observations):
        calls.append([observation["id"] for observation in observations])
        return [{"actions": observation["id"]} for observation in observations]

    async def main():
        batcher = DynamicBatcher(infer_batch, max_batch_size=3, max_wait_ms=50)
        task = asyncio.create_task(batcher.run())
        results = await asyncio.gather(*(batcher.infer({"id": index}) for index in range(5)))
        task.cancel()
        return results

    results = asyncio.run(main())
    assert [result["actions"] for result in results] == list(range(5))
    assert calls == [[0, 1, 2], [3, 4]]
    assert [result["server_timing"]["batch_size"] for result in results] == [3, 3, 3, 2, 2]
    assert {result["server_timing"]["max_batch_size"] for result in results} == {3}


def test_batcher_fails_only_the_batch_that_raised():
    def infer_batch(observations):
        if any(observation["id"] == 0 for observation in observations):
            raise RuntimeError("boom")
        return [{"actions": observation["id"]} for observation in observations]

    async def main():
        batcher = DynamicBatcher(infer_batch, max_batch_size=1, max_wait_ms=0)
        task = asyncio.create_task(batcher.run())
        results = await asyncio.gather(batcher.infer({"id": 0}), batcher.infer({"id": 1}), return_exceptions=True)
        task.cancel()
        return results

    failed, ok = asyncio.run(main())
    assert isinstance(failed, RuntimeError) and ok["actions"] == 1


def test_batched_rows_draw_each_requests_own_noise():
    torch = pytest.importorskip("torch")
    proxy = _NoiseTorch(torch, [11, 22])
    noise = proxy.randn(size=(2, 16, 32), dtype=torch.float32, device="cpu")
    for row, seed in enumerate((11, 22)):
        torch.manual_seed(seed)
        assert torch.equal(noise[row:row + 1], torch.randn(size=(1, 16, 32), dtype=torch.float32, device="cpu"))
    with pytest.raises(RuntimeError, match="unexpected GR00T noise draw"):
        proxy.randn(size=(2, 16, 32), dtype=torch.float32, device="cpu")


def test_handshake_records_and_checks_request_shaping():
    batching = policy_server.batching_identity(_server())
    assert batching == {"implementation": "dynamic_batching", "max_batch_size": 4, "max_wait_ms": 5.0, "batch_padding": "none"}
    assert policy_server.batching_identity(_server(implementation="single", max_batch_size=1, max_wait_ms=0)) is None
    metadata = policy_server.build_metadata("pi05_libero", "gs://c", 10, batching=batching)
    policy_server.validate_metadata(metadata, openpi_config="pi05_libero", checkpoint="gs://c", action_chunk_steps=10, batching=batching)
    with pytest.raises(RuntimeError, match="max_batch_size"):
        policy_server.validate_metadata(metadata, openpi_config="pi05_libero", checkpoint="gs://c", action_chunk_steps=10)


def test_schema_two_studies_accept_dynamic_batching_but_not_stray_batch_settings():
    base = {"schema_version": 2, "name": "s", "base_experiment": "x.yaml", "sweep": {"inference.latency_ms": [0]}}
    StudySpec.model_validate({**base, "policy": {"implementation": "dynamic_batching", "max_batch_size": 4, "max_wait_ms": 5}})
    for policy in ({"max_batch_size": 4}, {"implementation": "vanilla"}):
        with pytest.raises(ValueError, match="schema 2"):
            StudySpec.model_validate({**base, "policy": policy})
