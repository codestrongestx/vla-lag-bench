"""Check shared inference isolation without loading a neural model or GPU."""
import json
import queue
from types import SimpleNamespace

import pytest
import torch

from vlalagbench.benchmarks.dynamicvla.policy import _install_scheduler, _streaming_action
from vlalagbench.benchmarks.dynamicvla.shared_backend import (
    ChunkTransport, SharedFrontend, _check_endpoint, handle_request, infer_chunk,
    model_identity, with_streaming_factory,
)


IDENTITY = {"checkpoint": "test"}


def message(client="a", generation=1, request_id=0, index=5, noise=1):
    return {"kind": "infer", "identity": IDENTITY,
            "tags": {"client_id": client, "generation": generation,
                     "request_id": request_id, "observation_step": index, "episode_seed": 17},
            "batch": {"index": index}, "noise": noise}


def bare_transport(client="a"):
    transport = object.__new__(ChunkTransport)
    transport.client_id = client
    transport.generation = 1
    transport.identity = IDENTITY
    transport.pending = None
    transport.sent = []
    transport._socket = SimpleNamespace(send_pyobj=transport.sent.append)
    transport.replies = queue.Queue()
    transport._receive = lambda timeout: transport.replies.get(timeout=timeout)
    return transport


def test_same_indices_are_routed_independently_and_reset_discards_only_stale_client_reply():
    a, b = bare_transport("a"), bare_transport("b")
    for client in (a, b):
        client.submit({"index": 5}, client.client_id, request_id=0, episode_seed=17)
    a.reset()
    a.submit({"index": 5}, "a-new", request_id=0, episode_seed=18)
    for request, client in ((a.sent[0], a), (b.sent[0], b), (a.sent[1], a)):
        client.replies.put(handle_request(request, client.client_id.encode(), IDENTITY,
                                         lambda batch, noise: noise))
    assert b.receive(1) == {"index": 5, "actions": "b"}
    assert a.receive(1) == {"index": 5, "actions": "a-new"}
    assert a.pending is b.pending is None


@pytest.mark.parametrize("field,value", [("client_id", "b"), ("request_id", 7),
                                         ("observation_step", 6), ("episode_seed", 18),
                                         ("generation", 2)])
def test_frontend_rejects_wrong_reply_tags(field, value):
    client = bare_transport()
    client.submit({"index": 5}, 1, request_id=0, episode_seed=17)
    tags = {**client.pending, field: value}
    client.replies.put({"kind": "chunk", "tags": tags, "actions": 1})
    with pytest.raises(RuntimeError, match="another client|different request tags"):
        client.receive(1)


def test_outstanding_request_cannot_be_overwritten_and_timeout_is_preserved():
    client = bare_transport()
    client.submit({"index": 5}, 1, request_id=0, episode_seed=17)
    with pytest.raises(RuntimeError, match="outstanding"):
        client.submit({"index": 6}, 2, request_id=1, episode_seed=17)
    with pytest.raises(queue.Empty):
        client.receive(0.001)
    assert len(client.sent) == 1 and client.pending is not None


@pytest.mark.parametrize("damage", ["identity", "route", "index", "negative", "noise_error"])
def test_backend_rejects_bad_requests_and_reports_inference_errors(damage):
    request = message()
    route = b"a"
    if damage == "identity":
        request["identity"] = {}
    elif damage == "route":
        route = b"b"
    elif damage == "index":
        request["batch"]["index"] = 9
    elif damage == "negative":
        request["tags"]["request_id"] = -1
    def infer(*_):
        assert damage == "noise_error", "invalid request reached inference"
        raise ValueError("inference failure")
    response = handle_request(request, route, IDENTITY, infer)
    assert response["tags"] == request["tags"]
    assert "error" in response and "actions" not in response


def test_handshake_checks_checkpoint_and_client_route():
    request = {"kind": "hello", "client_id": "a", "hello": "nonce", "identity": IDENTITY}
    assert handle_request(request, b"a", IDENTITY, None) == {
        "kind": "hello", "hello": "nonce", "identity": IDENTITY}
    assert "error" in handle_request(request, b"b", IDENTITY, None)
    assert "error" in handle_request(request, b"a", {}, None)


def test_chunk_matches_streaming_normalization_delta_and_owned_noise():
    class Model:
        config = SimpleNamespace(use_delta_action=True)
        def _prepare_batch(self, batch):
            return {**batch, "observation.state": batch["observation.state"] * 0.5}
        def _get_action_chunk(self, batch, noise):
            assert not torch.is_grad_enabled()
            torch.testing.assert_close(batch["observation.state"], torch.tensor([[[1., 2.], [3., 4.]]]))
            noise.add_(2)
            return noise
    batch = {"observation.state": torch.tensor([[[2., 4.], [6., 8.]]]), "index": 5}
    noise = torch.zeros(1, 3, 3)
    result = infer_chunk(Model(), batch, noise, device="cpu")
    torch.testing.assert_close(result, torch.tensor([[[8., 10., 2.]]] * 3))
    assert torch.equal(noise, torch.zeros_like(noise))
    assert batch["observation.state"][0, -1].tolist() == [6., 8.]
    with pytest.raises(ValueError, match="explicit"):
        infer_chunk(Model(), batch, None, device="cpu")


def test_factory_preserves_upstream_config_parsing_without_neural_allocation(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": "dynamicvla"}))
    class Model:
        @staticmethod
        def get_streaming_model(*args):
            pytest.fail("neural wrapper was allocated")
    original = Model.__dict__["get_streaming_model"]
    upstream = SimpleNamespace(utils=SimpleNamespace(helpers=SimpleNamespace(get_policy_class=lambda _: Model)))
    def get_model(path, delta, streaming):
        assert delta and streaming
        config = SimpleNamespace(n_action_steps=20, enable_streaming=True)
        return Model.get_streaming_model(path, config), {"delta_timestamps": [-2, 0]}
    model, metadata = with_streaming_factory(upstream, get_model, tmp_path, True, True,
                                             lambda path, config: SimpleNamespace(config=config))
    assert model.config.n_action_steps == 20
    assert metadata == {"delta_timestamps": [-2, 0]}
    assert Model.__dict__["get_streaming_model"] is original
    with pytest.raises(RuntimeError):
        with_streaming_factory(upstream, get_model, tmp_path, True, True,
                               lambda *_: (_ for _ in ()).throw(RuntimeError("failed")))
    assert Model.__dict__["get_streaming_model"] is original
    first = model_identity(tmp_path, model.config)
    model.config.n_action_steps = 10
    assert first != model_identity(tmp_path, model.config)


def test_shared_frontend_uses_existing_fixed_step_scheduler():
    client = bare_transport()
    frontend = SharedFrontend(SimpleNamespace(n_action_steps=6, chunk_size=6, max_action_dim=2), client)
    _install_scheduler(frontend, "continuous", 4, 2, 1)
    frontend._vlalagbench_episode_seed = 17
    assert _streaming_action(frontend, {"index": 5}) is None
    assert client.pending["request_id"] == 0
    assert _streaming_action(frontend, {"index": 6}) is None
    client.replies.put({"kind": "chunk", "tags": dict(client.pending),
                        "actions": torch.arange(12).reshape(6, 1, 2).float()})
    assert _streaming_action(frontend, {"index": 7}).tolist() == [[4., 5.]]
    assert frontend.last_action_meta["release"]["release_step"] == 7
    assert frontend.last_action_meta["action_index"] == 2
    frontend.reset()
    assert client.generation == 2 and client.pending is None


@pytest.mark.parametrize("endpoint", ["tcp://0.0.0.0:9000", "tcp://127.0.0.1:70000", "ipc://test"])
def test_only_loopback_tcp_is_supported(endpoint):
    with pytest.raises(ValueError):
        _check_endpoint(endpoint)
