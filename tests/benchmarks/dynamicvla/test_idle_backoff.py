"""The idle experiment changes polling only, leaving the vendor worker intact."""
import json
import multiprocessing
from types import SimpleNamespace

import pytest

from vlalagbench.benchmarks.dynamicvla.idle_backoff import (
    ENV_BACKOFF_MS, IdleInput, backoff_seconds, paced_inference_loop, with_idle_backoff,
)
from vlalagbench.benchmarks.dynamicvla.shared_backend import ENV_ENDPOINT


def test_idle_input_sleeps_only_when_empty_and_forwards_clear(monkeypatch):
    waits = []
    monkeypatch.setattr("vlalagbench.benchmarks.dynamicvla.idle_backoff.time.sleep", waits.append)
    source = {}
    wrapped = IdleInput(source, 0.001)
    assert wrapped.get("obs") is None
    source["obs"] = ("observation", "noise")
    assert wrapped.get("obs") == source["obs"]
    assert waits == [0.001]
    wrapped.clear()
    assert source == {}


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "bad"])
def test_invalid_backoff_is_rejected(value):
    with pytest.raises(ValueError):
        backoff_seconds(value)


def test_worker_factory_restores_original_on_success_and_failure(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_BACKOFF_MS, "1")
    (tmp_path / "config.json").write_text(json.dumps({"type": "dynamicvla"}))
    class Model:
        @staticmethod
        def _inference_loop(*_):
            pass
    original = Model.__dict__["_inference_loop"]
    upstream = SimpleNamespace(utils=SimpleNamespace(helpers=SimpleNamespace(get_policy_class=lambda _: Model)))
    def get_model(*args):
        assert Model._inference_loop is paced_inference_loop
        return "model"
    assert with_idle_backoff(upstream, get_model, tmp_path, True, True) == "model"
    assert Model.__dict__["_inference_loop"] is original
    with pytest.raises(RuntimeError):
        with_idle_backoff(upstream, lambda *_: (_ for _ in ()).throw(RuntimeError()), tmp_path)
    assert Model.__dict__["_inference_loop"] is original


def test_spawn_entrypoint_calls_pristine_upstream_loop(tmp_path, monkeypatch):
    package = tmp_path / "policies/dynamicvla"
    package.mkdir(parents=True)
    (tmp_path / "policies/__init__.py").touch()
    (package / "__init__.py").touch()
    (package / "modeling_dynamicvla.py").write_text(
        "class DynamicVLAPolicy:\n"
        "    @staticmethod\n"
        "    def _inference_loop(checkpoint, config, q_in, q_out):\n"
        "        q_out.put((checkpoint, config, q_in.get('obs'), q_in.seconds))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(ENV_BACKOFF_MS, "1")
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    worker = context.Process(target=paced_inference_loop,
                             args=("checkpoint", {"test": True}, {"obs": 42}, output))
    worker.start()
    try:
        assert output.get(timeout=15) == ("checkpoint", {"test": True}, 42, 0.001)
        worker.join(timeout=15)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join()
        output.close()
        output.join_thread()


def test_shared_and_backoff_experiments_are_mutually_exclusive(monkeypatch):
    from vlalagbench.benchmarks.dynamicvla.policy import main

    monkeypatch.setenv(ENV_BACKOFF_MS, "1")
    monkeypatch.setenv(ENV_ENDPOINT, "tcp://127.0.0.1:5766")
    with pytest.raises(SystemExit) as error:
        main(["--dynamicvla-root", "/unused", "--weights", "/unused", "--mode", "lazy",
              "--execution-horizon-steps", "14", "--latency-steps", "3", "--reply-timeout-s", "1"])
    assert error.value.code == 2
