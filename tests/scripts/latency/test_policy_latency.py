import numpy as np

from vlalagbench.benchmarks.libero.policy import REQUEST_KEY
from scripts.latency.latency_table import row
from scripts.latency.policy_latency import (
    Loaded, measure, request_elements, resolve_checkpoint, stats, synthetic_frames, timed,
)


def test_timed_drains_the_device_around_the_call():
    events, sink = [], []

    def finish(result):
        events.append(("finish", result))
        return result

    wrapped = timed(lambda value: events.append(("call", value)) or value * 2, finish, sink)
    assert wrapped(3) == 6
    assert events == [("finish", None), ("call", 3), ("finish", 6)]
    assert len(sink) == 1 and sink[0] >= 0


def test_measure_excludes_warmup_and_keeps_first_call_and_agreement():
    model_ms: list[float] = []
    seen = []

    def call(element):
        seen.append(element)
        model_ms.append(1.0)
        return np.full((2, 7), float(element))

    loaded = Loaded(call=call, elements=[0, 1, 2], model_ms=model_ms, config={},
                    references=[np.zeros((2, 7)), np.ones((2, 7)), np.full((2, 7), 5.0)])
    result = measure(loaded, warmup=2, calls=4)
    assert seen == [0, 1, 2, 0, 1, 2]
    assert len(result["e2e_ms"]) == 4 and result["model_ms"] == [1.0] * 4
    assert result["first_call_ms"] is not None
    # Timed calls start at element 2, so each output meets its own reference.
    assert result["agreement"]["requests"] == 3
    assert result["agreement"]["arm_max_abs"] == 3.0


def test_synthetic_requests_match_the_evaluator_request():
    elements = request_elements(synthetic_frames("put the bowl on the plate", count=3))
    assert len(elements) == 3
    first = elements[0]
    for key in ("observation/image", "observation/wrist_image"):
        assert first[key].shape == (224, 224, 3) and first[key].dtype == np.uint8
    assert first["observation/state"].shape == (8,)
    assert first["prompt"] == "put the bowl on the plate"
    assert first[REQUEST_KEY] == {"latency_request": 0}


def test_checkpoint_variables_expand_and_uris_stay(monkeypatch):
    monkeypatch.setenv("VLALAGBENCH_CHECKPOINTS", "/checkpoints")
    assert resolve_checkpoint("$VLALAGBENCH_CHECKPOINTS/pi05_libero_pytorch") == "/checkpoints/pi05_libero_pytorch"
    uri = "gs://openpi-assets/checkpoints/pi05_libero"
    assert resolve_checkpoint(uri) == uri


def test_stats_and_table_row():
    summary = stats([10.0, 20.0, 30.0, 40.0])
    assert summary["p50"] == 25.0 and summary["n"] == 4 and summary["min"] == 10.0
    record = {
        "model": "pi05", "stack": "flashrt", "precision": "bf16", "transport": "in-process",
        "inputs": "recorded", "period_ms": 0.0, "control_hz": 20.0, "e2e_ms": summary,
        "model_ms": stats([5.0, 6.0]),
        "gpu": {"exclusive": True, "start": {"name": "NVIDIA GeForce RTX 3090"}},
    }
    line = row(record)
    assert line["gpu"] == "GeForce RTX 3090" and line["steps"] == 0.5 and line["notes"] == ""
