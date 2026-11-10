"""Nested host timings preserve calls and isolate concurrently persisted traces."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace

import pytest

from vlalagbench.benchmarks.dynamicvla.profiling import HostTimings, install_simulator_timing


class Clock:
    def __init__(self):
        self.local = threading.local()
    def __call__(self):
        return getattr(self.local, "value", 0)
    def advance(self, value):
        self.local.value = self() + value


def test_nested_exclusive_spans_do_not_double_count_terminal_callback():
    clock = Clock()
    profile = HostTimings(clock=clock)
    with profile.span("physics"):
        clock.advance(10)
        with profile.span("terminal"):
            clock.advance(20)
            with profile.span("state_copy"):
                clock.advance(30)
        clock.advance(40)
    totals = profile.snapshot()["totals"]
    assert totals["physics"]["inclusive_seconds"] == 100 / 1e9
    assert totals["physics"]["exclusive_seconds"] == 50 / 1e9
    assert totals["terminal"]["inclusive_seconds"] == 50 / 1e9
    assert totals["terminal"]["exclusive_seconds"] == 20 / 1e9
    assert totals["state_copy"]["exclusive_seconds"] == 30 / 1e9
    assert sum(row["exclusive_seconds"] for row in totals.values()) == pytest.approx(100 / 1e9)


def test_exception_closes_stack_and_preserves_original_error():
    clock = Clock()
    profile = HostTimings(clock=clock)
    with pytest.raises(ValueError, match="original"):
        with profile.span("failed"):
            clock.advance(5)
            raise ValueError("original")
    with profile.span("next"):
        clock.advance(3)
    assert profile.snapshot()["totals"]["next"]["exclusive_seconds"] == 3 / 1e9
    assert profile.snapshot()["totals"]["failed"]["calls"] == 1


def test_threads_have_independent_stacks_and_episode_context(tmp_path):
    clock = Clock()
    profile = HostTimings(clock=clock)
    first = profile.begin_episode("one")
    profile.bind_episode(0, 10)
    second = profile.begin_episode("two")
    profile.bind_episode(0, 11)
    barrier = threading.Barrier(2)
    def worker(episode, duration):
        with profile.episode_scope(episode), profile.span("persist"):
            barrier.wait(timeout=5)
            clock.advance(duration)
            with profile.span("checksum"):
                clock.advance(5)
        assert profile.episode is None
        profile.mark("persisted", episode)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, first, 10), executor.submit(worker, second, 20)]
        for future in futures:
            future.result()
    assert profile.episode == second
    assert profile.episode_for(0, 10) == first
    rows = [row for row in profile.snapshot()["spans"] if row["span"] == "persist"]
    assert {row["episode_ordinal"] for row in rows} == {first, second}
    assert sorted(row["exclusive_seconds"] for row in rows) == [10 / 1e9, 20 / 1e9]
    profile.output = tmp_path / "host-timing.json"
    profile.write()
    data = json.loads(profile.output.read_text())
    assert len(data["episodes"]) == 2
    assert all("persisted" in row["marks_seconds"] for row in data["episodes"])


def test_wrappers_preserve_calls_sockets_and_async_trace_attribution(monkeypatch, tmp_path):
    from vlalagbench.benchmarks.dynamicvla import tracing

    profile = HostTimings()
    profile.output = tmp_path / "host-timing.json"
    calls = []
    sentinel = object()
    native_action = object()
    native_observation = SimpleNamespace(send_pyobj=lambda value: calls.append(value) or sentinel)
    trace = SimpleNamespace(meta=SimpleNamespace(task_id=0, episode=7))
    monkeypatch.setattr(tracing.DynamicVLATraceRecorder, "finish", lambda *_: trace)
    class Driver:
        def apply(self, *args):
            return sentinel
    class Writer:
        def _commit_oldest(self):
            return sentinel
    simulator = SimpleNamespace(PlanarDriver=Driver, _TraceReceiptWriter=Writer)
    for name in ("simulate_injected", "read_policy_observation", "owned_cpu_policy_state", "policy_observation_message",
                 "receive_reply", "wait_for_action", "step_and_capture_terminal",
                 "evaluate_termination_terms", "_json_action"):
        setattr(simulator, name, lambda *args, **kwargs: sentinel)
    simulator._persist_trace = lambda *args, **kwargs: sentinel
    simulator._write_receipt = lambda *args, **kwargs: sentinel
    original_read = simulator.read_policy_observation
    upstream = SimpleNamespace(
        get_sim_results=lambda *args: simulator.simulate_injected(),
        get_test_env=lambda *args: sentinel,
        get_frames=lambda *args: sentinel,
        sim=SimpleNamespace(get_frames=lambda *args: sentinel, dump_video=lambda *args: sentinel),
        get_zmq_sockets=lambda: (native_observation, native_action),
    )
    original_results = upstream.get_sim_results
    original_results.cfg = {"seed": 7}
    try:
        install_simulator_timing(profile, simulator, upstream)
        assert not calls
        assert simulator.read_policy_observation() is sentinel
        observation, action = upstream.get_zmq_sockets()
        assert action is native_action
        assert observation.send_pyobj({"index": 5}) is sentinel
        assert calls == [{"index": 5}]
        assert upstream.get_sim_results.cfg == {"seed": 7}
        assert upstream.get_sim_results({}, "environment.json") is sentinel
        assert tracing.DynamicVLATraceRecorder.finish(None) is trace
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(simulator._persist_trace, trace).result() is sentinel
        assert simulator._write_receipt(None, task_id=0, episode=7) is sentinel
        data = json.loads(profile.output.read_text())
        assert data["episodes"][0]["episode_id"] == 7
        assert "receipt_committed" in data["episodes"][0]["marks_seconds"]
        persist = next(row for row in data["spans"] if row["span"] == "trace.persist")
        assert persist["episode_ordinal"] == 0
        assert persist["thread"].startswith("ThreadPoolExecutor")
        assert data["totals"]["observation.read"]["calls"] == 1
    finally:
        profile.restore()
    assert simulator.read_policy_observation is original_read
    assert upstream.get_sim_results is original_results
