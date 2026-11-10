"""Opt-in host wall timing for DynamicVLA, without GPU synchronization.

Inclusive spans contain nested spans on the same thread; exclusive spans subtract
those children. Threads may overlap, so summed times are not suite elapsed time.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

ENV_PROFILE = "VLALAGBENCH_DYNAMICVLA_PROFILE"


class HostTimings:
    def __init__(self, *, clock: Callable[[], int] = time.perf_counter_ns):
        self.clock = clock
        self.started_ns = clock()
        self.output: Path | None = None
        self._local = threading.local()
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stats: dict[tuple, dict] = {}
        self._episodes: dict[int, dict] = {}
        self._episode_keys: dict[tuple[int, int], int] = {}
        self._restore: list[tuple[Any, str, Any]] = []

    @property
    def episode(self):
        return getattr(self._local, "episode", None)

    @contextmanager
    def episode_scope(self, episode):
        previous = self.episode
        self._local.episode = episode
        try:
            yield
        finally:
            self._local.episode = previous

    @contextmanager
    def span(self, name: str):
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        frame = [self.clock(), 0]
        episode = self.episode
        stack.append(frame)
        try:
            yield
        finally:
            elapsed = self.clock() - frame[0]
            stack.pop()
            if stack:
                stack[-1][1] += elapsed
            key = (episode, threading.current_thread().name, name)
            with self._lock:
                row = self._stats.setdefault(key, {"calls": 0, "inclusive_ns": 0,
                                                   "exclusive_ns": 0, "max_ns": 0})
                row["calls"] += 1
                row["inclusive_ns"] += elapsed
                row["exclusive_ns"] += max(0, elapsed - frame[1])
                row["max_ns"] = max(row["max_ns"], elapsed)

    def begin_episode(self, environment: str) -> int:
        with self._lock:
            ordinal = len(self._episodes)
            self._episodes[ordinal] = {"ordinal": ordinal, "environment": environment,
                                       "marks_seconds": {"setup_start": (self.clock() - self.started_ns) / 1e9}}
        self._local.episode = ordinal
        return ordinal

    def mark(self, name: str, episode=None) -> None:
        episode = self.episode if episode is None else episode
        if episode is not None:
            with self._lock:
                self._episodes[episode]["marks_seconds"][name] = (self.clock() - self.started_ns) / 1e9

    def bind_episode(self, task_id: int, episode_id: int) -> None:
        if self.episode is not None:
            with self._lock:
                self._episode_keys[task_id, episode_id] = self.episode
                self._episodes[self.episode].update(task_id=task_id, episode_id=episode_id)

    def episode_for(self, task_id: int, episode_id: int):
        with self._lock:
            return self._episode_keys.get((task_id, episode_id))

    def snapshot(self) -> dict:
        with self._lock:
            rows = [{"episode_ordinal": key[0], "thread": key[1], "span": key[2],
                     "calls": value["calls"],
                     **{name.replace("_ns", "_seconds"): value[name] / 1e9
                        for name in ("inclusive_ns", "exclusive_ns", "max_ns")}}
                    for key, value in self._stats.items()]
            episodes = [{**row, "marks_seconds": dict(row["marks_seconds"])}
                        for row in self._episodes.values()]
        totals = {}
        for row in rows:
            total = totals.setdefault(row["span"], {"calls": 0, "inclusive_seconds": 0.,
                                                    "exclusive_seconds": 0., "max_seconds": 0.})
            for field in ("calls", "inclusive_seconds", "exclusive_seconds"):
                total[field] += row[field]
            total["max_seconds"] = max(total["max_seconds"], row["max_seconds"])
        return {"schema_version": 1, "clock": "perf_counter_ns", "pid": os.getpid(),
                "elapsed_seconds": (self.clock() - self.started_ns) / 1e9,
                "scope": "Host wall time from simulator.main; excludes interpreter/owned-module imports. No CUDA synchronization; threads overlap; active spans appear after return.",
                "totals": totals, "episodes": episodes, "spans": rows}

    def write(self) -> None:
        if self.output is not None:
            from vlalagbench.files import atomic_text
            with self._write_lock:
                atomic_text(self.output, json.dumps(self.snapshot(), indent=2) + "\n")

    def patch(self, owner: Any, attribute: str, replacement: Any) -> None:
        self._restore.append((owner, attribute, getattr(owner, attribute)))
        setattr(owner, attribute, replacement)

    def wrap(self, owner: Any, attribute: str, name: str) -> None:
        original = getattr(owner, attribute)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.span(name):
                return original(*args, **kwargs)
        self.patch(owner, attribute, wrapped)

    def restore(self) -> None:
        for owner, attribute, original in reversed(self._restore):
            setattr(owner, attribute, original)
        self._restore.clear()


def install_simulator_timing(profile: HostTimings, simulator: Any, upstream: Any) -> None:
    from vlalagbench.benchmarks.dynamicvla import tracing
    from vlalagbench.simulators.isaaclab import tracing as isaac_tracing

    for attribute, name in (
        ("simulate_injected", "rollout.simulate"),
        ("read_policy_observation", "observation.read"),
        ("owned_cpu_policy_state", "observation.owned_cpu_snapshot"),
        ("policy_observation_message", "observation.message"),
        ("receive_reply", "ipc.reply_wait"),
        ("wait_for_action", "ipc.control_wait"),
        ("step_and_capture_terminal", "physics.step_with_terminal_capture"),
        ("evaluate_termination_terms", "termination.evaluate"),
        ("_json_action", "action.receipt_copy"),
    ):
        profile.wrap(simulator, attribute, name)
    profile.wrap(simulator.PlanarDriver, "apply", "motion.apply")
    profile.wrap(tracing, "capture_env_state", "trace.full_state")
    profile.wrap(isaac_tracing, "_capture_rng_state", "trace.rng_state")
    profile.wrap(isaac_tracing, "_capture_external_wrenches", "trace.external_wrenches")
    for attribute, name in (("__init__", "trace.recorder_init"),
                            ("record_observation", "trace.observation_copy"),
                            ("capture_post_step", "trace.terminal_capture"),
                            ("record_step", "trace.record_step")):
        profile.wrap(tracing.DynamicVLATraceRecorder, attribute, name)
    original_finish = tracing.DynamicVLATraceRecorder.finish
    @functools.wraps(original_finish)
    def finish(*args, **kwargs):
        with profile.span("trace.finish_stack_validate"):
            trace = original_finish(*args, **kwargs)
        profile.bind_episode(trace.meta.task_id, trace.meta.episode)
        profile.mark("trace_finalized")
        return trace
    profile.patch(tracing.DynamicVLATraceRecorder, "finish", finish)

    original_get_results = upstream.get_sim_results
    @functools.wraps(original_get_results)
    def get_results(*args, **kwargs):
        environment = args[1] if len(args) > 1 else kwargs.get("environment_path", "unknown")
        profile.begin_episode(str(environment))
        try:
            with profile.span("episode.get_sim_results"):
                return original_get_results(*args, **kwargs)
        finally:
            profile.mark("rollout_returned")
    profile.patch(upstream, "get_sim_results", get_results)
    profile.wrap(upstream, "get_test_env", "environment.construct_reset")
    profile.wrap(upstream, "get_frames", "video.collect_frames")
    profile.wrap(upstream.sim, "get_frames", "video.compose_frames")
    profile.wrap(upstream.sim, "dump_video", "video.encode")

    original_sockets = upstream.get_zmq_sockets
    class ObservationSocket:
        def __init__(self, socket):
            self.socket = socket
        def send_pyobj(self, *args, **kwargs):
            with profile.span("ipc.observation_send"):
                return self.socket.send_pyobj(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(self.socket, name)
    @functools.wraps(original_sockets)
    def get_sockets(*args, **kwargs):
        observation, action = original_sockets(*args, **kwargs)
        return ObservationSocket(observation), action
    profile.patch(upstream, "get_zmq_sockets", get_sockets)

    original_persist = simulator._persist_trace
    @functools.wraps(original_persist)
    def persist(trace, *args, **kwargs):
        episode = profile.episode_for(trace.meta.task_id, trace.meta.episode)
        with profile.episode_scope(episode), profile.span("trace.persist"):
            result = original_persist(trace, *args, **kwargs)
        profile.mark("trace_persisted", episode)
        return result
    profile.patch(simulator, "_persist_trace", persist)
    profile.wrap(simulator._TraceReceiptWriter, "_commit_oldest", "trace.writer_wait_commit")
    original_receipt = simulator._write_receipt
    @functools.wraps(original_receipt)
    def receipt(*args, **kwargs):
        episode = profile.episode_for(kwargs["task_id"], kwargs["episode"])
        with profile.episode_scope(episode), profile.span("trace.receipt_write"):
            result = original_receipt(*args, **kwargs)
        profile.mark("receipt_committed", episode)
        with profile.span("profiling.snapshot_write"):
            profile.write()
        return result
    profile.patch(simulator, "_write_receipt", receipt)
