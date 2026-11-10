from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vlalagbench.benchmarks.dynamicvla.continuation import (
    _validate_policy_snapshot,
    merge_continuation,
    policy_snapshot_at,
    preflight_continuation,
    validate_resume_point,
)
from vlalagbench.benchmarks.dynamicvla.protocol import (
    POLICY_NOISE_SCHEME,
    audited_trace_receipt,
    derive_request_seed,
    receipt_sha256,
)
from vlalagbench.benchmarks.dynamicvla.tracing import (
    TRACE_SCHEMA_VERSION,
    ContinuationMeta,
    DynamicVLATrace,
    DynamicVLATraceMeta,
)
from vlalagbench.benchmarks.dynamicvla import trace_tool
from vlalagbench.benchmarks.dynamicvla.trace_tool import (
    _observation,
    _recorded_observation,
    _rendered_observation,
    _resume_history,
)
from vlalagbench.timing import ChunkScheduler, TimingConfig


def _trace() -> DynamicVLATrace:
    return DynamicVLATrace(
        meta=DynamicVLATraceMeta.model_validate(
            {
                "schema_version": 5,
                "task_id": 0,
                "task_name": "example-task",
                "episode": 1,
                "environment_index": 0,
                "replica_id": 1,
                "environment_path": "/assets/example.json",
                "environment_sha256": "a" * 64,
                "environment": {"seed": 7},
                "asset_sha256": {},
                "instruction": "move",
                "seed": 7,
                "upstream_revision": "b" * 40,
                "checkpoint": "/checkpoint",
                "checkpoint_sha256": "c" * 64,
                "checkpoint_config_sha256": "d" * 64,
                "policy_noise_scheme": POLICY_NOISE_SCHEME,
                "policy_torch_version": "2.4.0",
                "policy_seed": 0,
                "policy_episode_seed": 41,
                "episode_length_s": 8.0,
                "control_hz": 25.0,
                "physics_time_step": 0.04,
                "termination_tolerance": 0.07,
                "disable_fabric": False,
                "path_tracing": False,
                "simulator_versions": {
                    "isaaclab": "2.2.1",
                    "isaacsim": "4.5.0",
                },
                "inference_mode": "continuous",
                "latency_steps": 1,
                "action_chunk_steps": 4,
                "execution_horizon_steps": 3,
                "success_term": "success",
                "timeout_terms": ["time_out"],
                "num_steps": 2,
                "status": 0,
                "success": True,
                "recorded_at": "2026-08-28T00:00:00+00:00",
                "python_version": "3.11",
                "numpy_version": np.__version__,
            }
        ),
        states={
            ("scene", "rigid_object", "object", "root_pose"): np.asarray(
                [[[0.0]], [[1.0]], [[2.0]]], dtype=np.float32
            ),
            ("runtime", "rng", "python_gauss"): np.full(
                (3, 1), np.nan, dtype=np.float64
            ),
        },
        observations={
            ("images", "front"): np.asarray(
                [[[[[0.0]]]], [[[[1.0]]]], [[[[2.0]]]]], dtype=np.float32
            ),
            ("state", "end_effector", "pos"): np.asarray(
                [[[0.0]], [[1.0]], [[2.0]]], dtype=np.float32
            ),
        },
        actions=np.asarray(
            [[[1.0] + [0.0] * 7], [[1.0] + [0.0] * 7]], dtype=np.float32
        ),
        success=np.asarray([False, True]),
        done=np.asarray([False, True]),
        terminations=[
            {"success": False, "time_out": False},
            {"success": True, "time_out": False},
        ],
        protocol=[{}, {}],
        final_policy={},
    )


def _trace_with_steps(num_steps: int, *, latency_steps: int = 1) -> DynamicVLATrace:
    trace = _trace()
    trace.meta = trace.meta.model_copy(
        update={
            "latency_steps": latency_steps,
            "execution_horizon_steps": 4 - latency_steps,
            "execution_horizon_basis": "post_latency_action_count",
            "num_steps": num_steps,
        }
    )
    trace.states = {
        path: np.repeat(values[:1], num_steps + 1, axis=0)
        for path, values in trace.states.items()
    }
    trace.observations = {
        path: np.repeat(values[:1], num_steps + 1, axis=0)
        for path, values in trace.observations.items()
    }
    trace.actions = np.zeros((num_steps, 1, 8), dtype=np.float32)
    trace.success = np.asarray([False] * (num_steps - 1) + [True])
    trace.done = np.asarray([False] * (num_steps - 1) + [True])
    trace.terminations = [
        {"success": False, "time_out": False} for _ in range(num_steps - 1)
    ] + [{"success": True, "time_out": False}]
    trace.protocol = [{} for _ in range(num_steps)]
    trace.validate()
    return trace


def _chunk(length: int = 4) -> list[list[list[float]]]:
    return [[[float(index)] * 7] for index in range(length)]


def _policy_snapshot(
    *,
    next_step: int = 5,
    next_request_id: int = 0,
    pending: dict | None = None,
) -> dict:
    return {
        "scheduler": {
            "next_step": next_step,
            "next_request_id": next_request_id,
            "pending": pending,
            "queue": [],
        },
        "action_index": -1,
        "episode_seed": 41,
        "get_action_count": 0,
        "base_state": [[0.0]],
    }


def _deadline_trace() -> DynamicVLATrace:
    trace = _trace_with_steps(6)
    trace.success[:] = False
    trace.terminations[-1] = {"success": False, "time_out": True}
    trace.meta = trace.meta.model_copy(update={"status": 1, "success": False})
    trace.final_policy = _policy_snapshot(next_step=6)
    return trace


def test_policy_snapshot_at_internal_state_uses_recorded_future_chunk():
    trace = _trace_with_steps(8)
    trace.protocol[5] = {
        "policy": _policy_snapshot(
            next_step=6,
            next_request_id=1,
            pending={
                "request": {"id": 0, "sent_step": 5, "available_step": 6},
                "chunk": None,
            },
        ),
        "get_action_count": -2,
        "base_state": [[0.0]],
    }
    trace.protocol[6] = {
        "release": {
            "request_id": 0,
            "observation_step": 5,
            "release_step": 6,
            "chunk_index": 0,
        },
        "released_chunk": {"index": 5, "actions": _chunk()},
    }

    snapshot = policy_snapshot_at(trace, 6)

    assert snapshot["scheduler"]["pending"]["chunk"] == _chunk()
    assert snapshot["get_action_count"] == -2
    assert snapshot["base_state"] == [[0.0]]


def test_resume_step_one_restores_task_and_first_control_observation():
    trace = _trace()
    history = _resume_history(trace, 1)

    assert history[0] == {"task": "move"}
    assert history[1]["index"] == 0
    assert history[1]["task"] == "move"
    assert "task" not in _recorded_observation(trace, 1)
    np.testing.assert_array_equal(
        history[1]["observation.images.front"], np.asarray([[[[0.0]]]])
    )


def test_state_only_resume_history_re_renders_images_at_restored_states(monkeypatch):
    trace = _without_images(_trace())
    restored = []
    monkeypatch.setattr(
        trace_tool,
        "restore_env_state",
        lambda _env, state: restored.append(
            float(state[("scene", "rigid_object", "object", "root_pose")].squeeze())
        ),
    )
    monkeypatch.setattr(trace_tool, "refresh_sensors", lambda _env: None)
    monkeypatch.setattr(
        trace_tool,
        "read_policy_observation",
        lambda *_args: (
            {"front": {"rgb": np.full((1, 1, 1, 3), restored[-1])}},
            {"end_effector": {"pos": np.full((1, 3), -1.0)}},
        ),
    )
    env = SimpleNamespace(unwrapped=object())

    history = _resume_history(
        trace, 2, lambda index: _rendered_observation(object(), env, trace, index)
    )

    assert restored == [0.0, 1.0]
    assert [item["index"] for item in history] == [0, 1]
    assert all(item["task"] == "move" for item in history)
    for index, item in enumerate(history):
        np.testing.assert_array_equal(
            item["observation.images.front"], np.full((1, 1, 1, 3), float(index))
        )
        np.testing.assert_array_equal(
            item["observation.state"]["end_effector"]["pos"], [[float(index)]]
        )
    with pytest.raises(ValueError, match="missing images"):
        _recorded_observation(trace, 1)


def test_live_continuation_control_observation_omits_task(monkeypatch):
    monkeypatch.setattr(
        "vlalagbench.benchmarks.dynamicvla.trace_tool.read_policy_observation",
        lambda *_args: ({}, {"end_effector": {}}),
    )

    assert "task" not in _observation(object(), object(), _trace(), 1)


def test_trace_schedule_validates_horizon_modes():
    metadata = _trace().meta.model_dump(mode="python")
    metadata.update(
        inference_mode="synchronous",
        action_index_origin="release",
        execution_horizon_steps=4,
    )
    synchronous = DynamicVLATraceMeta.model_validate(metadata)
    assert synchronous.execution_horizon_steps == 4

    metadata.update(inference_mode="continuous", execution_horizon_steps=3)
    with pytest.raises(ValueError, match="invalid execution window"):
        DynamicVLATraceMeta.model_validate(metadata)

    metadata.update(
        inference_mode="lazy", latency_steps=1, execution_horizon_steps=3
    )
    lazy = DynamicVLATraceMeta.model_validate(metadata)
    assert lazy.execution_horizon_steps == 3
    metadata["execution_horizon_steps"] = 4
    lazy = DynamicVLATraceMeta.model_validate(metadata)
    assert lazy.execution_horizon_steps == 4


def test_trace_schema_distinguishes_legacy_and_release_relative_continuous():
    legacy = _trace().meta
    assert legacy.schema_version == 5
    assert legacy.action_index_origin == "observation"

    corrected = DynamicVLATraceMeta.model_validate(
        {
            **legacy.model_dump(mode="python"),
            "schema_version": 7,
            "action_index_origin": "release",
            "execution_horizon_steps": legacy.action_chunk_steps,
        }
    )
    assert corrected.action_index_origin == "release"

    missing_origin = corrected.model_dump(mode="python")
    missing_origin.pop("action_index_origin")
    with pytest.raises(ValueError, match="action_index_origin"):
        DynamicVLATraceMeta.model_validate(missing_origin)


def test_policy_snapshot_at_endpoint_uses_finalized_inference():
    trace = _deadline_trace()
    trace.final_policy = _policy_snapshot(
        next_step=6,
        next_request_id=1,
        pending={
            "request": {"id": 0, "sent_step": 5, "available_step": 6},
            "chunk": _chunk(),
        },
    )

    assert policy_snapshot_at(trace, trace.num_steps) == trace.final_policy


def test_policy_snapshot_uses_finalized_chunk_due_after_trace_endpoint():
    trace = _trace_with_steps(7, latency_steps=3)
    trace.protocol[5] = {
        "policy": _policy_snapshot(
            next_step=6,
            next_request_id=1,
            pending={
                "request": {"id": 0, "sent_step": 5, "available_step": 8},
                "chunk": None,
            },
        ),
        "get_action_count": 0,
        "base_state": [[0.0]],
    }
    trace.final_policy = _policy_snapshot(
        next_step=7,
        next_request_id=1,
        pending={
            "request": {"id": 0, "sent_step": 5, "available_step": 8},
            "chunk": _chunk(),
        },
    )

    snapshot = policy_snapshot_at(trace, 6)

    assert snapshot["scheduler"]["pending"]["chunk"] == _chunk()


def test_continuation_rejects_stale_pending_request_identity():
    trace = _deadline_trace()
    trace.final_policy = _policy_snapshot(
        next_step=6,
        next_request_id=2,
        pending={
            "request": {"id": 0, "sent_step": 0, "available_step": 1},
            "chunk": _chunk(),
        },
    )

    with pytest.raises(ValueError, match="request identity is inconsistent"):
        preflight_continuation(trace, trace.num_steps)


def test_continuation_preflight_rejects_failed_and_legacy_traces():
    failed = _deadline_trace()
    failed.meta = failed.meta.model_copy(update={"status": 3})
    failed.final_policy = {}
    with pytest.raises(ValueError, match="failed policy traces"):
        preflight_continuation(failed, failed.num_steps)

    legacy = _deadline_trace()
    legacy.meta = legacy.meta.model_copy(
        update={"policy_noise_scheme": "sha256-seeded-cpu-torch-randn-v1"}
    )
    with pytest.raises(ValueError, match="unsupported continuation noise scheme"):
        preflight_continuation(legacy, legacy.num_steps)


@pytest.mark.parametrize("chunk_length", [3, 5])
def test_continuation_rejects_wrong_pending_chunk_length(chunk_length):
    trace = _deadline_trace()
    trace.final_policy = _policy_snapshot(
        next_step=6,
        next_request_id=1,
        pending={
            "request": {"id": 0, "sent_step": 5, "available_step": 6},
            "chunk": _chunk(chunk_length),
        },
    )

    with pytest.raises(ValueError, match="pending policy chunk length"):
        preflight_continuation(trace, trace.num_steps)


def test_continuation_rejects_wrong_backfilled_release_identity():
    trace = _trace_with_steps(8)
    trace.protocol[5] = {
        "policy": _policy_snapshot(
            next_step=6,
            next_request_id=1,
            pending={
                "request": {"id": 0, "sent_step": 5, "available_step": 6},
                "chunk": None,
            },
        ),
        "get_action_count": -2,
        "base_state": [[0.0]],
    }
    trace.protocol[6] = {
        "release": {
            "request_id": 99,
            "observation_step": 5,
            "release_step": 6,
            "chunk_index": 99,
        },
        "released_chunk": {"index": 5, "actions": _chunk()},
    }

    with pytest.raises(ValueError, match="release identity"):
        policy_snapshot_at(trace, 6)


def test_continuation_rejects_malformed_scheduler_queue():
    trace = _deadline_trace()
    trace.final_policy["scheduler"]["queue"] = [{}]

    with pytest.raises(ValueError, match="queue state is invalid"):
        preflight_continuation(trace, trace.num_steps)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"active": "false"}, "queue state is invalid"),
        ({"reason": None}, "queue state is invalid"),
        ({"chunk_id": "0"}, "queue provenance is invalid"),
        ({"action": None}, "queue action is invalid"),
        ({"action": [[0.0] * 8]}, "queue action is invalid"),
    ],
)
def test_continuation_rejects_semantically_invalid_scheduler_queue(update, message):
    trace = _deadline_trace()
    trace.meta = trace.meta.model_copy(
        update={
            "inference_mode": "synchronous",
            "latency_steps": 0,
            "execution_horizon_steps": 4,
        }
    )
    item = {
        "action": [[0.0] * 7],
        "active": True,
        "reason": "model",
        "chunk_id": 0,
        "index_in_chunk": 1,
        "observation_step": 5,
        **update,
    }
    trace.final_policy = _policy_snapshot(next_step=6, next_request_id=1)
    trace.final_policy["scheduler"]["queue"] = [item]

    with pytest.raises(ValueError, match=message):
        preflight_continuation(trace, trace.num_steps)


@pytest.mark.parametrize("indices", [[3], [2, 3], [1, 2]])
def test_continuation_rejects_an_inexact_scheduler_queue_suffix(indices):
    trace = _deadline_trace()
    trace.meta = trace.meta.model_copy(
        update={
            "inference_mode": "synchronous",
            "latency_steps": 0,
            "execution_horizon_steps": 4,
        }
    )
    trace.final_policy = _policy_snapshot(next_step=6, next_request_id=1)
    trace.final_policy["scheduler"]["queue"] = [
        {
            "action": [[float(index)] * 7],
            "active": True,
            "reason": "model",
            "chunk_id": 0,
            "index_in_chunk": index,
            "observation_step": 5,
        }
        for index in indices
    ]

    with pytest.raises(ValueError, match="exact action suffix"):
        preflight_continuation(trace, trace.num_steps)


def test_continuation_rejects_stale_scheduler_queue_lineage():
    trace = _deadline_trace()
    trace.meta = trace.meta.model_copy(
        update={
            "inference_mode": "synchronous",
            "latency_steps": 0,
            "execution_horizon_steps": 4,
        }
    )
    trace.final_policy = _policy_snapshot(next_step=6, next_request_id=2)
    trace.final_policy["scheduler"]["queue"] = [
        {
            "action": [[float(index)] * 7],
            "active": True,
            "reason": "model",
            "chunk_id": 0,
            "index_in_chunk": index,
            "observation_step": 5,
        }
        for index in range(1, 4)
    ]

    with pytest.raises(ValueError, match="request lineage"):
        preflight_continuation(trace, trace.num_steps)


def test_continuation_accepts_a_coherent_scheduler_queue():
    trace = _deadline_trace()
    trace.meta = trace.meta.model_copy(
        update={
            "inference_mode": "synchronous",
            "latency_steps": 0,
            "execution_horizon_steps": 4,
        }
    )
    trace.final_policy = _policy_snapshot(next_step=6, next_request_id=1)
    trace.final_policy["scheduler"]["queue"] = [
        {
            "action": [[float(index)] * 7],
            "active": True,
            "reason": "model",
            "chunk_id": 0,
            "index_in_chunk": index,
            "observation_step": 5,
        }
        for index in range(1, 4)
    ]

    assert preflight_continuation(trace, trace.num_steps) == trace.final_policy


@pytest.mark.parametrize(
    ("mode", "latency_steps", "horizon_steps"),
    [
        ("continuous", 0, 6),
        ("continuous", 2, 4),
        ("continuous", 4, 2),
        ("synchronous", 0, 4),
        ("synchronous", 2, 4),
        ("lazy", 0, 4),
        ("lazy", 2, 4),
        ("lazy", 4, 2),
    ],
)
@pytest.mark.parametrize("basis", ["action_count", "post_latency_action_count"])
def test_continuation_accepts_shared_scheduler_snapshots(
    mode, latency_steps, horizon_steps, basis
):
    scheduler = ChunkScheduler(
        TimingConfig(
            mode, 6, horizon_steps, latency_steps,
            promotion_offset_steps=0 if mode == "synchronous" else latency_steps,
            execution_horizon_basis=basis,
        ), start_step=5
    )
    trace = SimpleNamespace(
        meta=SimpleNamespace(
            policy_episode_seed=41,
            inference_mode=mode,
            action_index_origin="release" if mode == "synchronous" else "observation",
            latency_steps=latency_steps,
            action_chunk_steps=6,
            execution_horizon_steps=horizon_steps,
            execution_horizon_basis=basis,
        )
    )
    for step in range(5, 15):
        request = scheduler.request(step)
        if request is not None:
            scheduler.deliver(request, _chunk(6))
        scheduler.action(step)
        snapshot = {
            "scheduler": scheduler.snapshot(),
            "action_index": step,
            "episode_seed": 41,
            "get_action_count": step,
            "base_state": None,
        }

        _validate_policy_snapshot(trace, snapshot, step + 1)


def test_legacy_horizon_is_preserved_but_not_used_for_new_async_continuation(tmp_path):
    legacy = _trace()
    path = legacy.save(tmp_path / "legacy.npz")
    restored = DynamicVLATrace.load(path)
    assert restored.meta.execution_horizon_basis == "action_count"
    assert restored.meta.execution_horizon_steps == 3
    with pytest.raises(ValueError, match="different execution horizon basis"):
        preflight_continuation(restored, restored.num_steps)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"type": "other", "n_action_steps": 4}, "not DynamicVLA"),
        (
            {
                "type": "dynamicvla",
                "n_action_steps": 20,
                "delta_timestamps": {"observation": [-2, 0]},
            },
            "action chunk",
        ),
        (
            {
                "type": "dynamicvla",
                "n_action_steps": 4,
                "delta_timestamps": {"observation": [-1, 0]},
            },
            "observation window",
        ),
    ],
)
def test_continuation_preflight_rejects_incompatible_checkpoint(config, message):
    trace = _deadline_trace()

    with pytest.raises(ValueError, match=message):
        preflight_continuation(trace, trace.num_steps, config)


def test_resume_rejects_a_non_timeout_terminal_state():
    trace = _trace()
    trace.success[:] = False
    trace.terminations[-1] = {"success": False, "time_out": False, "fell": True}
    trace.meta = trace.meta.model_copy(update={"status": 1, "success": False})

    with pytest.raises(ValueError, match="non-timeout termination"):
        validate_resume_point(trace, trace.num_steps)


def _without_images(trace: DynamicVLATrace) -> DynamicVLATrace:
    trace.observations = {
        path: values
        for path, values in trace.observations.items()
        if path[0] != "images"
    }
    return trace


def _continuation_pair(
    source_basis: str = "control_steps",
) -> tuple[DynamicVLATrace, DynamicVLATrace]:
    source = _trace()
    source.meta = source.meta.model_copy(update={"deadline_basis": source_basis})
    suffix = _trace()
    source.success[:] = False
    source.done[-1] = True
    source.terminations[-1] = {"success": False, "time_out": True}
    source.meta = source.meta.model_copy(update={"status": 1, "success": False})
    source.protocol = [
        {
            "reply_to": step,
            "request": None,
            "release": None,
            "wait_reason": "policy_warmup",
            "chunk_index": None,
            "observation_step": None,
            "action_index": None,
            "blocked_s": 0.0,
        }
        for step in range(2)
    ]
    suffix.success[:] = False
    suffix.done[-1] = True
    suffix.terminations[-1] = {"success": False, "time_out": True}
    suffix.protocol = [
        {
            "reply_to": step,
            "request": None,
            "release": None,
            "wait_reason": "policy_warmup",
            "chunk_index": None,
            "observation_step": None,
            "action_index": None,
            "blocked_s": 0.0,
        }
        for step in range(1, 3)
    ]
    suffix.states[next(iter(suffix.states))] += 1
    for values in suffix.observations.values():
        values += 1
    suffix.meta = suffix.meta.model_copy(
        update={"episode_length_s": 12.0, "status": 2, "success": False}
    )
    return source, suffix


@pytest.mark.parametrize("source_basis", ["control_steps", "active_action_steps"])
@pytest.mark.parametrize(
    "source_images,suffix_images",
    [(True, True), (True, False), (False, False), (False, True)],
)
def test_continuation_merge_preserves_prefix_and_reopens_old_deadline(
    tmp_path, source_basis, source_images, suffix_images
):
    source, suffix = _continuation_pair(source_basis)
    source.meta = source.meta.model_copy(update={"schema_version": 8})
    suffix.meta = suffix.meta.model_copy(update={"schema_version": TRACE_SCHEMA_VERSION})
    state_path = next(iter(suffix.states))
    if not source_images:
        _without_images(source)
    if not suffix_images:
        _without_images(suffix)

    merged = merge_continuation(
        source,
        suffix,
        resume_step=1,
        source_path="source.npz",
        source_sha256="d" * 64,
        max_additional_steps=2,
    )

    np.testing.assert_array_equal(
        merged.states[state_path].squeeze(), [0.0, 1.0, 2.0, 3.0]
    )
    assert merged.num_steps == 3
    np.testing.assert_array_equal(
        merged.observations[("state", "end_effector", "pos")].squeeze(),
        [0.0, 1.0, 2.0, 3.0],
    )
    assert merged.done.tolist() == [False, False, True]
    assert merged.meta.continuation is not None
    assert merged.meta.continuation.resume_step == 1
    assert merged.meta.continuation.original_num_steps == 2
    assert len(merged.meta.continuation.protocol_receipt_sha256) == 64
    assert merged.meta.episode_length_s == 12.0
    assert merged.meta.schema_version == TRACE_SCHEMA_VERSION
    assert merged.has_images is (source_images and suffix_images)
    assert merged.meta.continuation.history_images == (
        "recorded" if source_images else "rerendered"
    )
    loaded = DynamicVLATrace.load(merged.save(tmp_path / "continued.npz"))
    assert loaded.has_images is merged.has_images
    assert loaded.meta.continuation == merged.meta.continuation


def test_continuation_metadata_requires_a_protocol_receipt_digest():
    metadata = _trace().meta.model_dump(mode="python")
    metadata["continuation"] = {
        "source_trace": "source.npz",
        "source_trace_sha256": "e" * 64,
        "resume_step": 1,
        "original_num_steps": 2,
        "max_additional_steps": 2,
    }

    with pytest.raises(ValueError, match="protocol_receipt_sha256"):
        DynamicVLATraceMeta.model_validate(metadata)


def test_audited_continuation_trace_revalidates_its_embedded_receipt(tmp_path):
    trace = _trace_with_steps(8)
    trace.success[:] = False
    trace.done[:] = False
    trace.done[-1] = True
    trace.terminations = [
        {"success": False, "time_out": False} for _ in range(7)
    ] + [{"success": False, "time_out": True}]
    trace.meta = trace.meta.model_copy(update={"status": 1, "success": False})
    trace.protocol = []
    for step in range(8):
        request = None
        release = None
        if step >= 5:
            request_id = step - 5
            request = {
                "request_id": request_id,
                "observation_step": step,
                "release_step": step + 1,
                "request_seed": derive_request_seed(41, request_id),
                "noise_sha256": f"{request_id:064x}",
            }
        if step >= 6:
            release = {
                "request_id": step - 6,
                "observation_step": step - 1,
                "release_step": step,
                "chunk_index": step - 6,
            }
        wait_reason = (
            "model"
            if step >= 6
            else "policy_warmup" if step < 5 else "inference_wait"
        )
        trace.protocol.append(
            {
                "request": request,
                "release": release,
                "reply_to": step,
                "wait_reason": wait_reason,
                "chunk_index": step - 6 if step >= 6 else None,
                "observation_step": step - 1 if step >= 6 else None,
                "action_index": 1 if step >= 6 else None,
                "blocked_s": 0.0,
            }
        )
    receipt = audited_trace_receipt(trace)
    assert receipt["exact"] is True
    trace.meta = trace.meta.model_copy(
        update={
            "continuation": ContinuationMeta.model_validate(
                {
                    "source_trace": "source.npz",
                    "source_trace_sha256": "e" * 64,
                    "resume_step": 6,
                    "original_num_steps": 6,
                    "max_additional_steps": 2,
                    "protocol_receipt_sha256": receipt_sha256(receipt),
                }
            )
        }
    )

    loaded = DynamicVLATrace.load(trace.save(tmp_path / "continued.npz"))
    assert loaded.meta.continuation.protocol_receipt_sha256 == receipt_sha256(receipt)
    loaded.protocol[6]["reply_to"] = 5
    with pytest.raises(ValueError, match="protocol receipt is invalid"):
        loaded.validate()


def test_continuation_merge_compares_images_only_when_both_traces_store_them():
    source, suffix = _continuation_pair()
    suffix.observations[("images", "front")][0] += 5
    arguments = dict(
        resume_step=1,
        source_path="source.npz",
        source_sha256="d" * 64,
        max_additional_steps=2,
    )

    with pytest.raises(ValueError, match="selected observation"):
        merge_continuation(source, suffix, **arguments)
    merged = merge_continuation(_without_images(source), suffix, **arguments)
    assert not merged.has_images

    suffix.observations[("state", "end_effector", "pos")][0] += 1
    with pytest.raises(ValueError, match="selected observation"):
        merge_continuation(source, suffix, **arguments)
    suffix.observations.pop(("state", "end_effector", "pos"))
    with pytest.raises(ValueError, match="observation structure"):
        merge_continuation(source, suffix, **arguments)


def test_continuation_merge_rejects_different_provenance():
    source = _trace()
    suffix = _trace()
    suffix.meta = suffix.meta.model_copy(update={"checkpoint_sha256": "e" * 64})

    with pytest.raises(ValueError, match="metadata does not match"):
        merge_continuation(
            source,
            suffix,
            resume_step=1,
            source_path="source.npz",
            source_sha256="d" * 64,
            max_additional_steps=2,
        )


@pytest.mark.parametrize("mode", ["lazy", "continuous"])
def test_continuation_rejects_release_relative_async_trace(mode):
    trace = _trace()
    trace.meta = trace.meta.model_copy(update={
        "inference_mode": mode,
        "action_index_origin": "release",
        "execution_horizon_steps": trace.meta.action_chunk_steps,
    })
    with pytest.raises(ValueError, match="cannot continue a release-relative async trace"):
        preflight_continuation(trace, trace.num_steps)


@pytest.mark.parametrize("mode", ["lazy", "continuous"])
@pytest.mark.parametrize("latency", [6, 7, 9, 12])
def test_new_trace_metadata_enforces_async_chunk_capacity(mode, latency):
    data = _trace().meta.model_dump(mode="python")
    data.update(inference_mode=mode, action_index_origin="observation",
                execution_horizon_basis="post_latency_action_count",
                action_chunk_steps=20, execution_horizon_steps=14,
                latency_steps=latency)
    if latency + 14 > 20:
        with pytest.raises(ValueError, match="latency plus execution horizon"):
            DynamicVLATraceMeta.model_validate(data)
    else:
        assert DynamicVLATraceMeta.model_validate(data).execution_horizon_steps == 14


def test_replay_does_not_add_wait_steps_twice_to_a_continued_deadline():
    import math
    from vlalagbench.benchmarks.dynamicvla.trace_tool import _make_env, _recorder_metadata

    source = _trace()
    source.meta = source.meta.model_copy(update={"deadline_basis": "active_action_steps"})
    source.protocol = [{"wait_reason": "policy_warmup"}, {"wait_reason": "model"}]
    configured = []
    def make_env(environment, *args):
        configured.append(environment["episode_length_s"])
        return SimpleNamespace(reset=lambda **kwargs: None)
    upstream = SimpleNamespace(get_test_env=make_env, _get_env_cfg=lambda source: None)
    args = SimpleNamespace(device="cpu")
    _make_env(upstream, source, Path("/assets"), args)
    assert math.ceil(configured[-1] * source.meta.control_hz) == 201

    metadata = _recorder_metadata(source, episode_length_s=14 * 0.04, status=1, success=False)
    assert metadata["deadline_basis"] == "control_steps"
    source.meta = source.meta.model_copy(update=metadata)
    _make_env(upstream, source, Path("/assets"), args)
    assert math.ceil(configured[-1] / 0.04) == 14
