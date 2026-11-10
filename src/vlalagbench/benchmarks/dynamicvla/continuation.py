"""Resume DynamicVLA traces from recorded simulator and protocol state."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vlalagbench.benchmarks.dynamicvla.protocol import (
    POLICY_NOISE_SCHEME,
    POLICY_STARTUP_STEPS,
    audited_trace_receipt,
    receipt_sha256,
)
from vlalagbench.benchmarks.dynamicvla.tracing import (
    ContinuationMeta,
    DynamicVLATrace,
    is_image_observation,
)
from vlalagbench.timing import ChunkScheduler, TimingConfig

MODEL_ACTION_SHAPE = (1, 7)


def preflight_continuation(
    trace: DynamicVLATrace,
    resume_step: int,
    checkpoint_config: Mapping[str, Any] | None = None,
) -> dict:
    """Validate a trace continuation without starting policy or simulator processes."""

    if (
        trace.meta.inference_mode != "synchronous"
        and trace.meta.action_index_origin != "observation"
    ):
        raise ValueError(
            "cannot continue a release-relative async trace with observation-relative inference"
        )
    if (
        trace.meta.inference_mode != "synchronous"
        and trace.meta.execution_horizon_basis != "post_latency_action_count"
    ):
        raise ValueError(
            "cannot continue an async trace with a different execution horizon basis"
        )
    if trace.meta.status not in (0, 1):
        raise ValueError("failed policy traces cannot be continued")
    if trace.meta.policy_noise_scheme != POLICY_NOISE_SCHEME:
        raise ValueError("trace uses an unsupported continuation noise scheme")
    if checkpoint_config is not None:
        if not isinstance(checkpoint_config, Mapping):
            raise ValueError("continuation checkpoint config must be an object")
        if checkpoint_config.get("type") != "dynamicvla":
            raise ValueError("continuation checkpoint config is not DynamicVLA")
        if checkpoint_config.get("n_action_steps") != trace.meta.action_chunk_steps:
            raise ValueError("continuation checkpoint action chunk does not match trace")
        delta_timestamps = checkpoint_config.get("delta_timestamps")
        timestamps = (
            delta_timestamps.get("observation")
            if isinstance(delta_timestamps, Mapping)
            else None
        )
        if timestamps != [-2, 0]:
            raise ValueError(
                "continuation checkpoint must use the verified [-2, 0] observation window"
            )
    return policy_snapshot_at(trace, resume_step)


def validate_resume_point(trace: DynamicVLATrace, resume_step: int) -> None:
    """Require a non-success state, reopening only deadline terminations."""

    if not 1 <= resume_step <= trace.num_steps:
        raise ValueError("resume step must select a recorded post-action state")
    if trace.success[:resume_step].any():
        raise ValueError("successful trace prefixes cannot be deadline-extended")
    if trace.done[resume_step - 1]:
        fired = {
            name
            for name, value in trace.terminations[resume_step - 1].items()
            if value
        }
        if not fired.issubset(trace.meta.timeout_terms):
            raise ValueError("non-timeout termination states cannot be deadline-extended")


def policy_snapshot_at(trace: DynamicVLATrace, resume_step: int) -> dict:
    """Return the exact policy state immediately before ``resume_step``."""

    validate_resume_point(trace, resume_step)
    try:
        if resume_step == trace.num_steps:
            snapshot = copy.deepcopy(trace.final_policy)
        else:
            step = trace.protocol[resume_step - 1]
            snapshot = copy.deepcopy(step["policy"])
            snapshot["get_action_count"] = int(step["get_action_count"])
            snapshot["base_state"] = copy.deepcopy(step["base_state"])
            _backfill_pending_chunk(trace, snapshot)
        _validate_policy_snapshot(trace, snapshot, resume_step)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"trace policy snapshot is not restorable: {exc}") from None
    return snapshot


def _backfill_pending_chunk(trace: DynamicVLATrace, snapshot: dict) -> None:
    pending = snapshot["scheduler"]["pending"]
    if pending is None or pending["chunk"] is not None:
        return
    request = pending["request"]
    due = int(request["available_step"])
    if due < trace.num_steps:
        protocol = trace.protocol[due]
        release = protocol["release"]
        released = protocol["released_chunk"]
        expected_release = {
            "request_id": request["id"],
            "observation_step": request["sent_step"],
            "release_step": request["available_step"],
            "chunk_index": request["id"],
        }
        if not isinstance(release, dict) or any(
            release.get(key) != value for key, value in expected_release.items()
        ):
            raise ValueError("pending chunk release identity does not match its request")
        if (
            not isinstance(released, dict)
            or released.get("index") != request["sent_step"]
        ):
            raise ValueError("pending chunk source does not match its request")
        pending["chunk"] = copy.deepcopy(released["actions"])
    else:
        final_pending = trace.final_policy["scheduler"]["pending"]
        if (
            final_pending is None
            or final_pending["request"] != request
            or final_pending["chunk"] is None
        ):
            raise ValueError("trace does not contain the pending chunk at resume step")
        pending["chunk"] = copy.deepcopy(final_pending["chunk"])


def _validate_policy_snapshot(
    trace: DynamicVLATrace, snapshot: dict, resume_step: int
) -> None:
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError("policy snapshot is empty")
    if snapshot.get("episode_seed") != trace.meta.policy_episode_seed:
        raise ValueError("policy snapshot episode seed does not match trace")
    for key in ("action_index", "get_action_count"):
        if type(snapshot.get(key)) is not int:
            raise TypeError(f"policy snapshot {key} must be an integer")
    if "base_state" not in snapshot:
        raise KeyError("base_state")
    scheduler = snapshot.get("scheduler")
    if not isinstance(scheduler, dict):
        raise TypeError("policy scheduler snapshot must be an object")
    next_step = scheduler.get("next_step")
    next_request_id = scheduler.get("next_request_id")
    if type(next_step) is not int or next_step != max(
        POLICY_STARTUP_STEPS, resume_step
    ):
        raise ValueError("policy scheduler next step does not match resume step")
    if type(next_request_id) is not int or next_request_id < 0:
        raise ValueError("policy scheduler next request ID is invalid")
    queue = scheduler.get("queue")
    if not isinstance(queue, list) or not all(isinstance(item, dict) for item in queue):
        raise TypeError("policy scheduler queue is invalid")
    first_action_index = (
        trace.meta.latency_steps
        if trace.meta.action_index_origin == "observation"
        else 0
    )
    last_action_index = trace.meta.execution_horizon_steps
    if trace.meta.execution_horizon_basis in ("action_count", "post_latency_action_count"):
        last_action_index += first_action_index
    queue_identity = None
    previous_action_index = None
    for item in queue:
        if item.get("active") is not True or item.get("reason") != "model":
            raise ValueError("policy scheduler queue state is invalid")
        if any(
            type(item.get(key)) is not int
            for key in ("chunk_id", "index_in_chunk", "observation_step")
        ):
            raise TypeError("policy scheduler queue provenance is invalid")
        if (
            not 0 <= item["chunk_id"] < next_request_id
            or not first_action_index
            <= item["index_in_chunk"]
            < last_action_index
            or not POLICY_STARTUP_STEPS
            <= item["observation_step"]
            < next_step
        ):
            raise ValueError("policy scheduler queue provenance is inconsistent")
        action = np.asarray(item.get("action"))
        if (
            action.shape != MODEL_ACTION_SHAPE
            or not np.issubdtype(action.dtype, np.number)
            or not np.isfinite(action).all()
        ):
            raise ValueError("policy scheduler queue action is invalid")
        identity = (item["chunk_id"], item["observation_step"])
        if queue_identity is None:
            queue_identity = identity
        elif identity != queue_identity:
            raise ValueError("policy scheduler queue mixes action chunks")
        if (
            previous_action_index is not None
            and item["index_in_chunk"] != previous_action_index + 1
        ):
            raise ValueError("policy scheduler queue action indices are not contiguous")
        previous_action_index = item["index_in_chunk"]
    pending = scheduler.get("pending")
    if queue and pending is not None and trace.meta.inference_mode == "synchronous":
        raise ValueError("synchronous policy scheduler has queue and pending state")
    if pending is not None:
        if not isinstance(pending, dict) or not isinstance(
            pending.get("request"), dict
        ):
            raise TypeError("policy scheduler pending request is invalid")
        request = pending["request"]
        if any(
            type(request.get(key)) is not int
            for key in ("id", "sent_step", "available_step")
        ):
            raise TypeError("policy scheduler request identity is invalid")
        if (
            request["id"] != next_request_id - 1
            or request["sent_step"] < POLICY_STARTUP_STEPS
            or not request["sent_step"] < next_step <= request["available_step"]
            or request["available_step"]
            != request["sent_step"] + trace.meta.latency_steps
        ):
            raise ValueError("policy scheduler request identity is inconsistent")
        chunk = pending.get("chunk")
        if not isinstance(chunk, list) or len(chunk) != trace.meta.action_chunk_steps:
            raise ValueError(
                "pending policy chunk length does not match trace action chunk"
            )
        if any(
            (action := np.asarray(value)).shape != MODEL_ACTION_SHAPE
            or not np.issubdtype(action.dtype, np.number)
            or not np.isfinite(action).all()
            for value in chunk
        ):
            raise ValueError("pending policy chunk actions are invalid")
    if queue:
        observation_step = queue[0]["observation_step"]
        expected_first = next_step - observation_step
        if trace.meta.action_index_origin == "release":
            expected_first -= trace.meta.latency_steps
        expected_last = last_action_index - 1
        expected_chunk_id = next_request_id - (2 if pending is not None else 1)
        if (
            queue[0]["index_in_chunk"] != expected_first
            or queue[-1]["index_in_chunk"] != expected_last
        ):
            raise ValueError("policy scheduler queue is not the exact action suffix")
        if queue[0]["chunk_id"] != expected_chunk_id:
            raise ValueError("policy scheduler queue request lineage is inconsistent")
    ChunkScheduler.from_snapshot(
        TimingConfig(
            scheme=trace.meta.inference_mode,
            action_chunk_steps=trace.meta.action_chunk_steps,
            execution_horizon_steps=trace.meta.execution_horizon_steps,
            latency_steps=trace.meta.latency_steps,
            promotion_offset_steps=first_action_index,
            execution_horizon_basis=trace.meta.execution_horizon_basis,
        ),
        scheduler,
        lambda action: action,
    )


def merge_continuation(
    source: DynamicVLATrace,
    suffix: DynamicVLATrace,
    *,
    resume_step: int,
    source_path: str | Path,
    source_sha256: str,
    max_additional_steps: int,
) -> DynamicVLATrace:
    """Join a recorded prefix to a newly executed continuation suffix.

    Camera frames are kept only when both traces store them. Without frames in
    the source, the suffix's policy history was re-rendered from saved states,
    which the continuation metadata records.
    """

    validate_resume_point(source, resume_step)
    if not 1 <= suffix.num_steps <= max_additional_steps:
        raise ValueError("continuation suffix length is outside its requested budget")
    ignored_meta = {
        "schema_version",
        "episode_length_s",
        "deadline_basis",
        "num_steps",
        "status",
        "success",
        "recorded_at",
        "continuation",
    }
    if source.meta.model_dump(exclude=ignored_meta) != suffix.meta.model_dump(
        exclude=ignored_meta
    ):
        raise ValueError("continuation metadata does not match source trace")
    if source.states.keys() != suffix.states.keys():
        raise ValueError("continuation state structure does not match source trace")
    source_images = {path for path in source.observations if is_image_observation(path)}
    suffix_images = {path for path in suffix.observations if is_image_observation(path)}
    if (
        source.observations.keys() - source_images
        != suffix.observations.keys() - suffix_images
        or (source_images and suffix_images and source_images != suffix_images)
    ):
        raise ValueError("continuation observation structure does not match source trace")
    observation_paths = [
        path
        for path in source.observations
        if not is_image_observation(path) or (source_images and suffix_images)
    ]
    if any(
        not np.array_equal(
            source.states[path][resume_step], suffix.states[path][0], equal_nan=True
        )
        for path in source.states
    ):
        raise ValueError("continuation did not start from the selected source state")
    if any(
        not np.array_equal(
            source.observations[path][resume_step], suffix.observations[path][0]
        )
        for path in observation_paths
    ):
        raise ValueError("continuation did not start from the selected observation")
    done_prefix = source.done[:resume_step].copy()
    done_prefix[-1] = False
    termination_prefix = copy.deepcopy(source.terminations[:resume_step])
    for name in source.meta.timeout_terms:
        termination_prefix[-1][name] = False
    states = {
        path: np.concatenate(
            [source.states[path][: resume_step + 1], suffix.states[path][1:]],
            axis=0,
        )
        for path in source.states
    }
    observations = {
        path: np.concatenate(
            [
                source.observations[path][: resume_step + 1],
                suffix.observations[path][1:],
            ],
            axis=0,
        )
        for path in observation_paths
    }
    actions = np.concatenate(
        [source.actions[:resume_step], suffix.actions], axis=0
    )
    success = np.concatenate(
        [source.success[:resume_step], suffix.success], axis=0
    )
    done = np.concatenate([done_prefix, suffix.done], axis=0)
    meta = suffix.meta.model_copy(
        update={
            "num_steps": len(actions),
            "success": bool(success.any()),
            "continuation": ContinuationMeta.model_validate(
                {
                    "source_trace": str(source_path),
                    "source_trace_sha256": source_sha256,
                    "resume_step": resume_step,
                    "original_num_steps": source.num_steps,
                    "max_additional_steps": max_additional_steps,
                    "protocol_receipt_sha256": "0" * 64,
                    "history_images": "recorded" if source_images else "rerendered",
                }
            ),
        }
    )
    meta = type(meta).model_validate(meta.model_dump(mode="json"))
    trace = DynamicVLATrace(
        meta=meta,
        states=states,
        observations=observations,
        actions=actions,
        success=success,
        done=done,
        terminations=[
            *termination_prefix,
            *suffix.terminations,
        ],
        protocol=[*source.protocol[:resume_step], *suffix.protocol],
        final_policy=suffix.final_policy,
    )
    receipt = audited_trace_receipt(trace)
    if receipt["errors"]:
        raise ValueError(f"continuation latency receipt is invalid: {receipt['errors']}")
    trace.meta = trace.meta.model_copy(
        update={
            "continuation": trace.meta.continuation.model_copy(
                update={"protocol_receipt_sha256": receipt_sha256(receipt)}
            )
        }
    )
    trace.validate()
    return trace
