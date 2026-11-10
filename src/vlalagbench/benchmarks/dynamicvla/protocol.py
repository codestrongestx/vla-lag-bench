"""Receipt validation for DynamicVLA control-step scheduling."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

POLICY_STARTUP_STEPS = 5
POLICY_NOISE_SCHEME = "sha256-policy-task-environment-replica-request-cpu-torch-v2"


def derive_episode_seed(
    policy_seed: int, task_name: str, environment_sha256: str, replica_id: int
) -> int:
    return _derive_seed(
        "episode",
        {
            "policy_seed": policy_seed,
            "task_name": task_name,
            "environment_sha256": environment_sha256,
            "replica_id": replica_id,
        },
    )


def derive_request_seed(episode_seed: int, request_id: int) -> int:
    return _derive_seed(
        "request", {"episode_seed": episode_seed, "request_id": request_id}
    )


def _derive_seed(kind: str, identity: dict[str, Any]) -> int:
    payload = json.dumps(
        {"scheme": POLICY_NOISE_SCHEME, "kind": kind, **identity},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def receipt_errors(
    receipt: Mapping[str, Any],
    *,
    mode: str,
    latency_steps: int,
    action_chunk_steps: int,
    execution_horizon_steps: int,
) -> list[str]:
    """Return falsified protocol invariants for an episode receipt."""

    if mode not in ("continuous", "synchronous", "lazy"):
        raise ValueError(f"unsupported DynamicVLA inference mode: {mode!r}")
    try:
        return _receipt_errors(
            receipt,
            mode=mode,
            latency_steps=latency_steps,
            action_chunk_steps=action_chunk_steps,
            execution_horizon_steps=execution_horizon_steps,
        )
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return ["receipt structure is invalid"]


def trace_receipt(trace: Any) -> dict[str, Any]:
    """Reconstruct a latency receipt from a recorded DynamicVLA trace."""

    request_events = []
    release_events = []
    applied_events = []
    control_events = []
    gap_steps = []
    blocked_s = 0.0
    violations = 0
    for step, (protocol, action) in enumerate(
        zip(trace.protocol, trace.actions, strict=True)
    ):
        if not isinstance(protocol, Mapping):
            raise TypeError("trace protocol event must be an object")
        request = protocol.get("request")
        if request is not None:
            request_events.append({**request, "step": step})
        release = protocol.get("release")
        if release is not None:
            release_events.append({**release, "step": step})
        if protocol.get("wait_reason") == "model":
            applied_events.append(
                {
                    "step": step,
                    "chunk_index": protocol.get("chunk_index"),
                    "observation_step": protocol.get("observation_step"),
                    "action_index": protocol.get("action_index"),
                }
            )
        else:
            gap_steps.append(step)
        control_events.append(
            {
                "step": step,
                "action": action.tolist(),
                "reply_to": protocol.get("reply_to"),
                "wait_reason": protocol.get("wait_reason"),
                "chunk_index": protocol.get("chunk_index"),
                "observation_step": protocol.get("observation_step"),
                "action_index": protocol.get("action_index"),
            }
        )
        blocked_s += float(protocol.get("blocked_s") or 0.0)
        violations += protocol.get("reply_to") != step

    startup_gap = 0
    for expected, gap in enumerate(gap_steps):
        if gap != expected:
            break
        startup_gap += 1
    staleness_hist: dict[int, int] = {}
    for event in applied_events:
        staleness = event["step"] - event["observation_step"]
        staleness_hist[staleness] = staleness_hist.get(staleness, 0) + 1
    n_steps = trace.num_steps
    applied = len(applied_events)
    deadline_fields = {}
    if hasattr(trace.meta, "episode_length_s") and hasattr(trace.meta, "control_hz"):
        deadline_basis = getattr(trace.meta, "deadline_basis", "control_steps")
        deadline_fields = {
            "deadline_basis": deadline_basis,
            "active_action_deadline_steps": round(
                trace.meta.episode_length_s * trace.meta.control_hz
            ),
            "deadline_paused_steps": (
                len(gap_steps) if deadline_basis == "active_action_steps" else 0
            ),
        }
    return {
        "episode_seed": trace.meta.policy_episode_seed,
        "mode": trace.meta.inference_mode,
        "action_index_origin": trace.meta.action_index_origin,
        "execution_horizon_steps": trace.meta.execution_horizon_steps,
        **(
            {"execution_horizon_basis": "post_latency_action_count"}
            if getattr(trace.meta, "execution_horizon_basis", "action_count")
            == "post_latency_action_count"
            else {}
        ),
        "injected_steps": trace.meta.latency_steps,
        "status": trace.meta.status,
        "success": trace.meta.success,
        **deadline_fields,
        "n_steps": n_steps,
        "applied": applied,
        "coverage": applied / max(n_steps, 1),
        "startup_gap": startup_gap,
        "steady_gaps": len(gap_steps) - startup_gap,
        "gap_steps": gap_steps,
        "requests": len(request_events),
        "request_events": request_events,
        "releases": len(release_events),
        "release_events": release_events,
        "applied_events": applied_events,
        "control_events": control_events,
        "staleness_hist": dict(sorted(staleness_hist.items())),
        "blocked_s": round(blocked_s, 4),
        "violations": violations,
        "timeouts": 0,
    }


def receipt_sha256(receipt: Mapping[str, Any]) -> str:
    normalized = json.loads(json.dumps(receipt, allow_nan=False))
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def audited_trace_receipt(trace: Any) -> dict[str, Any]:
    receipt = trace_receipt(trace)
    errors = receipt_errors(
        receipt,
        mode=trace.meta.inference_mode,
        latency_steps=trace.meta.latency_steps,
        action_chunk_steps=trace.meta.action_chunk_steps,
        execution_horizon_steps=trace.meta.execution_horizon_steps,
    )
    receipt["errors"] = errors
    receipt["exact"] = not errors
    return receipt


def _receipt_errors(
    receipt: Mapping[str, Any],
    *,
    mode: str,
    latency_steps: int,
    action_chunk_steps: int,
    execution_horizon_steps: int,
) -> list[str]:
    errors: list[str] = []
    n_steps = int(receipt.get("n_steps", -1))
    applied = int(receipt.get("applied", -1))
    gap_steps = [int(step) for step in receipt.get("gap_steps", [])]
    request_events = receipt.get("request_events", [])
    release_events = receipt.get("release_events", [])
    applied_events = receipt.get("applied_events", [])
    control_events = receipt.get("control_events", [])

    action_index_origin = receipt.get("action_index_origin")
    horizon_basis = receipt.get("execution_horizon_basis", "action_count")
    if (
        receipt.get("mode") != mode
        or receipt.get("execution_horizon_steps") != execution_horizon_steps
        or action_index_origin not in ("observation", "release")
        or (mode == "synchronous" and action_index_origin != "release")
        or horizon_basis not in ("action_count", "post_latency_action_count")
    ):
        errors.append("inference schedule identity is invalid")
    if horizon_basis == "post_latency_action_count":
        if not 0 < execution_horizon_steps <= action_chunk_steps:
            errors.append("execution horizon exceeds the action chunk")
        if mode != "synchronous" and (
            latency_steps < 0
            or latency_steps + execution_horizon_steps > action_chunk_steps
        ):
            errors.append("async latency plus execution horizon must fit in the action chunk")
        if mode != "synchronous" and action_index_origin != "observation":
            errors.append("async action index origin must be observation")
        if errors:
            return errors
    if n_steps <= 0 or not 0 <= applied <= n_steps:
        errors.append("step counts are invalid")
    if applied + len(gap_steps) != n_steps:
        errors.append("applied and gap steps do not partition the episode")
    deadline_basis = receipt.get("deadline_basis")
    if deadline_basis is not None:
        if deadline_basis not in ("control_steps", "active_action_steps"):
            errors.append("deadline basis is invalid")
        else:
            deadline_steps = int(receipt.get("active_action_deadline_steps", -1))
            paused_steps = int(receipt.get("deadline_paused_steps", -1))
            expected_paused = (
                len(gap_steps) if deadline_basis == "active_action_steps" else 0
            )
            if deadline_steps <= 0 or paused_steps != expected_paused:
                errors.append("deadline accounting is invalid")
            elif n_steps > deadline_steps + paused_steps:
                errors.append("episode exceeded its paused deadline")
    if len(applied_events) != applied:
        errors.append("not every applied action has a provenance event")
    if receipt.get("requests") != len(request_events):
        errors.append("request count does not match request events")
    if receipt.get("releases") != len(release_events):
        errors.append("release count does not match release events")
    if len(control_events) != n_steps:
        errors.append("control events do not account for every step")
    expected_coverage = applied / max(n_steps, 1)
    if not math.isclose(
        float(receipt.get("coverage", -1)), expected_coverage, abs_tol=1e-12
    ):
        errors.append("coverage does not match step counts")
    if receipt.get("timeouts") != 0:
        errors.append("policy reply timed out")
    if receipt.get("violations") != 0:
        errors.append("lockstep schedule was violated")
    if receipt.get("status") == 2 and applied != 0:
        errors.append("status 2 episode applied policy actions")
    if receipt.get("releases", 0) <= 0 and receipt.get("status") != 2:
        errors.append("no action chunk was released")
    episode_seed = receipt.get("episode_seed")
    if not isinstance(episode_seed, int) or any(
        event.get("request_seed")
        != derive_request_seed(episode_seed, int(event.get("request_id", -1)))
        or re.fullmatch(r"[0-9a-f]{64}", str(event.get("noise_sha256"))) is None
        for event in request_events
    ):
        errors.append("request noise identities are invalid")

    histogram_total = sum(int(count) for count in receipt.get("staleness_hist", {}).values())
    if histogram_total != applied:
        errors.append("staleness histogram does not account for every applied action")
    if mode == "continuous":
        expected = _continuous_schedule(
            n_steps,
            latency_steps,
            (
                latency_steps + execution_horizon_steps
                if horizon_basis == "post_latency_action_count"
                else action_chunk_steps
            ),
            action_index_origin=str(action_index_origin),
        )
    elif mode == "synchronous":
        expected = _synchronous_schedule(
            n_steps, latency_steps, execution_horizon_steps
        )
    elif mode == "lazy":
        expected = _lazy_schedule(
            n_steps,
            latency_steps,
            action_chunk_steps,
            execution_horizon_steps,
            action_index_origin=str(action_index_origin),
        )
    expected_requests, expected_releases, expected_applied, expected_gaps = expected
    if _normalize_requests(request_events) != expected_requests:
        errors.append("request events do not match the configured schedule")
    if _normalize_releases(release_events) != expected_releases:
        errors.append("release events do not match the configured schedule")
    if _normalize_applied(applied_events) != expected_applied:
        errors.append("applied-action provenance does not match the release schedule")
    if gap_steps != expected_gaps:
        errors.append(f"gap steps were {gap_steps}, expected {expected_gaps}")

    expected_histogram: dict[int, int] = {}
    for event in expected_applied:
        staleness = event["step"] - event["observation_step"]
        expected_histogram[staleness] = expected_histogram.get(staleness, 0) + 1
    normalized_histogram = {
        int(position): int(count)
        for position, count in receipt.get("staleness_hist", {}).items()
    }
    if normalized_histogram != expected_histogram:
        errors.append("staleness histogram does not match provenance events")

    expected_wait_reasons = {
        step: ("policy_warmup" if step < POLICY_STARTUP_STEPS else "inference_wait")
        for step in expected_gaps
    }
    normalized_controls = [
        {
            "step": int(event.get("step", -1)),
            "wait_reason": event.get("wait_reason"),
        }
        for event in control_events
    ]
    if any(
        event["step"] != index
        or event["wait_reason"]
        != expected_wait_reasons.get(index, "model")
        for index, event in enumerate(normalized_controls)
    ):
        errors.append("control wait reasons do not match the release schedule")

    startup_gap = 0
    for expected, gap in enumerate(gap_steps):
        if expected != gap:
            break
        startup_gap += 1
    if receipt.get("startup_gap") != startup_gap:
        errors.append("startup gap count does not match gap events")
    if receipt.get("steady_gaps") != len(gap_steps) - startup_gap:
        errors.append("steady gap count does not match gap events")
    return errors


def _continuous_schedule(
    n_steps: int,
    latency_steps: int,
    action_chunk_steps: int,
    *,
    action_index_origin: str = "release",
) -> tuple[list[dict[str, int]], list[dict[str, int]], list[dict[str, int]], list[int]]:
    if action_index_origin not in ("observation", "release"):
        raise ValueError("continuous action index origin is invalid")
    period = max(latency_steps, 1)
    sources = list(range(POLICY_STARTUP_STEPS, n_steps, period))
    requests = [
        {
            "step": source,
            "request_id": request_id,
            "observation_step": source,
            "release_step": source + latency_steps,
        }
        for request_id, source in enumerate(sources)
    ]
    releases = [
        {**request, "step": request["release_step"], "chunk_index": request["request_id"]}
        for request in requests
        if request["release_step"] < n_steps
    ]
    releases_by_step = {event["step"]: event for event in releases}
    active = None
    applied = []
    gaps = []
    for step in range(n_steps):
        if step in releases_by_step:
            active = releases_by_step[step]
        action_index = (
            step - active[action_index_origin + "_step"]
            if active is not None
            else -1
        )
        if active is None or action_index >= action_chunk_steps:
            gaps.append(step)
        else:
            applied.append(
                {
                    "step": step,
                    "chunk_index": active["chunk_index"],
                    "observation_step": active["observation_step"],
                    "action_index": action_index,
                }
            )
    return requests, releases, applied, gaps


def _synchronous_schedule(
    n_steps: int,
    latency_steps: int,
    execution_horizon_steps: int,
) -> tuple[list[dict[str, int]], list[dict[str, int]], list[dict[str, int]], list[int]]:
    requests = []
    releases = []
    applied = []
    request_step = POLICY_STARTUP_STEPS
    request_id = 0
    while request_step < n_steps:
        release_step = request_step + latency_steps
        request = {
            "step": request_step,
            "request_id": request_id,
            "observation_step": request_step,
            "release_step": release_step,
        }
        requests.append(request)
        if release_step >= n_steps:
            break
        releases.append(
            {**request, "step": release_step, "chunk_index": request_id}
        )
        for action_index in range(
            min(execution_horizon_steps, n_steps - release_step)
        ):
            applied.append(
                {
                    "step": release_step + action_index,
                    "chunk_index": request_id,
                    "observation_step": request_step,
                    "action_index": action_index,
                }
            )
        request_step = release_step + execution_horizon_steps
        request_id += 1
    applied_steps = {event["step"] for event in applied}
    gaps = [step for step in range(n_steps) if step not in applied_steps]
    return requests, releases, applied, gaps


def _lazy_schedule(
    n_steps: int,
    latency_steps: int,
    action_chunk_steps: int,
    execution_horizon_steps: int,
    *,
    action_index_origin: str = "release",
) -> tuple[list[dict[str, int]], list[dict[str, int]], list[dict[str, int]], list[int]]:
    first = latency_steps if action_index_origin == "observation" else 0
    executed_steps = execution_horizon_steps
    if executed_steps <= 0 or first + executed_steps > action_chunk_steps:
        raise ValueError("lazy execution horizon exceeds the action chunk")
    period = max(latency_steps, executed_steps)
    sources = list(range(POLICY_STARTUP_STEPS, n_steps, period))
    requests = [
        {
            "step": source,
            "request_id": request_id,
            "observation_step": source,
            "release_step": source + latency_steps,
        }
        for request_id, source in enumerate(sources)
    ]
    releases = [
        {
            **request,
            "step": request["release_step"],
            "chunk_index": request["request_id"],
        }
        for request in requests
        if request["release_step"] < n_steps
    ]
    applied = [
        {
            "step": release["step"] + offset,
            "chunk_index": release["chunk_index"],
            "observation_step": release["observation_step"],
            "action_index": first + offset,
        }
        for release in releases
        for offset in range(
            min(executed_steps, n_steps - release["step"])
        )
    ]
    applied_steps = {event["step"] for event in applied}
    gaps = [step for step in range(n_steps) if step not in applied_steps]
    return requests, releases, applied, gaps


def _normalize_requests(events: Any) -> list[dict[str, int]]:
    return [
        {
            "step": int(event.get("step", -1)),
            "request_id": int(event.get("request_id", -1)),
            "observation_step": int(event.get("observation_step", -1)),
            "release_step": int(event.get("release_step", -1)),
        }
        for event in events
    ]


def _normalize_releases(events: Any) -> list[dict[str, int]]:
    return [
        {
            **event,
            "chunk_index": int(source.get("chunk_index", -1)),
        }
        for source, event in zip(events, _normalize_requests(events), strict=True)
    ]


def _normalize_applied(events: Any) -> list[dict[str, int]]:
    return [
        {
            "step": int(event.get("step", -1)),
            "chunk_index": int(event.get("chunk_index", -1)),
            "observation_step": int(event.get("observation_step", -1)),
            "action_index": int(event.get("action_index", -1)),
        }
        for event in events
    ]
