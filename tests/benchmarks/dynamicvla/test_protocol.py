from __future__ import annotations

import collections
import json
import queue
from types import SimpleNamespace

import pytest
import torch

from vlalagbench.benchmarks.dynamicvla.policy import (
    _final_policy_snapshot,
    _install_scheduler,
    _noise_checksum,
    _request_noise,
    _restore_policy_snapshot,
    _run_tests,
    _streaming_action,
)
from vlalagbench.benchmarks.dynamicvla.protocol import (
    POLICY_NOISE_SCHEME,
    derive_episode_seed,
    derive_request_seed,
    receipt_errors,
    receipt_sha256,
)
from vlalagbench.benchmarks.dynamicvla.trace_tool import _audit_continuation


IDENTITY = {
    "policy_seed": 11,
    "task_name": "example-task",
    "environment_sha256": "a" * 64,
    "replica_id": 3,
}


class _WorkerInput(dict):
    def __init__(self, output, action_chunk_steps, numeric=False):
        super().__init__()
        self.output = output
        self.action_chunk_steps = action_chunk_steps
        self.numeric = numeric

    def update(self, values):
        super().update(values)
        batch = values["obs"][0]
        source = int(batch["index"])
        actions = [
            (
                torch.tensor([float(source), float(index)])
                if self.numeric
                else f"{source}:{index}"
            )
            for index in range(self.action_chunk_steps)
        ]
        self.output.put(
            {
                "index": source,
                "actions": actions,
            }
        )


def _model(
    latency_steps: int,
    action_chunk_steps: int = 6,
    numeric=False,
    *,
    mode: str = "continuous",
    execution_horizon_steps: int = 4,
):
    output = queue.Queue(maxsize=1)
    model = SimpleNamespace(
        q_out=output,
        q_in=_WorkerInput(output, action_chunk_steps, numeric),
        config=SimpleNamespace(
            n_action_steps=action_chunk_steps,
            chunk_size=action_chunk_steps,
            max_action_dim=2,
        ),
        _action_index=0,
        reset=lambda: None,
    )
    _install_scheduler(
        model, mode, execution_horizon_steps, latency_steps, 1.0
    )
    model._vlalagbench_episode_seed = 17
    return model


def _run_policy(
    latency_steps: int,
    steps: int,
    action_chunk_steps: int = 6,
    *,
    mode: str = "continuous",
    execution_horizon_steps: int = 4,
):
    model = _model(
        latency_steps,
        action_chunk_steps,
        mode=mode,
        execution_horizon_steps=execution_horizon_steps,
    )
    actions = []
    metadata = []
    for step in range(5, steps):
        actions.append(_streaming_action(model, {"index": step}))
        metadata.append(model.last_action_meta)
    return model, actions, metadata


def _observation_relative_continuous_oracle(
    latency_steps: int, steps: int, chunk_steps: int
):
    actions = []
    period = max(latency_steps, 1)
    chunks = {
        source + latency_steps: {
            "index": source,
            "actions": [f"{source}:{index}" for index in range(chunk_steps)],
        }
        for source in range(5, steps, period)
        if source + latency_steps < steps
    }
    active = None
    for step in range(5, steps):
        chunk = chunks.get(step)
        if chunk is not None:
            active = {**chunk, "release_step": step}
        action_index = step - active["index"] if active is not None else -1
        if active is not None and action_index < chunk_steps:
            actions.append(active["actions"][action_index])
        else:
            actions.append(None)
    return actions


@pytest.mark.parametrize("mode", ["synchronous", "lazy", "continuous"])
def test_scheduler_rejects_a_missing_horizon_without_changing_the_model(mode):
    model = SimpleNamespace(config=SimpleNamespace(n_action_steps=20, chunk_size=20))
    with pytest.raises(ValueError, match="requires an execution horizon"):
        _install_scheduler(model, mode, None, 12, 1.0)
    assert vars(model.config) == {"n_action_steps": 20, "chunk_size": 20}


@pytest.mark.parametrize("mode", ["synchronous", "lazy", "continuous"])
def test_policy_cli_requires_execution_horizon_before_loading_model(mode, capsys):
    from vlalagbench.benchmarks.dynamicvla.policy import main

    with pytest.raises(SystemExit) as error:
        main([
            "--dynamicvla-root", "/unused", "--weights", "/unused",
            "--mode", mode, "--latency-steps", "12", "--reply-timeout-s", "1",
        ])
    assert error.value.code == 2
    assert "--execution-horizon-steps" in capsys.readouterr().err


def test_shared_scheduler_skips_latency_on_every_continuous_release():
    _, actions, metadata = _run_policy(2, 14)

    assert actions == _observation_relative_continuous_oracle(2, 14, 6)
    assert actions == [
        None,
        None,
        "5:2",
        "5:3",
        "7:2",
        "7:3",
        "9:2",
        "9:3",
        "11:2",
    ]
    assert [meta["action_index"] for meta in metadata[2:]] == [2, 3, 2, 3, 2, 3, 2]
    assert [meta["observation_step"] for meta in metadata[2:]] == [5, 5, 7, 7, 9, 9, 11]


def test_shared_scheduler_matches_observation_relative_oracle_across_latency_chunk_grid():
    for chunk_steps in (2, 3, 4, 6, 20):
        for latency_steps in range(chunk_steps):
            steps = 5 + 3 * chunk_steps
            _, actions, metadata = _run_policy(
                latency_steps, steps, chunk_steps, execution_horizon_steps=chunk_steps - latency_steps
            )

            assert actions == _observation_relative_continuous_oracle(
                latency_steps, steps, chunk_steps
            ), (latency_steps, chunk_steps)
            for step, (action, meta) in enumerate(
                zip(actions, metadata, strict=True), start=5
            ):
                if action is None:
                    continue
                source, position = (int(value) for value in action.split(":"))
                assert meta["observation_step"] == source
                assert meta["action_index"] == position
                assert position == step - source


def test_zero_latency_replaces_the_chunk_on_every_control_tick():
    _, actions, metadata = _run_policy(0, 9)

    assert actions == _observation_relative_continuous_oracle(0, 9, 6)
    assert actions == ["5:0", "6:0", "7:0", "8:0"]
    assert [meta["request"]["request_id"] for meta in metadata] == [0, 1, 2, 3]
    assert [meta["release"]["chunk_index"] for meta in metadata] == [0, 1, 2, 3]


@pytest.mark.parametrize("mode", ["synchronous", "lazy", "continuous"])
def test_saved_action_index_matches_the_control_step(mode):
    _, _, metadata = _run_policy(
        6, 80, 20, mode=mode,
        execution_horizon_steps=14,
    )
    for step, meta in enumerate(metadata, start=5):
        if meta["wait_reason"] == "model":
            assert meta["policy"]["action_index"] == step


@pytest.mark.parametrize("latency_steps", [0, 3, 6, 9, 12])
def test_formal_continuous_sweep_skips_stale_prefix_and_waits_on_exhaustion(latency_steps):
    if latency_steps + 14 > 20:
        with pytest.raises(ValueError, match="latency plus execution horizon"):
            _scheduled_receipt(latency_steps, 20, 80, execution_horizon_steps=14)
        return
    receipt = _scheduled_receipt(latency_steps, 20, 80, execution_horizon_steps=14)

    assert receipt["execution_horizon_steps"] == 14
    assert receipt["action_index_origin"] == "observation"
    expected_actions = _observation_relative_continuous_oracle(latency_steps, 80, latency_steps + 14)
    assert receipt["gap_steps"] == list(range(5)) + [
        step for step, action in enumerate(expected_actions, start=5) if action is None
    ]
    assert (receipt["steady_gaps"] > 0) == (latency_steps > 14)
    assert all(
        event["action_index"] == latency_steps
        for event in receipt["applied_events"]
        if any(
            release["step"] == event["step"]
            for release in receipt["release_events"]
        )
    )
    releases = {event["step"] for event in receipt["release_events"]}
    followup_requests = {
        event["step"]
        for event in receipt["request_events"]
        if event["request_id"] > 0
    }
    assert releases - {5 + latency_steps} <= followup_requests


def _scheduled_receipt(
    latency_steps: int,
    action_chunk_steps: int,
    n_steps: int,
    *,
    mode: str = "continuous",
    execution_horizon_steps: int = 4,
):
    model, _, metadata = _run_policy(
        latency_steps,
        n_steps,
        action_chunk_steps,
        mode=mode,
        execution_horizon_steps=execution_horizon_steps,
    )
    all_metadata = [
        {
            "request": None,
            "release": None,
            "chunk_index": None,
            "observation_step": None,
            "action_index": None,
            "wait_reason": "policy_warmup",
        }
        for _ in range(5)
    ] + metadata
    request_events = [
        {"step": step, **meta["request"]}
        for step, meta in enumerate(all_metadata)
        if meta["request"] is not None
    ]
    release_events = [
        {"step": step, **meta["release"]}
        for step, meta in enumerate(all_metadata)
        if meta["release"] is not None
    ]
    applied_events = [
        {
            "step": step,
            "chunk_index": meta["chunk_index"],
            "observation_step": meta["observation_step"],
            "action_index": meta["action_index"],
        }
        for step, meta in enumerate(all_metadata)
        if meta["wait_reason"] == "model"
    ]
    gap_steps = [
        step
        for step, meta in enumerate(all_metadata)
        if meta["wait_reason"] != "model"
    ]
    staleness = collections.Counter(
        event["step"] - event["observation_step"] for event in applied_events
    )
    startup_gap = 0
    for expected, gap in enumerate(gap_steps):
        if expected != gap:
            break
        startup_gap += 1
    control_events = [
        {
            "step": step,
            "action": [[float(step)] * 8],
            "wait_reason": meta["wait_reason"],
            "chunk_index": meta["chunk_index"],
            "observation_step": meta["observation_step"],
            "action_index": meta["action_index"],
        }
        for step, meta in enumerate(all_metadata)
    ]
    return {
        "n_steps": n_steps,
        "episode_seed": model._vlalagbench_episode_seed,
        "mode": mode,
        "action_index_origin": "release" if mode == "synchronous" else "observation",
        "execution_horizon_basis": "post_latency_action_count",
        "status": 1,
        "success": False,
        "execution_horizon_steps": (
            model._vlalagbench_timing.execution_horizon_steps
        ),
        "applied": len(applied_events),
        "coverage": len(applied_events) / n_steps,
        "startup_gap": startup_gap,
        "steady_gaps": len(gap_steps) - startup_gap,
        "gap_steps": gap_steps,
        "requests": len(request_events),
        "request_events": request_events,
        "releases": len(release_events),
        "release_events": release_events,
        "applied_events": applied_events,
        "control_events": control_events,
        "staleness_hist": dict(staleness),
        "timeouts": 0,
        "violations": 0,
    }


def _trace_from_receipt(receipt):
    requests = {
        event["step"]: {key: value for key, value in event.items() if key != "step"}
        for event in receipt["request_events"]
    }
    releases = {
        event["step"]: {key: value for key, value in event.items() if key != "step"}
        for event in receipt["release_events"]
    }
    protocol = [
        {
            "request": requests.get(event["step"]),
            "release": releases.get(event["step"]),
            "reply_to": event["step"],
            "wait_reason": event["wait_reason"],
            "chunk_index": event["chunk_index"],
            "observation_step": event["observation_step"],
            "action_index": event["action_index"],
            "blocked_s": 0.0,
        }
        for event in receipt["control_events"]
    ]
    return SimpleNamespace(
        meta=SimpleNamespace(
            policy_episode_seed=receipt["episode_seed"],
            inference_mode=receipt["mode"],
            action_index_origin=receipt["action_index_origin"],
            execution_horizon_steps=receipt["execution_horizon_steps"],
            execution_horizon_basis=receipt["execution_horizon_basis"],
            latency_steps=2,
            action_chunk_steps=6,
            status=receipt["status"],
            success=receipt["success"],
        ),
        num_steps=receipt["n_steps"],
        protocol=protocol,
        actions=torch.tensor(
            [event["action"] for event in receipt["control_events"]]
        ),
    )


def test_receipt_checks_request_release_action_and_wait_events():
    receipt = _scheduled_receipt(2, 6, 14)

    assert receipt_errors(
        receipt,
        mode="continuous",
        latency_steps=2,
        action_chunk_steps=6,
        execution_horizon_steps=receipt["execution_horizon_steps"],
    ) == []
    receipt["applied_events"][0]["action_index"] = 3
    assert receipt_errors(
        receipt,
        mode="continuous",
        latency_steps=2,
        action_chunk_steps=6,
        execution_horizon_steps=4,
    )


def test_status_two_is_an_exact_failure_before_any_policy_action():
    receipt = _scheduled_receipt(2, 6, 5)
    receipt["status"] = 2

    assert receipt["applied"] == 0
    assert receipt["releases"] == 0
    assert receipt_errors(
        receipt,
        mode="continuous",
        latency_steps=2,
        action_chunk_steps=6,
        execution_horizon_steps=receipt["execution_horizon_steps"],
    ) == []

    receipt = _scheduled_receipt(2, 6, 14)
    receipt["status"] = 2
    assert "status 2 episode applied policy actions" in receipt_errors(
        receipt,
        mode="continuous",
        latency_steps=2,
        action_chunk_steps=6,
        execution_horizon_steps=4,
    )


def test_receipt_digest_is_stable_across_json_round_trip():
    receipt = {"staleness_hist": {2: 4, 10: 3}}

    assert receipt_sha256(receipt) == receipt_sha256(
        json.loads(json.dumps(receipt))
    )


def test_continuation_receipt_rebuild_uses_the_independent_schedule_oracle():
    source = _scheduled_receipt(
        2,
        6,
        16,
        mode="synchronous",
        execution_horizon_steps=3,
    )
    trace = _trace_from_receipt(source)

    rebuilt = _audit_continuation(trace)

    assert rebuilt["exact"] is True
    assert rebuilt["errors"] == []
    trace.protocol[7]["chunk_index"] = 99
    with pytest.raises(RuntimeError, match="applied-action provenance"):
        _audit_continuation(trace)
    trace.protocol[7]["chunk_index"] = 1
    trace.protocol[7]["reply_to"] = 6
    with pytest.raises(RuntimeError, match="lockstep schedule was violated"):
        _audit_continuation(trace)

    trace = _trace_from_receipt(source)
    trace.protocol[6]["request"] = {**trace.protocol[5]["request"], "step": 5}
    trace.protocol[5]["request"] = None
    with pytest.raises(RuntimeError, match="request events"):
        _audit_continuation(trace)


def test_receipt_rejects_timeout_and_missing_control_action():
    receipt = _scheduled_receipt(0, 6, 10)
    receipt["timeouts"] = 1
    receipt["control_events"].pop()

    errors = receipt_errors(
        receipt,
        mode="continuous",
        latency_steps=0,
        action_chunk_steps=6,
        execution_horizon_steps=4,
    )

    assert "policy reply timed out" in errors
    assert "control events do not account for every step" in errors


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_steps", None),
        ("n_steps", float("inf")),
        ("request_events", None),
        ("applied_events", [1]),
        ("staleness_hist", None),
        ("coverage", "invalid"),
    ],
)
def test_receipt_returns_a_falsified_invariant_for_malformed_data(field, value):
    receipt = _scheduled_receipt(2, 6, 14)
    receipt[field] = value

    assert receipt_errors(
        receipt,
        mode="continuous",
        latency_steps=2,
        action_chunk_steps=6,
        execution_horizon_steps=4,
    ) == ["receipt structure is invalid"]


def test_receipt_returns_a_falsified_invariant_for_non_mapping_data():
    assert receipt_errors(
        None,
        mode="continuous",
        latency_steps=2,
        action_chunk_steps=6,
        execution_horizon_steps=4,
    ) == ["receipt structure is invalid"]


@pytest.mark.parametrize("horizon", [1, 6])
def test_synchronous_receipt_accepts_horizon_boundaries(horizon):
    receipt = _scheduled_receipt(
        2,
        6,
        20,
        mode="synchronous",
        execution_horizon_steps=horizon,
    )

    assert receipt_errors(
        receipt,
        mode="synchronous",
        latency_steps=2,
        action_chunk_steps=6,
        execution_horizon_steps=horizon,
    ) == []


def test_synchronous_staleness_is_observation_age_not_chunk_position():
    receipt = _scheduled_receipt(
        2,
        6,
        16,
        mode="synchronous",
        execution_horizon_steps=3,
    )

    assert receipt["applied_events"][0]["action_index"] == 0
    assert receipt["staleness_hist"] == {2: 2, 3: 2, 4: 2}
    assert receipt_errors(
        receipt,
        mode="synchronous",
        latency_steps=2,
        action_chunk_steps=6,
        execution_horizon_steps=3,
    ) == []


def test_synchronous_zero_latency_executes_from_chunk_zero_until_horizon():
    _, actions, metadata = _run_policy(
        0,
        13,
        mode="synchronous",
        execution_horizon_steps=3,
    )

    assert actions == [
        "5:0",
        "5:1",
        "5:2",
        "8:0",
        "8:1",
        "8:2",
        "11:0",
        "11:1",
    ]
    assert [
        meta["request"]["observation_step"]
        for meta in metadata
        if meta["request"] is not None
    ] == [5, 8, 11]


def test_synchronous_positive_latency_holds_between_horizons():
    _, actions, metadata = _run_policy(
        2,
        16,
        mode="synchronous",
        execution_horizon_steps=3,
    )

    assert actions == [
        None,
        None,
        "5:0",
        "5:1",
        "5:2",
        None,
        None,
        "10:0",
        "10:1",
        "10:2",
        None,
    ]
    assert [meta["wait_reason"] for meta in metadata] == [
        "inference_wait",
        "inference_wait",
        "model",
        "model",
        "model",
        "inference_wait",
        "inference_wait",
        "model",
        "model",
        "model",
        "inference_wait",
    ]


def test_lazy_skips_latency_including_the_first_chunk():
    _, actions, metadata = _run_policy(
        2,
        18,
        action_chunk_steps=10,
        mode="lazy",
        execution_horizon_steps=5,
    )

    assert actions == [None, None, "5:2", "5:3", "5:4", "5:5", "5:6",
                       "10:2", "10:3", "10:4", "10:5", "10:6", "15:2"]
    assert [meta["request"]["observation_step"] for meta in metadata
            if meta["request"] is not None] == [5, 10, 15]


def test_lazy_zero_latency_matches_synchronous_request_and_action_sequences():
    _, sync_actions, sync_metadata = _run_policy(
        0,
        18,
        action_chunk_steps=10,
        mode="synchronous",
        execution_horizon_steps=5,
    )
    _, lazy_actions, lazy_metadata = _run_policy(
        0,
        18,
        action_chunk_steps=10,
        mode="lazy",
        execution_horizon_steps=5,
    )

    assert lazy_actions == sync_actions
    assert [meta["request"] for meta in lazy_metadata] == [
        meta["request"] for meta in sync_metadata
    ]
    assert [meta["release"] for meta in lazy_metadata] == [
        meta["release"] for meta in sync_metadata
    ]


@pytest.mark.parametrize("mode", ["synchronous", "lazy", "continuous"])
@pytest.mark.parametrize("latency", [0, 3, 6, 9, 12])
def test_generated_tail_does_not_change_the_fixed_execution_horizon(mode, latency):
    sequences = []
    for chunk_steps in (14, 20, 32):
        if mode != "synchronous" and latency + 14 > chunk_steps:
            with pytest.raises(ValueError, match="latency plus execution horizon"):
                _run_policy(latency, 100, chunk_steps, mode=mode, execution_horizon_steps=14)
            continue
        _, actions, metadata = _run_policy(
            latency, 100, chunk_steps, mode=mode, execution_horizon_steps=14,
        )
        sequences.append((actions, [meta["request"] for meta in metadata]))
    # Noise checksums depend on generated tensor shape, so compare request
    # timing and identity independently of their random-noise payloads.
    reference_actions, reference_requests = sequences[0]
    reference_steps = [None if r is None else r["observation_step"] for r in reference_requests]
    for actions, requests in sequences[1:]:
        assert actions == reference_actions
        assert [None if r is None else r["observation_step"] for r in requests] == reference_steps


def test_lazy_rejects_latency_plus_horizon_exceeding_the_chunk():
    with pytest.raises(ValueError, match="latency plus execution horizon"):
        _scheduled_receipt(4, 6, 20, mode="lazy", execution_horizon_steps=4)


def test_lazy_receipt_matches_independent_oracle_across_schedule_grid():
    for chunk_steps in (2, 3, 6, 20):
        for latency_steps in range(chunk_steps):
            for horizon_steps in range(1, chunk_steps - latency_steps + 1):
                n_steps = 6 + 3 * max(latency_steps, horizon_steps)
                receipt = _scheduled_receipt(
                    latency_steps,
                    chunk_steps,
                    n_steps,
                    mode="lazy",
                    execution_horizon_steps=horizon_steps,
                )

                errors = receipt_errors(
                    receipt,
                    mode="lazy",
                    latency_steps=latency_steps,
                    action_chunk_steps=chunk_steps,
                    execution_horizon_steps=horizon_steps,
                )
                assert errors == [], (
                    latency_steps,
                    horizon_steps,
                    chunk_steps,
                    errors,
                )


def test_synchronous_episode_end_finalizes_pending_request():
    model = _model(2, mode="synchronous", execution_horizon_steps=4)
    _streaming_action(model, {"index": 5})

    snapshot = _final_policy_snapshot(model, SimpleNamespace())

    assert snapshot["scheduler"]["pending"] == {
        "request": {"id": 0, "sent_step": 5, "available_step": 7},
        "chunk": [f"5:{index}" for index in range(6)],
        "executed_steps": 0,
    }


@pytest.mark.parametrize("mode", ["synchronous", "lazy"])
def test_horizon_mode_failed_receipt_reports_timeout(mode):
    receipt = _scheduled_receipt(
        2,
        6,
        15,
        mode=mode,
        execution_horizon_steps=4,
    )
    receipt["timeouts"] = 1

    errors = receipt_errors(
        receipt,
        mode=mode,
        latency_steps=2,
        action_chunk_steps=6,
        execution_horizon_steps=4,
    )

    assert "policy reply timed out" in errors


@pytest.mark.parametrize("mode", ["synchronous", "lazy"])
def test_horizon_mode_policy_rejects_wrong_chunk_length(mode):
    model = _model(0, mode=mode, execution_horizon_steps=3)
    model.q_in = SimpleNamespace(
        update=lambda _message: model.q_out.put(
            {"index": 5, "actions": ["short"]}
        )
    )

    with pytest.raises(ValueError, match="expected 6"):
        _streaming_action(model, {"index": 5})


def test_policy_acknowledges_a_task_before_processing_control_observations():
    observations = iter(
        [
            {"episode_identity": IDENTITY},
            {"task": "move the object"},
            {"success_rates": {}},
        ]
    )
    sent = []
    model = _model(2)
    upstream = SimpleNamespace(
        get_latest_observation=lambda _socket: next(observations),
        _handle_test_case_finished=lambda *_args: None,
        _get_action=lambda *_args: None,
        _get_observations=lambda values, _timestamps: values,
        np=SimpleNamespace(round=round),
        torch=torch,
    )

    _run_tests(
        upstream,
        2,
        object(),
        SimpleNamespace(send_pyobj=sent.append),
        {"delta_timestamps": [-2, 0]},
        model,
        "weights",
        0,
        "euler",
        True,
        None,
    )

    assert sent == [
        {
            "identity_ack": True,
            "episode_seed": derive_episode_seed(
                IDENTITY["policy_seed"],
                IDENTITY["task_name"],
                IDENTITY["environment_sha256"],
                IDENTITY["replica_id"],
            ),
        },
        {
            "task_ack": True,
            "instruction": "move the object",
            "policy_noise_scheme": POLICY_NOISE_SCHEME,
            "policy_torch_version": torch.__version__,
            "episode_seed": derive_episode_seed(
                IDENTITY["policy_seed"],
                IDENTITY["task_name"],
                IDENTITY["environment_sha256"],
                IDENTITY["replica_id"],
            ),
            "inference_mode": "continuous",
            "action_index_origin": "observation",
            "execution_horizon_steps": 4,
            "execution_horizon_basis": "post_latency_action_count",
            "action_chunk_steps": 6,
        },
        {"suite_ack": True},
    ]


def test_policy_loop_preserves_pinned_history_and_records_first_request():
    observations = iter(
        [
            {"episode_identity": IDENTITY},
            {"task": "move the object"},
            *[{"index": index, "seed": 7, "latency_steps": 2} for index in range(6)],
            {"success_rates": {}},
        ]
    )
    sent = []
    model = _model(2)

    def get_action(policy, selected, *_args):
        latest = selected[-1]
        if any("index" not in observation for observation in selected):
            return None
        count = getattr(get_action, "count", -3)
        get_action.count = count + 1
        if count < 0:
            return None
        return _streaming_action(policy, {"index": latest["index"]})

    upstream = SimpleNamespace(
        get_latest_observation=lambda _socket: next(observations),
        _handle_test_case_finished=lambda *_args: None,
        _get_action=get_action,
        _get_observations=lambda values, timestamps: [
            values[max(0, len(values) - 1 + timestamp)] for timestamp in timestamps
        ],
        np=SimpleNamespace(round=round),
        torch=torch,
    )

    _run_tests(
        upstream,
        2,
        object(),
        SimpleNamespace(send_pyobj=sent.append),
        {"delta_timestamps": [-2, 0]},
        model,
        "weights",
        0,
        "euler",
        True,
        None,
    )

    replies = [message for message in sent if "meta" in message]
    assert [reply["meta"]["wait_reason"] for reply in replies] == [
        "policy_warmup"
    ] * 5 + ["inference_wait"]
    request = replies[-1]["meta"]["request"]
    assert {
        key: request[key]
        for key in ("request_id", "observation_step", "release_step")
    } == {"request_id": 0, "observation_step": 5, "release_step": 7}
    assert request["request_seed"] == derive_request_seed(
        derive_episode_seed(
            IDENTITY["policy_seed"],
            IDENTITY["task_name"],
            IDENTITY["environment_sha256"],
            IDENTITY["replica_id"],
        ),
        0,
    )
    assert len(request["noise_sha256"]) == 64


def test_policy_snapshot_restores_shared_scheduler_and_delta_base():
    model, _, _ = _run_policy(2, 9)

    def get_action():
        pass

    get_action.count = 5
    get_action.state = [[0.1, 0.2]]
    snapshot = _final_policy_snapshot(model, get_action)
    scheduler_state = snapshot["scheduler"]
    for item in scheduler_state["queue"]:
        source, index = item["action"].split(":")
        item["action"] = [float(source), float(index)]
    pending = scheduler_state["pending"]
    if pending is not None and pending["chunk"] is not None:
        pending["chunk"] = [
            [float(value) for value in action.split(":")]
            for action in pending["chunk"]
        ]
    restored = _model(2)
    restored._vlalagbench_numpy = SimpleNamespace(asarray=lambda value: value)

    _restore_policy_snapshot(restored, get_action, snapshot)

    assert restored._vlalagbench_scheduler.pending_request.id == pending["request"]["id"]
    assert restored._vlalagbench_scheduler.snapshot()["next_step"] == scheduler_state[
        "next_step"
    ]
    assert restored._vlalagbench_episode_seed == 17
    assert get_action.count == 5
    assert get_action.state == [[0.1, 0.2]]


@pytest.mark.parametrize(
    ("mode", "horizon"),
    [("continuous", 4), ("synchronous", 4), ("lazy", 4)],
)
@pytest.mark.parametrize("cut_step", [5, 6])
def test_resumed_release_metadata_matches_uninterrupted_execution(
    mode, horizon, cut_step
):
    uninterrupted = _model(
        2,
        numeric=True,
        mode=mode,
        execution_horizon_steps=horizon,
    )
    source = _model(
        2,
        numeric=True,
        mode=mode,
        execution_horizon_steps=horizon,
    )
    for step in range(5, cut_step + 1):
        _streaming_action(uninterrupted, {"index": step})
        _streaming_action(source, {"index": step})

    def get_action():
        pass

    snapshot = _final_policy_snapshot(source, get_action)
    resumed = _model(
        2,
        numeric=True,
        mode=mode,
        execution_horizon_steps=horizon,
    )
    resumed._vlalagbench_numpy = SimpleNamespace(asarray=lambda value: value)
    _restore_policy_snapshot(resumed, get_action, snapshot)

    for step in range(cut_step + 1, 10):
        expected_action = _streaming_action(uninterrupted, {"index": step})
        resumed_action = _streaming_action(resumed, {"index": step})
        if expected_action is None:
            assert resumed_action is None
        else:
            torch.testing.assert_close(resumed_action, expected_action)
        fields = (
            "request",
            "release",
            "chunk_index",
            "observation_step",
            "action_index",
            "wait_reason",
        )
        assert {
            field: resumed.last_action_meta[field] for field in fields
        } == {
            field: uninterrupted.last_action_meta[field] for field in fields
        }
        expected_chunk = uninterrupted.last_action_meta["released_chunk"]
        resumed_chunk = resumed.last_action_meta["released_chunk"]
        if expected_chunk is None:
            assert resumed_chunk is None
        else:
            assert resumed_chunk["index"] == expected_chunk["index"]
            assert len(resumed_chunk["actions"]) == len(
                expected_chunk["actions"]
            )
            for resumed_value, expected_value in zip(
                resumed_chunk["actions"],
                expected_chunk["actions"],
                strict=True,
            ):
                torch.testing.assert_close(
                    torch.as_tensor(resumed_value),
                    torch.as_tensor(expected_value),
                )


def test_policy_noise_is_stable_across_process_restarts_and_request_order():
    model = SimpleNamespace(
        _vlalagbench_episode_seed=derive_episode_seed(7, "task", "b" * 64, 4),
        config=SimpleNamespace(chunk_size=4, max_action_dim=3),
    )

    first = _request_noise(model, 12)
    _request_noise(model, 99)
    restarted = SimpleNamespace(
        _vlalagbench_episode_seed=derive_episode_seed(7, "task", "b" * 64, 4),
        config=model.config,
    )

    torch.testing.assert_close(_request_noise(restarted, 12), first, rtol=0, atol=0)
    assert not torch.equal(_request_noise(restarted, 13), first)
    assert derive_request_seed(restarted._vlalagbench_episode_seed, 12) == derive_request_seed(
        model._vlalagbench_episode_seed, 12
    )


def test_twenty_replicas_have_distinct_paired_diffusion_streams():
    episode_seeds = [
        derive_episode_seed(7, "task", "b" * 64, replica_id)
        for replica_id in range(20)
    ]

    assert len(set(episode_seeds)) == 20
    assert episode_seeds == [
        derive_episode_seed(7, "task", "b" * 64, replica_id)
        for replica_id in range(20)
    ]
    assert derive_request_seed(episode_seeds[3], 9) == derive_request_seed(
        derive_episode_seed(7, "task", "b" * 64, 3), 9
    )
    assert episode_seeds[3] != derive_episode_seed(7, "task", "c" * 64, 3)
    checksums = {
        _noise_checksum(
            _request_noise(
                SimpleNamespace(
                    _vlalagbench_episode_seed=episode_seed,
                    config=SimpleNamespace(chunk_size=4, max_action_dim=3),
                ),
                0,
            )
        )
        for episode_seed in episode_seeds
    }
    assert len(checksums) == 20


def test_request_noise_is_paired_across_modes_and_latency_setups():
    models = [
        _model(0),
        _model(2),
        _model(2, mode="synchronous", execution_horizon_steps=3),
        _model(2, mode="lazy", execution_horizon_steps=4),
    ]

    for model in models:
        _streaming_action(model, {"index": 5})

    requests = [model.last_action_meta["request"] for model in models]
    assert len({request["request_seed"] for request in requests}) == 1
    assert len({request["noise_sha256"] for request in requests}) == 1


@pytest.mark.parametrize("mode", ["lazy", "continuous"])
def test_async_receipt_rejects_replaying_the_stale_prefix(mode):
    receipt = _scheduled_receipt(
        2, 10, 30, mode=mode,
        execution_horizon_steps=5,
    )
    for event in receipt["applied_events"]:
        assert event["action_index"] == event["step"] - event["observation_step"]
        event["action_index"] -= 2
    assert "applied-action provenance does not match the release schedule" in receipt_errors(
        receipt, mode=mode, latency_steps=2, action_chunk_steps=10,
        execution_horizon_steps=receipt["execution_horizon_steps"],
    )


@pytest.mark.parametrize("mode", ["lazy", "continuous"])
@pytest.mark.parametrize("latency", [6, 7, 9, 12])
def test_policy_enforces_async_chunk_capacity_without_changing_model(mode, latency):
    model = SimpleNamespace(config=SimpleNamespace(n_action_steps=20))
    if latency + 14 > 20:
        with pytest.raises(ValueError, match="latency plus execution horizon"):
            _install_scheduler(model, mode, 14, latency, 1.0)
        assert not hasattr(model, "_vlalagbench_scheduler")
    else:
        _install_scheduler(model, mode, 14, latency, 1.0)
        assert model._vlalagbench_timing.promotion_offset == 6
        assert model._vlalagbench_timing.execution_horizon_steps == 14
    assert model.config.n_action_steps == 20


@pytest.mark.parametrize("mode", ["lazy", "continuous"])
def test_new_receipts_reject_insufficient_chunk_capacity(mode):
    receipt = _scheduled_receipt(6, 20, 80, mode=mode, execution_horizon_steps=14)
    receipt["execution_horizon_steps"] = 15
    errors = receipt_errors(receipt, mode=mode, latency_steps=6,
                            action_chunk_steps=20, execution_horizon_steps=15)
    assert errors
    assert any("horizon" in error for error in errors)


def test_lazy_horizon_sets_request_interval_after_skipping_latency():
    _, _, metadata = _run_policy(6, 80, 20, mode="lazy", execution_horizon_steps=14)
    assert [m["request"]["observation_step"] for m in metadata if m["request"]] == [5, 19, 33, 47, 61, 75]
    assert [m["action_index"] for m in metadata[6:20]] == list(range(6, 20))
