import pytest

from vlalagbench.timing import ChunkScheduler, TimingConfig, effective_steps


def test_effective_steps_uses_the_next_control_boundary():
    assert effective_steps(0, 20) == 0
    assert effective_steps(50, 20) == 1
    assert effective_steps(51, 20) == 2


def test_synchronous_waits_then_executes_five_actions_from_zero():
    requests, actions = _run(_scheduler("synchronous", latency_steps=2), steps=10)

    assert requests == [0, 7]
    assert actions == [None, None, "0:0", "0:1", "0:2", "0:3", "0:4", None, None, "1:0"]


def test_lazy_requests_early_and_executes_full_horizons_from_zero():
    requests, actions = _run(_scheduler("lazy", latency_steps=2), steps=13)

    assert requests == [0, 5, 10]
    assert actions == [None, None, "0:0", "0:1", "0:2", "0:3", "0:4", "1:0", "1:1", "1:2", "1:3", "1:4", "2:0"]


def test_continuous_replaces_the_old_plan_when_each_request_arrives():
    requests, actions = _run(_scheduler("continuous", latency_steps=2), steps=9)

    assert requests == [0, 2, 4, 6, 8]
    assert actions == [None, None, "0:0", "0:1", "1:0", "1:1", "2:0", "2:1", "3:0"]


def test_default_rule_skips_the_prefix_executed_while_in_flight():
    # Chunk 0 arrives after a frozen wait, so nothing was overtaken; later
    # chunks skip exactly the latency prefix executed meanwhile.
    requests, actions = _run(_scheduler("lazy", latency_steps=2, offset=None), steps=13)

    assert requests == [0, 5, 10]
    assert actions == [None, None, "0:0", "0:1", "0:2", "0:3", "0:4", "1:2", "1:3", "1:4", "1:5", "1:6", "2:2"]

    continuous = ChunkScheduler(
        TimingConfig(
            scheme="continuous",
            action_chunk_steps=10,
            execution_horizon_steps=8,
            latency_steps=2,
        )
    )
    requests, actions = _run(continuous, steps=9)

    assert requests == [0, 2, 4, 6, 8]
    assert actions == [None, None, "0:0", "0:1", "1:2", "1:3", "2:2", "2:3", "3:2"]


def test_async_cold_start_skips_nothing_because_nothing_was_executed():
    requests, actions = _run(_scheduler("lazy", latency_steps=4, offset=None), steps=15)

    assert requests == [0, 5, 10]
    assert actions[:9] == [None] * 4 + ["0:0", "0:1", "0:2", "0:3", "0:4"]
    assert actions[9:14] == ["1:4", "1:5", "1:6", "1:7", "1:8"]


def test_explicit_promotion_offset_is_applied_verbatim_even_at_cold_start():
    scheduler = ChunkScheduler(
        TimingConfig(
            scheme="lazy",
            action_chunk_steps=10,
            execution_horizon_steps=5,
            latency_steps=0,
            promotion_offset_steps=4,
        )
    )
    _, actions = _run(scheduler, steps=6)

    assert actions == ["0:4", "0:5", "0:6", "0:7", "0:8", "1:4"]


def test_snapshot_round_trip_keeps_the_in_flight_executed_count():
    config = TimingConfig(
        scheme="lazy", action_chunk_steps=10, execution_horizon_steps=5, latency_steps=2
    )
    scheduler = ChunkScheduler(config)
    _run(scheduler, steps=7)
    assert scheduler.pending_request is not None
    snapshot = scheduler.snapshot()
    assert snapshot["pending"]["executed_steps"] == 2

    restored = ChunkScheduler.from_snapshot(config, snapshot, lambda action: action)
    assert restored.snapshot() == snapshot
    restored.request(7)
    assert restored.action(7).action == "1:2"

    legacy = {**snapshot, "pending": {k: v for k, v in snapshot["pending"].items() if k != "executed_steps"}}
    assert ChunkScheduler.from_snapshot(config, legacy, lambda action: action).snapshot()["pending"]["executed_steps"] == 0


def test_zero_latency_sync_and_lazy_are_identical():
    sync = _run(_scheduler("synchronous", latency_steps=0), steps=12)
    lazy = _run(_scheduler("lazy", latency_steps=0), steps=12)
    assert sync == lazy


def test_only_model_actions_are_active():
    scheduler = _scheduler("synchronous", latency_steps=2)
    active = []
    for step in range(4):
        request = scheduler.request(step)
        if request is not None:
            scheduler.deliver(request, [f"{request.id}:{index}" for index in range(10)])
        active.append(scheduler.action(step).active)
    assert active == [False, False, True, True]


def test_lazy_latency_does_not_shorten_the_execution_horizon():
    requests, actions = _run(_scheduler("lazy", latency_steps=6), steps=13)

    assert requests == [0, 6, 12]
    assert actions == [None] * 6 + ["0:0", "0:1", "0:2", "0:3", "0:4", None, "1:0"]


def test_continuous_latency_does_not_shorten_the_released_action_window():
    requests, actions = _run(_scheduler("continuous", latency_steps=6), steps=13)

    assert requests == [0, 6, 12]
    assert actions == [None] * 6 + [f"0:{index}" for index in range(6)] + ["1:0"]


def test_dynamicvla_lazy_horizon_must_fit_after_latency_prefix():
    with pytest.raises(ValueError, match="promotion offset plus execution horizon"):
        TimingConfig(
            promotion_offset_steps=12,
            scheme="lazy",
            action_chunk_steps=20,
            execution_horizon_steps=14,
            latency_steps=12,
        )


def test_scheduler_rejects_the_wrong_chunk_length():
    scheduler = _scheduler("synchronous", latency_steps=0)
    request = scheduler.request(0)
    assert request is not None
    with pytest.raises(ValueError, match="expected 10"):
        scheduler.deliver(request, ["short"])


def _scheduler(scheme, latency_steps, offset=0):
    return ChunkScheduler(
        TimingConfig(
            promotion_offset_steps=offset,
            scheme=scheme,
            action_chunk_steps=10,
            execution_horizon_steps=10 if scheme == "continuous" else 5,
            latency_steps=latency_steps,
        )
    )


def _run(scheduler, steps):
    requests = []
    actions = []
    for step in range(steps):
        request = scheduler.request(step)
        if request is not None:
            requests.append(step)
            scheduler.deliver(
                request,
                [
                    f"{request.id}:{index}"
                    for index in range(scheduler.config.action_chunk_steps)
                ],
            )
        actions.append(scheduler.action(step).action)
    return requests, actions


@pytest.mark.parametrize("mode", ["lazy", "continuous"])
def test_explicit_zero_offset_cannot_bypass_dynamicvla_capacity_check(mode):
    with pytest.raises(ValueError, match="latency plus execution horizon"):
        TimingConfig(mode, 20, 14, 12, promotion_offset_steps=0,
                     execution_horizon_basis="post_latency_action_count")
    with pytest.raises(ValueError, match="offset must equal latency"):
        TimingConfig(mode, 20, 14, 6, promotion_offset_steps=0,
                     execution_horizon_basis="post_latency_action_count")
