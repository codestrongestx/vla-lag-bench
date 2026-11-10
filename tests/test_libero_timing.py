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


def test_lazy_requests_early_and_executes_five_actions_after_stale_prefix():
    requests, actions = _run(_scheduler("lazy", latency_steps=2), steps=13)

    assert requests == [0, 5, 10]
    assert actions == [None, None, "0:0", "0:1", "0:2", "0:3", "0:4", "1:2", "1:3", "1:4", "1:5", "1:6", "2:2"]


def test_async_cold_start_skips_nothing_because_nothing_was_executed():
    # The first chunk of a lazy episode arrives after a frozen wait, so no
    # prefix was overtaken; later chunks skip exactly the latency prefix.
    requests, actions = _run(_scheduler("lazy", latency_steps=4), steps=15)

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


def test_continuous_replaces_the_old_plan_when_each_request_arrives():
    requests, actions = _run(_scheduler("continuous", latency_steps=2), steps=9)

    assert requests == [0, 2, 4, 6, 8]
    assert actions == [None, None, "0:0", "0:1", "1:2", "1:3", "2:2", "2:3", "3:2"]


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


def test_async_horizon_and_stale_prefix_must_fit_chunk():
    with pytest.raises(ValueError, match="latency plus execution horizon"):
        _scheduler("lazy", latency_steps=6)


def test_scheduler_rejects_the_wrong_chunk_length():
    scheduler = _scheduler("synchronous", latency_steps=0)
    request = scheduler.request(0)
    assert request is not None
    with pytest.raises(ValueError, match="expected 10"):
        scheduler.deliver(request, ["short"])


def _scheduler(scheme, latency_steps):
    return ChunkScheduler(
        TimingConfig(
            scheme=scheme,
            action_chunk_steps=10,
            execution_horizon_steps=5,
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
            scheduler.deliver(request, [f"{request.id}:{index}" for index in range(10)])
        actions.append(scheduler.action(step).action)
    return requests, actions
