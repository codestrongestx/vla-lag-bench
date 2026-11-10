import pytest

from vlalagbench.statistics import wilson_interval


def test_wilson_interval_for_observed_post_support_plane_rate():
    low, high = wilson_interval(194, 200)

    assert low == pytest.approx(0.9361057075)
    assert high == pytest.approx(0.9861796857)


def test_wilson_interval_handles_all_successes_without_exceeding_one():
    low, high = wilson_interval(200, 200)

    assert low == pytest.approx(0.9811539941)
    assert high == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("successes", "episodes"),
    [(-1, 200), (201, 200), (0, 0)],
)
def test_wilson_interval_rejects_invalid_counts(successes: int, episodes: int):
    with pytest.raises(ValueError):
        wilson_interval(successes, episodes)
