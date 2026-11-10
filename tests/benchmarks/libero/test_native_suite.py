from pathlib import Path
import hashlib
import json

import pytest

from vlalagbench.authoring import paired_trajectory_schedule
from vlalagbench.config import load_experiment
from vlalagbench.trajectory import sample_trajectory
from vlalagbench.files import sha256


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs/experiments/libero"
BASKET = CONFIGS / "libero_object_task1_native_basket_ellipse"
AUDITED_REVISION = "7cb43108678f849c437f19648ca16d2e1607400d"


@pytest.mark.parametrize("directory,filename,samples,revision", [
    (BASKET, "basket_ellipse_paired_200.yaml", 114400, AUDITED_REVISION),
])
def test_native_release_preserves_its_historical_audit(directory, filename, samples, revision):
    manifest = json.loads((directory / "release_manifest.json").read_text())
    assert manifest["passed"]
    assert manifest["config_sha256"] == sha256(directory / filename)
    # These receipts describe the original release, not subsequent physics changes.
    for key, path in [
        ("audit_source_sha256", "suite.py.txt"),
        ("dynamics_source_sha256", "dynamics.py.txt"),
    ]:
        source = (ROOT / "tests/fixtures/native_audits" / revision / path).read_bytes()
        assert manifest[key] == hashlib.sha256(source).hexdigest()
    assert manifest["failures"] == manifest["first_forbidden_contacts"] == []
    assert manifest["maximum_sample_travel_m"] <= 0.001
    assert manifest["mujoco_forward_samples"] == samples
    assert manifest["maximum_support_penetration_m"] < 0.001
    assert len(set(manifest["official_state_sha256"])) == 50
    assert len({row["pose_sha256"] for row in manifest["episodes"]}) == 100
    assert len({row["condition_sha256"] for row in manifest["episodes"]}) == 200


def test_native_basket_preserves_its_only_override_and_fixed_world_ellipse():
    experiment = load_experiment(BASKET / "basket_ellipse_paired_200.yaml")
    reference = [-0.254, 0.326998, 0.035]
    assert len(experiment.initial_state_overrides) == 50
    for index, override in enumerate(experiment.initial_state_overrides):
        assert override.task_id == 1
        assert override.initial_state_index == index
        assert override.object_positions_m == {"basket_1_main": reference}
        assert override.object_yaws_rad == {}
    trajectory = experiment.trajectory
    assert trajectory.release_on_grasp_steps is None
    world = [
        [reference[axis] + frame.position_offset_m[axis] for axis in (0, 1)]
        for frame in trajectory.keyframes
    ]
    assert world[0] == world[-1]
    for axis, center, radius in [(0, -0.144, 0.11), (1, 0.327, 0.069845)]:
        assert min(point[axis] for point in world) == pytest.approx(center - radius)
        assert max(point[axis] for point in world) == pytest.approx(center + radius)


@pytest.mark.parametrize("path,family", [
    (BASKET / "basket_ellipse_paired_200.yaml", "ellipse"),
])
def test_released_schedules_keep_exact_phases_order_and_direction_pairing(path, family):
    experiment = load_experiment(path)
    assert experiment.benchmark.initial_state_indices == list(range(50))
    assert experiment.benchmark.episodes_per_task == 200
    schedule = [row.model_dump() for row in experiment.trajectory.episode_schedule]
    assert paired_trajectory_schedule(family) == schedule
    phases = {row["initial_phase"] for row in schedule}
    assert len(phases) == 100
    assert {int(phase * 100) for phase in phases} == set(range(100))
    for layout in range(50):
        pairs = {}
        for row in schedule[layout::50]:
            pairs.setdefault(row["initial_phase"], set()).add(row["initial_direction"])
        assert len(pairs) == 2
        assert all(directions == {-1, 1} for directions in pairs.values())
        left, right = pairs
        gap = abs(left - right)
        assert min(gap, 1 - gap) >= 0.45
