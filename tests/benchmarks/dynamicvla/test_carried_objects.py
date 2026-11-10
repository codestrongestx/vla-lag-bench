from pathlib import Path
from types import SimpleNamespace
import json
import math

import pytest
import yaml

from vlalagbench.config import DynamicVLAExperiment, load_experiment
from vlalagbench.benchmarks.dynamicvla.adapter import _check_authored_targets
from vlalagbench.benchmarks.dynamicvla.carriers import install_carriers
from vlalagbench.benchmarks.dynamicvla.motion import EpisodeMotion, PlanarDriver, episode_actors

ROOT = Path(__file__).resolve().parents[3]
# The cradle bank the evaluated ribbed bank was derived from; kept only as a fixture.
BANK = ROOT / "tests/fixtures/configs/experiments/dynamicvla/fcan04_moving_can_20261006/experiment.yaml"
RIBBED = ROOT / "configs/experiments/dynamicvla/fcan04_moving_can_20261006/ribbed-experiment.yaml"


def test_moving_can_bank_drives_only_the_carrier():
    experiment = load_experiment(BANK)
    assert experiment.trajectory.target == "carrier"
    assert experiment.trajectory.carried_objects == ["object"]
    assert len(experiment.trajectory.episode_schedule) == 200
    starts = {(row.initial_phase, row.initial_direction) for row in experiment.trajectory.episode_schedule}
    assert len(starts) == 200


def test_dynamicvla_rejects_carrying_a_driven_target():
    document = yaml.safe_load(BANK.read_text())
    document["trajectory"]["carried_objects"] = ["carrier"]
    with pytest.raises(ValueError, match="carried objects"):
        DynamicVLAExperiment.model_validate(document)


def test_ribbed_bank_matches_the_cradle_bank_except_its_carrier():
    cradle = load_experiment(BANK)
    ribbed = load_experiment(RIBBED)
    assert ribbed.trajectory.episode_schedule == cradle.trajectory.episode_schedule
    assert ribbed.trajectory.keyframes == cradle.trajectory.keyframes
    assert [o.object_positions_m for o in ribbed.initial_state_overrides] == [
        o.object_positions_m for o in cradle.initial_state_overrides
    ]
    assert ribbed.benchmark.tasks[0].environments == ["derived/fcan04-moving-can-ribbed.json"]


@pytest.mark.parametrize("shared", [False, True])
def test_a_passenger_cannot_be_shifted_twice(shared):
    document = yaml.safe_load(BANK.read_text())
    if shared:
        actor = episode_actors(load_experiment(BANK), 0)[0].model_dump(mode="json")
        document["dynamics"] = [{**actor, "target": "container"}]
    else:
        document["trajectory"]["carried_objects"] = ["object", "object"]
    with pytest.raises(ValueError, match="exactly one carrier"):
        DynamicVLAExperiment.model_validate(document)


def test_preflight_checks_carried_object_names():
    document = yaml.safe_load(BANK.read_text())
    document["trajectory"]["carried_objects"] = ["missing"]
    environment = BANK.with_name("fcan04-moving-can.json")
    with pytest.raises(ValueError, match="no rigid assets named: missing"):
        _check_authored_targets(
            DynamicVLAExperiment.model_validate(document), environment,
            json.loads(environment.read_text()),
        )


def test_carrier_hook_only_disables_gravity_on_declared_carriers():
    def asset():
        return SimpleNamespace(spawn=SimpleNamespace(rigid_props=SimpleNamespace(disable_gravity=False)))

    cfg = SimpleNamespace(scene=SimpleNamespace(carrier=asset(), object=asset(), container=asset()))
    upstream = SimpleNamespace(_get_env_cfg=lambda source: cfg)
    install_carriers(upstream)
    hook = upstream._get_env_cfg
    install_carriers(upstream)
    assert upstream._get_env_cfg is hook
    assert not upstream._get_env_cfg({}).scene.carrier.spawn.rigid_props.disable_gravity
    upstream._get_env_cfg({"dynabench_carriers": ["carrier"]})
    assert cfg.scene.carrier.spawn.rigid_props.disable_gravity
    assert not cfg.scene.object.spawn.rigid_props.disable_gravity
    assert not cfg.scene.container.spawn.rigid_props.disable_gravity


def test_can_is_shifted_at_start_and_never_driven_afterward():
    torch = pytest.importorskip("torch")

    class Asset:
        def __init__(self):
            self.data = SimpleNamespace(root_state_w=torch.arange(13, dtype=torch.float64)[None])
            self.writes = []

        def write_root_state_to_sim(self, state):
            self.writes.append(state.clone())
            self.data.root_state_w = state

    can, carrier = Asset(), Asset()
    before = can.data.root_state_w.clone()
    env = SimpleNamespace(scene={"object": can, "carrier": carrier})
    driver = PlanarDriver(EpisodeMotion(
        control_hz=25.0, robot_position_w=[0.0, 0.0, 0.0], robot_yaw_rad=math.pi / 2,
        reset_positions_w={"carrier": [0.0, 1.0, 2.0]}, driven_friction=0.0,
        actors=episode_actors(load_experiment(BANK), 150),
    ))
    driver.apply(env, 0)
    driver.place_carried(env)
    shift = driver.command(0)["carrier"][0] - [0.0, 1.0]
    assert can.writes[0][0, :2].numpy() == pytest.approx(before[0, :2].numpy() + shift)
    assert torch.equal(can.writes[0][:, 2:], before[:, 2:])
    for step in range(1, 25):
        driver.apply(env, step)
    assert len(can.writes) == 1
