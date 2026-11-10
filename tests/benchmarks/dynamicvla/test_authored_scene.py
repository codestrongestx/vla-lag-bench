import copy
import json
import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import numpy as np

from vlalagbench.benchmarks.dynamicvla.adapter import _build_manifest, _check_authored_targets, select_manifest_episodes
from vlalagbench.benchmarks.dynamicvla.motion import (
    EpisodeMotion,
    PlanarDriver,
    actors_from_json,
    episode_actors,
)
from vlalagbench.benchmarks.dynamicvla.scene import (
    RobotFrame,
    apply_initial_state,
    euler_xyz_from_quat,
    official_state,
)
from vlalagbench.config import (
    DynamicVLAExperiment,
    DynamicVLAInitialStateOverride,
    load_experiment,
)
from vlalagbench.files import sha256
from vlalagbench.trajectory import offset_and_velocity

ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENT = "tests/1-2_place_franka_avocado06d_O03_01050247_9766.json"
FIXTURES = ROOT / "tests/fixtures/configs/experiments/dynamicvla"
AUTHORING = FIXTURES / "avocado06_bowl04_authoring/experiment.yaml"


def test_episode_subset_preserves_manifest_identity(tmp_path):
    experiment = load_experiment(AUTHORING)
    assets, _ = _assets_root(tmp_path)
    original = _build_manifest(experiment, assets, tmp_path / "states")
    selected = {(row["task_id"], row["episode_id"]) for row in original["episodes"][1::2]}
    subset = select_manifest_episodes(original, selected)
    assert subset["episodes"] == original["episodes"][1::2]
    assert subset["episodes_per_environment"] == original["episodes_per_environment"]
    assert subset["environments"] == original["environments"]
    assert sum(task["resolved_episode_count"] for task in subset["tasks"]) == len(selected)
    assert len(original["episodes"]) > len(subset["episodes"])
    for invalid in (set(), {(0, 100000)}):
        with pytest.raises(ValueError, match="episode selection"):
            select_manifest_episodes(original, invalid)
RIGID = "isaaclab.assets.rigid_object.rigid_object:RigidObject"


def _asset(pos, rot, lin_vel=(0.0, 0.0, 0.0)):
    return {
        "class_type": RIGID,
        "init_state": {"pos": list(pos), "rot": list(rot), "lin_vel": list(lin_vel), "ang_vel": [0.0, 0.0, 0.0]},
    }


# The official avocado06 -> bowl04 reset state (world frame).
OFFICIAL = {
    "scene": {
        "robot": {
            "class_type": "isaaclab.assets.articulation.articulation:Articulation",
            "init_state": {"pos": [-3.8647470097637173, -3.1541299709926323, 0.4317886531352997], "rot": [0, 0, 0, 1]},
        },
        "object": _asset(
            [-3.9990074283349433, -3.416171698518181, 0.4742886535823345],
            [0.28110755796325065, -0.2811075579632506, -0.6488285912750282, 0.6488285912750282],
            [-0.4312656167654592, 0.46005073613762115, 0.0],
        ),
        "house": {"init_state": {"pos": [0, 0, 0], "rot": [0.7071068, 0.7071068, 0, 0]}},
        "object01": _asset(
            [-4.1132115604221475, -3.2122173002604035, 0.45928865298628807],
            [0.7026384337760762, -0.702638433776076, -0.07936769733778866, 0.07936769733778867],
        ),
        "container": _asset(
            [-4.199687676668644, -3.0170562658060147, 0.5067886561155319],
            [0.41824698794538157, 0.0, 0.0, -0.9083333402857212],
        ),
    },
}


def _experiment(**updates):
    data = yaml.safe_load(AUTHORING.read_text())
    for key, value in updates.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return data


def test_official_state_is_expressed_in_the_robot_base_frame():
    state = official_state(OFFICIAL)
    assert state["object"]["position_m"] == pytest.approx([0.13426, 0.262042, 0.0425], abs=1e-6)
    assert state["container"]["position_m"] == pytest.approx([0.334941, -0.137074, 0.075], abs=1e-6)
    assert state["object"]["velocity_mps"] == pytest.approx([0.431266, -0.460051, 0.0], abs=1e-6)
    assert math.degrees(state["object"]["roll_rad"]) == pytest.approx(-90.0)
    assert state["object01"]["velocity_mps"] == pytest.approx([0.0, 0.0, 0.0])


def test_robot_frame_round_trips_world_positions():
    frame = RobotFrame(OFFICIAL)
    point = [-4.2, -3.0, 0.5]
    assert frame.to_world(frame.to_robot(point)) == pytest.approx(point, abs=1e-12)


def test_unchanged_state_reproduces_the_official_json():
    assert apply_initial_state(OFFICIAL, None, start_at_rest=False) == OFFICIAL


def test_authored_state_moves_rotates_and_rests_objects():
    override = DynamicVLAInitialStateOverride(
        environment=ENVIRONMENT,
        initial_state_index=1,
        object_positions_m={"container": [0.4, 0.1, 0.075]},
        object_yaws_rad={"object": 0.25},
    )
    result = apply_initial_state(OFFICIAL, override, start_at_rest=True)
    state = official_state(result)
    assert state["container"]["position_m"] == pytest.approx([0.4, 0.1, 0.075], abs=1e-12)
    assert state["object"]["yaw_rad"] == pytest.approx(0.25, abs=1e-12)
    roll, pitch, _ = euler_xyz_from_quat(result["scene"]["object"]["init_state"]["rot"])
    official_roll, official_pitch, _ = euler_xyz_from_quat(OFFICIAL["scene"]["object"]["init_state"]["rot"])
    assert (roll, pitch) == pytest.approx((official_roll, official_pitch), abs=1e-12)
    for name in ("object", "object01", "container"):
        assert result["scene"][name]["init_state"]["lin_vel"] == [0.0, 0.0, 0.0]
    assert result["scene"]["object01"]["init_state"]["pos"] == OFFICIAL["scene"]["object01"]["init_state"]["pos"]
    assert OFFICIAL["scene"]["object"]["init_state"]["lin_vel"][0] != 0.0


def test_unknown_assets_are_rejected():
    override = DynamicVLAInitialStateOverride(
        environment=ENVIRONMENT, initial_state_index=0, object_positions_m={"basket": [0, 0, 0]}
    )
    with pytest.raises(ValueError, match="unknown DOM assets: basket"):
        apply_initial_state(OFFICIAL, override, start_at_rest=False)


def test_authoring_config_is_valid_and_reloads_identically():
    experiment = load_experiment(AUTHORING)
    assert isinstance(experiment, DynamicVLAExperiment)
    assert experiment.benchmark.start_at_rest
    assert experiment.benchmark.initial_states_per_environment == 2
    assert DynamicVLAExperiment.model_validate(experiment.model_dump(mode="json")) == experiment


def test_native_configs_record_default_friction():
    path = FIXTURES / "avocado06_bowl04_20260926/experiment.yaml"
    experiment = load_experiment(path)
    assert not experiment.has_authored_scene
    expected = yaml.safe_load(path.read_text())
    expected["benchmark"]["driven_friction"] = "native"
    assert experiment.model_dump(mode="json") == expected


@pytest.mark.parametrize(
    ("benchmark", "overrides", "message"),
    [
        (
            {"initial_states_per_environment": 2},
            [{"environment": "tests/other.json", "initial_state_index": 1, "object_yaws_rad": {"container": 0.0}}],
            "unconfigured environment",
        ),
        (
            {},
            [{"environment": ENVIRONMENT, "initial_state_index": 1, "object_yaws_rad": {"container": 0.0}}],
            "below initial_states_per_environment",
        ),
        (
            {"initial_states_per_environment": 2},
            [{"environment": ENVIRONMENT, "initial_state_index": 1, "object_positions_m": {"robot": [0, 0, 0]}}],
            "not the robot",
        ),
    ],
)
def test_initial_state_contract(benchmark, overrides, message):
    data = _experiment(initial_state_overrides=overrides or None, trajectory=None)
    data["benchmark"] = {**data["benchmark"], "initial_states_per_environment": 1, **benchmark}
    with pytest.raises(ValidationError, match=message):
        DynamicVLAExperiment.model_validate(data)


@pytest.mark.parametrize("explicit", [False, True])
def test_repeated_placements_preserve_episode_motion(tmp_path, explicit):
    data = _experiment()
    data["initial_state_overrides"] = []
    if explicit:
        for index in range(2):
            data["initial_state_overrides"].append({
                "environment": ENVIRONMENT,
                "initial_state_index": index,
                "object_positions_m": {"object": [0.4, 0.05, 0.05]},
            })
    data["trajectory"]["episode_schedule"][0] = {"initial_phase": 0.1, "initial_direction": 1}
    data["trajectory"]["episode_schedule"][1] = {"initial_phase": 0.7, "initial_direction": -1}
    experiment = DynamicVLAExperiment.model_validate(data)
    assets, _ = _assets_root(tmp_path)
    episodes = _build_manifest(experiment, assets, tmp_path / "states")["episodes"]
    assert [episode["initial_state_index"] for episode in episodes[:2]] == [0, 1]
    assert episodes[0]["environment_sha256"] == episodes[1]["environment_sha256"]
    for episode, phase, direction in zip(episodes[:2], [0.1, 0.7], [1, -1]):
        [actor] = episode["motion_actors"]
        assert actor["motion"]["initial_phase"] == phase
        assert actor["motion"]["initial_direction"] == direction


def test_duplicate_override_for_same_state_is_still_rejected():
    data = _experiment()
    data["initial_state_overrides"].append(copy.deepcopy(data["initial_state_overrides"][0]))
    with pytest.raises(ValidationError, match="one entry per environment and state"):
        DynamicVLAExperiment.model_validate(data)


def test_trajectories_need_an_explicit_schedule():
    data = _experiment()
    trajectory = copy.deepcopy(data["trajectory"])
    trajectory.pop("episode_schedule")
    trajectory["phase_bins"] = 2
    with pytest.raises(ValidationError, match="explicit episode_schedule"):
        DynamicVLAExperiment.model_validate({**data, "trajectory": trajectory})
    trajectory = {**data["trajectory"], "episode_schedule": data["trajectory"]["episode_schedule"][:-1]}
    with pytest.raises(ValidationError, match="length must equal"):
        DynamicVLAExperiment.model_validate({**data, "trajectory": trajectory})


def _assets_root(tmp_path):
    environment = copy.deepcopy(OFFICIAL)
    environment.update(
        seed=1050247,
        episode_length_s=12,
        instruction={
            "task": "place",
            "objects": ["avocado"],
            "containers": ["bowl"],
        },
    )
    path = tmp_path / "assets" / ENVIRONMENT
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(environment))
    return tmp_path / "assets", environment


def test_manifest_materializes_each_authored_state(tmp_path):
    assets, environment = _assets_root(tmp_path)
    experiment = load_experiment(AUTHORING)
    manifest = _build_manifest(experiment, assets, tmp_path / "states")
    episodes = manifest["episodes"]
    assert [episode["initial_state_index"] for episode in episodes[:4]] == [0, 1, 0, 1]
    paths = {episode["initial_state_index"]: episode["environment"] for episode in episodes}
    assert len(set(paths.values())) == 2
    for index, path in paths.items():
        derived = json.loads(Path(path).read_text())
        override = next(
            (o for o in experiment.initial_state_overrides if o.initial_state_index == index),
            None,
        )
        assert derived == apply_initial_state(environment, override, start_at_rest=True)
        assert derived["scene"]["object"]["init_state"]["lin_vel"] == [0.0, 0.0, 0.0]
    for episode in episodes:
        assert sha256(Path(episode["environment"])) == episode["environment_sha256"]
        [actor] = episode["motion_actors"]
        scheduled = experiment.trajectory.episode_schedule[episode["replica_id"]]
        assert actor["target"] == "container"
        assert actor["motion"]["initial_phase"] == scheduled.initial_phase
        assert actor["motion"]["initial_direction"] == scheduled.initial_direction
    # Before materialization only the digests are known, and they agree.
    preview = _build_manifest(experiment, assets)
    assert [e["environment_sha256"] for e in preview["episodes"]] == [
        e["environment_sha256"] for e in episodes
    ]
    assert {e["environment"] for e in preview["episodes"]} == {None}


def test_native_manifest_is_unchanged(tmp_path):
    assets, _ = _assets_root(tmp_path)
    experiment = load_experiment(
        FIXTURES / "avocado06_bowl04_20260926/experiment.yaml"
    )
    manifest = _build_manifest(experiment, assets, tmp_path / "states")
    assert not (tmp_path / "states").exists()
    official = str((assets / ENVIRONMENT).resolve())
    for episode in manifest["episodes"]:
        assert episode["environment"] == official
        assert "initial_state_index" not in episode and "motion_actors" not in episode


def test_identical_scene_reuse_preserves_episode_semantics(tmp_path, monkeypatch):
    assets, _ = _assets_root(tmp_path)
    data = load_experiment(AUTHORING).model_dump(mode="json")
    first = data["initial_state_overrides"][0]
    data["initial_state_overrides"] = [
        {**copy.deepcopy(first), "initial_state_index": index} for index in range(2)
    ]
    experiment = DynamicVLAExperiment.model_validate(data)
    ordinary = _build_manifest(experiment, assets, tmp_path / "ordinary")
    monkeypatch.setenv("VLALAGBENCH_DYNAMICVLA_REUSE_IDENTICAL_SCENES", "1")
    reused = _build_manifest(experiment, assets, tmp_path / "reused")
    assert len({e["environment"] for e in ordinary["episodes"]}) == 2
    assert len({e["environment"] for e in reused["episodes"]}) == 1
    for before, after in zip(ordinary["episodes"], reused["episodes"], strict=True):
        assert {k: v for k, v in before.items() if k != "environment"} == {
            k: v for k, v in after.items() if k != "environment"
        }
        assert Path(before["environment"]).read_bytes() == Path(after["environment"]).read_bytes()


def test_planar_driver_rotates_robot_frame_motion_into_the_world():
    experiment = load_experiment(AUTHORING)
    actors = episode_actors(experiment, 2)
    motion = EpisodeMotion(
        control_hz=25.0,
        driven_friction=0.0,
        robot_position_w=[-3.8647, -3.1541, 0.4318],
        robot_yaw_rad=math.pi,
        reset_positions_w={"container": [-4.2, -3.017, 0.4743]},
        actors=actors,
    )
    driver = PlanarDriver(motion)
    for step in (0, 7, 40):
        offset, velocity = offset_and_velocity(actors[0], step, 25.0)
        position, planar_velocity = driver.command(step)["container"]
        # Robot yaw pi flips both planar axes.
        assert position == pytest.approx([-4.2 - offset[0], -3.017 - offset[1]], abs=1e-12)
        assert planar_velocity == pytest.approx([-velocity[0], -velocity[1]], abs=1e-12)
        assert float(np.hypot(*planar_velocity)) == pytest.approx(0.05, abs=1e-3)
    assert EpisodeMotion.model_validate(motion.model_dump(mode="json")) == motion
    assert actors_from_json([a.model_dump(mode="json") for a in actors]) == actors


@pytest.mark.parametrize("friction", [0.0, None])
def test_planar_driver_writes_only_planar_position_and_velocity(friction):
    torch = pytest.importorskip("torch")

    class Asset:
        def __init__(self):
            self.data = type("Data", (), {})()
            self.data.root_state_w = torch.arange(13, dtype=torch.float32)[None]
            self.written = None

        def write_root_state_to_sim(self, state):
            self.written = state

    asset = Asset()
    env = type("Env", (), {"scene": {"container": asset}})()
    motion = EpisodeMotion(
        control_hz=25.0,
        driven_friction=friction,
        native_materials={"container": [[[0.7, 0.4, 0.1]]]} if friction is None else {},
        robot_position_w=[0.0, 0.0, 0.0],
        robot_yaw_rad=0.0,
        reset_positions_w={"container": [1.0, 2.0, 3.0]},
        actors=episode_actors(load_experiment(AUTHORING), 0),
    )
    class View:
        count = 1

        def __init__(self):
            self.materials = torch.tensor([[[0.5, 0.5, 0.1]]])

        def get_material_properties(self):
            return self.materials.clone()

        def set_material_properties(self, materials, indices):
            self.materials = materials

    asset.root_physx_view = View()
    PlanarDriver(motion).prepare(env)
    expected = [0.7, 0.4, 0.1] if friction is None else [0.0, 0.0, 0.1]
    assert asset.root_physx_view.materials[0, 0].tolist() == pytest.approx(expected)
    assert EpisodeMotion.model_validate_json(motion.model_dump_json()) == motion
    PlanarDriver(motion).apply(env, 0)
    written = asset.written[0].tolist()
    position, velocity = PlanarDriver(motion).command(0)["container"]
    assert written[0:2] == pytest.approx(position.tolist())
    assert written[7:9] == pytest.approx(velocity.tolist())
    assert written[2:7] == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert written[9:] == [9.0, 10.0, 11.0, 12.0]


def test_friction_manifest_preserves_paired_scene_identity(tmp_path):
    assets, _ = _assets_root(tmp_path)
    experiment = load_experiment(AUTHORING)
    assert experiment.benchmark.driven_friction == "native"
    with_friction = _build_manifest(experiment, assets, tmp_path / "states")
    frictionless = experiment.model_copy(update={"benchmark": experiment.benchmark.model_copy(
        update={"driven_friction": "zero"})})
    zero = _build_manifest(frictionless, assets, tmp_path / "states")
    assert experiment.benchmark.model_dump()["driven_friction"] == "native"
    assert frictionless.benchmark.model_dump()["driven_friction"] == "zero"
    for left, right in zip(zero["episodes"], with_friction["episodes"], strict=True):
        assert right.pop("driven_friction") == "native"
        assert left == right


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: {**t, "target": "object"}, "cannot drive the grasped target"),
        (lambda t: {**t, "release_on_grasp_steps": 3}, "release_on_grasp_steps"),
    ],
)
def test_unsupported_motion_fails_loudly(change, message):
    data = _experiment()
    with pytest.raises(ValidationError, match=message):
        DynamicVLAExperiment.model_validate({**data, "trajectory": change(data["trajectory"])})


def test_preflight_rejects_unknown_scene_assets(tmp_path):
    assets, environment = _assets_root(tmp_path)
    data = _experiment()
    data["trajectory"] = {**data["trajectory"], "target": "object02"}
    with pytest.raises(ValueError, match="no rigid assets named: object02"):
        _check_authored_targets(
            DynamicVLAExperiment.model_validate(data), assets / ENVIRONMENT, environment
        )
