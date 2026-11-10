"""Evaluation environments render only the camera outputs the policy consumes."""
import types

import pytest

from vlalagbench.benchmarks.dynamicvla import camera_outputs


class CameraCfg:
    def __init__(self, data_types):
        self.data_types = data_types


@pytest.fixture(autouse=True)
def reset_active(monkeypatch):
    monkeypatch.setattr(camera_outputs, "_active", None)


def upstream_with(scene):
    return types.SimpleNamespace(_get_env_cfg=lambda source: types.SimpleNamespace(scene=scene))


def test_rgb_only_drops_segmentation_from_every_camera():
    scene = types.SimpleNamespace(wrist_cam=CameraCfg(["rgb", "semantic_segmentation"]),
                                  side_cam=CameraCfg(["semantic_segmentation", "rgb"]),
                                  robot=types.SimpleNamespace(data_types=["other"]))
    upstream = upstream_with(scene)
    camera_outputs.install_camera_outputs(upstream, camera_outputs.RGB_ONLY)
    upstream._get_env_cfg({})
    assert scene.wrist_cam.data_types == ["rgb"] and scene.side_cam.data_types == ["rgb"]
    assert scene.robot.data_types == ["other"]
    assert camera_outputs.active_camera_outputs() == "rgb"


def test_upstream_mode_leaves_configs_and_rejects_conflicting_install():
    scene = types.SimpleNamespace(cam=CameraCfg(["rgb", "semantic_segmentation"]))
    upstream = upstream_with(scene)
    camera_outputs.install_camera_outputs(upstream, camera_outputs.UPSTREAM)
    upstream._get_env_cfg({})
    assert scene.cam.data_types == ["rgb", "semantic_segmentation"]
    with pytest.raises(ValueError, match="already installed"):
        camera_outputs.install_camera_outputs(upstream, camera_outputs.RGB_ONLY)


def test_camera_without_rgb_is_rejected():
    upstream = upstream_with(types.SimpleNamespace(cam=CameraCfg(["depth"])))
    camera_outputs.install_camera_outputs(upstream, camera_outputs.RGB_ONLY)
    with pytest.raises(ValueError, match="must render rgb"):
        upstream._get_env_cfg({})


def test_modes_from_environment_and_recorded_traces(monkeypatch):
    monkeypatch.delenv(camera_outputs.ENV_CAMERA_OUTPUTS, raising=False)
    assert camera_outputs.simulator_camera_outputs() == "upstream"
    monkeypatch.setenv(camera_outputs.ENV_CAMERA_OUTPUTS, "rgb")
    assert camera_outputs.simulator_camera_outputs() == "rgb"
    monkeypatch.setenv(camera_outputs.ENV_CAMERA_OUTPUTS, "yes")
    with pytest.raises(ValueError):
        camera_outputs.simulator_camera_outputs()
    assert camera_outputs.recorded_camera_outputs({}) == "upstream"
    assert camera_outputs.recorded_camera_outputs({"renderer_camera_outputs": "rgb"}) == "rgb"
