import types

import numpy as np

from vlalagbench.simulators.mujoco.rendering import MjvSceneRenderer, configure_free_camera


class _FakeRenderer:
    instances = []

    def __init__(self, model, *, height, width, max_geom):
        self.model = model
        self.height = height
        self.width = width
        self.max_geom = max_geom
        self.scene = types.SimpleNamespace(markers=[])
        self.updated = []
        self.closed = False
        self.instances.append(self)

    def update_scene(self, data, *, camera):
        self.updated.append((data, camera))
        self.scene.markers.clear()

    def render(self):
        return np.full((self.height, self.width, 3), len(self.scene.markers), dtype=np.uint8)

    def close(self):
        self.closed = True


def test_scene_renderer_applies_overlay_after_every_scene_update(monkeypatch):
    import mujoco

    monkeypatch.setattr(mujoco, "Renderer", _FakeRenderer)
    visual = types.SimpleNamespace(offwidth=320, offheight=240)
    raw_model = types.SimpleNamespace(vis=types.SimpleNamespace(global_=visual))
    raw_data = object()
    sim = types.SimpleNamespace(
        model=types.SimpleNamespace(_model=raw_model),
        data=types.SimpleNamespace(_data=raw_data),
    )

    def overlay(scene):
        scene.markers.extend(["red", "green"])

    with MjvSceneRenderer(
        sim, camera="agentview", width=512, height=400, overlay=overlay
    ) as renderer:
        first = renderer.render()
        second = renderer.render()
        backend = renderer._renderer

    assert visual.offwidth == 512
    assert visual.offheight == 400
    assert backend.updated == [(raw_data, "agentview"), (raw_data, "agentview")]
    assert np.all(first == 2)
    assert np.all(second == 2)
    assert backend.closed


def test_scene_renderer_rejects_nonpositive_dimensions():
    sim = types.SimpleNamespace()

    try:
        MjvSceneRenderer(sim, width=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("expected invalid width to fail")


def test_configure_free_camera_clears_fixed_camera_mode(monkeypatch):
    import mujoco

    model = object()
    camera = types.SimpleNamespace(
        type=mujoco.mjtCamera.mjCAMERA_FIXED,
        fixedcamid=2,
        trackbodyid=4,
        lookat=np.zeros(3),
        distance=0.0,
        azimuth=0.0,
        elevation=0.0,
    )
    initialized = []
    monkeypatch.setattr(
        mujoco, "mjv_defaultFreeCamera", lambda actual_model, _: initialized.append(actual_model)
    )

    configure_free_camera(
        model,
        camera,
        lookat=[0.1, 0.2, 0.3],
        distance=1.4,
        azimuth=90.0,
        elevation=-55.0,
    )

    assert initialized == [model]
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert camera.fixedcamid == -1
    assert camera.trackbodyid == -1
    np.testing.assert_allclose(camera.lookat, [0.1, 0.2, 0.3])
    assert camera.distance == 1.4
    assert camera.azimuth == 90.0
    assert camera.elevation == -55.0
