"""Render MuJoCo frames with optional non-physical scene annotations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


# An overlay appends visualization-only geoms to the freshly updated MjvScene.
# It never changes MjModel / MjData and therefore cannot affect simulation.
SceneOverlay = Callable[[Any], Any]


def configure_free_camera(
    model: Any,
    camera: Any,
    *,
    lookat: Any | None = None,
    distance: float | None = None,
    azimuth: float | None = None,
    elevation: float | None = None,
) -> None:
    """Put a viewer camera in mouse-orbitable free-camera mode."""

    import mujoco

    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    if lookat is not None:
        camera.lookat[:] = lookat
    if distance is not None:
        camera.distance = distance
    if azimuth is not None:
        camera.azimuth = azimuth
    if elevation is not None:
        camera.elevation = elevation


class MjvSceneRenderer:
    """Reusable offscreen renderer that can annotate each visual scene.

    ``mujoco.Renderer.update_scene`` rebuilds the scene for every frame.  The
    overlay is applied immediately afterwards, so its geoms participate in the
    normal camera projection, depth test, and occlusion without becoming
    collidable MuJoCo model geoms.
    """

    def __init__(
        self,
        sim: Any,
        *,
        camera: Any = "agentview",
        width: int = 512,
        height: int = 512,
        max_geom: int = 10000,
        overlay: SceneOverlay | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("renderer width and height must be positive")

        import mujoco

        self._model = getattr(sim.model, "_model", sim.model)
        self._data = getattr(sim.data, "_data", sim.data)
        self.camera = camera
        self.overlay = overlay

        # robosuite / LIBERO commonly compile a 640x480 offscreen buffer.  The
        # settings are mutable and enlarging them does not alter model physics.
        visual = self._model.vis.global_
        visual.offwidth = max(int(visual.offwidth), int(width))
        visual.offheight = max(int(visual.offheight), int(height))
        self._renderer = mujoco.Renderer(
            self._model,
            height=int(height),
            width=int(width),
            max_geom=int(max_geom),
        )

    @property
    def scene(self) -> Any:
        return self._renderer.scene

    def render(self) -> np.ndarray:
        """Render the current simulation state, then add the visual overlay."""

        self._renderer.update_scene(self._data, camera=self.camera)
        if self.overlay is not None:
            self.overlay(self._renderer.scene)
        return np.ascontiguousarray(self._renderer.render())

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> "MjvSceneRenderer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
