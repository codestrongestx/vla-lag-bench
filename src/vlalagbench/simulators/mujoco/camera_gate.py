"""Skip robosuite camera renders whose images nobody reads.

robosuite samples every camera observable once per control step, inside
``env.step``. A chunked policy reads the observation only on request steps,
so most of those renders are discarded. :class:`CameraGate` wraps each image
sensor: while the gate is open the original sensor runs unchanged; while it
is closed the sensor returns the observable's previous value and marks it
stale. Rendering has no effect on physics, and the sampling schedule is left
untouched, so an image read after an open step is the very render the
ungated environment would have produced.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from vlalagbench.simulators.mujoco.dynamics import _simulation_host


class CameraGate:
    """Open/close all image observables of one robosuite environment."""

    def __init__(self, env: Any) -> None:
        host = _simulation_host(env)
        self.open = True
        self._stale: set[str] = set()
        self._names: list[str] = []
        for name, observable in host._observables.items():
            if observable.modality != "image":
                continue
            observable._sensor = self._gated(name, observable, observable._sensor)
            self._names.append(name)
        if not self._names:
            raise ValueError("environment has no image observables to gate")

    def _gated(self, name: str, observable: Any, sensor: Callable) -> Callable:
        @functools.wraps(sensor)
        def gated(obs_cache):
            if self.open:
                self._stale.discard(name)
                return sensor(obs_cache)
            self._stale.add(name)
            return observable._current_observed_value

        return gated

    def require_fresh(self) -> None:
        """Fail unless every current image came from an actual render."""

        if self._stale:
            raise RuntimeError(f"policy observation would use unrendered images: {sorted(self._stale)}")
