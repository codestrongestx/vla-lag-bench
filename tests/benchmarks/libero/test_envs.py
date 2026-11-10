import types

import pytest
from pydantic import ValidationError

from vlalagbench.benchmarks.libero import envs
from vlalagbench.simulators.mujoco.tracing import TRACE_SCHEMA_VERSION, TraceMeta


@pytest.mark.parametrize("camera_observations", [True, False])
def test_make_env_selects_camera_mode(tmp_path, monkeypatch, camera_observations):
    class FakeEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def seed(self, seed):
            self.seed_value = seed

    class ControlEnv(FakeEnv):
        pass

    monkeypatch.setattr(envs._vendor, "libero", lambda: types.SimpleNamespace(
        get_libero_path=lambda _: tmp_path,
        ControlEnv=ControlEnv,
    ))

    result = envs._make_env(
        "problem", "task.bddl", 256, 7,
        hard_reset=False, camera_observations=camera_observations,
    )

    assert isinstance(result, ControlEnv)
    assert result.seed_value == 7
    assert result.kwargs["use_camera_obs"] is camera_observations
    assert result.kwargs["has_offscreen_renderer"] is camera_observations
    assert result.kwargs["camera_heights"] == 256


def test_make_env_from_trace_rejects_incomplete_libero_metadata():
    meta = TraceMeta.model_validate(
        {
            "schema_version": TRACE_SCHEMA_VERSION,
            "environment": {
                "kind": "libero",
                "problem_folder": "libero_spatial",
                "bddl_file": "task.bddl",
                "resolution": 256,
            },
            "control_freq": 20.0,
            "num_steps": 0,
            "success": False,
        }
    )

    with pytest.raises(ValidationError, match="seed"):
        envs.make_env_from_trace(meta)
