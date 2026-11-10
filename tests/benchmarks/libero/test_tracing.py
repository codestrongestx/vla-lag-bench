import json

import numpy as np
import pytest
from pydantic import ValidationError

from vlalagbench.benchmarks.libero.tracing import load_libero_trace
from vlalagbench.simulators.mujoco.tracing import TRACE_SCHEMA_VERSION


def _write_trace(path, environment):
    np.savez(
        path,
        meta=json.dumps(
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                "environment": environment,
                "control_freq": 20.0,
                "num_steps": 0,
                "success": False,
            }
        ),
        states=np.zeros((1, 1)),
        actions=np.zeros((0, 1)),
        rewards=np.zeros(0),
        dones=np.zeros(0, dtype=bool),
        acting=np.zeros(0, dtype=bool),
        chunk_ids=np.zeros(0, dtype=np.int64),
        action_indices=np.zeros(0, dtype=np.int64),
        observation_steps=np.zeros(0, dtype=np.int64),
        request_steps=np.zeros(0, dtype=np.int64),
        arrival_steps=np.zeros(0, dtype=np.int64),
        chunks=np.zeros((0, 0, 0)),
    )


def _environment(**updates):
    environment = {
        "kind": "libero",
        "problem_folder": "libero_spatial",
        "bddl_file": "task.bddl",
        "resolution": 256,
        "seed": 7,
    }
    environment.update(updates)
    return environment


def test_load_libero_trace_accepts_complete_metadata(tmp_path):
    path = tmp_path / "valid.npz"
    _write_trace(path, _environment())

    trace = load_libero_trace(path)

    assert trace.meta.environment.seed == 7


@pytest.mark.parametrize(
    ("environment", "error_field"),
    [
        ({key: value for key, value in _environment().items() if key != "seed"}, "seed"),
        (_environment(sead=7), "sead"),
    ],
    ids=["missing-seed", "misspelled-field"],
)
def test_load_libero_trace_rejects_invalid_metadata(tmp_path, environment, error_field):
    path = tmp_path / "invalid.npz"
    _write_trace(path, environment)

    with pytest.raises(ValidationError, match=error_field):
        load_libero_trace(path)
