"""Opt-in public-policy equivalence test; run in the isolated GR00T environment."""
import os

import numpy as np
import pytest


def test_pinned_public_policy_decodes_equivalent_actions(request):
    checkpoint = os.environ.get("GR00T_TEST_CHECKPOINT_ROOT")
    if not checkpoint:
        pytest.skip("set GR00T_TEST_CHECKPOINT_ROOT after enabling backbone access")
    import time
    import torch
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from vlalagbench.benchmarks.libero.gr00t_adapter import (
        AXES, Gr00tRequestPolicy, backbone_processor_identity, checkpoint_identity, format_observation, isolated_rng,
    )
    from vlalagbench.benchmarks.libero.policy import REQUEST_KEY, request_seed
    from tests.benchmarks.libero.test_gr00t import model, observation
    from pathlib import Path

    spec = model().model_copy(update={"checkpoint": checkpoint})
    path = Path(checkpoint) / spec.checkpoint_subdir
    upstream = Path(__file__).resolve().parents[3] / "third_party/Isaac-GR00T"
    checkpoint_identity(spec, path, upstream)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    public = Gr00tPolicy(spec.embodiment, str(path), device="cuda:0")
    processors = backbone_processor_identity(public)
    assert processors == backbone_processor_identity(public)
    assert "observation/tokenizer.json" in processors and "collator/tokenizer.json" in processors
    obs = format_observation(observation(), "move")
    flat = {f"{kind}.{key}": value for kind in ("video", "state") for key, value in obs[kind].items()}
    flat["annotation.human.action.task_description"] = ["move"]
    identity = {"policy_seed": 7, "task_id": 0, "episode_id": 0, "request_id": 0}
    with isolated_rng(request_seed(identity)):
        reference, _ = Gr00tSimPolicyWrapper(public).get_action(flat)
    reference = np.concatenate([reference[f"action.{key}"] for key in (*AXES, "gripper")], axis=-1)[0]
    reference[:, -1] = -np.sign(2 * reference[:, -1] - 1)
    actual = Gr00tRequestPolicy(public).infer({**obs, REQUEST_KEY: identity})["actions"]
    torch.cuda.synchronize()
    request.node.user_properties.extend([("load_and_comparison_seconds", time.monotonic() - started),
                                         ("peak_vram_bytes", torch.cuda.max_memory_allocated())])
    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
