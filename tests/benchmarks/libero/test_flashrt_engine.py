import hashlib

import numpy as np
import pytest
from pydantic import ValidationError

from vlalagbench.benchmarks.libero import policy_server
from vlalagbench.benchmarks.libero.policy import REQUEST_KEY, RESPONSE_KEY, deterministic_noise
from vlalagbench.config import ModelSpec

CHECKPOINT = "/checkpoints/pi05_libero_pytorch"


def test_openpi_model_spec_serializes_as_before():
    model = ModelSpec(openpi_config="pi05_libero", checkpoint=CHECKPOINT)
    assert model.model_dump() == {"openpi_config": "pi05_libero", "checkpoint": CHECKPOINT}
    assert policy_server.engine_identity(model) is None


def test_openpi_engine_rejects_other_precisions():
    with pytest.raises(ValidationError, match="openpi serves"):
        ModelSpec(openpi_config="pi05_libero", checkpoint=CHECKPOINT, precision="fp8", calibration="frames.npz")


def test_handshake_requires_the_engine_and_precision():
    model = ModelSpec(openpi_config="pi05_libero", checkpoint=CHECKPOINT, engine="flashrt")
    kwargs = {"model": model, "openpi_config": "pi05_libero", "checkpoint": CHECKPOINT, "action_chunk_steps": 10}
    served = policy_server.build_metadata("pi05_libero", CHECKPOINT, 10, engine=policy_server.engine_identity(model))
    policy_server.validate_metadata(served, **kwargs)
    openpi = policy_server.build_metadata("pi05_libero", CHECKPOINT, 10)
    with pytest.raises(RuntimeError, match="engine"):
        policy_server.validate_metadata(openpi, **kwargs)
    fp8 = ModelSpec(openpi_config="pi05_libero", checkpoint=CHECKPOINT, engine="flashrt", precision="fp8",
                    calibration="frames.npz")
    with pytest.raises(RuntimeError, match="precision"):
        policy_server.validate_metadata(served, **{**kwargs, "model": fp8})


def test_openpi_client_rejects_an_engine_server():
    served = policy_server.build_metadata("pi05_libero", CHECKPOINT, 10, engine={"engine": "flashrt", "precision": "bf16"})
    model = ModelSpec(openpi_config="pi05_libero", checkpoint=CHECKPOINT)
    with pytest.raises(RuntimeError, match="engine"):
        policy_server.validate_metadata(served, model=model, openpi_config="pi05_libero", checkpoint=CHECKPOINT,
                                        action_chunk_steps=10)


def test_fp8_and_nvfp4_require_calibration_frames():
    for precision in ("fp8", "nvfp4"):
        with pytest.raises(ValidationError, match="calibration"):
            ModelSpec(openpi_config="pi05_libero", checkpoint=CHECKPOINT, engine="flashrt", precision=precision)
        model = ModelSpec(openpi_config="pi05_libero", checkpoint=CHECKPOINT, engine="flashrt", precision=precision,
                          calibration="frames.npz")
        assert model.model_dump()["calibration"] == "frames.npz"
        assert policy_server.engine_identity(model) == {"engine": "flashrt", "precision": precision}
    with pytest.raises(ValidationError, match="calibration"):
        ModelSpec(openpi_config="pi05_libero", checkpoint=CHECKPOINT, engine="flashrt", calibration="frames.npz")



class _FakeFrontend:
    """Records FlashRT's batched calls; each row's actions are its noise's first seven columns."""

    def __init__(self, checkpoint, **kwargs):
        self.width = None
        self.built = []
        self.passes = []

    def set_batched_mode(self, *, enable, batch_size):
        assert enable
        self.width = batch_size

    def set_prompt_batch(self, prompts):
        assert len(prompts) == self.width
        self.built.append((self.width, tuple(prompts)))

    def calibrate_batch(self, frames):
        assert len(frames) == 1

    def infer_batch(self, rows, *, noise):
        assert len(rows) == self.width == len(noise)
        self.passes.append(self.width)
        return [{"actions": row_noise[:, :7]} for row_noise in noise]


@pytest.fixture
def batch_policy(monkeypatch):
    import sys
    import types

    pytest.importorskip("torch")
    module = types.ModuleType("flash_rt.frontends.torch.pi05_rtx")
    module.Pi05TorchFrontendRtx = _FakeFrontend
    monkeypatch.setitem(sys.modules, "flash_rt.frontends.torch.pi05_rtx", module)
    from vlalagbench.benchmarks.libero import flashrt_policy
    from vlalagbench.benchmarks.libero.flashrt_policy import FlashRTBatchPolicy

    monkeypatch.setattr(flashrt_policy, "_use_flash_attn_where_fa2_is_missing", lambda: None)

    def make(**kwargs):
        return FlashRTBatchPolicy(CHECKPOINT, precision="bf16", action_horizon=10, action_dim=32, **kwargs)

    return make


PROMPT = "put the bowl on the plate"


def _request(episode, prompt=PROMPT):
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    return {"observation/image": image, "observation/wrist_image": image, "prompt": prompt,
            REQUEST_KEY: {"policy_seed": 0, "task_id": 0, "episode_id": episode, "request_id": 0}}


def _noise(episode):
    return deterministic_noise(_request(episode)[REQUEST_KEY], action_horizon=10, action_dim=32)


def test_fixed_padding_runs_every_pass_at_one_width_with_each_requests_noise(batch_policy):
    policy = batch_policy(max_batch_size=4, batch_padding="fixed")
    outputs = policy.infer_batch([_request(episode) for episode in range(3)])
    assert policy.model.built == [(4, (PROMPT,) * 4)]
    assert policy.model.passes == [4] and len(outputs) == 3
    for episode, output in enumerate(outputs):
        np.testing.assert_array_equal(output["actions"], _noise(episode)[:, :7])
        assert output[RESPONSE_KEY]["episode_id"] == episode
        assert output[RESPONSE_KEY]["noise_sha256"] == hashlib.sha256(_noise(episode).tobytes()).hexdigest()


def test_short_passes_are_padded_and_overfull_ones_refused(batch_policy):
    policy = batch_policy(max_batch_size=3, batch_padding="fixed")
    policy.warm_up({**_request(0), REQUEST_KEY: {"warm_up": True}})
    alone = policy.infer_batch([_request(1)])
    together = policy.infer_batch([_request(0), _request(1)])
    assert policy.model.built == [(3, (PROMPT,) * 3)] and policy.model.passes == [3, 3, 3]
    np.testing.assert_array_equal(alone[0]["actions"], together[1]["actions"])
    with pytest.raises(ValueError, match="exceed"):
        policy.infer_batch([_request(episode) for episode in range(4)])


def test_batched_server_refuses_other_prompts_and_precisions(batch_policy):
    from vlalagbench.benchmarks.libero.flashrt_policy import FlashRTBatchPolicy

    policy = batch_policy(max_batch_size=2, batch_padding="fixed")
    policy.infer_batch([_request(0)])
    with pytest.raises(ValueError, match="serves"):
        policy.infer_batch([_request(1, prompt="open the drawer")])
    with pytest.raises(ValueError, match="bf16 only"):
        FlashRTBatchPolicy(CHECKPOINT, precision="fp8", action_horizon=10, action_dim=32,
                           max_batch_size=2, batch_padding="fixed")
    with pytest.raises(ValueError, match="fixed"):
        FlashRTBatchPolicy(CHECKPOINT, precision="bf16", action_horizon=10, action_dim=32,
                           max_batch_size=2, batch_padding="none")


def test_handshake_checks_batching_on_engine_servers():
    model = ModelSpec(openpi_config="pi05_libero", checkpoint=CHECKPOINT, engine="flashrt")
    batching = {"implementation": "dynamic_batching", "max_batch_size": 32, "max_wait_ms": 0.0, "batch_padding": "fixed"}
    served = policy_server.build_metadata("pi05_libero", CHECKPOINT, 10, batching=batching,
                                          engine=policy_server.engine_identity(model))
    kwargs = {"model": model, "openpi_config": "pi05_libero", "checkpoint": CHECKPOINT, "action_chunk_steps": 10}
    policy_server.validate_metadata(served, **kwargs, batching=batching)
    with pytest.raises(RuntimeError, match="implementation"):
        policy_server.validate_metadata(served, **kwargs)
