"""Serve pi0.5 through FlashRT with openpi's LIBERO inputs and request noise.

The client sends the observation openpi's LIBERO policy consumes: both camera
images already resized with padding to 224 px as uint8, the state, and the
prompt. FlashRT receives those two images, the prompt, and the same
episode-scoped diffusion noise as the openpi path, and returns unnormalized
actions; only the network's kernels and arithmetic differ.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from vlalagbench.benchmarks.libero.policy import REQUEST_KEY, RESPONSE_KEY, deterministic_noise, validate_actions
from vlalagbench.files import local_path

logger = logging.getLogger(__name__)

IMAGE_KEYS = ("observation/image", "observation/wrist_image")
CALIBRATION_SEED = 0


def calibration_frames(path: Path, prompt: str) -> dict[str, Any]:
    """The recorded requests for one prompt, as FlashRT calibration frames."""
    data = np.load(path)
    rows = np.flatnonzero(data["prompt"] == prompt)
    if not len(rows):
        raise ValueError(f"{path} holds no requests for {prompt!r}")
    return {
        "frames": [{"images": [data["image"][row], data["wrist_image"][row]]} for row in rows],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sources": [str(source) for source in data["source"][rows]],
    }


class FlashRTPolicy:
    """One FlashRT pi0.5 frontend serving a single LIBERO prompt.

    FlashRT captures its CUDA graph for a prompt before its first request;
    with FP8 the held-out calibration requests also set the activation
    scales, which then serve every later frame. A server therefore serves
    one task's prompt and refuses any other.
    """

    def __init__(self, checkpoint: str, *, precision: str, action_horizon: int, action_dim: int,
                 calibration: dict[str, Any] | None = None) -> None:
        import torch
        from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

        quantized = precision in ("fp8", "nvfp4")
        if quantized and torch.cuda.get_device_capability() < (8, 9):
            raise RuntimeError("FlashRT FP8 needs an sm_89 or newer GPU")
        if precision == "nvfp4" and torch.cuda.get_device_capability() < (12, 0):
            raise RuntimeError("FlashRT's NVFP4 prefix needs an sm_120 GPU")
        self.precision = precision
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        # nvfp4 quantizes the prefix (SigLIP and Gemma-2B) to NVFP4 and keeps
        # the calibrated FP8 action expert.
        self.model = Pi05TorchFrontendRtx(
            local_path(checkpoint), num_views=len(IMAGE_KEYS), chunk_size=action_horizon,
            use_fp8=quantized, **({"prefix_precision": precision} if precision == "nvfp4" else {}),
        )
        if quantized != (calibration is not None):
            raise ValueError("FlashRT takes calibration frames for fp8 and nvfp4 and only for them")
        self.calibration = calibration
        self._prompt: str | None = None

    def identity(self) -> dict[str, Any]:
        import subprocess

        import flash_rt

        source = Path(flash_rt.__file__).resolve().parent
        # A FlashRT installed from its git checkout records the commit; a packaged install has none.
        git = subprocess.run(["git", "-C", str(source), "rev-parse", "--show-toplevel", "HEAD"],
                             capture_output=True, text=True)
        lines = git.stdout.split()
        own_checkout = git.returncode == 0 and (Path(lines[0]) / "flash_rt").resolve() == source
        commit = lines[1] if own_checkout else None
        # FlashRT rounds the supplied float32 noise into its bf16 buffer; the
        # audit hash covers the float32 request noise before that cast.
        return {"engine": "flashrt", "precision": self.precision, "flashrt_commit": commit,
                "flashrt_version": importlib.metadata.version("flash-rt"), "noise_dtype": "bfloat16",
                **({"calibration_sha256": self.calibration["sha256"],
                    "calibration_requests": len(self.calibration["frames"])} if self.calibration else {})}

    def _images(self, observation: dict[str, Any]) -> dict[str, Any]:
        images = []
        for key in IMAGE_KEYS:
            image = np.asarray(observation[key])
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(f"{key} must be a 224x224x3 uint8 image; got {image.shape} {image.dtype}")
            images.append(image)
        return {"images": images}

    def _prepare(self, prompt: str, frame: dict[str, Any]) -> None:
        import torch

        started = time.monotonic()
        self.model.set_prompt(prompt)
        # Graph capture runs the network on these frames; FP8 also sets its
        # activation scales from them, so it uses the held-out recorded
        # requests and bf16 the first frame. Seed the pass's own noise draw.
        torch.manual_seed(CALIBRATION_SEED)
        self.model.calibrate(self.calibration["frames"] if self.calibration else [frame])
        self._prompt = prompt
        logger.info("FlashRT prompt %r ready in %.1f s", prompt, time.monotonic() - started)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        observation = dict(observation)
        identity = observation.pop(REQUEST_KEY)
        if not isinstance(identity, dict):
            raise ValueError(f"{REQUEST_KEY} must be a request identity")
        frame = self._images(observation)
        prompt = str(observation["prompt"])
        if self._prompt is None:
            self._prepare(prompt, frame)
        elif prompt != self._prompt:
            raise ValueError(f"this FlashRT server serves {self._prompt!r}; got {prompt!r}")
        noise = deterministic_noise(identity, action_horizon=self.action_horizon, action_dim=self.action_dim)
        started = time.monotonic()
        actions = validate_actions(self.model.infer(frame, noise=noise)["actions"], self.action_horizon)
        return {
            "actions": actions,
            "policy_timing": {"infer_ms": (time.monotonic() - started) * 1000.0},
            RESPONSE_KEY: {**identity, "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest()},
        }


def _use_flash_attn_where_fa2_is_missing() -> None:
    """Let FlashRT's batched pi0.5 attention fall back to pip flash_attn.

    FlashRT's batched backend calls its own FA2 module at every attention
    site, and FlashRT builds that module only for sm_80/86/89/120, not for
    the H100's sm_90. Its single-request backend falls back to pip
    ``flash_attn_func`` at any site FA2 is off (``FVK_RTX_FA2=0``); this
    gives the batched backend the same fallback on the same B-folded
    buffers. Sites with FA2 run FlashRT's own code unchanged.
    """
    from flash_rt.frontends.torch import pi05_rtx
    from flash_rt.hardware.rtx.attn_backend_batched_pi05 import RtxFlashAttnBatchedBackendPi05

    if getattr(pi05_rtx.RtxFlashAttnBatchedBackendPi05, "flash_attn_fallback", False):
        return

    class Backend(RtxFlashAttnBatchedBackendPi05):
        flash_attn_fallback = True

        def _attend(self, site: Any, q: Any, k: Any, v: Any) -> int:
            out = self._flash_attn_func(q, k, v, causal=False)
            # The returned pointer feeds the next GEMM; keep its storage alive.
            self.__dict__.setdefault("_flash_attn_outputs", {})[site] = out
            return out.data_ptr()

        def vision_attn_batched(self, stream: int = 0) -> int:
            if self._fa2_sites["siglip"]:
                return super().vision_attn_batched(stream=stream)
            return self._attend("siglip", self.vis_Q_b2, self.vis_K_b2, self.vis_V_b2)

        def encoder_attn_batched(self, layer_idx: int, seq: int, stream: int = 0) -> int:
            if self._fa2_sites["encoder"]:
                return super().encoder_attn_batched(layer_idx, seq, stream=stream)
            return self._attend(("encoder", layer_idx), self.enc_Q_b2[:, :seq],
                                self.enc_K_b2[layer_idx, :, :seq], self.enc_V_b2[layer_idx, :, :seq])

        def decoder_attn_batched(self, layer_idx: int, enc_seq: int, dec_seq: int, stream: int = 0) -> int:
            if self._fa2_sites["decoder"]:
                return super().decoder_attn_batched(layer_idx, enc_seq, dec_seq, stream=stream)
            total_kv = enc_seq + dec_seq
            return self._attend(("decoder", layer_idx), self.dec_Q_b2[:, :dec_seq],
                                self.enc_K_b2[layer_idx, :, :total_kv], self.enc_V_b2[layer_idx, :, :total_kv])

    # set_batched_mode builds and type-checks the backend through this name.
    pi05_rtx.RtxFlashAttnBatchedBackendPi05 = Backend


class FlashRTBatchPolicy(FlashRTPolicy):
    """Several LIBERO requests per pass through FlashRT's batched pi0.5 pipelines.

    Every pass is padded to ``max_batch_size`` rows (``batch_padding:
    fixed``), and rows of a pass are independent, so a request's actions do
    not depend on how many others shared its pass or in what order. FlashRT
    captures one pipeline per width; one per size up to 32 does not fit an
    80 GB H100, so unpadded batching is refused. Batched FP8 would take its
    activation scales from a single frame instead of the held-out recorded
    requests, so only bf16 is served batched.
    """

    def __init__(self, checkpoint: str, *, precision: str, action_horizon: int, action_dim: int,
                 max_batch_size: int, batch_padding: str) -> None:
        if precision != "bf16":
            raise ValueError(f"batched FlashRT serving runs bf16 only; got {precision}")
        if batch_padding != "fixed":
            raise ValueError("batched FlashRT serving pads every pass; use batch_padding: fixed")
        _use_flash_attn_where_fa2_is_missing()
        super().__init__(checkpoint, precision=precision, action_horizon=action_horizon, action_dim=action_dim)
        self.width = max_batch_size

    def identity(self) -> dict[str, Any]:
        sites = self.model.attn_backend._fa2_sites
        return {**super().identity(),
                "attention": {site: "flash_rt_fa2" if fa2 else "flash_attn" for site, fa2 in sorted(sites.items())}}

    def _prepare(self, prompt: str, frame: dict[str, Any]) -> None:
        import torch

        started = time.monotonic()
        self.model.set_batched_mode(enable=True, batch_size=self.width)
        self.model.set_prompt_batch([prompt] * self.width)
        torch.manual_seed(CALIBRATION_SEED)
        self.model.calibrate_batch([frame])
        self._prompt = prompt
        logger.info("FlashRT prompt %r ready at width %d in %.1f s", prompt, self.width, time.monotonic() - started)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self.infer_batch([observation])[0]

    def infer_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        identities, frames = [], []
        for observation in observations:
            observation = dict(observation)
            identity = observation.pop(REQUEST_KEY)
            if not isinstance(identity, dict):
                raise ValueError(f"{REQUEST_KEY} must be a request identity")
            frame = self._images(observation)
            prompt = str(observation["prompt"])
            if self._prompt is None:
                self._prepare(prompt, frame)
            elif prompt != self._prompt:
                raise ValueError(f"this FlashRT server serves {self._prompt!r}; got {prompt!r}")
            identities.append(identity)
            frames.append(frame)
        count = len(frames)
        if count > self.width:
            raise ValueError(f"{count} requests exceed the batch width {self.width}")
        noise = np.stack([deterministic_noise(identity, action_horizon=self.action_horizon, action_dim=self.action_dim)
                          for identity in identities])
        # Padding rows repeat the first request; their outputs are dropped.
        rows = frames + [frames[0]] * (self.width - count)
        row_noise = np.concatenate([noise, np.repeat(noise[:1], self.width - count, axis=0)])
        started = time.monotonic()
        results = self.model.infer_batch(rows, noise=row_noise)
        infer_ms = (time.monotonic() - started) * 1000.0
        return [{
            "actions": validate_actions(result["actions"], self.action_horizon),
            "policy_timing": {"infer_ms": infer_ms},
            RESPONSE_KEY: {**identity, "noise_sha256": hashlib.sha256(noise[index].tobytes()).hexdigest()},
        } for index, (identity, result) in enumerate(zip(identities, results[:count], strict=True))]

    def warm_up(self, example: dict[str, Any]) -> None:
        """Capture the batched pipeline on the example's prompt before clients connect."""
        self.infer_batch([{**example, REQUEST_KEY: {"warm_up": index}} for index in range(self.width)])
