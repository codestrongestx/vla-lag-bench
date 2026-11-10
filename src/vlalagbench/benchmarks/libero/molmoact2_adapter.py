"""MolmoAct2 public-checkpoint adapter for LIBERO's simulator conventions.

Observations follow the pinned reference evaluation (allenai/lerobot branch
``molmoact2-hf-inference``, LeRobot v0.5.1): both 256 px cameras rotated by
180 degrees and passed through LeRobot's float32 [0, 1] round trip, and an
8-D state of end-effector position, axis-angle and gripper joints. Actions come
back in robot scale from the checkpoint's own ``norm_stats.json`` (tag
``libero``), so they go to the simulator unchanged.

Batched inference runs the steps of ``predict_action`` (continuous mode, no
depth reasoning) for many requests in one model call. Each request draws its
initial flow noise from its own generator seeded from the request identity,
exactly as a lone ``predict_action`` call with that generator would.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vlalagbench.benchmarks.libero.policy import REQUEST_KEY, RESPONSE_KEY, RNG_SCHEME, request_seed, validate_actions
from vlalagbench.files import sha256

CHECKPOINT_REVISION = "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
REFERENCE_EVAL = "allenai/lerobot@28b4f721 (molmoact2-hf-inference)"
NORM_TAG = "libero"
ACTION_CHUNK_STEPS = 10
PRECISIONS = ("float32", "tf32", "bfloat16")
_CALL_LOCK = threading.Lock()


def lerobot_image(frame: np.ndarray) -> np.ndarray:
    """Rotate 180 degrees and replay LeRobot's uint8 -> float [0, 1] -> uint8 path."""
    frame = np.asarray(frame)
    if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
        raise ValueError("MolmoAct2 LIBERO requires native 256x256 uint8 RGB cameras")
    scaled = frame[::-1, ::-1].astype(np.float32) / np.float32(255)
    return np.clip(scaled * np.float32(255.0), 0, 255).astype(np.uint8)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """LeRobot's ``LiberoProcessorStep._quat2axisangle`` in float32."""
    import torch

    q = torch.as_tensor(np.asarray(quat, dtype=np.float32)).reshape(1, 4)
    w = q[:, 3].clamp(-1.0, 1.0)
    den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    angle = 2.0 * torch.acos(w)
    axis = q[:, :3] / den.unsqueeze(1)
    result = axis * angle.unsqueeze(1)
    return torch.where(den.unsqueeze(1) > 1e-10, result, torch.zeros_like(result))[0].numpy()


def format_observation(obs: dict[str, Any], language: str) -> dict[str, Any]:
    state = np.concatenate((
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        quat2axisangle(obs["robot0_eef_quat"]),
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    )).astype(np.float32)
    return {
        "images": np.stack([lerobot_image(obs["agentview_image"]),
                            lerobot_image(obs["robot0_eye_in_hand_image"])]),
        "state": state,
        "task": str(language),
    }


class _NoiseTorch:
    """Stand-in for ``torch`` inside MolmoAct2's modeling module.

    Its one ``torch.randn`` call per model pass draws the initial flow noise
    for the whole batch; each row is drawn from a fresh CUDA generator at that
    request's seed, so a row equals a lone call with ``generator=seed``.
    """

    def __init__(self, torch: Any, seeds: Sequence[int]) -> None:
        self._torch = torch
        self._seeds = list(seeds)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._torch, name)

    def randn(self, size: Any, *, device: Any = None, dtype: Any = None, generator: Any = None) -> Any:
        if not self._seeds or generator is not None or size[0] != len(self._seeds):
            raise RuntimeError(f"unexpected MolmoAct2 noise draw: size={size}")
        torch, seeds, self._seeds = self._torch, self._seeds, []
        rows = [torch.randn((1, *size[1:]), device=device, dtype=dtype,
                            generator=torch.Generator(device=device).manual_seed(seed)) for seed in seeds]
        return torch.cat(rows)


def _batch_attention_bias(model: Any) -> None:
    """Build the prefill attention bias one row at a time.

    The checkpoint's ``_build_native_attention_bias`` merges the image tokens'
    bidirectional mask into a causal mask that has batch size one, so it fails
    for more than one prompt. Each row's bias is the one a lone request gets.
    """
    import torch

    build = model._build_native_attention_bias

    def per_row(*, inputs_embeds, attention_mask, token_type_ids, past_key_values):
        if inputs_embeds.shape[0] == 1 or token_type_ids is None:
            return build(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                         token_type_ids=token_type_ids, past_key_values=past_key_values)
        return torch.cat([
            build(inputs_embeds=inputs_embeds[row:row + 1],
                  attention_mask=None if attention_mask is None else attention_mask[row:row + 1],
                  token_type_ids=token_type_ids[row:row + 1], past_key_values=past_key_values)
            for row in range(inputs_embeds.shape[0])
        ])

    model._build_native_attention_bias = per_row


def _graph_per_shape(manager: Any) -> None:
    """Keep one flow CUDA graph per input shape instead of only the last.

    The checkpoint's manager recaptures whenever the batch shape changes; with
    one graph per padded batch size, every size replays its own graph.
    """
    helpers = sys.modules[type(manager).__module__]
    graphs: dict[Any, Any] = {}

    def run_action_flow(inputs: Any, steps: int, run_loop: Any) -> Any:
        key = helpers._cuda_graph_key(inputs, steps)
        cache = graphs.get(key)
        if cache is None:
            static_inputs = helpers._clone_static_inputs(inputs)
            graph, output = helpers._capture_cuda_graph(
                lambda: run_loop(static_inputs, steps), inputs.trajectory.device,
                after_warmup=lambda: static_inputs.trajectory.copy_(inputs.trajectory),
            )
            cache = graphs[key] = helpers._ActionFlowCudaGraph(
                key=key, graph=graph, static_inputs=static_inputs, output=output)
        else:
            helpers._copy_inputs_(cache.static_inputs, inputs)
        cache.graph.replay()
        return cache.output.clone()

    manager.run_action_flow = run_action_flow


class MolmoAct2BatchPolicy:
    """Run several MolmoAct2 requests through one forward pass."""

    def __init__(self, checkpoint: Path, *, precision: str, max_batch_size: int, batch_padding: str,
                 cuda_graph: bool = True, num_steps: int | None = None) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
        torch.backends.cudnn.allow_tf32 = precision == "tf32"
        self._torch = torch
        self.precision = precision
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, trust_remote_code=True, dtype=torch.float32, low_cpu_mem_usage=True,
        ).to("cuda").eval().requires_grad_(False)
        self._module = sys.modules[type(self.model).__module__]
        _batch_attention_bias(self.model.model)
        self.stats = self.model._get_robot_stats()
        self.metadata = self.stats.get_metadata(NORM_TAG)
        self.action_dim = int(self.stats.get_action_dim(NORM_TAG))
        self.horizon = int(self.stats.get_action_horizon(NORM_TAG))
        if self.horizon != ACTION_CHUNK_STEPS or int(self.stats.get_n_action_steps(NORM_TAG)) != ACTION_CHUNK_STEPS:
            raise ValueError("MolmoAct2-LIBERO must return 10-step chunks")
        self.cuda_graph = cuda_graph
        self.num_steps = num_steps
        self._rows = max_batch_size if batch_padding == "fixed" else 0
        self._max_size = max_batch_size
        # Without fixed padding, passes pad to the next of a few sizes so the
        # flow graphs stay few; each size keeps its own graph.
        self._sizes = [self._rows] if self._rows else sorted(
            {size for size in (8, 16, 24) if size < max_batch_size} | {max_batch_size})
        _graph_per_shape(self.model.model.action_cuda_graph_manager)
        # cuBLAS handles and workspaces are per thread. One thread runs every
        # pass, so they exist after warm-up, before clients' renderers fill
        # the GPU.
        self._gpu_thread = ThreadPoolExecutor(1, thread_name_prefix="molmoact2-gpu")

    def _text(self, observation: dict[str, Any]) -> str:
        module = self._module
        state = np.asarray(self.stats.normalize_state(observation["state"], NORM_TAG), dtype=np.float32)
        return module._build_robot_text(
            task=module._normalize_question_text(str(observation["task"])),
            style="robot_action",
            discrete_state_string=module._build_discrete_state_string(state, int(self.model.config.num_state_tokens)),
            setup_type=str(self.metadata.get("setup_type", "") or ""),
            control_mode=str(self.metadata.get("control_mode", "") or ""),
            add_setup_tokens=bool(self.model.config.add_setup_tokens),
            add_control_tokens=bool(self.model.config.add_control_tokens),
            num_images=len(observation["images"]),
        )

    def prepare(self, observations: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """CPU half of a pass: prompts, image crops and padding rows."""
        identities, rows = [], []
        for observation in observations:
            clean = dict(observation)
            identity = clean.pop(REQUEST_KEY, None)
            if not isinstance(identity, dict):
                raise ValueError(f"{REQUEST_KEY} must be a request identity")
            identities.append(identity)
            rows.append(clean)
        seeds = [request_seed(identity) for identity in identities]
        target = next(size for size in self._sizes if size >= len(rows))
        padded = rows + [rows[0]] * (target - len(rows))
        inputs = self.processor(text=[self._text(row) for row in padded],
                                images=[list(row["images"]) for row in padded], return_tensors="pt")
        if inputs["input_ids"].shape[0] != len(padded):
            raise RuntimeError("processor did not return one prompt per request")
        return {"identities": identities, "seeds": seeds + [seeds[0]] * (len(padded) - len(rows)),
                "inputs": {key: value.pin_memory() if hasattr(value, "pin_memory") else value
                           for key, value in inputs.items()}}

    def execute(self, prepared: dict[str, Any]) -> list[dict[str, Any]]:
        """GPU half of a pass, always on the policy's GPU thread."""
        return self._gpu_thread.submit(self._execute, prepared).result()

    def _execute(self, prepared: dict[str, Any]) -> list[dict[str, Any]]:
        torch, model = self._torch, self.model
        seeds = prepared["seeds"]
        inputs = model._drop_trivial_attention_mask(model._move_inputs_to_device(prepared["inputs"], model.device))
        if "attention_mask" in inputs:
            raise RuntimeError("batched MolmoAct2 prompts must share one length")
        pad = model._build_action_dim_is_pad(action_dim=self.action_dim, max_action_dim=int(model.config.max_action_dim),
                                             batch_size=len(seeds), device=model.device)
        # No autocast cast cache: the flow CUDA graph must read the weights
        # themselves, not cached casts that are freed when autocast exits.
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.precision == "bfloat16",
                                  cache_enabled=False)
        with _CALL_LOCK, torch.inference_mode(), autocast:
            model.model.action_cuda_graph_manager.set_enabled(self.cuda_graph)
            model.depth_decode_cuda_graph_manager.set_enabled(self.cuda_graph)
            original = self._module.torch
            self._module.torch = _NoiseTorch(original, seeds)
            try:
                actions = model.model.generate_actions_from_inputs(
                    **inputs, action_dim_is_pad=pad, action_horizon=self.horizon, num_steps=self.num_steps,
                )
            finally:
                self._module.torch = original
            actions = model._slice_action_chunk(model._slice_action_dim(actions, self.action_dim),
                                                int(model.config.n_obs_steps), self.horizon)
            actions = torch.as_tensor(self.stats.unnormalize_action(actions, NORM_TAG)).to(dtype=torch.float32)
            # Poll instead of letting the device-to-host copy spin a core.
            done = torch.cuda.Event()
            done.record()
            while not done.query():
                time.sleep(0.0005)
            actions = actions.cpu().numpy()
        outputs = []
        for index, identity in enumerate(prepared["identities"]):
            chunk = validate_actions(actions[index], self.horizon)
            outputs.append({"actions": chunk, RESPONSE_KEY: {**identity, "seed": seeds[index], "rng_scheme": RNG_SCHEME}})
        return outputs

    def infer_batch(self, observations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.execute(self.prepare(observations))

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self.infer_batch([observation])[0]

    def warm_up(self, examples: Sequence[dict[str, Any]]) -> None:
        """Capture the flow graph before clients connect.

        The graph's shape includes the prompt length, so the last example
        should carry the instruction the clients will send.
        """
        for example in examples:
            for size in self._sizes:
                for _ in range(2):
                    self.infer_batch([{**example, REQUEST_KEY: {"warm_up": index}} for index in range(size)])


def task_instructions(suite: str, task_ids: Sequence[int]) -> list[str]:
    """LIBERO's instruction for each task, as ``libero.libero.benchmark`` derives it."""
    import runpy

    root = Path(__file__).resolve().parents[4] / "third_party/openpi/third_party/libero/libero/libero"
    names = runpy.run_path(str(root / "benchmark/libero_suite_task_map.py"))["libero_task_map"][suite]
    return [" ".join(names[task_id].split("_")) for task_id in task_ids]


def checkpoint_path(model: Any) -> Path:
    root = Path(model.checkpoint).expanduser()
    if root.is_dir():
        return root
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model.checkpoint, revision=model.revision))


def checkpoint_identity(model: Any, path: Path) -> dict[str, Any]:
    files = [p for p in sorted(path.iterdir()) if p.is_file() and p.suffix in (".json", ".safetensors", ".py", ".jinja")]
    digests = {file.name: sha256(file) for file in files}
    if "norm_stats.json" not in digests or "modeling_molmoact2.py" not in digests:
        raise ValueError("MolmoAct2 checkpoint must include norm_stats.json and its modeling code")
    return {**model.model_dump(), "files_sha256": digests, "action_chunk_steps": ACTION_CHUNK_STEPS,
            "reference_eval": REFERENCE_EVAL, "norm_tag": NORM_TAG,
            "model_config": json.loads((path / "config.json").read_text())}
