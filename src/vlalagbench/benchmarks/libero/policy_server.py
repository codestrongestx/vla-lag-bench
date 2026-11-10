"""Serve OpenPI through its single-request API with audited identity metadata."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import importlib.metadata
import os
from pathlib import Path
from typing import Any

from vlalagbench.config import Experiment, Gr00tModelSpec, ModelSpec, MolmoAct2ModelSpec, Runtime, load_experiment, load_runtime
from vlalagbench.files import atomic_text, local_path
from vlalagbench.benchmarks.libero.policy import SingleRequestPolicy


# This key is part of the client/server wire protocol and must remain stable.
METADATA_KEY = "mujoco_dynamic"


def validate_runtime(runtime: Runtime) -> None:
    """Keep historical records readable without executing their retired protocols."""
    if (
        runtime.policy_server.implementation not in ("single", "dynamic_batching")
        or runtime.environment.reuse_task_envs
        or runtime.environment.initial_state_strategy != "cycle_official"
    ):
        raise ValueError(
            "historical LIBERO runtimes are read-only; use configs/runtime/libero.yaml "
            "or a schema-2 study for single-request or batched inference and fresh official-state episodes"
        )


def build_metadata(
    openpi_config: str,
    checkpoint: str,
    action_chunk_steps: int,
    *,
    pytorch_compile_mode: str = "max-autotune-no-cudagraphs",
    deterministic_ops: bool = False,
    batching: dict[str, Any] | None = None,
    engine: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        METADATA_KEY: {
            "openpi_config": openpi_config,
            "checkpoint": checkpoint,
            "action_chunk_steps": action_chunk_steps,
            "implementation": "single",
            "pytorch_compile_mode": pytorch_compile_mode,
            "max_batch_size": 1,
            "max_wait_ms": 0.0,
            "batch_padding": "none",
            "deterministic_ops": deterministic_ops,
            **(batching or {}),
            **(engine or {}),
        }
    }


def engine_identity(model: Any) -> dict[str, Any] | None:
    """The engine fields a client expects; None for openpi's own model."""
    if not isinstance(model, ModelSpec) or model.engine == "openpi":
        return None
    return {"engine": model.engine, "precision": model.precision}


def batching_identity(server: Any) -> dict[str, Any] | None:
    """How a batching server shapes requests; None for single-request serving."""
    if server.implementation != "dynamic_batching":
        return None
    return {"implementation": "dynamic_batching", "max_batch_size": server.max_batch_size,
            "max_wait_ms": float(server.max_wait_ms), "batch_padding": server.batch_padding}


def validate_metadata(
    metadata: dict[str, Any],
    *,
    model: Any = None,
    openpi_config: str,
    checkpoint: str,
    action_chunk_steps: int,
    pytorch_compile_mode: str = "max-autotune-no-cudagraphs",
    deterministic_ops: bool = False,
    batching: dict[str, Any] | None = None,
) -> None:
    expected = build_metadata(
        openpi_config,
        checkpoint,
        action_chunk_steps,
        pytorch_compile_mode=pytorch_compile_mode,
        deterministic_ops=deterministic_ops,
        batching=batching,
        engine=engine_identity(model),
    )[METADATA_KEY]
    if isinstance(model, Gr00tModelSpec):
        expected = {
            **model.model_dump(), "action_chunk_steps": action_chunk_steps,
            "implementation": "single", "precision": "bfloat16", "inference": "eager",
            "max_batch_size": 1, "deterministic_ops": deterministic_ops, **(batching or {}),
        }
    if isinstance(model, MolmoAct2ModelSpec):
        from vlalagbench.benchmarks.libero.policy import RNG_SCHEME
        expected = {
            **model.model_dump(), "action_chunk_steps": action_chunk_steps, "implementation": "single",
            "max_batch_size": 1, "deterministic_ops": deterministic_ops, "rng_scheme": RNG_SCHEME, **(batching or {}),
        }
    actual = metadata.get(METADATA_KEY)
    if not isinstance(actual, dict):
        raise RuntimeError(
            "policy server identity is missing; start it with "
            "src/vlalagbench/benchmarks/libero/policy_server.py"
        )
    if isinstance(model, MolmoAct2ModelSpec):
        files = actual.get("files_sha256", {})
        if not isinstance(files, dict) or "norm_stats.json" not in files or "modeling_molmoact2.py" not in files:
            raise RuntimeError("policy server handshake failed: checkpoint file digests are missing")
    if isinstance(model, Gr00tModelSpec):
        from vlalagbench.benchmarks.libero.gr00t_adapter import UPSTREAM_REVISION
        from vlalagbench.benchmarks.libero.policy import RNG_SCHEME
        expected.update(upstream_revision=UPSTREAM_REVISION, rng_scheme=RNG_SCHEME)
        files = actual.get("files_sha256", {})
        if not isinstance(files, dict) or not files or any(
            not isinstance(value, str) or len(value) != 64 for value in files.values()
        ):
            raise RuntimeError("policy server handshake failed: checkpoint file digests are missing")
        processors = actual.get("backbone_processors_sha256", {})
        if not isinstance(processors, dict) or not all(
            any(isinstance(key, str) and key.startswith(name + "/") for key in processors)
            for name in ("observation", "collator")
        ) or any(not isinstance(value, str) or len(value) != 64 for value in processors.values()):
            raise RuntimeError("policy server handshake failed: loaded backbone processor digests are missing")
    mismatches = [
        f"{field}: expected {value!r}, got {actual.get(field)!r}"
        for field, value in expected.items()
        if actual.get(field) != value
    ]
    if "engine" not in expected and actual.get("engine", "openpi") != "openpi":
        # Servers from before engines existed carry no engine field.
        mismatches.append(f"engine: expected 'openpi', got {actual['engine']!r}")
    if mismatches:
        raise RuntimeError("policy server handshake failed: " + "; ".join(mismatches))


def _warm_up(policy: Any, action_horizon: int | None = None) -> None:
    """Compile the model before any client connects.

    The first inference triggers minutes of JIT compilation. Serving it lazily
    blocks the websocket event loop, so every connected worker's keepalive
    ping times out and asynchronous episodes run without actions meanwhile.
    """
    import time

    from openpi.policies import libero_policy

    started = time.monotonic()
    result = policy.infer(libero_policy.make_libero_example())
    if action_horizon is not None:
        from vlalagbench.benchmarks.libero.policy import validate_actions
        validate_actions(result["actions"], action_horizon)
    logging.getLogger(__name__).info("policy warm-up took %.1f s", time.monotonic() - started)


def serve(openpi_config: str, checkpoint: str, runtime: Runtime, identity_path: Path | None = None) -> None:
    validate_runtime(runtime)
    from openpi.policies import policy_config
    from openpi.training import config

    server = runtime.policy_server
    train_config = config.get_config(openpi_config)
    train_config = dataclasses.replace(
        train_config,
        model=dataclasses.replace(
            train_config.model,
            pytorch_compile_mode=server.pytorch_compile_mode,
        ),
    )
    # Remote checkpoints (gs://) load as given; local ones may name $VLALAGBENCH_CHECKPOINTS.
    source = checkpoint if "://" in checkpoint else str(local_path(checkpoint))
    policy = policy_config.create_trained_policy(train_config, source)
    _warm_up(policy, train_config.model.action_horizon)
    batching = batching_identity(server)
    if batching is not None:
        import time

        from openpi.policies import libero_policy
        from vlalagbench.benchmarks.libero.batching import OpenPIBatchPolicy

        batch_policy = OpenPIBatchPolicy(
            policy, action_horizon=train_config.model.action_horizon, action_dim=train_config.model.action_dim,
            max_batch_size=server.max_batch_size, batch_padding=server.batch_padding,
        )
        started = time.monotonic()
        batch_policy.warm_up(libero_policy.make_libero_example())
        logging.getLogger(__name__).info("batch shapes compiled in %.1f s", time.monotonic() - started)
    metadata = dict(policy.metadata)
    metadata.update(
        build_metadata(
            openpi_config,
            checkpoint,
            train_config.model.action_horizon,
            pytorch_compile_mode=server.pytorch_compile_mode,
            deterministic_ops=server.deterministic_ops,
            batching=batching,
        )
    )
    if identity_path is not None:
        import yaml

        packages = {}
        for package in ("jax", "jaxlib", "torch", "numpy"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        atomic_text(identity_path, yaml.safe_dump({"policy": metadata[METADATA_KEY], "packages": packages,
                    "xla_flags": os.environ.get("XLA_FLAGS", "")}, sort_keys=True))
    if batching is not None:
        from vlalagbench.benchmarks.libero.batching import BatchedWebsocketPolicyServer

        BatchedWebsocketPolicyServer(
            batch_policy, host="0.0.0.0", port=server.port, metadata=metadata,
            max_batch_size=server.max_batch_size, max_wait_ms=server.max_wait_ms,
        ).serve_forever()
        return
    from openpi.serving import websocket_policy_server

    websocket_policy_server.WebsocketPolicyServer(
        policy=SingleRequestPolicy(
            policy, action_horizon=train_config.model.action_horizon,
            action_dim=train_config.model.action_dim,
        ),
        host="0.0.0.0",
        port=server.port,
        metadata=metadata,
    ).serve_forever()


def serve_flashrt(model: ModelSpec, runtime: Runtime, prompt: str, identity_path: Path | None = None) -> None:
    """Serve an openpi checkpoint through FlashRT, warmed on the study's prompt."""
    import time

    import yaml
    from openpi.policies import libero_policy
    from openpi.training import config

    from vlalagbench.benchmarks.libero.flashrt_policy import FlashRTBatchPolicy, FlashRTPolicy, calibration_frames
    from vlalagbench.benchmarks.libero.policy import REQUEST_KEY

    validate_runtime(runtime)
    server = runtime.policy_server
    batching = batching_identity(server)
    train_config = config.get_config(model.openpi_config)
    started = time.monotonic()
    example = libero_policy.make_libero_example()
    example.update(prompt=prompt, **{REQUEST_KEY: {"warm_up": True}})
    if batching is None:
        calibration = calibration_frames(Path(model.calibration), prompt) if model.calibration is not None else None
        policy = FlashRTPolicy(model.checkpoint, precision=model.precision,
                               action_horizon=train_config.model.action_horizon,
                               action_dim=train_config.model.action_dim, calibration=calibration)
        policy.infer(example)
    else:
        policy = FlashRTBatchPolicy(model.checkpoint, precision=model.precision,
                                    action_horizon=train_config.model.action_horizon,
                                    action_dim=train_config.model.action_dim,
                                    max_batch_size=batching["max_batch_size"], batch_padding=batching["batch_padding"])
        policy.warm_up(example)
    logging.getLogger(__name__).info("flashrt %s warm-up took %.1f s", model.precision, time.monotonic() - started)
    metadata = build_metadata(
        model.openpi_config, model.checkpoint, train_config.model.action_horizon,
        pytorch_compile_mode=server.pytorch_compile_mode, deterministic_ops=server.deterministic_ops,
        batching=batching, engine=engine_identity(model),
    )
    metadata[METADATA_KEY].update(policy.identity())
    if identity_path is not None:
        packages = {}
        for package in ("torch", "numpy", "flash-rt"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        atomic_text(identity_path, yaml.safe_dump({"policy": metadata[METADATA_KEY], "packages": packages,
                                                   "prompt": prompt}, sort_keys=True))
    if batching is not None:
        from vlalagbench.benchmarks.libero.batching import BatchedWebsocketPolicyServer

        BatchedWebsocketPolicyServer(
            policy, host="0.0.0.0", port=server.port, metadata=metadata,
            max_batch_size=batching["max_batch_size"], max_wait_ms=batching["max_wait_ms"],
        ).serve_forever()
        return
    from openpi.serving import websocket_policy_server

    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=server.port, metadata=metadata,
    ).serve_forever()


def serve_gr00t(model: Gr00tModelSpec, runtime: Runtime, action_chunk_steps: int, identity_path: Path | None = None) -> None:
    import time
    import numpy as np
    import torch
    import yaml
    from vlalagbench.benchmarks.libero.gr00t_adapter import (
        Gr00tRequestPolicy, checkpoint_path, checkpoint_identity, check_backbone_access,
        format_observation, backbone_processor_identity,
    )
    from vlalagbench.benchmarks.libero.policy import REQUEST_KEY, validate_actions

    validate_runtime(runtime)
    path = checkpoint_path(model)
    backbones = check_backbone_access(path)
    upstream = Path(__file__).resolve().parents[4] / "third_party/Isaac-GR00T"
    identity = checkpoint_identity(model, path, upstream)
    identity["backbones"] = backbones
    identity["deterministic_ops"] = runtime.policy_server.deterministic_ops
    batching = batching_identity(runtime.policy_server)
    identity.update(batching or {})
    torch.use_deterministic_algorithms(runtime.policy_server.deterministic_ops)
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    policy = Gr00tRequestPolicy(Gr00tPolicy(model.embodiment, str(path), device="cuda:0"))
    identity["backbone_processors_sha256"] = backbone_processor_identity(policy.policy)
    example = format_observation({
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0., 0., 0., 1.]),
        "robot0_gripper_qpos": np.zeros(2),
    }, "pick up the object")
    example[REQUEST_KEY] = {"policy_seed": 0, "task_id": 0, "episode_id": 0, "request_id": 0}
    validate_actions(policy.infer(example)["actions"], action_chunk_steps)
    if batching is not None:
        from vlalagbench.benchmarks.libero.batching import Gr00tBatchPolicy

        batch_policy = Gr00tBatchPolicy(policy.policy, max_batch_size=batching["max_batch_size"],
                                        batch_padding=batching["batch_padding"])
        batch_policy.warm_up(example)
    torch.cuda.synchronize()
    logging.info("GR00T load/warm-up %.3f s; peak allocated VRAM %.3f GiB",
                 time.monotonic() - started, torch.cuda.max_memory_allocated() / 2**30)
    packages = {}
    for package in (
        "torch", "torchvision", "transformers", "diffusers", "numpy", "flash-attn",
        "albumentations", "albucore", "pillow", "opencv-python-headless", "scipy",
        "tokenizers", "triton", "safetensors", "gr00t", "huggingface-hub",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    import sys
    identity["packages"] = packages
    identity["python"] = sys.version.split()[0]
    identity["cuda"] = torch.version.cuda
    identity["cudnn"] = torch.backends.cudnn.version()
    identity["matmul_precision"] = torch.get_float32_matmul_precision()
    if identity_path is not None:
        atomic_text(identity_path, yaml.safe_dump({"policy": identity, "packages": packages}, sort_keys=True))
    if batching is not None:
        from vlalagbench.benchmarks.libero.batching import BatchedWebsocketPolicyServer

        BatchedWebsocketPolicyServer(
            batch_policy, host="0.0.0.0", port=runtime.policy_server.port, metadata={METADATA_KEY: identity},
            max_batch_size=batching["max_batch_size"], max_wait_ms=batching["max_wait_ms"],
        ).serve_forever()
        return
    from openpi.serving import websocket_policy_server
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=runtime.policy_server.port,
        metadata={METADATA_KEY: identity},
    ).serve_forever()


def serve_molmoact2(model: MolmoAct2ModelSpec, runtime: Runtime, action_chunk_steps: int,
                    instructions: list[str], identity_path: Path | None = None) -> None:
    import sys
    import time
    import numpy as np
    import torch
    import yaml
    from vlalagbench.benchmarks.libero.batching import BatchedWebsocketPolicyServer, PipelinedBatcher
    from vlalagbench.benchmarks.libero.molmoact2_adapter import (
        MolmoAct2BatchPolicy, checkpoint_identity, checkpoint_path, format_observation,
    )
    from vlalagbench.benchmarks.libero.policy import REQUEST_KEY, RNG_SCHEME, validate_actions

    validate_runtime(runtime)
    server = runtime.policy_server
    batching = batching_identity(server)
    if batching is None:
        batching_args = {"max_batch_size": 1, "batch_padding": "none"}
    else:
        batching_args = {"max_batch_size": batching["max_batch_size"], "batch_padding": batching["batch_padding"]}
    path = checkpoint_path(model)
    identity = checkpoint_identity(model, path)
    identity.update({"implementation": "single", "max_batch_size": 1, "deterministic_ops": server.deterministic_ops,
                     "rng_scheme": RNG_SCHEME, "inference": "cuda_graph", **(batching or {})})
    torch.use_deterministic_algorithms(server.deterministic_ops)
    started = time.monotonic()
    policy = MolmoAct2BatchPolicy(path, precision=model.precision, **batching_args)
    examples = [format_observation({
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0., 0., 0., 1.]),
        "robot0_gripper_qpos": np.zeros(2),
    }, instruction) for instruction in instructions]
    validate_actions(policy.infer({**examples[0], REQUEST_KEY: {"warm_up": 0}})["actions"], action_chunk_steps)
    policy.warm_up(examples)
    torch.cuda.synchronize()
    logging.info("MolmoAct2 load/warm-up %.3f s; peak allocated VRAM %.3f GiB",
                 time.monotonic() - started, torch.cuda.max_memory_allocated() / 2**30)
    packages = {}
    for package in ("torch", "torchvision", "transformers", "accelerate", "numpy", "pillow",
                    "huggingface-hub", "safetensors", "tokenizers", "einops"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    identity.update(packages=packages, python=sys.version.split()[0], cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version())
    if identity_path is not None:
        atomic_text(identity_path, yaml.safe_dump({"policy": identity, "packages": packages}, sort_keys=True))
    # No keepalive pings: a client blocked in an EGL call holds its GIL and
    # cannot answer, which closed connections mid-episode.
    batcher = PipelinedBatcher(policy.prepare, policy.execute, max_batch_size=batching_args["max_batch_size"],
                               max_wait_ms=0.0 if batching is None else batching["max_wait_ms"])
    BatchedWebsocketPolicyServer(policy, host="0.0.0.0", port=server.port, metadata={METADATA_KEY: identity},
                                 max_batch_size=batching_args["max_batch_size"],
                                 max_wait_ms=0.0 if batching is None else batching["max_wait_ms"],
                                 batcher=batcher, ping_interval=None).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--openpi-config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--identity-out", type=Path)
    parser.add_argument("--prompt", help="the task instruction a FlashRT server is warmed on")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    runtime = load_runtime(args.runtime)
    # Replace an inherited value so the recorded setting actually controls XLA.
    flags = [flag for flag in os.environ.get("XLA_FLAGS", "").split() if not flag.startswith("--xla_gpu_deterministic_ops")]
    flags.append(f"--xla_gpu_deterministic_ops={str(runtime.policy_server.deterministic_ops).lower()}")
    os.environ["XLA_FLAGS"] = " ".join(flags)
    if args.experiment:
        if args.openpi_config or args.checkpoint:
            parser.error("--experiment cannot be combined with --openpi-config or --checkpoint")
        experiment = load_experiment(args.experiment)
        if not isinstance(experiment, Experiment):
            parser.error("policy-server --experiment requires a LIBERO experiment")
        if isinstance(experiment.model, Gr00tModelSpec):
            serve_gr00t(experiment.model, runtime, experiment.inference.action_chunk_steps, args.identity_out)
            return
        if isinstance(experiment.model, MolmoAct2ModelSpec):
            from vlalagbench.benchmarks.libero.molmoact2_adapter import task_instructions
            task_ids = experiment.benchmark.task_ids
            if task_ids == "all":
                parser.error("MolmoAct2 serving needs explicit task_ids to warm up its prompt shapes")
            serve_molmoact2(experiment.model, runtime, experiment.inference.action_chunk_steps,
                            task_instructions(experiment.benchmark.suite, task_ids), args.identity_out)
            return
        if engine_identity(experiment.model) is not None:
            if not args.prompt:
                parser.error("a FlashRT server needs --prompt")
            serve_flashrt(experiment.model, runtime, args.prompt, args.identity_out)
            return
        args.openpi_config, args.checkpoint = experiment.model.openpi_config, experiment.model.checkpoint
    elif not args.openpi_config or not args.checkpoint:
        parser.error("use --experiment or both --openpi-config and --checkpoint")
    serve(args.openpi_config, args.checkpoint, runtime, args.identity_out)


if __name__ == "__main__":
    main()
