"""Measure how long pi0.5 takes to return one action chunk at batch 1.

One process loads pi0.5 with one inference stack on one GPU and sends it
recorded policy requests one at a time, back to back: a warm-up (excluded,
its first call reported separately as the cold call) and then the timed calls.
Each call is timed from the request the evaluator would send to the action
chunk being on the host, so it includes every host step of the deployed path
(tokenization, normalization, unnormalization, image upload). The network
call is also timed alone, with the device synchronized on both sides.

Stacks: openpi (``--engine openpi``; JAX or PyTorch, whichever the checkpoint
is) and FlashRT bf16 (``--engine flashrt``, an openpi PyTorch checkpoint).

The run refuses a GPU that other processes use, unless ``--allow-shared-gpu``
(the record then says the GPU was shared, and the numbers are not clean).

Inputs
------
``--frames``  ``scripts/latency/policy_frames.py`` output: 224 px requests
              rebuilt from recorded LIBERO traces, with the chunks the
              recording run returned (the record reports agreement with them).
``--synthetic`` random requests of the same shapes and dtypes; the record says
              ``inputs: synthetic``.

Commands
--------
From the repository root, on a GPU nobody else uses:

    export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4
    S=scripts/latency/policy_latency.py F=FRAMES.npz

The PyTorch checkpoint is openpi's ``pi05_libero`` converted with
``third_party/openpi/examples/convert_jax_model_to_pytorch.py --config_name
pi05_libero --output_path $VLALAGBENCH_CHECKPOINTS/pi05_libero_pytorch``.

pi0.5 openpi PyTorch (openpi's environment with its PyTorch support installed;
torch.compile mode as the policy server, max-autotune-no-cudagraphs):
    PYTHONPATH=src uv run --project third_party/openpi python $S OUT.json --frames $F \\
        --checkpoint $VLALAGBENCH_CHECKPOINTS/pi05_libero_pytorch
pi0.5 openpi JAX:
    PYTHONPATH=src XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run --project third_party/openpi python $S OUT.json \\
        --frames $F --checkpoint gs://openpi-assets/checkpoints/pi05_libero
pi0.5 FlashRT bf16 (a Python environment with FlashRT and openpi; on the
H100, which FlashRT builds no FA2 module for, also set FVK_RTX_FA2=0):
    PYTHONPATH=src:third_party/openpi/src $FLASHRT_PYTHON $S OUT.json --frames $F --engine flashrt \\
        --checkpoint $VLALAGBENCH_CHECKPOINTS/pi05_libero_pytorch

On a shared multi-socket host, pin the run to cores of the GPU's NUMA node
(``taskset -c``); the run holds every thread on the CPUs it started with.

``scripts/latency/latency_table.py OUT.json [...]`` tabulates many runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

SCHEMA_VERSION = 1
# LIBERO's controller consumes one action per 50 ms.
CONTROL_HZ = 20.0
SYNTHETIC_REQUESTS = 16
PACKAGES = ("torch", "jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "flash-rt", "transformers",
            "flash-attn", "numpy", "openpi")


@dataclasses.dataclass
class Loaded:
    """A ready policy and the requests it is timed on."""

    call: Callable[[Any], np.ndarray]  # one request in, its action chunk on the host out
    elements: list[Any]
    model_ms: list[float]  # the network call's times, appended as they happen
    config: dict[str, Any]
    references: list[np.ndarray | None] | None = None  # recorded chunks, to check the path is the real one


def device_sync() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def torch_finish(result: Any) -> Any:
    import torch

    torch.cuda.synchronize()
    return result


def jax_finish(result: Any) -> Any:
    import jax

    return jax.block_until_ready(result)


def timed(function: Callable, finish: Callable[[Any], Any], sink: list[float]) -> Callable:
    """The same call, with the device drained before it starts and after it returns."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        finish(None)
        started = time.perf_counter_ns()
        result = finish(function(*args, **kwargs))
        sink.append((time.perf_counter_ns() - started) / 1e6)
        return result

    return wrapper


def file_record(path: Path, full_hash: bool) -> dict[str, Any]:
    """Sizes of every file, digests of the small ones (all of them with full_hash)."""
    from vlalagbench.files import sha256

    files = [path] if path.is_file() else sorted(file for file in path.rglob("*") if file.is_file())
    manifest = hashlib.sha256()
    digests, total = {}, 0
    for file in files:
        size = file.stat().st_size
        total += size
        name = file.name if path.is_file() else file.relative_to(path).as_posix()
        manifest.update(f"{name}\0{size}\n".encode())
        if full_hash or size < 1 << 20:
            digests[name] = sha256(file)
    record = {"path": str(path), "files": len(files), "bytes": total, "manifest_sha256": manifest.hexdigest(),
              "full_hash": full_hash}
    if len(digests) <= 64:
        record["sha256"] = digests
    return record


def resolve_checkpoint(checkpoint: str) -> str:
    """A local checkpoint with its ``$VARIABLE`` expanded; a remote URI unchanged."""
    if "://" in checkpoint:
        return checkpoint
    from vlalagbench.files import local_path

    return str(local_path(checkpoint))


def checkpoint_record(checkpoint: str, full_hash: bool) -> dict[str, Any]:
    record: dict[str, Any] = {"checkpoint": checkpoint}
    if checkpoint.startswith("gs://"):
        try:
            from openpi.shared import download

            local = Path(download.maybe_download(checkpoint))
        except Exception as error:  # noqa: BLE001 - the record says why it is missing
            record["local_error"] = repr(error)
            return record
    else:
        local = Path(resolve_checkpoint(checkpoint))
    if local.exists():
        record.update(file_record(local, full_hash))
    return record


def gpu_index() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    return visible or "0"


def nvidia_smi(*args: str) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True, timeout=30).stdout


GPU_FIELDS = ("index", "uuid", "name", "driver_version", "pci.bus_id", "pstate", "clocks.sm", "clocks.mem",
              "clocks.max.sm", "clocks.max.mem", "power.draw", "power.limit", "temperature.gpu",
              "utilization.gpu", "memory.used", "memory.total", "persistence_mode")


def cpu_model() -> str | None:
    # Launch-bound stacks scale with the host CPU, so the record names it.
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def gpu_snapshot(own_context: bool = False) -> dict[str, Any]:
    """The GPU's state and the compute processes other than this one.

    In a container nvidia-smi reports host PIDs, so this process may not be
    found by its own PID; once it holds a CUDA context (own_context) it is
    then counted out by number instead.
    """
    values = nvidia_smi("-i", gpu_index(), f"--query-gpu={','.join(GPU_FIELDS)}",
                        "--format=csv,noheader,nounits").strip().split(", ")
    snapshot = dict(zip(GPU_FIELDS, (value.strip() for value in values), strict=True))
    apps = nvidia_smi("--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits")
    processes = [
        {"pid": int(pid), "memory_mib": int(memory)}
        for uuid, pid, memory in (line.split(", ") for line in apps.strip().splitlines() if line.strip())
        if uuid == snapshot["uuid"]
    ]
    others = [process for process in processes if process["pid"] != os.getpid()]
    if own_context and others and len(others) == len(processes):
        others = sorted(others, key=lambda process: process["memory_mib"])[:-1]
        snapshot["own_pid_visible"] = False
    snapshot["processes"] = processes
    snapshot["other_processes"] = others
    snapshot["time"] = time.time()
    # Other tenants' CPU load slows launch-bound stacks even on an idle GPU.
    snapshot["host_loadavg"] = list(os.getloadavg())
    return snapshot


def environment_record() -> dict[str, Any]:
    header = nvidia_smi()
    cuda_driver = next((line.split("CUDA Version:")[1].split()[0] for line in header.splitlines()
                        if "CUDA Version:" in line), None)
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    record: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": packages,
        "cuda_driver_api": cuda_driver,
        "env": {key: os.environ[key] for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS",
                                                 "XLA_FLAGS", "XLA_PYTHON_CLIENT_MEM_FRACTION",
                                                 "XLA_PYTHON_CLIENT_PREALLOCATE", "PYTHONPATH", "FVK_RTX_FA2",
                                                 "TRITON_LIBCUDA_PATH")
                if key in os.environ},
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "git": git_record(),
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        record["torch_cuda"] = torch.version.cuda
        record["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            record["torch_device"] = torch.cuda.get_device_name()
            record["torch_capability"] = list(torch.cuda.get_device_capability())
            record["torch_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    jax = sys.modules.get("jax")
    if jax is not None:
        try:
            device = jax.devices()[0]
            record["jax_device"] = device.device_kind
            record["jax_platform_version"] = device.client.platform_version
        except Exception as error:  # noqa: BLE001
            record["jax_error"] = repr(error)
    return record


def git_record() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {"n": int(array.size), "mean": float(array.mean()), "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "min": float(array.min()), "max": float(array.max()),
            **{f"p{q}": float(np.percentile(array, q)) for q in (50, 90, 95, 99)}}


def frames_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# pi0.5 -----------------------------------------------------------------------


def synthetic_frames(prompt: str, count: int = SYNTHETIC_REQUESTS) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {"image": rng.integers(0, 256, (count, 224, 224, 3), dtype=np.uint8),
            "wrist_image": rng.integers(0, 256, (count, 224, 224, 3), dtype=np.uint8),
            "state": rng.standard_normal((count, 8)),
            "prompt": np.array([prompt] * count),
            "identity": np.array([json.dumps({"latency_request": index}) for index in range(count)])}


def request_elements(frames: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """The policy requests the LIBERO evaluator sends, one per frame row."""
    from vlalagbench.benchmarks.libero.policy import REQUEST_KEY

    return [{
        "observation/image": frames["image"][index],
        "observation/wrist_image": frames["wrist_image"][index],
        "observation/state": frames["state"][index],
        "prompt": str(frames["prompt"][index]),
        REQUEST_KEY: json.loads(str(frames["identity"][index])),
    } for index in range(len(frames["prompt"]))]


def make_policy(engine: str, checkpoint: str, openpi_config: str, pytorch_compile_mode: str | None = None):
    """The single-request policy the LIBERO policy server serves for this engine."""
    from openpi.training import config

    train_config = config.get_config(openpi_config)
    if pytorch_compile_mode is not None:
        # The policy server's setting ("eager" skips torch.compile); JAX
        # checkpoints ignore it.
        mode = None if pytorch_compile_mode == "eager" else pytorch_compile_mode
        model = dataclasses.replace(train_config.model, pytorch_compile_mode=mode)
        train_config = dataclasses.replace(train_config, model=model)
    horizon, dim = train_config.model.action_horizon, train_config.model.action_dim
    if engine == "flashrt":
        from vlalagbench.benchmarks.libero.flashrt_policy import FlashRTPolicy
        return FlashRTPolicy(checkpoint, precision="bf16", action_horizon=horizon, action_dim=dim)
    from openpi.policies import policy_config

    from vlalagbench.benchmarks.libero.policy import SingleRequestPolicy
    return SingleRequestPolicy(policy_config.create_trained_policy(train_config, resolve_checkpoint(checkpoint)),
                               action_horizon=horizon, action_dim=dim)


def load_pi05(args: argparse.Namespace) -> Loaded:
    frames = synthetic_frames(args.prompt) if args.synthetic else dict(np.load(args.frames))
    prompts = sorted(set(frames["prompt"].tolist()))
    if args.engine == "flashrt" and len(prompts) != 1:
        raise SystemExit(f"a FlashRT policy serves one prompt; {args.frames} has {len(prompts)}")
    policy = make_policy(args.engine, args.checkpoint, args.openpi_config, args.pytorch_compile_mode)
    elements = request_elements(frames)
    references = list(frames["recorded_chunk"]) if "recorded_chunk" in frames else None

    model_ms: list[float] = []
    config: dict[str, Any] = {"openpi_config": args.openpi_config}
    if args.engine == "flashrt":
        import flash_rt.frontends.torch.pi05_rtx as frontend

        stack = "flashrt"
        policy.model.infer = timed(policy.model.infer, torch_finish, model_ms)
        config.update(policy.identity(), denoising_steps=getattr(frontend, "NUM_STEPS_DEFAULT", None),
                      model_scope="FlashRT frontend infer (its image upload, CUDA graph replay, readback)")
    else:
        inner = policy.policy
        pytorch = inner._is_pytorch_model
        stack = "openpi-pytorch" if pytorch else "openpi-jax"
        inner._sample_actions = timed(inner._sample_actions, torch_finish if pytorch else jax_finish, model_ms)
        config.update(denoising_steps=inner._sample_kwargs.get("num_steps", 10),
                      model_scope="openpi model.sample_actions (prefix + all denoising steps), inputs on device")
        if pytorch:
            config["pytorch_compile_mode"] = args.pytorch_compile_mode
    from openpi.training import config as openpi_config

    model_config = openpi_config.get_config(args.openpi_config).model
    config.update(
        model="pi05", stack=stack, precision="bf16", action_horizon=model_config.action_horizon,
        action_dim=model_config.action_dim, max_token_len=model_config.max_token_len,
        cameras=[{"name": key, "shape": list(np.shape(elements[0][key])), "dtype": str(elements[0][key].dtype)}
                 for key in ("observation/image", "observation/wrist_image")],
        prompts=prompts,
        checkpoint=checkpoint_record(args.checkpoint, args.hash_checkpoint),
    )

    def call(element: dict) -> np.ndarray:
        return np.asarray(policy.infer(element)["actions"])

    return Loaded(call=call, elements=elements, model_ms=model_ms, config=config, references=references)


# Measurement -----------------------------------------------------------------


def agreement(chunks: list[np.ndarray], references: list[np.ndarray | None]) -> dict[str, Any] | None:
    pairs = [(np.asarray(chunk, dtype=np.float64), np.asarray(reference, dtype=np.float64))
             for chunk, reference in zip(chunks, references, strict=True) if reference is not None]
    pairs = [(chunk, reference) for chunk, reference in pairs if chunk.shape == reference.shape]
    if not pairs:
        return None
    # The last action dimension is the gripper; the arm is the rest.
    diff = np.abs(np.stack([chunk - reference for chunk, reference in pairs]))[..., :-1]
    return {"requests": len(pairs), "arm_mean_abs": float(diff.mean()), "arm_max_abs": float(diff.max()),
            "note": "against the recording run's chunk; differences within the stack's rerun noise are expected"}


class AffinityGuard:
    """Hold every thread of this process on the CPUs it started with.

    Some hosts run numad, which moves large processes between NUMA nodes and
    overrides taskset. Restoring the set between calls (outside the timed
    window) keeps the host work on the CPUs the run was given.
    """

    def __init__(self, cpus: set[int] | None) -> None:
        self.cpus = cpus
        self.restores = 0

    def hold(self) -> None:
        if self.cpus is None or os.sched_getaffinity(0) == self.cpus:
            return
        self.restores += 1
        for task in os.listdir("/proc/self/task"):
            try:
                os.sched_setaffinity(int(task), self.cpus)
            except OSError:
                pass  # a thread that exited meanwhile


def measure(loaded: Loaded, warmup: int, calls: int, guard: AffinityGuard | None = None) -> dict[str, Any]:
    elements = loaded.elements
    e2e, model, outputs, picked = [], [], [], []
    first_ms = None
    for index in range(warmup + calls):
        element = elements[index % len(elements)]
        loaded.model_ms.clear()
        if guard is not None:
            guard.hold()
        started = time.perf_counter_ns()
        chunk = loaded.call(element)
        device_sync()
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if index == 0:
            first_ms = elapsed
        if index >= warmup:
            e2e.append(elapsed)
            if len(loaded.model_ms) == 1:
                model.append(loaded.model_ms[0])
            if index - warmup < len(elements):
                outputs.append(chunk)
                picked.append(index % len(elements))
    references = loaded.references
    return {
        "first_call_ms": first_ms,
        "e2e_ms": e2e,
        "model_ms": model if len(model) == len(e2e) else [],
        "agreement": agreement(outputs, [references[i] for i in picked]) if references else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--frames", type=Path, help="recorded requests (see Inputs)")
    parser.add_argument("--synthetic", action="store_true", help="random inputs of the native shapes")
    parser.add_argument("--prompt", default="put the black bowl on the plate", help="synthetic inputs' instruction")
    parser.add_argument("--checkpoint", required=True, help="a gs:// URI or a local path; $VARIABLES expand")
    parser.add_argument("--hash-checkpoint", action="store_true", help="hash every checkpoint file, weights too")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--calls", type=int, default=500)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--label", help="free text kept in the record")
    parser.add_argument("--engine", choices=["openpi", "flashrt"], default="openpi")
    parser.add_argument("--openpi-config", default="pi05_libero")
    parser.add_argument("--pytorch-compile-mode", default="max-autotune-no-cudagraphs",
                        choices=["max-autotune-no-cudagraphs", "max-autotune", "reduce-overhead", "default", "eager"],
                        help="openpi PyTorch only; the default is the LIBERO policy server's")
    args = parser.parse_args()
    for name in ("out", "frames"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.synthetic == (args.frames is not None):
        parser.error("pass --frames or --synthetic")

    gpu_start = gpu_snapshot()
    start_cpus = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    if gpu_start["other_processes"] and not args.allow_shared_gpu:
        raise SystemExit(f"GPU {gpu_start['index']} is in use by {gpu_start['other_processes']}; "
                         "pick a free GPU or pass --allow-shared-gpu")
    started = time.perf_counter()
    loaded = load_pi05(args)
    load_s = time.perf_counter() - started
    gpu_before = gpu_snapshot(own_context=True)
    guard = AffinityGuard(start_cpus)
    guard.hold()
    result = measure(loaded, args.warmup, args.calls, guard)
    gpu_end = gpu_snapshot(own_context=True)
    e2e = stats(result["e2e_ms"])
    record = {
        "schema_version": SCHEMA_VERSION,
        "label": args.label,
        "model": "pi05",
        "stack": loaded.config["stack"],
        "precision": loaded.config["precision"],
        "batch_size": 1,
        "inputs": "synthetic" if args.synthetic else "recorded",
        "frames": None if args.synthetic else {**frames_record(args.frames), "requests": len(loaded.elements)},
        "transport": "in-process",
        "warmup_calls": args.warmup,
        "timed_calls": args.calls,
        "period_ms": 0.0,
        "control_hz": CONTROL_HZ,
        "load_s": load_s,
        "first_call_ms": result["first_call_ms"],
        "e2e_ms": e2e,
        "model_ms": stats(result["model_ms"]),
        "steps_stale": {key: e2e[key] * CONTROL_HZ / 1000 for key in ("p50", "p95", "p99")},
        "agreement": result["agreement"],
        "config": {key: value for key, value in loaded.config.items() if key not in ("stack", "precision", "model")},
        "cpu": {"model": cpu_model(), "logical_cpus": os.cpu_count(),
                "affinity": None if start_cpus is None else sorted(start_cpus),
                "affinity_restores": guard.restores},
        "gpu": {"start": gpu_start, "before_timing": gpu_before, "end": gpu_end,
                "exclusive": not (gpu_start["other_processes"] or gpu_before["other_processes"]
                                  or gpu_end["other_processes"])},
        "environment": environment_record(),
        "argv": sys.argv,
        "e2e_ms_raw": result["e2e_ms"],
        "model_ms_raw": result["model_ms"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, default=str) + "\n")
    model = record["model_ms"]
    print(f"pi05 {record['stack']} {record['precision']} on {gpu_start['name']}: e2e p50 {e2e['p50']:.1f} "
          f"p95 {e2e['p95']:.1f} p99 {e2e['p99']:.1f} ms"
          + (f"; model p50 {model['p50']:.1f} ms" if model else "")
          + f"; first call {record['first_call_ms'] / 1000:.1f} s; shared GPU: {not record['gpu']['exclusive']}")
    # Leave no GPU memory behind.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
