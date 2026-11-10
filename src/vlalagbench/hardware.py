"""Minimal provider-neutral hardware audit."""

from __future__ import annotations

import socket
import subprocess
import os
import time
from collections import Counter

from vlalagbench.config import GPUAllocation, Hardware


def collect_hardware() -> Hardware:
    """Record only the hostname and visible GPU type/count."""

    counts = Counter(_nvidia_gpu_types())
    if not counts:
        raise RuntimeError("no NVIDIA GPUs discovered")
    return Hardware(
        hostname=socket.gethostname(),
        cpu_count=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1),
        gpus=[GPUAllocation(type=gpu_type, count=count) for gpu_type, count in sorted(counts.items())],
    )


def _nvidia_gpu_types(attempts: int = 5) -> list[str]:
    # nvidia-smi can stall for tens of seconds on a busy shared host without
    # persistence mode, so a slow or failed query is retried before giving up.
    for attempt in range(attempts):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError:
            return []
        except (OSError, subprocess.SubprocessError):
            if attempt + 1 < attempts:
                time.sleep(5)
            continue
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return []
