"""Apply a Torch allocator budget before starting an independent policy process."""

from __future__ import annotations

import runpy
import sys


def main() -> None:
    fraction = float(sys.argv[1])
    if not 0 < fraction <= 1:
        raise ValueError("the CUDA memory fraction must be in (0, 1]")
    import torch

    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    sys.argv = sys.argv[2:]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
