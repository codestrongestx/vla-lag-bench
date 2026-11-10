"""Benchmark adapters exposed to the shared runner."""

from __future__ import annotations

from vlalagbench.benchmarks.base import BenchmarkAdapter


def get_adapter(kind: str) -> BenchmarkAdapter:
    if kind == "libero":
        from vlalagbench.benchmarks.libero.adapter import ADAPTER

        return ADAPTER
    if kind == "dynamicvla":
        from vlalagbench.benchmarks.dynamicvla.adapter import ADAPTER

        return ADAPTER
    raise ValueError(f"unknown benchmark adapter: {kind!r}")
