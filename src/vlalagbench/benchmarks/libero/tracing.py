"""Strict LIBERO metadata validation at the benchmark boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from vlalagbench.simulators.mujoco.tracing import EpisodeTrace, TraceMeta


class LiberoEnvironmentMeta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["libero"]
    problem_folder: Annotated[str, Field(min_length=1)]
    bddl_file: Annotated[str, Field(min_length=1)]
    resolution: Annotated[int, Field(gt=0)]
    seed: Annotated[int, Field(ge=0)]


def parse_libero_environment(meta: TraceMeta) -> LiberoEnvironmentMeta:
    """Validate and return the LIBERO-specific portion of shared trace metadata."""

    return LiberoEnvironmentMeta.model_validate(meta.environment.model_dump(mode="python"))


def load_libero_trace(path: str | Path) -> EpisodeTrace:
    """Load a trace and reject incomplete or misspelled LIBERO environment metadata."""

    trace = EpisodeTrace.load(path)
    parse_libero_environment(trace.meta)
    return trace
