"""Small contract between benchmark-specific evaluators and the shared runner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict

try:
    from typing import NotRequired
except ImportError:  # Python 3.10 in the pinned Isaac Lab environment
    from typing_extensions import NotRequired

if TYPE_CHECKING:
    from vlalagbench.config import ExperimentDocument, RuntimeDocument


class EpisodeOutput(TypedDict):
    task_id: int
    episode: int
    seed: int
    success: bool
    status: NotRequired[int | None]
    steps: int
    control_steps: int
    wait_steps: int
    policy_requests: int
    trace: str
    error: str | None


class BenchmarkAdapter(Protocol):
    def validate(
        self, experiment: ExperimentDocument, runtime: RuntimeDocument
    ) -> None: ...

    def evaluate(
        self,
        experiment: ExperimentDocument,
        runtime: RuntimeDocument,
        artifact_dir: Path,
    ) -> list[EpisodeOutput]: ...
