"""Resolve source-config paths without weakening checksum verification."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from vlalagbench.config import SourceConfig


LEGACY_EXPERIMENT_ROOT = Path("configs/experiments")
LIBERO_EXPERIMENT_ROOT = LEGACY_EXPERIMENT_ROOT / "libero"


def resolve_source_config(
    source: SourceConfig,
    repo_root: str | Path,
    *,
    resolved_root: str | Path | None = None,
) -> Path:
    """Return the current path whose bytes match a recorded source checksum."""

    repo_root = Path(repo_root).resolve()
    recorded = Path(source.source)
    candidates = [repo_root / recorded]
    try:
        relative = recorded.relative_to(LEGACY_EXPERIMENT_ROOT)
    except ValueError:
        pass
    else:
        if not relative.parts or relative.parts[0] != "libero":
            candidates.append(repo_root / LIBERO_EXPERIMENT_ROOT / relative)
    if resolved_root is not None:
        candidates.append(Path(resolved_root) / source.resolved)

    for candidate in candidates:
        if candidate.is_file() and _sha256(candidate) == source.sha256:
            return candidate
    raise FileNotFoundError(
        f"no source config matches {source.source!r} with sha256 {source.sha256}"
    )


def read_source_config(
    source: SourceConfig,
    repo_root: str | Path,
    *,
    resolved_root: str | Path | None = None,
    commit: str | None = None,
) -> bytes:
    """Read checksum-verified source bytes from disk or the recorded Git commit."""

    try:
        return resolve_source_config(
            source,
            repo_root,
            resolved_root=resolved_root,
        ).read_bytes()
    except FileNotFoundError:
        if commit is None:
            raise

    completed = subprocess.run(
        ["git", "-C", str(Path(repo_root).resolve()), "show", f"{commit}:{source.source}"],
        check=False,
        capture_output=True,
    )
    if (
        completed.returncode == 0
        and hashlib.sha256(completed.stdout).hexdigest() == source.sha256
    ):
        return completed.stdout
    raise FileNotFoundError(
        f"no source config matches {source.source!r} with sha256 {source.sha256}"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
