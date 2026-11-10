"""Checksums and atomic records for completed work."""

import hashlib
import os
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    """Hash a directory's relative file names and contents in canonical order."""
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def local_path(value: str | Path) -> Path:
    """Expand ``$VARIABLE`` and ``~`` in a configured local path, e.g. a checkpoint."""
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    if "$" in expanded:
        raise ValueError(f"{value}: set the environment variables this path uses")
    return Path(expanded)


def atomic_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".record-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
