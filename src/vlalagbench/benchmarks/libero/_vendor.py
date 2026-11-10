"""Import glue for the pinned third-party LIBERO snapshot.

LIBERO is vendored as a git submodule under ``third_party/openpi`` and is not
pip-installed (its declared dependencies pull in far more than the simulator
needs). Instead, mirror what openpi's own LIBERO Dockerfile does: put the
checkout on ``sys.path`` and pre-write LIBERO's config file so importing
``libero`` never prompts interactively and always resolves asset paths inside
this repository.

Use :func:`libero` to access the package after initialization.
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LIBERO_ROOT = (
    REPO_ROOT / "third_party" / "openpi" / "third_party" / "libero"
)


def _configured_libero_root() -> Path:
    return Path(
        os.environ.get("VLALAGBENCH_LIBERO_ROOT", DEFAULT_LIBERO_ROOT)
    ).expanduser().resolve()


LIBERO_ROOT = _configured_libero_root()
LIBERO_CONFIG_DIR = REPO_ROOT / ".libero"


@functools.cache
def libero() -> SimpleNamespace:
    """Return the LIBERO API after its import prerequisites are satisfied."""

    _bootstrap()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    return SimpleNamespace(
        benchmark=benchmark,
        get_libero_path=get_libero_path,
        ControlEnv=ControlEnv,
    )


def _bootstrap() -> None:
    """Make ``import libero`` work against the pinned submodule. Idempotent."""
    if not (LIBERO_ROOT / "libero" / "libero").is_dir():
        raise RuntimeError(
            f"LIBERO submodule not found at {LIBERO_ROOT}. Run:\n"
            "  git submodule update --init third_party/openpi/third_party/libero"
        )
    path = str(LIBERO_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    _ensure_libero_config()


def _ensure_libero_config() -> None:
    """Write our default LIBERO config without touching a user-owned config.

    Must happen before ``import libero.libero``, which reads (or interactively
    creates) the config on import. When this project owns the default config,
    absolute paths are rewritten whenever they become stale.
    """
    if "LIBERO_CONFIG_PATH" in os.environ:
        return
    os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_CONFIG_DIR)
    config_dir = LIBERO_CONFIG_DIR
    pkg = LIBERO_ROOT / "libero" / "libero"
    datasets_dir = REPO_ROOT / "artifacts" / "libero_datasets"
    config = "\n".join(
        [
            f"assets: {pkg / 'assets'}",
            f"bddl_files: {pkg / 'bddl_files'}",
            f"benchmark_root: {pkg}",
            f"datasets: {datasets_dir}",
            f"init_states: {pkg / 'init_files'}",
            "",
        ]
    )
    config_path = config_dir / "config.yaml"
    if not config_path.exists() or config_path.read_text() != config:
        config_dir.mkdir(parents=True, exist_ok=True)
        datasets_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config)
