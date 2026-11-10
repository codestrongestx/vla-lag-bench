"""Only a study's conclusion is committed under results/: no row-level data."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {".md", ".yaml", ".json", ".csv", ".png"}
MAX_FIGURE_BYTES = 500 * 1024
MAX_TEXT_BYTES = 100 * 1024  # a full study record is under 80 KB; a receipt list is not
MAX_CSV_ROWS = 200
# results/<suite>/<YYYYMMDD>_<task>/<model>[-<variant>]/; see results/README.md.
TASK_FOLDER = re.compile(r"\d{8}_[a-z0-9]+(?:-[a-z0-9]+)*")
MODEL_FOLDER = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _tracked_results() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "results"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return [ROOT / p for p in out.decode().split("\0") if p]


def test_results_hold_only_conclusions() -> None:
    offenders = []
    for path in _tracked_results():
        if path.name == ".gitkeep":
            continue
        if path.suffix not in ALLOWED:
            offenders.append(f"{path}: {path.suffix or 'no extension'} is not a conclusion file")
        elif path.stat().st_size > (MAX_FIGURE_BYTES if path.suffix == ".png" else MAX_TEXT_BYTES):
            offenders.append(f"{path}: {path.stat().st_size} bytes is row-level data, not a conclusion")
        elif path.suffix == ".csv" and sum(1 for _ in path.open()) > MAX_CSV_ROWS + 1:
            offenders.append(f"{path}: more rows than any study has points")
    assert not offenders, "\n".join(offenders)


def test_dated_task_folders_follow_the_layout() -> None:
    offenders = []
    tasks = set()
    for path in _tracked_results():
        parts = path.relative_to(ROOT).parts
        if len(parts) < 3 or not parts[2][:1].isdigit():
            continue
        tasks.add(path.parents[len(parts) - 4])
        if not TASK_FOLDER.fullmatch(parts[2]):
            offenders.append(f"{path}: task folder is not <YYYYMMDD>_<task>")
        elif len(parts) == 4 and parts[3] != "README.md":
            offenders.append(f"{path}: data belongs in a model subfolder")
        elif len(parts) > 4 and not MODEL_FOLDER.fullmatch(parts[3]):
            offenders.append(f"{path}: model folder is not <model>[-<variant>]")
    offenders += [f"{task}: no README.md" for task in sorted(tasks) if not (task / "README.md").is_file()]
    assert not offenders, "\n".join(offenders)
