"""Recompute the study hash a spec had when it was evaluated.

Two things make a portable spec hash differently from its evaluated study.
Specs write local checkpoints as ``$VLALAGBENCH_CHECKPOINTS/<name>``, while the
evaluated runs named an absolute path on their host. And the synchronous
reruns were evaluated while holding the pose during waits was the opt-in
setting ``inference.idle_control: hold_pose``; it is now the only behaviour,
so their specs no longer state it. ``configs/studies/evaluated_hashes.yaml``
lists, for each such spec, the checkpoint path and setting it was evaluated
with and the hash its study records;

    python scripts/studies/evaluated_hashes.py          # check every entry
    python scripts/studies/evaluated_hashes.py SPEC [PATH]  # print one hash
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlalagbench.study.spec import _canonical, expand, load_study, study_hash  # noqa: E402

RECORD = ROOT / "configs/studies/evaluated_hashes.yaml"


def evaluated_hash(spec_path: Path, checkpoint: str | None = None, idle_control: str | None = None) -> str:
    spec = load_study(spec_path)
    points = expand(spec, ROOT)
    if checkpoint is not None:
        if "model.checkpoint" in spec.override:
            spec = spec.model_copy(update={"override": {**spec.override, "model.checkpoint": checkpoint}})
        points = [
            dataclasses.replace(point, experiment=point.experiment.model_copy(
                update={"model": point.experiment.model.model_copy(update={"checkpoint": checkpoint})}))
            for point in points
        ]
    if idle_control is None:
        return study_hash(spec, points)
    # study_hash with the setting back in the override and in every point's inference.
    digest = hashlib.sha256()
    if spec.schema_version == 2:
        digest.update(_canonical({"schema_version": 2, "episode_reset": "fresh"}))
    digest.update(_canonical({
        "sweep": spec.sweep,
        "override": {**spec.override, "inference.idle_control": idle_control},
        "policy": spec.policy.model_dump(mode="python"),
    }))
    for point in sorted(points, key=lambda point: point.name):
        dump = point.experiment.model_dump(mode="python", exclude={"name"})
        dump["inference"]["idle_control"] = idle_control
        digest.update(_canonical(dump))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec", nargs="?", type=Path)
    parser.add_argument("checkpoint", nargs="?")
    args = parser.parse_args()
    if args.spec is not None:
        entry = yaml.safe_load(RECORD.read_text()).get(str(args.spec), {})
        print(evaluated_hash(args.spec, args.checkpoint or entry.get("checkpoint"), entry.get("idle_control")))
        return 0
    failures = 0
    for spec, entry in yaml.safe_load(RECORD.read_text()).items():
        found = evaluated_hash(ROOT / spec, entry.get("checkpoint"), entry.get("idle_control"))
        if found != entry["study_hash"]:
            print(f"{spec}: {found} != recorded {entry['study_hash']}")
            failures += 1
    print("all recorded hashes reproduce" if not failures else f"{failures} differ")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
