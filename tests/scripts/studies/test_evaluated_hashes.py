"""Every portable spec reproduces the study hash recorded when it was evaluated."""

import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("evaluated_hashes", ROOT / "scripts/studies/evaluated_hashes.py")
evaluated_hashes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluated_hashes)


def test_recorded_hashes_reproduce():
    record = yaml.safe_load(evaluated_hashes.RECORD.read_text())
    assert len(record) >= 76
    for path, entry in record.items():
        found = evaluated_hashes.evaluated_hash(ROOT / path, entry.get("checkpoint"), entry.get("idle_control"))
        assert found == entry["study_hash"], path


def test_dynamicvla_hashes_are_the_published_ones():
    record = yaml.safe_load(evaluated_hashes.RECORD.read_text())
    published = {
        run["study_hash"]
        for name in ("fcan04_native200_20261001", "fcan04_moving_can200_20261008")
        for runs in json.loads((ROOT / "results/dynamicvla" / name / "sources.json").read_text()).values()
        for run in runs
    }
    dynamicvla = {entry["study_hash"] for path, entry in record.items() if "fcan04" in path}
    assert dynamicvla == published
