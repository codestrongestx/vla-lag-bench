import hashlib

import pytest

from vlalagbench.config import SourceConfig
from vlalagbench.provenance import read_source_config, resolve_source_config


def test_legacy_experiment_paths_redirect_without_changing_recorded_hash(tmp_path):
    current = tmp_path / "configs/experiments/libero/example.yaml"
    current.parent.mkdir(parents=True)
    current.write_bytes(b"schema_version: 5\nname: example\n")
    source = SourceConfig(
        source="configs/experiments/example.yaml",
        sha256=hashlib.sha256(current.read_bytes()).hexdigest(),
        resolved="experiment.yaml",
    )

    assert resolve_source_config(source, tmp_path) == current

    mismatched = source.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(FileNotFoundError, match="no source config matches"):
        resolve_source_config(mismatched, tmp_path)


def test_resolved_snapshot_is_a_checksum_verified_fallback(tmp_path):
    snapshot = tmp_path / "runs/run-1/experiment.yaml"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"schema_version: 5\nname: generated\n")
    source = SourceConfig(
        source="configs/experiments/generated.yaml",
        sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        resolved="experiment.yaml",
    )

    assert (
        resolve_source_config(source, tmp_path, resolved_root=snapshot.parent)
        == snapshot
    )


def test_recorded_git_commit_is_the_final_verified_fallback(tmp_path, monkeypatch):
    content = b"schema_version: 5\nname: historical\n"
    source = SourceConfig(
        source="configs/experiments/historical.yaml",
        sha256=hashlib.sha256(content).hexdigest(),
        resolved="experiment.yaml",
    )

    class Completed:
        returncode = 0
        stdout = content

    monkeypatch.setattr(
        "vlalagbench.provenance.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )

    assert read_source_config(source, tmp_path, commit="a" * 40) == content
