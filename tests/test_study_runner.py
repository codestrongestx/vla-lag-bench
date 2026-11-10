"""The queue never double-claims, a raising evaluation still kills the server, report is one row per point."""

import csv
import datetime as dt
import threading
from pathlib import Path

import yaml
import pytest

from vlalagbench.config import Placement, StudySpec
from vlalagbench.study import runner
from vlalagbench.study.point import POINT_RECORD
from vlalagbench.study.report import report
from vlalagbench.study.spec import resolve
from tests.test_study_spec import PLACEMENT, ROOT, STUDY


def _plan(tmp_path, backend="openpi", **study_changes):
    if backend == "gr00t":
        from tests.benchmarks.libero.test_gr00t import model
        study_changes["override"] = {**STUDY.get("override", {}),
                                     "model": model().model_copy(update={"checkpoint_subdir": "libero_object"}).model_dump(),
                                     "inference.action_chunk_steps": 16}
    spec = StudySpec.model_validate({**STUDY, "schema_version": 2, "policy": {}, **study_changes})
    placement = Placement.model_validate({**PLACEMENT, "output_root": str(tmp_path)})
    return resolve(spec, placement, ROOT)


def test_two_shards_never_claim_the_same_point(tmp_path):
    plan = _plan(tmp_path)
    runner.prepare(plan)
    claims = {0: [], 1: []}

    def drain(shard):
        while (point := runner.claim(plan, shard)) is not None:
            claims[shard].append(point.name)

    threads = [threading.Thread(target=drain, args=(shard,)) for shard in claims]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(claims[0] + claims[1]) == sorted(point.name for point in plan.points)
    assert not set(claims[0]) & set(claims[1])
    assert runner.state(plan.study_dir, plan.points[0].name)[0] == "claimed"


def test_shard_kills_its_server_when_evaluation_raises(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    runner.prepare(plan)
    events = []

    class FakeServer:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout=None):
            events.append("wait")

    def fail(*args, **kwargs):
        raise RuntimeError("policy server dropped the connection")

    monkeypatch.setattr(runner.os, "sched_setaffinity", lambda pid, cpus: None, raising=False)
    def start(plan, shard, runtime):
        (plan.study_dir / "runtimes/shard00-policy.yaml").write_text("packages: {}\n")
        return FakeServer()

    monkeypatch.setattr(runner, "_start_server", start)
    monkeypatch.setattr(runner, "_wait_healthy", lambda port, server: None)
    monkeypatch.setattr(runner, "run_experiment", fail)

    assert runner.run_shard(plan, 0) == 1

    assert events == ["terminate", "wait"]
    assert all(runner.state(plan.study_dir, point.name)[0] == "failed" for point in plan.points)
    assert "dropped the connection" in (plan.study_dir / "points" / plan.points[0].name / "error.txt").read_text()

    for stale in (plan.study_dir / "queue").glob("*.failed"):
        stale.rename(stale.with_name(stale.stem + ".todo"))
    runner.run_shard(plan, 0)

    assert len(list((plan.study_dir / "failed").iterdir())) == len(plan.points)


@pytest.mark.parametrize("backend", ["openpi", "gr00t"])
def test_report_publishes_verified_data_without_generating_or_overwriting_notes(tmp_path, backend):
    plan = _plan(tmp_path, backend=backend)
    for index, point in enumerate(plan.points[:3]):
        record = {
            "name": point.name,
            "values": point.values,
            "run_id": f"run-{index}",
            "study_hash": plan.study_hash,
            "commit": "b" * 40,
            "deadline_steps": 600,
            "episodes": 200,
            "successes": 100 + index,
            "success_rate": (100 + index) / 200,
            "ci95_low": 0.4,
            "ci95_high": 0.6,
            "verified_at": dt.datetime.now(dt.timezone.utc),
        }
        path = plan.study_dir / "points" / point.name / POINT_RECORD
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(record))

    outputs = report(plan, publish=tmp_path / "results" / "probe")

    with Path(outputs["csv"]).open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["point"] for row in rows] == [point.name for point in plan.points[:3]]
    assert rows[0]["backend"] == backend
    assert rows[0]["action_chunk_steps"] == ("16" if backend == "gr00t" else "10")
    assert rows[1]["successes"] == "101" and rows[1]["trajectory.speed_mps"] in {"0.001", "0.125"}
    assert Path(outputs["figure"]).stat().st_size > 0
    published = sorted(path.name for path in (tmp_path / "results" / "probe").iterdir())
    assert published == ["probe.yaml", "probe_plot_data.csv", "probe_success_rates.png"]
    assert yaml.safe_load((tmp_path / "results/probe/probe.yaml").read_text())["points_verified"] == 3

    notes = tmp_path / "results/probe/README.md"
    notes.write_text("Reviewed findings.\n")
    report(plan, publish=notes.parent)
    assert notes.read_text() == "Reviewed findings.\n"


def test_report_places_uneven_speeds_on_linear_axis(tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("vlalagbench.study.report")
    speeds = [0.001, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4]
    plan = _plan(tmp_path, sweep={"trajectory.speed_mps": speeds})
    records = {
        point.name: SimpleNamespace(values=point.values, success_rate=0.5,
                                    ci95_low=0.3, ci95_high=0.7,
                                    episodes=20, deadline_steps=750)
        for point in plan.points
    }

    def inspect(figure, output, **kwargs):
        axis = figure.axes[0]
        line = axis.containers[0].lines[0]
        assert list(line.get_xdata()) == speeds
        assert axis.get_xscale() == "linear"
        assert axis.get_xlim()[0] < min(speeds)
        assert axis.get_xlim()[1] > max(speeds)

    monkeypatch.setattr(module, "save_confidence_interval_figure", inspect)
    module.plot(plan, records, tmp_path / "figure.png")


def test_lock_is_acquired_before_prepare_and_is_released(tmp_path, monkeypatch):
    import pytest

    plan = _plan(tmp_path)
    monkeypatch.setattr(runner, "prepare", lambda plan: (_ for _ in ()).throw(AssertionError("must not prepare")))
    with runner.study_lock(plan.study_dir):
        with pytest.raises(RuntimeError, match="holds"):
            runner.run(plan, Path("study"), Path("placement"), [])
    with runner.study_lock(plan.study_dir):
        pass


def test_completed_evaluation_is_recovered_without_starting_server(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    runner.prepare(plan)
    monkeypatch.setattr(runner, "_completed_result", lambda directory: True)
    verified = []
    monkeypatch.setattr(runner, "_verify", lambda plan, point: verified.append(point.name))
    monkeypatch.setattr(runner, "_start_server", lambda *args: (_ for _ in ()).throw(AssertionError("must not load a model")))
    assert runner.run(plan, Path("study"), Path("placement"), []) == 0
    assert verified == [point.name for point in plan.queue]
    assert all(runner.state(plan.study_dir, point.name) == ("done", None) for point in plan.points)


def test_resume_rejects_changed_evaluation_sources(tmp_path, monkeypatch):
    import pytest

    plan = _plan(tmp_path)
    runner.prepare(plan)
    monkeypatch.setattr(runner, "execution_identity", lambda: {"changed": True})
    with pytest.raises(ValueError, match="code or dependencies changed"):
        runner.prepare(plan)


def test_episode_errors_are_reevaluated_instead_of_recovered_as_finished(tmp_path, monkeypatch):
    from types import SimpleNamespace

    plan = _plan(tmp_path)
    runner.prepare(plan)
    point = runner.claim(plan, 0)
    point_dir = plan.study_dir / "points" / point.name
    result_path = point_dir / "runs/old/result.yaml"
    result_path.parent.mkdir(parents=True)
    result_path.write_text("interrupted policy episode\n")
    monkeypatch.setattr(runner, "load_run_result", lambda path: SimpleNamespace(
        status="completed", error=None, summary=SimpleNamespace(episode_errors=1),
    ))
    evaluations = []

    def evaluate(*args, **kwargs):
        evaluations.append(kwargs)
        point_dir.mkdir(parents=True)

    monkeypatch.setattr(runner, "run_experiment", evaluate)
    monkeypatch.setattr(runner, "_verify", lambda *args: None)
    assert runner._run_point(plan, point, Path("runtime.yaml"), 0)
    assert len(evaluations) == 1
    assert len(list((plan.study_dir / "failed").glob("*/runs/old/result.yaml"))) == 1
    assert runner.state(plan.study_dir, point.name) == ("done", None)


def test_partial_shard_launch_failure_cleans_up_started_process(tmp_path, monkeypatch):
    import pytest

    plan = _plan(tmp_path)
    monkeypatch.setattr(runner, "execution_identity", lambda: {})
    launched = object()
    calls = []

    def launch(*args, **kwargs):
        if calls:
            raise OSError("cannot launch second shard")
        calls.append(launched)
        return launched

    stopped = []
    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner, "_stop", lambda process, **kwargs: stopped.append(process))
    with pytest.raises(OSError, match="second shard"):
        runner.run(plan, Path("study"), Path("placement"), [])
    assert stopped == [launched]
    with runner.study_lock(plan.study_dir):
        pass


def test_process_stop_releases_a_real_owned_process():
    import subprocess
    import sys

    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    runner._stop(process, group=True)
    assert process.poll() is not None


@pytest.mark.parametrize("changed", ["revision", "processor_sha256", "upstream_revision", "backbone_processors_sha256"])
def test_gr00t_resume_rejects_changed_policy_identity_and_stops_server(tmp_path, monkeypatch, changed):
    plan = _plan(tmp_path, backend="gr00t")
    runner.prepare(plan)
    original = {"policy": {"backend": "gr00t", "revision": "a" * 40,
                           "processor_sha256": "a" * 64, "upstream_revision": "a" * 40,
                           "backbone_processors_sha256": {"observation/tokenizer.json": "a" * 64}}}
    (plan.study_dir / "policy-identity.yaml").write_text(yaml.safe_dump(original))
    changed_identity = {"policy": {**original["policy"], changed: "b" * len(original["policy"][changed])}}
    if changed == "backbone_processors_sha256":
        changed_identity["policy"][changed] = {"observation/tokenizer.json": "b" * 64}
    events = []
    class Server:
        def poll(self): return None
        def terminate(self): events.append("terminated")
        def wait(self, timeout=None): return 0
    def start(plan, shard, runtime):
        runtime.with_name("shard00-policy.yaml").write_text(yaml.safe_dump(changed_identity))
        return Server()
    monkeypatch.setattr(runner.os, "sched_setaffinity", lambda *args: None, raising=False)
    monkeypatch.setattr(runner, "_start_server", start)
    monkeypatch.setattr(runner, "_wait_healthy", lambda *args: None)
    with pytest.raises(ValueError, match="policy environment changed"):
        runner.run_shard(plan, 0)
    assert events == ["terminated"]
    assert all(runner.state(plan.study_dir, point.name) == ("todo", None) for point in plan.points)


def test_packaged_vendor_identity_uses_unchanged_tree(tmp_path, monkeypatch):
    import subprocess
    def git(*args):
        return subprocess.run(['git', *args], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
    git('init')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.invalid')
    vendor = tmp_path / 'third_party/openpi'
    vendor.mkdir(parents=True)
    (vendor / 'policy.py').write_text('original\n')
    git('add', '.')
    git('commit', '-m', 'Original vendor')
    original = git('rev-parse', 'HEAD')
    (tmp_path / 'dispatch.py').write_text('new orchestration\n')
    git('add', '.')
    git('commit', '-m', 'Update orchestration')
    current = git('rev-parse', 'HEAD')
    monkeypatch.setattr(runner, 'REPO_ROOT', tmp_path)
    old = {'evaluator_sha256': 'same', 'vendors': {'openpi': original}}
    new = {'evaluator_sha256': 'same', 'vendors': {'openpi': current}}
    assert runner.compatible_execution(old, new)
    assert not runner.compatible_execution(old, {**new, 'evaluator_sha256': 'changed'})
    (vendor / 'policy.py').write_text('changed\n')
    assert not runner.compatible_execution(old, new)
    git('add', '.')
    git('commit', '-m', 'Change vendor')
    assert not runner.compatible_execution(old, {**new, 'vendors': {'openpi': git('rev-parse', 'HEAD')}})


def test_report_can_leave_gap_at_missing_setting(tmp_path, monkeypatch):
    import importlib
    import math
    from types import SimpleNamespace

    module = importlib.import_module('vlalagbench.study.report')
    plan = _plan(tmp_path, sweep={'trajectory.speed_mps': [0.1, 0.2, 0.3]})
    records = {
        point.name: SimpleNamespace(values=point.values, success_rate=0.5,
                                    ci95_low=0.3, ci95_high=0.7,
                                    episodes=200, deadline_steps=750)
        for point in plan.points if point.values['trajectory.speed_mps'] != 0.2
    }

    def inspect(figure, output, **kwargs):
        axis = figure.axes[0]
        assert list(axis.lines[0].get_xdata()) == [0.1, 0.2, 0.3]
        assert math.isnan(axis.lines[0].get_ydata()[1])
        assert list(axis.containers[0].lines[0].get_xdata()) == [0.1, 0.3]
        assert axis.containers[0].lines[0].get_linestyle() == 'None'

    monkeypatch.setattr(module, 'save_confidence_interval_figure', inspect)
    module.plot(plan, records, tmp_path / 'figure.png', break_missing=True)
