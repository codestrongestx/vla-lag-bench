"""GR00T records use the existing point verification and video lifecycle."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from vlalagbench.config import Experiment, Videos, load_experiment, write_yaml
from vlalagbench.files import sha256
from vlalagbench.simulators.mujoco.tracing import EpisodeTrace
from vlalagbench.study import point, videos
from tests.test_study_point import EXPERIMENT, HASH, point_dir
from tests.benchmarks.libero.test_gr00t import model


def test_gr00t_records_verify_render_and_select(point_dir, monkeypatch):
    data = EXPERIMENT.model_dump()
    data["model"] = model().model_copy(update={"checkpoint_subdir": "libero_object"}).model_dump()
    data["inference"]["action_chunk_steps"] = 16
    experiment = Experiment.model_validate(data)
    experiment_path = point_dir / "runs/run-1/experiment.yaml"
    write_yaml(experiment, experiment_path)
    episode_path = point_dir / "artifacts/run-1/episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text().splitlines()]
    for row in rows:
        trace_path = Path(row["trace"])
        trace = EpisodeTrace.load(trace_path)
        trace.chunks = np.zeros((1, 16, 7))
        trace.meta = trace.meta.model_copy(update={"model": data["model"], "action_chunk_steps": 16,
                                                   "execution_horizon_steps": 5})
        trace.save(trace_path)
        row["trace_sha256"] = sha256(trace_path)
    episode_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result_path = point_dir / "runs/run-1/result.yaml"
    result = yaml.safe_load(result_path.read_text())
    result["experiment"]["sha256"] = sha256(experiment_path)
    result["episode_data"]["sha256"] = sha256(episode_path)
    result_path.write_text(yaml.safe_dump(result))
    verified = point.verify_point(point_dir, name="0", values={"inference.latency_ms": 0},
                                  experiment=experiment, study_hash=HASH)
    assert verified.episodes == 2

    def render(job):
        selection, camera, output, resolution = job
        trace = EpisodeTrace.load(selection.trace)
        assert trace.meta.model == data["model"] and trace.chunks.shape == (1, 16, 7)
        output.write_bytes(f"stub-video:{camera}:{resolution}".encode())
        return output
    monkeypatch.setattr(videos, "ProcessPoolExecutor", lambda **kw: ThreadPoolExecutor(max_workers=kw["max_workers"]))
    monkeypatch.setattr(videos, "_render_one", render)
    manifest = videos.film_point(point_dir, "0", Videos(clips_per_outcome=1, cameras=["agentview"], resolution=256), 1)
    assert manifest.complete and len(manifest.clips) == 2
    assert load_experiment(experiment_path) == experiment
