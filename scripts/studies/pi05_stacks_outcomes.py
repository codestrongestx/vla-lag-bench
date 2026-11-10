"""List every episode outcome of the verified points under a studies root.

Reads each verified point's traces (the run named in point.yaml) and writes
one CSV row per episode: study, point, task and episode ids, success, the
number of acting steps, and the median model time per request.

    .venv/bin/python scripts/studies/pi05_stacks_outcomes.py STUDIES_ROOT OUT.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rows = []
    for point_file in sorted(args.root.glob("*/points/*/point.yaml")):
        point = yaml.safe_load(point_file.read_text())
        study = point_file.parents[2].name
        for trace in sorted((point_file.parent / "artifacts" / point["run_id"]).glob("*.npz")):
            with np.load(trace) as data:
                meta = json.loads(str(data["meta"]))
                infer_ms = data["infer_ms"] if "infer_ms" in data.files else np.array([np.nan])
            rows.append({
                "study": study, "point": point["name"], "run_id": point["run_id"],
                "task_id": meta["task_id"], "episode_id": meta["episode_id"],
                "success": int(bool(meta["success"])), "num_steps": meta["num_steps"],
                "median_infer_ms": float(np.nanmedian(infer_ms)) if len(infer_ms) else float("nan"),
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["study"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} episodes to {args.out}")


if __name__ == "__main__":
    main()
