"""Publish per-task success of static LIBERO studies on the official initial states.

Each study has one point whose episodes cover every task of one suite. Every
point is checked against its record (digest, episode count, unique task and
episode identities, success total) before its rows are counted. Writes one
CSV with a row per task per study, a summary YAML, and one dot-plot figure
per suite comparing the studies on it.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vlalagbench.benchmarks.libero import _vendor  # noqa: E402
from vlalagbench.files import sha256  # noqa: E402
from vlalagbench.study.point import load_episodes, read_point  # noqa: E402
from vlalagbench.study.spec import open_study  # noqa: E402

# Categorical slots 1 and 2 of the dataviz reference palette (light surface).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, center - half), min(1.0, center + half)


def checked_episodes(study: Path) -> tuple[dict, list[dict]]:
    plan = open_study(study)
    if len(plan.points) != 1:
        raise ValueError(f"expected one point: {study}")
    point = plan.points[0]
    directory = plan.study_dir / "points" / point.name
    record = read_point(directory, plan.study_hash)
    relative = f"artifacts/{record.run_id}/episodes.jsonl"
    if sha256(directory / relative) != record.files[relative]:
        raise ValueError(f"episode checksum mismatch: {directory}")
    episodes = load_episodes(directory / relative)
    benchmark = point.experiment.benchmark
    expected = len(benchmark.task_ids) * benchmark.episodes_per_task
    if (record.episodes != expected or len(episodes) != expected
            or len({(row["task_id"], row["episode_id"]) for row in episodes}) != expected
            or sum(row["success"] for row in episodes) != record.successes
            or any(row["error"] for row in episodes)):
        raise ValueError(f"incomplete or inconsistent point: {directory}")
    model = point.experiment.model
    meta = {
        "study": plan.spec.name,
        "suite": benchmark.suite,
        "checkpoint": getattr(model, "checkpoint_subdir", None) or model.checkpoint,
        "run_id": record.run_id,
        "study_hash": plan.study_hash,
        "episodes_sha256": record.files[relative],
        "episodes": record.episodes,
        "successes": record.successes,
        "max_episode_steps": benchmark.max_episode_steps,
        "episodes_per_task": benchmark.episodes_per_task,
    }
    return meta, episodes


def plot_suite(suite: str, labels: dict[str, str], rows: list[dict], path: Path) -> None:
    names = {row["task_id"]: row["task"] for row in rows}
    tasks = sorted(names)
    series = list(dict.fromkeys(row["label"] for row in rows))
    rate = {(row["label"], row["task_id"]): row for row in rows}
    height = max(3.0, 0.24 * len(tasks) + 1.4)
    fig, ax = plt.subplots(figsize=(10, height), dpi=110)
    offsets = [(index - (len(series) - 1) / 2) * 0.22 for index in range(len(series))]
    for color, offset, label in zip(COLORS, offsets, series):
        ys = [len(tasks) - 1 - position + offset for position in range(len(tasks))]
        values = [rate[label, task]["success_rate"] * 100 for task in tasks]
        lows = [rate[label, task]["ci_low"] * 100 for task in tasks]
        highs = [rate[label, task]["ci_high"] * 100 for task in tasks]
        ax.hlines(ys, lows, highs, color=color, linewidth=1.2, alpha=0.45)
        ax.scatter(values, ys, s=26, color=color, edgecolors="white", linewidths=1.0, zorder=3,
                   label=labels[label])
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([f"{task:02d}  {names[task][:70]}" for task in reversed(tasks)], fontsize=7, color=INK)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.7, len(tasks) - 0.3)
    ax.set_xlabel("Success rate (%), 50 official initial states per task; lines are 95% Wilson intervals",
                  fontsize=8, color=MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", colors=MUTED, length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(series), frameon=False, fontsize=8)
    title = {"libero_10": "LIBERO-10", "libero_90": "LIBERO-90 (unseen by both models)"}.get(suite, suite)
    ax.set_title(f"{title} · synchronous, 0 ms", fontsize=10, color=INK, pad=24, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("studies", nargs="+", metavar="LABEL=STUDY_DIR",
                        help="series label (e.g. pi0.5) and a recorded study directory")
    parser.add_argument("--name", default="per_task", help="output file stem")
    args = parser.parse_args()
    benchmarks = _vendor.libero().benchmark.get_benchmark_dict()
    suites: dict[str, object] = {}
    rows, records, labels = [], [], {}
    for item in args.studies:
        label, _, directory = item.partition("=")
        meta, episodes = checked_episodes(Path(directory))
        labels[label] = f"{label} ({meta['checkpoint'].rsplit('/', 1)[-1]})"
        suite = suites.setdefault(meta["suite"], benchmarks[meta["suite"]]())
        by_task = defaultdict(list)
        for row in episodes:
            by_task[row["task_id"]].append(row["success"])
        for task_id, outcomes in sorted(by_task.items()):
            low, high = wilson(sum(outcomes), len(outcomes))
            rows.append({"suite": meta["suite"], "label": label, "checkpoint": meta["checkpoint"],
                         "task_id": task_id, "task": suite.get_task(task_id).language,
                         "episodes": len(outcomes), "successes": sum(outcomes),
                         "success_rate": round(sum(outcomes) / len(outcomes), 4),
                         "ci_low": round(low, 4), "ci_high": round(high, 4)})
        low, high = wilson(meta["successes"], meta["episodes"])
        records.append({**meta, "label": label, "success_rate": round(meta["successes"] / meta["episodes"], 4),
                        "ci95": [round(low, 4), round(high, 4)],
                        "tasks_with_no_success": sum(not any(v) for v in by_task.values()),
                        "tasks_all_success": sum(all(v) for v in by_task.values())})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f"{args.name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / f"{args.name}_summary.yaml").write_text(yaml.safe_dump({"studies": records}, sort_keys=False))
    for suite in suites:
        plot_suite(suite, labels, [row for row in rows if row["suite"] == suite],
                   args.output / f"{args.name}_{suite}.png")
    for record in records:
        print(f"{record['label']:>8} {record['suite']:<10} {record['checkpoint']:<45} "
              f"{record['successes']}/{record['episodes']} = {record['success_rate']:.1%}")


if __name__ == "__main__":
    main()
