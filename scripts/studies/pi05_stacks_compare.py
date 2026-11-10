"""Paired success comparison of pi0.5 inference stacks on identical episodes.

Every stack runs the same episodes (initial state, target motion and request
noise) at each point, so outcomes pair by (task, point, episode). For each
stack against the reference stack this reports the success difference per
point and pooled over all points, with a paired 95% interval
(normal approximation on the discordant pairs), plus each stack's median
model time per request.

    python scripts/studies/pi05_stacks_compare.py --reference xtra:jax_bf16 \
        xtra=RTX3090_OUTCOMES.csv r4090=RTX4090_OUTCOMES.csv --markdown OUT.md --csv OUT.csv

HOST labels the machine whose studies a CSV lists; the published tables use
xtra (RTX 3090), r4090 (RTX 4090), and bw (RTX PRO 4500 Blackwell).

The CSV holds one row per stack for the pooled comparison (task and point
empty) and one per stack, task and point.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

STUDY = re.compile(r"^pi05_stacks_(?P<stack>.+)_(?P<task>plate|bowl|basket|drawer)_\d{8}$")
TASKS = ("plate", "bowl", "basket", "drawer")


def load(sources: list[str]) -> dict[str, dict[tuple, int]]:
    """{host:stack -> {(task, point, episode) -> success}} plus median model times."""
    outcomes: dict[str, dict[tuple, int]] = defaultdict(dict)
    timing: dict[str, list[float]] = defaultdict(list)
    for source in sources:
        host, path = source.split("=", 1)
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                match = STUDY.match(row["study"])
                if not match:
                    continue
                name = f"{host}:{match['stack']}"
                outcomes[name][(match["task"], row["point"], int(row["episode_id"]))] = int(row["success"])
                timing[name].append(float(row["median_infer_ms"]))
    return outcomes, timing


def paired(stack: dict[tuple, int], reference: dict[tuple, int], keys: list[tuple]) -> dict[str, float]:
    n = len(keys)
    better = sum(stack[key] and not reference[key] for key in keys)
    worse = sum(reference[key] and not stack[key] for key in keys)
    diff = (better - worse) / n
    se = math.sqrt(max(better + worse - (better - worse) ** 2 / n, 0.0)) / n
    return {"n": n, "stack": sum(stack[key] for key in keys) / n, "reference": sum(reference[key] for key in keys) / n,
            "diff": diff, "low": diff - 1.96 * se, "high": diff + 1.96 * se, "discordant": (better + worse) / n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="+", help="HOST=OUTCOMES.csv")
    parser.add_argument("--reference", required=True, help="HOST:STACK every other stack is paired against")
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    outcomes, timing = load(args.sources)
    reference = outcomes[args.reference]
    lines = [f"Reference: `{args.reference}`. Δ = stack − reference in success points, paired 95% interval.", "",
             "| Stack | Episodes | Stack | Reference | Δ pooled [95% CI] | Discordant | Median model ms |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    per_point = ["", "| Stack | Task | Point | Stack | Reference | Δ [95% CI] |", "| --- | --- | --- | ---: | ---: | --- |"]
    rows = []

    def record(name: str, task: str, point: str, cell: dict[str, float], median_ms: float | None) -> None:
        # Point names are speed_mode_latency, e.g. 0.1_lazy_100.
        speed, mode, latency = point.split("_") if point else ("", "", "")
        rows.append({"stack": name, "reference": args.reference, "task": task, "speed_mps": speed,
                     "mode": mode, "latency_ms": latency,
                     "episodes": cell["n"], "stack_success": round(cell["stack"], 4),
                     "reference_success": round(cell["reference"], 4), "diff_pts": round(cell["diff"] * 100, 2),
                     "ci_low_pts": round(cell["low"] * 100, 2), "ci_high_pts": round(cell["high"] * 100, 2),
                     "discordant": round(cell["discordant"], 4),
                     "median_model_ms": "" if median_ms is None else round(median_ms, 1)})
    for name in sorted(outcomes):
        if name == args.reference:
            continue
        keys = sorted(set(outcomes[name]) & set(reference))
        if not keys:
            continue
        pooled = paired(outcomes[name], reference, keys)
        median_ms = sorted(timing[name])[len(timing[name]) // 2]
        lines.append(f"| {name} | {pooled['n']} | {pooled['stack']:.1%} | {pooled['reference']:.1%} | "
                     f"{pooled['diff'] * 100:+.1f} [{pooled['low'] * 100:+.1f}, {pooled['high'] * 100:+.1f}] | "
                     f"{pooled['discordant']:.1%} | {median_ms:.0f} |")
        record(name, "", "", pooled, median_ms)
        for task in TASKS:
            points = {key[1] for key in keys if key[0] == task}
            for point in sorted(points, key=lambda p: (float(p.split("_")[0]), float(p.split("_")[-1]))):
                cell = paired(outcomes[name], reference, [key for key in keys if key[:2] == (task, point)])
                speed, mode, latency = point.split("_")
                per_point.append(f"| {name} | {task} | {speed} m/s, {mode} {latency} ms | {cell['stack']:.1%} | "
                                 f"{cell['reference']:.1%} | {cell['diff'] * 100:+.1f} "
                                 f"[{cell['low'] * 100:+.1f}, {cell['high'] * 100:+.1f}] |")
                record(name, task, point, cell, None)
    ref_ms = sorted(timing[args.reference])[len(timing[args.reference]) // 2]
    lines.append(f"\nReference median model time: {ref_ms:.0f} ms per request.")
    text = "\n".join(lines + per_point) + "\n"
    print(text)
    if args.markdown:
        args.markdown.write_text(text)
    if args.csv and rows:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
