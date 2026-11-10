"""Tabulate policy_latency.py records as a markdown table.

One row per model, stack, precision, transport and GPU. Steps stale is the
latency in control periods of the benchmark's controller (LIBERO 20 Hz): how
many actions the robot executes before a chunk computed from the current
observation can arrive.

    python scripts/latency/latency_table.py RESULTS/*.json [--csv OUT.csv]

The CSV has the same rows plus each record's host load, CPU model when
recorded, agreement with the recorded chunks and its file name.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def row(record: dict) -> dict:
    e2e, model = record["e2e_ms"], record.get("model_ms") or {}
    flags = [] if record["gpu"]["exclusive"] else ["shared GPU"]
    if record["inputs"] != "recorded":
        flags.append(f"{record['inputs']} inputs")
    load = record["gpu"]["start"].get("host_loadavg")
    if load is not None:
        flags.append(f"host load {load[0]:.0f}")
    if record.get("period_ms"):
        flags.append(f"paced {record['period_ms']:g} ms")
    return {
        "model": record["model"], "stack": record["stack"], "precision": record["precision"],
        "transport": record["transport"], "gpu": record["gpu"]["start"]["name"].removeprefix("NVIDIA "),
        "p50": e2e["p50"], "p95": e2e["p95"], "p99": e2e["p99"],
        "model_p50": model.get("p50"), "hz": record["control_hz"],
        "steps": record["control_hz"] * e2e["p50"] / 1000, "steps_p99": record["control_hz"] * e2e["p99"] / 1000,
        "calls": e2e["n"], "notes": ", ".join(flags),
        "host_load": None if load is None else round(load[0], 1),
        "cpu_model": (record.get("cpu") or {}).get("model"),
        "arm_mean_abs": (record.get("agreement") or {}).get("arm_mean_abs"),
        "label": record.get("label"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = sorted(({**row(json.loads(path.read_text())), "record": f"{path.parent.name}/{path.name}"}
                   for path in args.records),
                  key=lambda item: (item["model"], item["gpu"], item["stack"], item["precision"], item["transport"]))
    if args.csv:
        fields = ["model", "stack", "precision", "transport", "gpu", "p50", "p95", "p99", "model_p50", "hz",
                  "steps", "steps_p99", "calls", "host_load", "cpu_model", "arm_mean_abs", "label", "record"]
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for item in rows:
                writer.writerow({key: round(value, 4 if key == "arm_mean_abs" else 2)
                                 if isinstance(value, float) else value for key, value in item.items()})
    print("| model | stack | precision | transport | GPU | p50 ms | p95 ms | p99 ms | model p50 ms "
          "| steps stale p50 / p99 | calls | notes |")
    print("|---|---|---|---|---|---:|---:|---:|---:|---|---:|---|")
    for item in rows:
        model = "" if item["model_p50"] is None else f"{item['model_p50']:.1f}"
        print(f"| {item['model']} | {item['stack']} | {item['precision']} | {item['transport']} | {item['gpu']} "
              f"| {item['p50']:.1f} | {item['p95']:.1f} | {item['p99']:.1f} | {model} "
              f"| {item['steps']:.2f} / {item['steps_p99']:.2f} at {item['hz']:g} Hz | {item['calls']} "
              f"| {item['notes']} |")


if __name__ == "__main__":
    main()
