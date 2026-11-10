"""Turn a study's verified points into its plot data, figure, clip index, and record.

Only verified points appear: a pending point is absent, never a zero. The
figure reads the sweep in its listed order: the first key is the x axis, the
second gives one panel per value, the third one series per value.
"""

from __future__ import annotations

import csv
import datetime as dt
import itertools
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vlalagbench.plotting import save_confidence_interval_figure  # noqa: E402
from vlalagbench.study.point import POINT_RECORD, PointRecord, read_point  # noqa: E402
from vlalagbench.study.spec import Plan, token  # noqa: E402
from vlalagbench.files import sha256  # noqa: E402

PUBLISHED_SUFFIXES = {".md", ".yaml", ".csv", ".png"}
SERIES_STYLE = [
    {"color": "#0072B2", "marker": "o"},
    {"color": "#009E73", "marker": "^"},
    {"color": "#D55E00", "marker": "v"},
    {"color": "#CC79A7", "marker": "s"},
    {"color": "#E69F00", "marker": "D"},
]
EXTRA_SERIES_STYLE = {"color": "#6A3D9A", "marker": "P"}


def report(plan: Plan, publish: Path | None = None) -> dict[str, Any]:
    study_dir = plan.study_dir
    study_dir.mkdir(parents=True, exist_ok=True)
    records = {
        point.name: read_point(study_dir / "points" / point.name, plan.study_hash)
        for point in plan.queue
        if (study_dir / "points" / point.name / POINT_RECORD).exists()
    }
    csv_path = write_plot_data(plan, records, study_dir / "plot_data.csv")
    figure_path = plot(plan, records, study_dir / "success_rates.png")
    outputs = {"points": len(records), "csv": csv_path, "figure": figure_path}
    if publish is not None:
        outputs["published"] = publish_results(plan, records, csv_path, figure_path, publish)
    return outputs


def write_plot_data(plan: Plan, records: dict[str, PointRecord], path: Path) -> Path:
    keys = list(plan.spec.sweep)
    if len(keys) > 3:
        raise ValueError("the built-in figure supports up to three sweep dimensions; select a smaller study")
    experiments = {point.name: point.experiment for point in plan.points}
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["point", *keys, "successes", "episodes", "success_rate", "ci95_low", "ci95_high", "run_id", "commit", "verified_at", "backend", "checkpoint", "checkpoint_revision", "action_chunk_steps", "execution_horizon_steps"])
        for name, record in records.items():
            writer.writerow(
                [
                    name,
                    *(token(record.values[key]) for key in keys),
                    record.successes,
                    record.episodes,
                    f"{record.success_rate:.4f}",
                    f"{record.ci95_low:.4f}",
                    f"{record.ci95_high:.4f}",
                    record.run_id,
                    record.commit,
                    record.verified_at.isoformat(),
                    getattr(experiments[name].model, "backend", "dynamicvla" if experiments[name].benchmark.kind == "dynamicvla" else "openpi"),
                    experiments[name].model.checkpoint,
                    getattr(experiments[name].model, "revision", ""),
                    experiments[name].inference.action_chunk_steps,
                    experiments[name].inference.execution_horizon_steps,
                ]
            )
    return path


def plot(
    plan: Plan, records: dict[str, PointRecord], output: Path, *,
    break_missing: bool = False,
    reference_series: tuple[Any, Any, str] | None = None,
    reference_panels: tuple[Any, ...] = (),
    measured_x_ticks: bool = False,
    reference_color: str | None = None,
) -> Path:
    """Optionally reuse a source panel/series as a labeled reference in empty series.

    `reference_color` draws that reference series in its own color and keeps it out of the other series, which take the
    rest of the palette plus one extra color.
    """
    keys = list(plan.spec.sweep)
    x_key, panel_key, series_key = (keys + [None, None])[:3]
    x_values = plan.spec.sweep[x_key]
    panels = plan.spec.sweep[panel_key] if panel_key else [None]
    series = plan.spec.sweep[series_key] if series_key else [None]
    numeric_x = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in x_values)
    position = {token(value): float(value) if numeric_x else index for index, value in enumerate(x_values)}
    figure, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6.6), sharey=True, squeeze=False)
    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.19, top=0.76, wspace=0.08)
    for axis, panel in zip(axes[0], panels, strict=True):
        palette = SERIES_STYLE if reference_color is None else [
            *(style for style in SERIES_STYLE if style["color"] != reference_color), EXTRA_SERIES_STYLE]
        colors = itertools.cycle(palette)
        for item in series:
            if reference_color is not None and reference_series is not None and item == reference_series[1]:
                style = {"color": reference_color, "marker": "o"}
            else:
                style = next(colors)
            points = sorted(
                (
                    record
                    for record in records.values()
                    if (panel is None or record.values[panel_key] == panel)
                    and (item is None or record.values[series_key] == item)
                ),
                key=lambda record: position[token(record.values[x_key])],
            )
            if (not points and reference_series is not None
                    and panel in reference_panels and item == reference_series[1]):
                points = sorted(
                    (record for record in records.values()
                     if record.values[panel_key] == reference_series[0]
                     and record.values[series_key] == reference_series[1]),
                    key=lambda record: position[token(record.values[x_key])],
                )
            rates = [100 * record.success_rate for record in points]
            coordinates = [position[token(record.values[x_key])] for record in points]
            if break_missing:
                # Draw the connecting line separately so absent measurements leave gaps.
                measured = dict(zip(coordinates, rates, strict=True))
                grid = sorted(position.values())
                axis.plot(grid, [measured.get(x, float("nan")) for x in grid],
                          color=style["color"], linewidth=1.9)
            axis.errorbar(
                coordinates,
                rates,
                yerr=[
                    [rate - 100 * record.ci95_low for record, rate in zip(points, rates, strict=True)],
                    [100 * record.ci95_high - rate for record, rate in zip(points, rates, strict=True)],
                ],
                linestyle="none" if break_missing else "-", linewidth=1.9, markersize=6.0, elinewidth=1.1, capsize=3.0,
                label=(reference_series[2] if reference_series is not None and item == reference_series[1]
                       else f"{token(item)}" if item is not None else plan.spec.name),
                **style,
            )
        if numeric_x:
            low, high = min(position.values()), max(position.values())
            padding = (high - low) * 0.04 if high > low else max(abs(low) * 0.04, 0.5)
            axis.set_xlim(low - padding, high + padding)
            if measured_x_ticks:
                axis.set_xticks([position[token(value)] for value in x_values])
                axis.set_xticklabels([token(value) for value in x_values], rotation=45, ha="right")
        else:
            axis.set_xticks(range(len(x_values)))
            axis.set_xticklabels([token(value) for value in x_values])
            axis.set_xlim(-0.5, len(x_values) - 0.5)
        if not numeric_x and len(x_values) > 10:
            for label in axis.get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment("right")
        axis.tick_params(axis="x", labelsize=8.5)
        axis.set_title(f"{panel_key} = {token(panel)}" if panel is not None else plan.spec.name, fontsize=14, fontweight="bold")
        axis.set_xlabel(x_key)
        axis.set_ylim(0, 105)
        axis.set_yticks(range(0, 101, 20))
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0][0].set_ylabel("Success rate (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=len(series), frameon=False, title=series_key)
    complete = len(records) == len(plan.points)
    figure.suptitle(plan.spec.name + ("" if complete else " · in progress"), fontsize=17, fontweight="bold", y=0.98)
    episodes = {record.episodes for record in records.values()}
    deadlines = {record.deadline_steps for record in records.values()}
    deadline_label = "per-environment deadlines (see traces)" if None in deadlines else ("/".join(map(str, sorted(deadlines))) or "?") + "-step deadline"
    figure.text(
        0.5, 0.035,
        f"{'/'.join(map(str, sorted(episodes))) or '?'} episodes per verified point · pointwise 95% Wilson CI · "
        f"{deadline_label} · {len(records)}/{len(plan.points)} points verified",
        ha="center", fontsize=9, color="#555555",
    )
    save_confidence_interval_figure(figure, output, expected_errorbar_series=len(panels) * len(series), dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def publish_results(
    plan: Plan,
    records: dict[str, PointRecord],
    csv_path: Path,
    figure_path: Path,
    target: Path,
) -> list[Path]:
    """Publish the record, plot data, and figure; leave optional notes untouched."""

    target.mkdir(parents=True, exist_ok=True)
    foreign = [path.name for path in target.iterdir() if path.suffix not in PUBLISHED_SUFFIXES]
    if foreign:
        raise ValueError(f"{target} holds files a study may not publish: {foreign}")
    name = plan.spec.name
    record = {
        "schema_version": 1,
        "study": name,
        "study_hash": plan.study_hash,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_experiment": plan.spec.base_experiment,
        "sweep": plan.spec.sweep,
        "override": plan.spec.override,
        "policy": plan.spec.policy.model_dump(mode="python"),
        "model": plan.points[0].experiment.model.model_dump(mode="python"),
        **({"policy_identity": yaml.safe_load((plan.study_dir / "policy-identity.yaml").read_text())}
           if (plan.study_dir / "policy-identity.yaml").exists() else {}),
        "horizons": [{"point": point.name,
                      "action_chunk_steps": point.experiment.inference.action_chunk_steps,
                      "execution_horizon_steps": point.experiment.inference.execution_horizon_steps}
                     for point in plan.points],
        "points_verified": len(records),
        "points_total": len(plan.points),
        "episodes": sum(record.episodes for record in records.values()),
        "successes": sum(record.successes for record in records.values()),
        "points": [
            {
                "name": point_name,
                "values": record.values,
                "run_id": record.run_id,
                "commit": record.commit,
                "code_dirty": record.code_dirty,
                "point_record_sha256": sha256(plan.study_dir / "points" / point_name / POINT_RECORD),
                "successes": record.successes,
                "episodes": record.episodes,
                "ci95": [record.ci95_low, record.ci95_high],
            }
            for point_name, record in records.items()
        ],
    }
    written = [target / f"{name}.yaml", target / f"{name}_plot_data.csv", target / f"{name}_success_rates.png"]
    written[0].write_text(yaml.safe_dump(record, sort_keys=False))
    written[1].write_bytes(csv_path.read_bytes())
    written[2].write_bytes(figure_path.read_bytes())
    return written
