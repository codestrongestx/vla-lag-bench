"""Publish FlashRT bf16 on set-down: the three-panel figure and its comparison with openpi's JAX pi0.5."""
from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from vlalagbench.files import sha256
from vlalagbench.study.point import load_episodes, read_point
from vlalagbench.study.report import plot, write_plot_data
from vlalagbench.study.spec import open_study

MODES = ['synchronous', 'lazy', 'continuous']
LATENCIES = [0, 50, 100, 150, 200, 250]
STUDIES = ['synchronous0_100', 'synchronous150_250', 'continuous50', 'continuous100', 'continuous150_250',
           *(f'lazy{latency}_{half}' for latency in (50, 100, 150, 200, 250) for half in ('slow', 'fast'))]
TASK = '20261001_task9-moving-plate'
TITLE = 'Moving plate · LIBERO-Spatial task 9 (pi0.5 served by FlashRT bf16, 600-step deadline)'
# openpi JAX pi0.5 on the same protocol: 0/100/200 ms and 50/150/250 ms.
JAX = ['pi05-h100-b32', 'pi05-latency-grid']
COLORS = {'synchronous': '#2a78d6', 'lazy': '#eb6834', 'continuous': '#1baf7a'}
MARKERS = {'synchronous': 'o', 'lazy': 's', 'continuous': '^'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def checked_record(plan, point):
    directory = plan.study_dir / 'points' / point.name
    record = read_point(directory, plan.study_hash)
    relative = f'artifacts/{record.run_id}/episodes.jsonl'
    if sha256(directory / relative) != record.files[relative]:
        raise ValueError(f'Episode checksum mismatch: {directory}')
    episodes = load_episodes(directory / relative)
    if (record.episodes != 200 or len(episodes) != 200
            or len({(r['task_id'], r['episode_id']) for r in episodes}) != 200
            or sum(r['success'] for r in episodes) != record.successes):
        raise ValueError(f'Incomplete or inconsistent point: {directory}')
    if record.values != point.values or record.deadline_steps != 600:
        raise ValueError(f'Protocol mismatch: {directory}')
    return record


def condition(row):
    return row['inference.mode'], int(float(row['inference.latency_ms'])), round(float(row['trajectory.speed_mps']), 3)


def read_rows(path):
    with path.open() as handle:
        return {condition(row): int(row['successes']) for row in csv.DictReader(handle)}


def compare(flashrt, jax, output):
    """Per-point table, and FlashRT minus JAX by mode and delay with 95% intervals."""
    rows = []
    for key in sorted(flashrt, key=lambda key: (MODES.index(key[0]), key[1], key[2])):
        mode, latency, speed = key
        rows.append({'inference.mode': mode, 'inference.latency_ms': latency, 'trajectory.speed_mps': speed,
                     'flashrt_successes': flashrt[key], 'jax_successes': jax[key], 'episodes': 200,
                     'difference_points': (flashrt[key] - jax[key]) / 2})
    with (output / 'vs_jax.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios': [1.25, 1]})
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.8, wspace=0.22)
    figure.suptitle('Set-down: pi0.5 served by FlashRT bf16 against openpi JAX', x=0.07, ha='left',
                    fontsize=14, color=INK, fontweight='bold')
    figure.text(0.07, 0.885, f'{len(flashrt)} conditions, 200 episodes each, same seeds and episodes; '
                'FlashRT batched 16 per pass on H100', fontsize=10, color=MUTED)
    offsets = {'synchronous': -9, 'lazy': 0, 'continuous': 9}
    for mode in MODES:
        latencies = [latency for latency in LATENCIES if any(key[:2] == (mode, latency) for key in flashrt)]
        means, halfwidths = [], []
        for latency in latencies:
            keys = [key for key in flashrt if key[:2] == (mode, latency)]
            diffs = [(flashrt[key] - jax[key]) / 200 for key in keys]
            variance = sum(p * (1 - p) / 200 for key in keys for p in (flashrt[key] / 200, jax[key] / 200))
            means.append(100 * sum(diffs) / len(keys))
            halfwidths.append(100 * 1.96 * variance ** 0.5 / len(keys))
        x = [latency + offsets[mode] for latency in latencies]
        left.errorbar(x, means, yerr=halfwidths, color=COLORS[mode], marker=MARKERS[mode], markersize=8,
                      linewidth=2, capsize=0, elinewidth=2, label=mode,
                      markeredgecolor='#fcfcfb', markeredgewidth=1.5)
    left.axhline(0, color=MUTED, linewidth=1)
    left.set_xticks(LATENCIES)
    left.set_xlabel('Delay (ms)', color=MUTED)
    left.set_ylabel('FlashRT minus JAX success (points)', color=MUTED)
    left.set_title('Mean over the nine speeds, 95% interval', loc='left', fontsize=11, color=INK)
    left.set_ylim(-8, 8)
    left.legend(frameon=False, loc='upper left', ncol=3)

    lim = (0, 100)
    right.plot(lim, lim, color=MUTED, linewidth=1)
    for mode in MODES:
        keys = [key for key in flashrt if key[0] == mode]
        right.scatter([jax[key] / 2 for key in keys], [flashrt[key] / 2 for key in keys], s=40,
                      color=COLORS[mode], marker=MARKERS[mode], edgecolors='#fcfcfb', linewidths=1, label=mode)
    pooled = (sum(flashrt.values()) - sum(jax.values())) / (2 * len(flashrt))
    right.text(4, 92, f'pooled difference {pooled:+.1f} points', fontsize=10, color=INK)
    right.set_xlim(*lim)
    right.set_ylim(*lim)
    right.set_aspect('equal')
    right.set_xlabel('openpi JAX success (%)', color=MUTED)
    right.set_ylabel('FlashRT bf16 success (%)', color=MUTED)
    right.set_title('Each condition', loc='left', fontsize=11, color=INK)
    right.legend(frameon=False, loc='lower right')
    for axis in (left, right):
        axis.grid(color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            axis.spines[side].set_color(GRID)
    figure.savefig(output / 'vs_jax.png', dpi=110, facecolor='#fcfcfb')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='directory holding the 15 flashrt_bf16_plate_*_b16_20261018 studies')
    parser.add_argument('results', type=Path, help='results/libero directory')
    parser.add_argument('--folder', default='pi05-flashrt-bf16')
    args = parser.parse_args()
    plans = [open_study(args.root / f'flashrt_bf16_plate_{study}_b16_20261018') for study in STUDIES]
    records, points = {}, []
    for plan in plans:
        for point in plan.points:
            if point.name in records:
                raise ValueError(f'Duplicate point: {point.name}')
            records[point.name] = checked_record(plan, point)
            points.append(point)
    speeds = plans[2].spec.sweep['trajectory.speed_mps']
    if len(records) != 16 * len(speeds):
        raise ValueError(f'expected {16 * len(speeds)} verified points, found {len(records)}')
    spec = plans[0].spec.model_copy(deep=True, update={
        'name': TITLE,
        'sweep': {'trajectory.speed_mps': speeds, 'inference.mode': MODES, 'inference.latency_ms': LATENCIES},
    })
    merged = dataclasses.replace(plans[0], spec=spec, points=points, queue=points)
    output = args.results / TASK / args.folder
    output.mkdir(parents=True, exist_ok=True)
    write_plot_data(merged, records, output / 'plot_data.csv')
    plot(merged, records, output / 'success_rates.png',
         reference_series=('synchronous', 0, '0 (synchronous reference)'),
         reference_panels=tuple(MODES), measured_x_ticks=True, reference_color='#0072B2')
    # The figure is opaque; dropping its alpha channel keeps it under the 500 KB results limit.
    with Image.open(output / 'success_rates.png') as image:
        if image.getchannel('A').getextrema() != (255, 255):
            raise ValueError('success_rates.png is not opaque')
        opaque = image.convert('RGB')
    opaque.save(output / 'success_rates.png', optimize=True)
    flashrt = read_rows(output / 'plot_data.csv')
    jax = {}
    for folder in JAX:
        jax.update(read_rows(args.results / TASK / folder / 'plot_data.csv'))
    missing = [key for key in flashrt if key not in jax]
    if missing:
        raise ValueError(f'no JAX point for {missing[:3]}')
    compare(flashrt, jax, output)


if __name__ == '__main__':
    main()
