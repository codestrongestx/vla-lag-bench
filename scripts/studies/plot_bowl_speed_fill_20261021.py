"""Publish the moving bowl on the 0.001-0.2 m/s grid of the other paper tasks.

The October 21 speed-fill studies add 0.025, 0.075, 0.125 and 0.175 m/s; the
other five speeds (0.001, 0.05, 0.1, 0.15, 0.2) come from the earlier
bowl-centered runs at the same delays. One figure and CSV per model.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from vlalagbench.files import sha256
from vlalagbench.study.point import load_episodes, read_point
from vlalagbench.study.report import plot, write_plot_data
from vlalagbench.study.spec import open_study

MODES = ['synchronous', 'lazy', 'continuous']
SPEEDS = [0.001, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
FILL = 'bowl_speed_fill_20261021'
FOLDER = '20261010_task9-moving-cabinet-bowl-centered'
TITLE = 'Moving cabinet, bowl centered · LIBERO-Spatial task 9 ({}, 600-step deadline)'
MODELS = {
    'pi05': ('pi0.5', [0, 50, 100, 150, 200, 250], [
        ('libero_spatial_task9_bowl_centered_20261010', 'pi05_task9_cabinet_bowl_centered_{}_b32_20261010', MODES),
        ('libero_latency_grid_20261014', 'pi05_bowl_{}_b32_20261014', MODES),
    ]),
    'gr00t': ('GR00T N1.7', [0, 100, 200], [
        ('libero_spatial_task9_bowl_centered_20261010', 'gr00t_task9_cabinet_bowl_centered_{}_20261010', MODES),
    ]),
    'molmoact2': ('MolmoAct2', [0, 50, 100, 150, 200, 250], [
        ('molmoact2_libero_20261015', 'molmoact2_bowl_{}_b32_20261015',
         ['synchronous0', 'synchronous', 'lazy', 'synchronous_ext', 'lazy_ext', 'continuous', 'continuous_ext']),
    ]),
}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='artifacts/studies, holding the speed-fill and earlier bowl studies')
    parser.add_argument('results', type=Path, help='results/libero directory')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    for model in args.models:
        label, latencies, earlier = MODELS[model]
        sources = [(FILL, f'{model}_bowl_speed_fill_{{}}_b32_20261021', MODES), *earlier]
        plans = [open_study(args.root / group / name.format(mode)) for group, name, modes in sources for mode in modes]
        records, points = {}, []
        for plan in plans:
            for point in plan.points:
                if point.values['trajectory.speed_mps'] not in SPEEDS:
                    continue
                if point.name in records:
                    raise ValueError(f'Duplicate point: {point.name}')
                records[point.name] = checked_record(plan, point)
                points.append(point)
        conditions = len(latencies) + 2 * (len(latencies) - 1)
        if len(records) != conditions * len(SPEEDS):
            raise ValueError(f'{model}: expected {conditions * len(SPEEDS)} verified points, found {len(records)}')
        spec = plans[0].spec.model_copy(deep=True, update={
            'name': TITLE.format(label),
            'sweep': {'trajectory.speed_mps': SPEEDS, 'inference.mode': MODES, 'inference.latency_ms': latencies},
        })
        merged = dataclasses.replace(plans[0], spec=spec, points=points, queue=points)
        output = args.results / FOLDER / f'{model}-speed-fill'
        output.mkdir(parents=True, exist_ok=True)
        write_plot_data(merged, records, output / 'plot_data.csv')
        plot(merged, records, output / 'success_rates.png',
             reference_series=('synchronous', 0, '0 (synchronous reference)'),
             reference_panels=tuple(MODES), measured_x_ticks=True, reference_color='#0072B2')


if __name__ == '__main__':
    main()
