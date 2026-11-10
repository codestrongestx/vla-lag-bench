"""Publish the pi0.5 50/150/250 ms latency grid as one three-panel figure per task."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from vlalagbench.files import sha256
from vlalagbench.study.point import load_episodes, read_point
from vlalagbench.study.report import plot, write_plot_data
from vlalagbench.study.spec import open_study

MODES = ['synchronous', 'lazy', 'continuous']
LATENCIES = [50, 150, 250]
TASKS = {
    'basket': ('20260908_moving-basket', 'Moving basket · LIBERO-Object task 1 (pi0.5 latency grid, 600-step deadline)'),
    'drawer': ('20260926_moving-drawer-fixed', 'Moving drawer · LIBERO-Goal task 0 (pi0.5 latency grid, 600-step deadline)'),
    'plate': ('20261001_task9-moving-plate', 'Moving plate · LIBERO-Spatial task 9 (pi0.5 latency grid, 600-step deadline)'),
    'bowl': ('20261010_task9-moving-cabinet-bowl-centered',
             'Moving cabinet, bowl centered · LIBERO-Spatial task 9 (pi0.5 latency grid, 600-step deadline)'),
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
    parser.add_argument('root', type=Path, help='directory holding the 12 pi05_<task>_<mode>_b32_20261014 studies')
    parser.add_argument('results', type=Path, help='results/libero directory')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--folder', default='pi05-latency-grid')
    args = parser.parse_args()
    for task in args.tasks:
        directory, title = TASKS[task]
        plans = [open_study(args.root / f'pi05_{task}_{mode}_b32_20261014') for mode in MODES]
        records, points = {}, []
        for plan in plans:
            for point in plan.points:
                if point.name in records:
                    raise ValueError(f'Duplicate point: {point.name}')
                records[point.name] = checked_record(plan, point)
                points.append(point)
        speeds = plans[0].spec.sweep['trajectory.speed_mps']
        if len(records) != len(MODES) * len(LATENCIES) * len(speeds):
            raise ValueError(f'{task}: expected {len(MODES) * len(LATENCIES) * len(speeds)} verified points, '
                             f'found {len(records)}')
        spec = plans[0].spec.model_copy(deep=True, update={
            'name': title,
            'sweep': {'trajectory.speed_mps': speeds, 'inference.mode': MODES, 'inference.latency_ms': LATENCIES},
        })
        merged = dataclasses.replace(plans[0], spec=spec, points=points, queue=points)
        output = args.results / directory / args.folder
        output.mkdir(parents=True, exist_ok=True)
        write_plot_data(merged, records, output / 'plot_data.csv')
        plot(merged, records, output / 'success_rates.png', measured_x_ticks=True)


if __name__ == '__main__':
    main()
