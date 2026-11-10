"""Publish a policy's Task 9 moving-plate sweep (or, with --tasks, other Task 9 studies) as one three-panel figure each."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from vlalagbench.files import sha256
from vlalagbench.study.point import load_episodes, read_point
from vlalagbench.study.report import plot, write_plot_data
from vlalagbench.study.spec import open_study

MODES = ['synchronous', 'lazy', 'continuous']
POLICIES = {'pi05': 'pi0.5', 'gr00t': 'GR00T N1.7'}
TASKS = {
    'plate': ('plate_ellipse_600', 'Moving plate · LIBERO-Spatial task 9 ({}, 600-step deadline)', 63),
    'cabinet': ('cabinet_ellipse_600', 'Moving cabinet · LIBERO-Spatial task 9 ({}, 600-step deadline)', 63),
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
    parser.add_argument('policy', choices=POLICIES)
    parser.add_argument('root', type=Path, help='directory holding the policy\'s six <policy>_task9_* studies')
    parser.add_argument('output', type=Path)
    parser.add_argument('--suffix', default='20261001', help='study name suffix after <policy>_task9_<task>_<mode>_')
    parser.add_argument('--name-format', default='{policy}_task9_{task}_{mode}_{suffix}',
                        help='study directory name with {policy}, {task}, {mode}, and {suffix} fields')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=['plate'])
    parser.add_argument('--variant', help='text added after the task name in figure titles')
    parser.add_argument('--flat', action='store_true', help='write a single task straight into OUTPUT')
    args = parser.parse_args()
    if args.flat and len(args.tasks) != 1:
        parser.error('--flat needs exactly one task')
    for task in args.tasks:
        directory, title, expected = TASKS[task]
        if args.variant:
            title = title.replace(' · ', f', {args.variant} · ', 1)
        plans = [open_study(args.root / args.name_format.format(policy=args.policy, task=task, mode=mode, suffix=args.suffix))
                 for mode in MODES]
        records, points = {}, []
        for plan in plans:
            for point in plan.points:
                if point.name in records:
                    raise ValueError(f'Duplicate point: {point.name}')
                records[point.name] = checked_record(plan, point)
                points.append(point)
        if len(records) != expected:
            raise ValueError(f'{task}: expected {expected} verified points, found {len(records)}')
        spec = plans[0].spec.model_copy(deep=True, update={
            'name': title.format(POLICIES[args.policy]),
            'sweep': {'trajectory.speed_mps': plans[0].spec.sweep['trajectory.speed_mps'],
                      'inference.mode': MODES, 'inference.latency_ms': [0, 100, 200]},
        })
        merged = dataclasses.replace(plans[0], spec=spec, points=points, queue=points)
        output = args.output if args.flat else args.output / directory
        output.mkdir(parents=True, exist_ok=True)
        write_plot_data(merged, records, output / 'plot_data.csv')
        plot(merged, records, output / 'success_rates.png',
             reference_series=('synchronous', 0, '0 (synchronous reference)'),
             reference_panels=('lazy',), measured_x_ticks=True)


if __name__ == '__main__':
    main()
