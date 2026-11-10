"""Publish a complete fixed-placement drawer rerun as one three-panel figure."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from vlalagbench.study.point import read_point, verify_files
from vlalagbench.study.report import plot, write_plot_data
from vlalagbench.study.spec import open_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--study-format', default='pi05_drawer_fixed_20260926_{mode}',
                        help='study directory name with a {mode} field')
    parser.add_argument('--modes', nargs=3, default=['sync', 'lazy', 'continuous'],
                        help='synchronous, lazy, and continuous study names, in that order')
    parser.add_argument('--policy', default='pi0.5', help='policy name in the figure title')
    args = parser.parse_args()
    plans = [open_study(args.root / args.study_format.format(mode=mode)) for mode in args.modes]
    records = {}
    for plan in plans:
        for point in plan.points:
            directory = plan.study_dir / 'points' / point.name
            record = read_point(directory, plan.study_hash)
            verify_files(directory, record)
            if record.name in records or record.episodes != 200:
                raise ValueError(f'Duplicate or incomplete point: {record.name}')
            records[record.name] = record
    if len(records) != 72:
        raise ValueError(f'Expected 72 verified points, found {len(records)}')
    spec = plans[0].spec.model_copy(deep=True, update={
        'name': f'Moving drawer · corrected layout ({args.policy}, 600-step deadline)',
        'sweep': {**plans[0].spec.sweep,
                  'inference.mode': ['synchronous', 'lazy', 'continuous']},
    })
    merged = dataclasses.replace(plans[0], spec=spec,
                                 points=[point for plan in plans for point in plan.points])
    args.output.mkdir(parents=True, exist_ok=True)
    write_plot_data(merged, records, args.output / 'plot_data.csv')
    plot(merged, records, args.output / 'success_rates.png')


if __name__ == '__main__':
    main()
