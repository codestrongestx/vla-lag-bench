"""Redraw a results folder's three-panel success_rates.png from its plot_data.csv.

    python scripts/studies/plot_results_table.py results/libero/20260908_moving-basket/pi05 --title "..." --deadline 600
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from vlalagbench.study.report import plot

MODES = ['synchronous', 'lazy', 'continuous']


def record(row: dict[str, str], deadline: int) -> SimpleNamespace:
    return SimpleNamespace(
        values={'trajectory.speed_mps': float(row['trajectory.speed_mps']), 'inference.mode': row['inference.mode'],
                'inference.latency_ms': int(row['inference.latency_ms'])},
        success_rate=float(row['success_rate']), ci95_low=float(row['ci95_low']), ci95_high=float(row['ci95_high']),
        episodes=int(row['episodes']), deadline_steps=deadline,
    )


def opaque_rgb(path: Path) -> None:
    """Drop the figure's fully opaque alpha channel, which keeps the larger panels under the 500 KB figure limit."""
    image = Image.open(path)
    image.load()
    if image.mode != 'RGBA' or image.getchannel('A').getextrema() != (255, 255):
        raise ValueError(f'{path}: expected an opaque RGBA figure')
    image.convert('RGB').save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('folder', type=Path)
    parser.add_argument('--title', required=True)
    parser.add_argument('--deadline', type=int, required=True, help='active-step deadline of every point')
    args = parser.parse_args()
    rows = list(csv.DictReader((args.folder / 'plot_data.csv').open()))
    # Lazy at 0 ms schedules like synchronous 0 ms, which every panel shows as its reference.
    rows = [row for row in rows if not (row['inference.mode'] == 'lazy' and row['inference.latency_ms'] == '0')]
    records = {f'{row["trajectory.speed_mps"]}_{row["inference.mode"]}_{row["inference.latency_ms"]}': record(row, args.deadline)
               for row in rows}
    plan = SimpleNamespace(
        spec=SimpleNamespace(name=args.title, sweep={
            'trajectory.speed_mps': sorted({float(row['trajectory.speed_mps']) for row in rows}),
            'inference.mode': [mode for mode in MODES if any(row['inference.mode'] == mode for row in rows)],
            'inference.latency_ms': sorted({int(row['inference.latency_ms']) for row in rows}),
        }),
        points=list(records),
    )
    output = args.folder / 'success_rates.png'
    if any(row['inference.mode'] == 'synchronous' and row['inference.latency_ms'] == '0' for row in rows):
        plot(plan, records, output, reference_series=('synchronous', 0, '0 (synchronous reference)'),
             reference_panels=tuple(plan.spec.sweep['inference.mode']), measured_x_ticks=True,
             reference_color='#0072B2')
    else:
        plot(plan, records, output, measured_x_ticks=True)
    opaque_rgb(output)
    print(f'{output}: {len(records)} points')


if __name__ == '__main__':
    main()
