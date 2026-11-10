"""Pool pilot and remainder counts in the shared LIBERO plotting layout.

An optional speed extension adds settings measured once on all 200 conditions.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from vlalagbench.files import sha256
from vlalagbench.statistics import wilson_interval
from vlalagbench.study.point import read_point, load_episodes
from vlalagbench.study.report import plot, write_plot_data
from vlalagbench.study.spec import open_study

MODES = ['synchronous', 'lazy', 'continuous']


def checked_record(plan, point, expected):
    directory = plan.study_dir / 'points' / point.name
    record = read_point(directory, plan.study_hash)
    relative = f'artifacts/{record.run_id}/episodes.jsonl'
    if sha256(directory / relative) != record.files[relative]:
        raise ValueError(f'Episode checksum mismatch: {directory}')
    episodes = load_episodes(directory / relative)
    if (record.episodes != expected or len(episodes) != expected
            or len({(r['task_id'], r['episode_id']) for r in episodes}) != expected
            or sum(r['success'] for r in episodes) != record.successes):
        raise ValueError(f'Incomplete or inconsistent point: {directory}')
    if record.values != point.values or record.deadline_steps != 750:
        raise ValueError(f'Protocol mismatch: {directory}')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('remainder', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--extension', type=Path, help='native-200 speed extension studies at new speeds')
    args = parser.parse_args()
    selection = json.loads(Path('configs/studies/fcan04_native_remaining180_20260928/selection.json').read_text())
    pilot_ids = selection['pilot_source_episode_ids']
    remaining_ids = selection['remaining_source_episode_ids']
    assert len(pilot_ids) == 20 and len(remaining_ids) == 180
    assert sorted(pilot_ids + remaining_ids) == list(range(200))
    records, points, sources, plans = {}, [], {}, []
    for mode in MODES:
        pilot = open_study(args.pilot / f'fcan04-native-random20-750steps-{mode}')
        remainder = open_study(args.remainder / f'fcan04-native-remaining180-750steps-{mode}')
        plans.append(pilot)
        remaining_points = {p.name: p for p in remainder.points}
        assert set(remaining_points) == {p.name for p in pilot.points}
        for point in pilot.points:
            name = f'{mode}_{point.name}'
            values = {**point.values, 'inference.mode': mode}
            points.append(dataclasses.replace(point, name=name, values=values))
            pair = [checked_record(pilot, point, 20),
                    checked_record(remainder, remaining_points[point.name], 180)]
            assert point.experiment.inference == remaining_points[point.name].experiment.inference
            assert point.experiment.model == remaining_points[point.name].experiment.model
            successes = sum(r.successes for r in pair)
            low, high = wilson_interval(successes, 200)
            records[name] = pair[0].model_copy(update=dict(
                name=name, values=values, episodes=200, successes=successes,
                success_rate=successes / 200, ci95_low=low, ci95_high=high,
                run_id=';'.join(r.run_id for r in pair),
                commit=';'.join(r.commit for r in pair),
                verified_at=max(r.verified_at for r in pair), files={},
            ))
            sources[name] = [dict(study_hash=r.study_hash, run_id=r.run_id,
                                  commit=r.commit, episodes=r.episodes, successes=r.successes,
                                  episode_sha256=r.files[f'artifacts/{r.run_id}/episodes.jsonl'])
                             for r in pair]
    assert len(records) == 49 and len(points) == 49
    speeds = list(plans[0].spec.sweep['trajectory.speed_mps'])
    if args.extension:
        reference = {(p.values['inference.mode'], p.values['inference.latency_ms']): p for p in points}
        for mode in MODES:
            extension = open_study(args.extension / f'fcan04-native200-speed-extension-20261001-{mode}')
            for point in extension.points:
                name = f'{mode}_{point.name}'
                values = {**point.values, 'inference.mode': mode}
                assert name not in records and values['trajectory.speed_mps'] not in speeds[:7]
                pooled = reference[mode, values['inference.latency_ms']]
                assert point.experiment.inference == pooled.experiment.inference
                assert point.experiment.model == pooled.experiment.model
                record = checked_record(extension, point, 200)
                points.append(dataclasses.replace(point, name=name, values=values))
                records[name] = record.model_copy(update=dict(name=name, values=values, files={}))
                sources[name] = [dict(study_hash=record.study_hash, run_id=record.run_id,
                                      commit=record.commit, episodes=record.episodes, successes=record.successes,
                                      episode_sha256=record.files[f'artifacts/{record.run_id}/episodes.jsonl'])]
                speeds.append(values['trajectory.speed_mps'])
        speeds = sorted(set(speeds))
        assert len(records) == 70 and len(speeds) == 10
    spec = plans[0].spec.model_copy(deep=True, update=dict(
        name='Moving target · food-can04 (DynamicVLA, 750-active-step budget)',
        sweep={'trajectory.speed_mps': speeds,
               'inference.mode': MODES, 'inference.latency_ms': [0, 120, 240]},
    ))
    merged = dataclasses.replace(plans[0], spec=spec, points=points, queue=points)
    args.output.mkdir(parents=True, exist_ok=True)
    write_plot_data(merged, records, args.output / 'plot_data.csv')
    plot(merged, records, args.output / 'success_rates.png', break_missing=True,
         reference_series=('synchronous', 0, '0 (synchronous reference)'),
         reference_panels=('lazy',), measured_x_ticks=True)
    (args.output / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')


if __name__ == '__main__':
    main()
