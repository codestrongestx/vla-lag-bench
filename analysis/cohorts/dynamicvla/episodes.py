"""Per-episode success table for the DynamicVLA moving-box and moving-can cohorts, from verified run records."""
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from .data import (BUDGET, CHECKPOINT_SHA256, CHUNK, CONDITIONS, EPISODE_FOLDER, HOLD_POSE_RESULTS, HOLD_POSE_STUDIES,
                   HORIZON, HZ, MODEL, RESULTS, SPEEDS, STUDIES, STUDY_ROOTS, TARGET, TASKS)
from analysis.paths import ARTIFACTS, ROOT, portable, resolve, sha, trace_path

FIELDS = ('model', 'task', 'speed', 'mode', 'latency', 'idle_control', 'episode_id', 'source_episode', 'initial_phase', 'direction',
          'success', 'steps', 'control_steps', 'wait_steps', 'policy_requests', 'study', 'run_id', 'trace',
          'trace_sha256')


def code_hashes():
    return {portable(p): sha(p) for p in (Path(__file__), Path(__file__).with_name('data.py'), ROOT / 'analysis/paths.py')}


def point_of(task, run_id):
    """The collected point directory holding a run's artifacts; exactly one must exist."""
    roots = STUDY_ROOTS[task] + (HOLD_POSE_STUDIES[task],)
    hits = [path for root in roots for path in (STUDIES / root).glob(f'*/points/*/artifacts/{run_id}')]
    if len(hits) != 1:
        raise ValueError(f'Expected one collected run directory for {run_id}: {hits}')
    return hits[0].parents[1]


def hold_pose_parts(task, sources):
    """(speed, latency) -> the hold_pose rerun parts of one task's synchronous settings."""
    return published_hold_pose_parts(task, sources) if task in HOLD_POSE_RESULTS else study_hold_pose_parts(task, sources)


def study_hold_pose_parts(task, sources):
    """The moving can's reruns from the point records of their study, pilot 20 before the remaining 180."""
    import yaml
    parts = {}
    for study in sorted((STUDIES / HOLD_POSE_STUDIES[task]).glob('*/study.yaml')):
        sources[portable(study.with_name('plot_data.csv'))] = sha(study.with_name('plot_data.csv'))
        with study.with_name('plot_data.csv').open() as stream:
            published = {r['run_id']: (int(r['successes']), int(r['episodes'])) for r in csv.DictReader(stream)}
        for path in sorted(study.parent.glob('points/*/point.yaml')):
            record = yaml.safe_load(path.read_text())
            inference = yaml.safe_load((path.parent / 'runs' / record['run_id'] / 'experiment.yaml').read_text())['inference']
            if inference.get('idle_control') != 'hold_pose':
                continue
            if published[record['run_id']] != (record['successes'], record['episodes']):
                raise ValueError(f'{path}: point record disagrees with its study table')
            key = (float(record['values']['trajectory.speed_mps']), inference['latency_ms'])
            parts.setdefault(key, []).append(dict(
                run_id=record['run_id'], study_hash=record['study_hash'], episodes=record['episodes'],
                successes=record['successes'], episode_sha256=record['files'][f'artifacts/{record["run_id"]}/episodes.jsonl']))
    for value in parts.values():
        value.sort(key=lambda part: part['episodes'])
    return parts


def published_hold_pose_parts(task, sources):
    """The moving box's reruns as the benchmark publishes them beside the cohort: the runs sources.yaml names, each
    checked against the folder's plot_data.csv (run, study, successes, episodes); setting_rows checks each run's point
    record, episode summary, and configuration against them."""
    import yaml
    folder = HOLD_POSE_RESULTS[task]
    for name in ('plot_data.csv', 'sources.yaml'):
        sources[portable(folder / name)] = sha(folder / name)
    with (folder / 'plot_data.csv').open() as stream:
        table = {r['point']: r for r in csv.DictReader(stream) if r['idle_control'] == 'hold_pose'}
    listed = yaml.safe_load((folder / 'sources.yaml').read_text())
    if set(listed) != set(table):
        raise ValueError(f'{folder}: sources.yaml and plot_data.csv name different hold_pose points')
    parts = {}
    for point, runs in listed.items():
        row = table[point]
        if row['inference.mode'] != 'synchronous' or \
                [(p['run_id'], p['study'], p['successes'], p['episodes']) for p in runs] != \
                [(row['run_id'], row['source'], int(row['successes']), int(row['episodes']))]:
            raise ValueError(f'{folder}: {point} differs between sources.yaml and plot_data.csv')
        parts[float(row['trajectory.speed_mps']), int(row['inference.latency_ms'])] = runs
    return parts


def conditions_of(config):
    """Each episode's condition, in episode order: the scene (environments and target path), the placement (the
    initial-state override replica r runs: state r % initial_states_per_environment), and the starting phase and
    direction. Target speed and episode order are not part of it, nor the replica index, which seeds the instruction
    wording and policy noise."""
    benchmark, trajectory = config['benchmark'], config['trajectory']
    scene = json.dumps([benchmark['tasks'], {k: v for k, v in trajectory.items() if k not in ('speed_mps', 'episode_schedule')}],
                       sort_keys=True)
    states = benchmark.get('initial_states_per_environment', 1)
    placements = {o['initial_state_index']: json.dumps({k: v for k, v in o.items() if k != 'initial_state_index'}, sort_keys=True)
                  for o in config.get('initial_state_overrides', [])}
    return [(scene, placements.get(r % states, 'null'), e['initial_phase'], e['initial_direction'])
            for r, e in enumerate(trajectory['episode_schedule'])]


def setting_rows(task, mode, latency, speed, parts, sources, idle_control='repeat_last_action'):
    """One setting's rows in run order, and the condition of each (conditions_of)."""
    import yaml
    rows, conditions = [], []
    for part in parts:
        run_id = part['run_id']
        point = point_of(task, run_id)
        record = yaml.safe_load((point / 'point.yaml').read_text())
        summary = f'artifacts/{run_id}/episodes.jsonl'
        experiment = f'runs/{run_id}/experiment.yaml'
        for name in ('point.yaml', summary, experiment):
            sources[portable(point / name)] = sha(point / name)
        if (record['run_id'], record['study_hash'], record['episodes'], record['successes'], record['deadline_steps']) != \
                (run_id, part['study_hash'], part['episodes'], part['successes'], BUDGET):
            raise ValueError(f'{point}: point record disagrees with the published sources')
        if part.get('study', point.parents[1].name) != point.parents[1].name:
            raise ValueError(f'{point}: run is not in the published study {part["study"]}')
        if not (sha(point / summary) == record['files'][summary] == part['episode_sha256']):
            raise ValueError(f'{point / summary}: episode summary hash mismatch')
        if sha(point / experiment) != record['files'][experiment]:
            raise ValueError(f'{point / experiment}: run configuration hash mismatch')
        config = yaml.safe_load((point / experiment).read_text())
        inference, trajectory, benchmark = config['inference'], config['trajectory'], config['benchmark']
        if config['model']['checkpoint_sha256'] != CHECKPOINT_SHA256 or benchmark['kind'] != 'dynamicvla':
            raise ValueError(f'{point}: wrong policy')
        if (inference['mode'], inference['latency_ms'], inference['action_chunk_steps'],
                inference['execution_horizon_steps'], inference.get('idle_control', 'repeat_last_action')) != \
                (mode, latency, CHUNK, HORIZON, idle_control):
            raise ValueError(f'{point}: wrong execution setting')
        if (benchmark['max_episode_steps'], benchmark['control_hz']) != (BUDGET, HZ):
            raise ValueError(f'{point}: wrong action budget or control rate')
        if abs(float(trajectory['speed_mps']) - speed) > 1e-12 or trajectory['target'] != TARGET[task]:
            raise ValueError(f'{point}: wrong trajectory')
        schedule = trajectory['episode_schedule']
        episodes = [json.loads(line) for line in (point / summary).read_text().splitlines() if line.strip()]
        if sorted(e['episode_id'] for e in episodes) != list(range(part['episodes'])) or len(schedule) != part['episodes']:
            raise ValueError(f'{point}: incomplete episode records')
        if sum(bool(e['success']) for e in episodes) != part['successes'] or any(e['error'] for e in episodes):
            raise ValueError(f'{point}: episode records disagree with the point record')
        conditions += conditions_of(config)
        for e in sorted(episodes, key=lambda e: e['episode_id']):
            trace = point / 'artifacts' / run_id / Path(e['trace']).name
            if not trace.is_file() or record['files'].get(f'artifacts/{run_id}/{trace.name}') != e['trace_sha256']:
                raise ValueError(f'{trace}: missing trace or inconsistent recorded hash')
            entry = schedule[e['episode_id']]
            rows.append(dict(model=MODEL, task=task, speed=speed, mode=mode, latency=latency, idle_control=idle_control,
                             episode_id=len(rows),
                             source_episode=e['episode_id'], initial_phase=entry['initial_phase'],
                             direction=entry['initial_direction'], success=int(bool(e['success'])), steps=e['steps'],
                             control_steps=e['control_steps'], wait_steps=e['wait_steps'],
                             policy_requests=e['policy_requests'], study=point.parents[1].name, run_id=run_id,
                             trace='artifacts/' + str(trace.relative_to(ARTIFACTS)), trace_sha256=e['trace_sha256']))
    return rows, conditions


def task_rows(task, sources):
    folder = RESULTS[task]
    for name in ('plot_data.csv', 'sources.json'):
        sources[portable(folder / name)] = sha(folder / name)
    published = {}
    with (folder / 'plot_data.csv').open() as stream:
        for r in csv.DictReader(stream):
            key = (float(r['trajectory.speed_mps']), r['inference.mode'], int(r['inference.latency_ms']))
            published[key] = (r['point'], int(r['successes']), int(r['episodes']))
    # The published table holds every setting but the synchronous ones at a nonzero latency, which the hold_pose
    # reruns replace.
    expected = {(s, m, lat) for s in SPEEDS[task] for m, lat in CONDITIONS if not (m == 'synchronous' and lat)}
    if set(published) != expected:
        raise ValueError(f'{folder}: unexpected settings {sorted(set(published) ^ expected)}')
    parts = json.loads((folder / 'sources.json').read_text())
    hold = hold_pose_parts(task, sources)
    if set(hold) != {(s, lat) for s in SPEEDS[task] for m, lat in CONDITIONS if m == 'synchronous' and lat}:
        raise ValueError(f'{HOLD_POSE_STUDIES[task]}: unexpected hold_pose settings {sorted(hold)}')
    rows, order = [], {}
    for (speed, mode, latency), (point, successes, episodes) in sorted(published.items()):
        values, conditions = setting_rows(task, mode, latency, speed, parts[point], sources)
        if len(values) != 200 or episodes != 200 or sum(r['success'] for r in values) != successes:
            raise ValueError(f'{task} {point}: pooled episodes disagree with the published table')
        if (mode, latency) == ('synchronous', 0):
            order[speed] = conditions
        rows.extend(values)
    for (speed, latency), runs in sorted(hold.items()):
        rerun, rerun_conditions = setting_rows(task, 'synchronous', latency, speed, runs, sources, 'hold_pose')
        # Within a speed every setting runs the same 200 conditions in the same order, so an episode_id names one
        # condition in every setting. A rerun may order them differently (the moving box's ran on the
        # speed-extension base): each rerun episode takes the episode_id of its condition in the speed's synchronous
        # 0 ms setting. It need not name one replica: a reordered rerun episode has another replica index, hence
        # another instruction wording and policy noise seed (README). No consumer pairs episodes across settings.
        index = {condition: i for i, condition in enumerate(rerun_conditions)}
        if len(index) != len(rerun) or sorted(index) != sorted(order[speed]):
            raise ValueError(f'{task} {speed} m/s {latency} ms: hold_pose rerun ran different conditions')
        rows.extend(dict(rerun[index[condition]], episode_id=i) for i, condition in enumerate(order[speed]))
    return rows


def ensure(*, workers=4, force=False):
    import yaml
    folder = (EPISODE_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)
    provenance_path = folder / 'provenance.json'
    code = code_hashes()
    environment = dict(python=sys.version.split()[0], pyyaml=yaml.__version__)
    if not force and provenance_path.exists():
        try:
            previous = json.loads(provenance_path.read_text())
            if (previous['code'] == code and previous['environment'] == environment
                    and previous['outputs']['episodes.csv'] == sha(folder / 'episodes.csv')
                    and all(sha(resolve(name)) == value for name, value in previous['sources'].items())):
                print('dynamicvla-episodes: run records and output verified; using cached table', flush=True)
                return folder
        except (OSError, ValueError, KeyError):
            pass
    rows, sources = [], {}
    for task in TASKS:
        rows.extend(task_rows(task, sources))
    with (folder / 'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    totals = {task: [sum(r['success'] for r in rows if r['task'] == task), sum(r['task'] == task for r in rows)]
              for task in TASKS}
    payload = dict(code=code, environment=environment, sources=sources, episodes=len(rows), successes=totals,
                   outputs={'episodes.csv': sha(folder / 'episodes.csv')})
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f'dynamicvla-episodes: {len(rows)} episodes from {len(sources)} records; successes {totals}', flush=True)
    return folder


if __name__ == '__main__':
    ensure(force='--force' in sys.argv)
