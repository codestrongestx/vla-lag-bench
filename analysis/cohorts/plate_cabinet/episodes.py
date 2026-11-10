"""Per-episode success table for the moving-plate and moving-cabinet studies, read from verified run records (no replay)."""
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from .data import (CHECKPOINT, CHUNK, CONDITIONS, EPISODE_FOLDER, MODELS, SPEEDS, STUDY, TASKS)
from analysis.paths import ARTIFACTS, ROOT, portable, resolve, sha, trace_path
from analysis import sync_hold

FIELDS = ('model', 'task', 'speed', 'mode', 'latency', 'episode_id', 'init_state', 'initial_phase', 'direction',
          'success', 'steps', 'control_steps', 'wait_steps', 'policy_requests', 'study', 'run_id', 'trace', 'trace_sha256')


def code_hashes():
    return {portable(p): sha(p) for p in (Path(__file__), Path(__file__).with_name('data.py'), ROOT / 'analysis/paths.py',
                                          ROOT / 'analysis/sync_hold.py')}


def study_rows(model, task, mode):
    """Rows of every grid setting of one model, task, and mode. A setting may come from any of the task's studies
    (the moving bowl combines the bowl-centered runs with the speed fill); each must be published exactly once, and
    settings outside the grid (the bowl's 0.25-0.4 m/s, the fill's 50/150/250 ms) are not read."""
    import yaml
    sources = {}
    published = {}
    expected = {(s, lat) for s in SPEEDS[task] for m, lat in CONDITIONS if m == mode}
    for study in STUDY[model](task, mode):
        table = study / 'plot_data.csv'
        sources[portable(table)] = sha(table)
        for r in csv.DictReader(table.open()):
            key = (float(r['trajectory.speed_mps']), int(r['inference.latency_ms']))
            if key not in expected:
                continue
            if key in published:
                raise ValueError(f'{study.name}: {key} is published by more than one study')
            published[key] = (study, r['run_id'], int(r['successes']), int(r['episodes']))
    if set(published) != expected:
        raise ValueError(f'{model} {task} {mode}: missing conditions {sorted(expected - set(published))}')
    if mode == 'synchronous':
        # Synchronous points at a nonzero latency come from the hold_pose reruns (analysis/sync_hold.py), which ran the
        # same episode schedule as the points they replace.
        for (speed, latency), (study, run_id, _, _) in list(published.items()):
            if latency == 0:
                continue
            run, config, episodes = sync_hold.rerun(model, task, speed, latency)
            old = yaml.safe_load((study / 'points' / f'{speed:g}_{mode}_{latency}' / 'runs' / run_id / 'experiment.yaml').read_text())
            if config['trajectory']['episode_schedule'] != old['trajectory']['episode_schedule']:
                raise ValueError(f'{model} {task} {speed} m/s {latency} ms: rerun episode schedule differs')
            table = sync_hold.study(model, task) / 'plot_data.csv'
            sources[portable(table)] = sha(table)
            published[(speed, latency)] = (sync_hold.study(model, task), run, sum(bool(e['success']) for e in episodes), 200)
    rows = []
    for (speed, latency), (study, run_id, successes, episodes) in sorted(published.items()):
        point = study / 'points' / f'{speed:g}_{mode}_{latency}'
        record = yaml.safe_load((point / 'point.yaml').read_text())
        experiment = point / 'runs' / run_id / 'experiment.yaml'
        jsonl = point / 'artifacts' / run_id / 'episodes.jsonl'
        for path in (point / 'point.yaml', experiment, jsonl):
            sources[portable(path)] = sha(path)
        if record['run_id'] != run_id or record['episodes'] != 200 or record['successes'] != successes or episodes != 200:
            raise ValueError(f'{point}: point record disagrees with the published table')
        config = yaml.safe_load(experiment.read_text())
        checkpoint = config['model']['checkpoint']
        if checkpoint != CHECKPOINT[model] or config['inference']['action_chunk_steps'] != CHUNK[model]:
            raise ValueError(f'{point}: wrong policy configuration')
        if (config['inference']['mode'], config['inference']['latency_ms']) != (mode, latency):
            raise ValueError(f'{point}: wrong execution setting')
        if config['inference']['execution_horizon_steps'] != 5 or config['benchmark']['max_episode_steps'] != 600:
            raise ValueError(f'{point}: wrong horizon or action budget')
        trajectory = config['trajectory']
        if abs(float(trajectory['speed_mps']) - speed) > 1e-12 or not trajectory['target'].startswith(('plate_1', 'wooden_cabinet_1')):
            raise ValueError(f'{point}: wrong trajectory')
        if (task == 'plate') != trajectory['target'].startswith('plate_1'):
            raise ValueError(f'{point}: trajectory target does not match the task')
        schedule = trajectory['episode_schedule']
        episodes = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        if len(episodes) != 200 or sorted(e['episode_id'] for e in episodes) != list(range(200)):
            raise ValueError(f'{point}: incomplete episode records')
        if sum(bool(e['success']) for e in episodes) != successes or any(e['error'] for e in episodes):
            raise ValueError(f'{point}: episode records disagree with the point record')
        for e in sorted(episodes, key=lambda e: e['episode_id']):
            trace = point / 'artifacts' / run_id / Path(e['trace']).name
            if not trace.is_file() or record['files'].get(f'artifacts/{run_id}/{trace.name}') != e['trace_sha256']:
                raise ValueError(f'{trace}: missing trace or inconsistent recorded hash')
            entry = schedule[e['episode_id']]
            rows.append(dict(model=model, task=task, speed=speed, mode=mode, latency=latency, episode_id=e['episode_id'],
                             init_state=e['episode_id'] % 50, initial_phase=entry['initial_phase'],
                             direction=entry['initial_direction'], success=int(bool(e['success'])), steps=e['steps'],
                             control_steps=e['control_steps'], wait_steps=e['wait_steps'],
                             policy_requests=e['policy_requests'], study=study.name, run_id=run_id,
                             trace='artifacts/' + str(trace.relative_to(ARTIFACTS)), trace_sha256=e['trace_sha256']))
    return rows, sources


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
                print('plate-cabinet-episodes: run records and output verified; using cached table', flush=True)
                return folder
        except (OSError, ValueError, KeyError):
            pass
    rows, sources = [], {}
    for model in MODELS:
        for task in TASKS:
            for mode in ('synchronous', 'lazy', 'continuous'):
                values, used = study_rows(model, task, mode)
                rows.extend(values)
                sources.update(used)
    keys = {(r['model'], r['task'], r['speed'], r['mode'], r['latency'], r['episode_id']) for r in rows}
    if len(rows) != 50400 or len(keys) != 50400:
        raise ValueError(f'Expected 50,400 unique moving-plate and moving-cabinet episodes, found {len(rows)}')
    with (folder / 'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = dict(code=code, environment=environment, sources=sources, episodes=len(rows),
                   outputs={'episodes.csv': sha(folder / 'episodes.csv')})
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f'plate-cabinet-episodes: {len(rows)} episodes from {len(sources)} run records', flush=True)
    return folder
