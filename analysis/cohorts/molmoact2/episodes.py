"""Per-episode success table of MolmoAct2-LIBERO on the four LIBERO tasks, read from verified run records (no replay)."""
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from .data import CHECKPOINT, FOLDER, MODES, SPEEDS, TARGET, TASKS, studies
from analysis.paths import ARTIFACTS, ROOT, portable, resolve, sha, trace_path

FIELDS = ('task', 'speed', 'mode', 'latency', 'episode_id', 'init_state', 'initial_phase', 'direction', 'success',
          'steps', 'control_steps', 'wait_steps', 'policy_requests', 'study', 'run_id', 'commit', 'trace', 'trace_sha256')


def code_hashes():
    return {portable(p): sha(p) for p in (Path(__file__), Path(__file__).with_name('data.py'), ROOT / 'analysis/paths.py')}


def study_rows(task, mode):
    """Rows of every grid setting of one task and mode, each published exactly once by one of the task's studies;
    the bowl's 0.25-0.4 m/s settings are not read."""
    import yaml
    sources, published, expected = {}, {}, set()
    for folder, latencies in studies(task, mode):
        expected |= {(s, lat) for s in SPEEDS[task] for lat in latencies}
        table = folder / 'plot_data.csv'
        sources[portable(table)] = sha(table)
        with table.open() as stream:
            for r in csv.DictReader(stream):
                key = (float(r['trajectory.speed_mps']), int(r['inference.latency_ms']))
                if key[0] not in SPEEDS[task]:
                    continue
                if key[1] not in latencies or key in published:
                    raise ValueError(f'{folder.name}: unexpected or repeated setting {key}')
                published[key] = (folder, r['run_id'], int(r['successes']), int(r['episodes']))
    if set(published) != expected:
        raise ValueError(f'{task} {mode}: missing conditions {sorted(expected - set(published))}')
    rows = []
    for (speed, latency), (folder, run_id, successes, episodes) in sorted(published.items()):
        point = folder / 'points' / f'{speed:g}_{mode}_{latency}'
        record = yaml.safe_load((point / 'point.yaml').read_text())
        experiment = point / 'runs' / run_id / 'experiment.yaml'
        jsonl = point / 'artifacts' / run_id / 'episodes.jsonl'
        for path in (point / 'point.yaml', experiment, jsonl):
            sources[portable(path)] = sha(path)
        if record['run_id'] != run_id or record['episodes'] != 200 or record['successes'] != successes or episodes != 200:
            raise ValueError(f'{point}: point record disagrees with the published table')
        config = yaml.safe_load(experiment.read_text())
        inference = config['inference']
        if config['model']['checkpoint'] != CHECKPOINT or config['model']['backend'] != 'molmoact2' or inference['action_chunk_steps'] != 10:
            raise ValueError(f'{point}: wrong policy configuration')
        if (inference['mode'], inference['latency_ms']) != (mode, latency):
            raise ValueError(f'{point}: wrong execution setting')
        if inference['execution_horizon_steps'] != 5 or config['benchmark']['max_episode_steps'] != 600:
            raise ValueError(f'{point}: wrong horizon or action budget')
        trajectory = config['trajectory']
        if abs(float(trajectory['speed_mps']) - speed) > 1e-12 or trajectory['target'] != TARGET[task]:
            raise ValueError(f'{point}: wrong trajectory')
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
            rows.append(dict(task=task, speed=speed, mode=mode, latency=latency, episode_id=e['episode_id'],
                             init_state=e['episode_id'] % 50, initial_phase=entry['initial_phase'],
                             direction=entry['initial_direction'], success=int(bool(e['success'])), steps=e['steps'],
                             control_steps=e['control_steps'], wait_steps=e['wait_steps'],
                             policy_requests=e['policy_requests'], study=folder.name, run_id=run_id,
                             commit=record['commit'], trace='artifacts/' + str(trace.relative_to(ARTIFACTS)),
                             trace_sha256=e['trace_sha256']))
    return rows, sources


def ensure(*, workers=4, force=False):
    import yaml
    folder = (FOLDER)
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
                print('molmoact2-episodes: run records and output verified; using cached table', flush=True)
                return folder
        except (OSError, ValueError, KeyError):
            pass
    rows, sources = [], {}
    for task in TASKS:
        for mode in MODES:
            values, used = study_rows(task, mode)
            rows.extend(values)
            sources.update(used)
    keys = {(r['task'], r['speed'], r['mode'], r['latency'], r['episode_id']) for r in rows}
    if len(rows) != 115200 or len(keys) != 115200:
        raise ValueError(f'Expected 115,200 unique MolmoAct2 episodes, found {len(rows)}')
    with (folder / 'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = dict(code=code, environment=environment, sources=sources, episodes=len(rows),
                   outputs={'episodes.csv': sha(folder / 'episodes.csv')})
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f'molmoact2-episodes: {len(rows)} episodes from {len(sources)} run records', flush=True)
    return folder
