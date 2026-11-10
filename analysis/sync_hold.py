"""Synchronous hold_pose reruns: the success of every synchronous point at a nonzero latency on the four LIBERO tasks.

Synchronous waits first sent a zero action, so the arm kept turning toward its last orientation goal while it waited.
Study sync_hold_pose_20261026 reran every such point with the arm holding its pose (idle_control hold_pose): pi0.5 and
MolmoAct2 at 50-250 ms, GR00T N1.7 at 100 and 200 ms, nine speeds, 200 episodes each. Its published tables
(results/libero/<task>/<policy>-sync-hold-pose/plot_data.csv) replace the original synchronous successes in
analysis.figures.tasks.success_rows and analysis.latency_model.data.load. The tables hold only totals per point:
within a replaced point the first k episodes by episode_id succeed, so totals are exact and splits by direction or
episode are not meaningful for these points. Measurements that replay traces read the reruns' own episode records and
traces instead (rerun, trace_rows), under VLALAGBENCH_ARTIFACTS (SYNC_HOLD_STUDY); rows built from them keep their real outcomes.
"""
import csv
from functools import cache

from analysis.paths import RESULTS, STUDIES

TASK_FOLDERS = {'basket': '20260908_moving-basket', 'plate': '20261001_task9-moving-plate',
                'cabinet': '20261010_task9-moving-cabinet-bowl-centered', 'drawer': '20260926_moving-drawer-fixed'}
POLICY_TAGS = {'pi0.5': 'pi05', 'GR00T N1.7': 'gr00t', 'MolmoAct2': 'molmoact2'}
SYNC_HOLD_STUDY = STUDIES / 'sync_hold_pose_20261026'
# The moving cabinet's rerun is named for the bowl the cabinet carries.
STUDY_TASKS = {'basket': 'basket', 'plate': 'plate', 'cabinet': 'bowl', 'drawer': 'drawer'}
# Episode-row source of every rerun point, in tables keyed by study source.
SOURCE = SYNC_HOLD_STUDY.name


def table(policy, task):
    """The published rerun table of one policy and task."""
    return RESULTS / f'libero/{TASK_FOLDERS[task]}/{POLICY_TAGS[policy]}-sync-hold-pose/plot_data.csv'


@cache
def successes(policy, task):
    """(speed, latency ms) -> successes of 200 for the policy's synchronous hold_pose reruns on the task."""
    out = {}
    with table(policy, task).open() as stream:
        for r in csv.DictReader(stream):
            if r['idle_control'] != 'hold_pose':
                continue
            if r['inference.mode'] != 'synchronous' or r['episodes'] != '200':
                raise ValueError(f'{policy} {task}: unexpected rerun row {r["point"]}')
            out[(float(r['trajectory.speed_mps']), int(r['inference.latency_ms']))] = int(r['successes'])
    return out


def study(policy, task):
    return SYNC_HOLD_STUDY / f'{POLICY_TAGS[policy]}_{STUDY_TASKS[task]}_synchronous_hold_pose_b32_20261026'


def point_dir(policy, task, speed, latency):
    return study(policy, task) / 'points' / f'{speed:g}_synchronous_{latency}'


@cache
def rerun(policy, task, speed, latency):
    """(run_id, configuration, episode records by episode_id) of one rerun point, checked against its point record and
    the published table."""
    import json
    import yaml
    point = point_dir(policy, task, speed, latency)
    record = yaml.safe_load((point / 'point.yaml').read_text())
    run = record['run_id']
    config = yaml.safe_load((point / 'runs' / run / 'experiment.yaml').read_text())
    inference = config['inference']
    if (inference['mode'], inference['latency_ms'], inference.get('idle_control')) != ('synchronous', latency, 'hold_pose') \
            or abs(float(config['trajectory']['speed_mps']) - speed) > 1e-12:
        raise ValueError(f'{point}: not the hold_pose rerun of this setting')
    jsonl = point / 'artifacts' / run / 'episodes.jsonl'
    episodes = sorted((json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()),
                      key=lambda e: e['episode_id'])
    if [e['episode_id'] for e in episodes] != list(range(200)) or any(e['error'] for e in episodes):
        raise ValueError(f'{point}: incomplete episode records')
    k = sum(bool(e['success']) for e in episodes)
    if record['episodes'] != 200 or k != record['successes'] or k != successes(policy, task)[(speed, latency)]:
        raise ValueError(f'{point}: episode records disagree with the point record or the published table')
    return run, config, episodes


def trace_rows(policy, task, rows):
    """Rows of the benchmark episode tables (pi0.5 and GR00T basket and drawer: source, episode_id, base_state,
    initial_phase, direction, success, steps, ...) with every synchronous point at a nonzero latency taken from its
    rerun's episode records; source becomes SOURCE. Each rerun episode must start from the same state and target
    phase and direction as the episode it replaces."""
    if not covered(policy, task):
        return rows
    out = []
    for r in rows:
        latency = int(r['latency'])
        if r['mode'] == 'synchronous' and latency > 0:
            speed = float(r['speed'])
            _, config, episodes = rerun(policy, task, speed, latency)
            e = episodes[int(r['episode_id'])]
            entry = config['trajectory']['episode_schedule'][e['episode_id']]
            states = config['benchmark']['initial_state_indices']
            if (abs(float(r['initial_phase']) - entry['initial_phase']) > 1e-9 or int(r['direction']) != entry['initial_direction']
                    or int(r['base_state']) != states[e['episode_id'] % len(states)]):
                raise ValueError(f'{policy} {task} {speed} m/s {latency} ms episode {e["episode_id"]}: rerun condition differs')
            r = dict(r, source=SOURCE, success=str(int(bool(e['success']))), steps=str(e['steps']),
                     control_steps=str(e['control_steps']), wait_steps=str(e['wait_steps']),
                     policy_requests=str(e['policy_requests']))
        out.append(r)
    return out


def from_rerun(row):
    """True for an episode row read from a rerun's own records (trace_rows, or analysis.cohorts.plate_cabinet's study rows)."""
    return row.get('source') == SOURCE or row.get('study', '').endswith('_synchronous_hold_pose_b32_20261026')


def covered(policy, task):
    return policy in POLICY_TAGS and task in TASK_FOLDERS


def replace_rows(policy, task, rows):
    """Per-episode success rows with every synchronous point at a nonzero latency taken from the reruns."""
    if not covered(policy, task):
        return rows
    hold = successes(policy, task)
    cells = {}
    for r in rows:
        # Rows already read from a rerun's episode records keep their real outcomes.
        if r['mode'] == 'synchronous' and int(r['latency']) > 0 and not from_rerun(r):
            cells.setdefault((float(r['speed']), int(r['latency'])), []).append(r)
    succeeds = {}
    for key, members in cells.items():
        if key not in hold:
            raise ValueError(f'{policy} {task}: no hold_pose rerun at {key[0]} m/s, {key[1]} ms')
        if len(members) != 200:
            raise ValueError(f'{policy} {task} {key}: {len(members)} episodes')
        members.sort(key=lambda r: int(r.get('episode_id') or 0))
        succeeds.update((id(r), i < hold[key]) for i, r in enumerate(members))
    return [dict(r, success='1' if succeeds[id(r)] else '0') if id(r) in succeeds else r for r in rows]


def replace_points(policy, task, rows):
    """latency_model.data point dicts with every synchronous point at a nonzero latency taken from the reruns."""
    if not covered(policy, task):
        return rows
    hold = successes(policy, task)
    out = []
    for r in rows:
        if r['mode'] == 'sync' and r['lat'] > 0:
            if (r['v'], r['lat']) not in hold:
                raise ValueError(f'{policy} {task}: no hold_pose rerun at {r["v"]} m/s, {r["lat"]} ms')
            r = dict(r, k=hold[(r['v'], r['lat'])])
        out.append(r)
    return out
