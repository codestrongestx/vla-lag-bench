"""Per-episode success tables of the moving basket (release) and moving drawer (articulated), pi0.5 and GR00T N1.7.

pi0.5: the basket sweep of September 8 (with its 0.4-1.5 m/s extension) and the fixed-placement drawer rerun of
September 26. GR00T N1.7: the September 15 basket rerun, selected by the run IDs of its published table, and the
fixed-placement drawer rerun of October 4. The synchronous settings at a nonzero latency come from the hold_pose
reruns (analysis/sync_hold.py), which the published tables list beside these. ensure() joins every published point's episode records
(episodes.jsonl) with the episode schedule of its run configuration, from benchmark artifact storage; the tables
ship in analysis/data/libero_episodes/.
"""
import csv
import json
import os
from pathlib import Path

from analysis import sync_hold
from analysis.paths import DATA, DERIVED, RESULTS, STUDIES, point_dir, read_rows

TABLES = {'pi0.5': DATA / 'libero_episodes/pi05.csv.gz', 'GR00T N1.7': DATA / 'libero_episodes/gr00t.csv.gz'}
FOLDER = DERIVED / 'libero_episodes'
PUBLISHED = {
    ('pi0.5', 'basket'): RESULTS / 'libero/20260908_moving-basket/pi05/plot_data.csv',
    ('pi0.5', 'drawer'): RESULTS / 'libero/20260926_moving-drawer-fixed/pi05/plot_data.csv',
    ('GR00T N1.7', 'basket'): RESULTS / 'libero/20260908_moving-basket/gr00t/plot_data.csv',
    ('GR00T N1.7', 'drawer'): RESULTS / 'libero/20260926_moving-drawer-fixed/gr00t/plot_data.csv',
}
# Point sources of the drawer reruns, whose published tables carry no source column.
DRAWER_SOURCE = {'pi0.5': ('pi05_drawer_fixed_20260926', {'synchronous': 'sync', 'lazy': 'lazy', 'continuous': 'continuous'}),
                 'GR00T N1.7': ('gr00t_drawer_fixed_20261004', {m: m for m in ('synchronous', 'lazy', 'continuous')})}
# GR00T's traces live in its reruns; each point is found by the run ID of its published row.
GR00T_ROOTS = {'basket': STUDIES / 'gr00t_speed_sweep_repeat_20260915', 'drawer': STUDIES / 'gr00t_drawer_fixed_20261004'}
FIELDS = ('task', 'speed', 'mode', 'latency', 'source', 'episode_id', 'base_state', 'success', 'steps', 'control_steps',
          'wait_steps', 'policy_requests', 'initial_phase', 'direction', 'error', 'trace_local')


def episode_rows(policy):
    """Validated per-episode rows of one policy's basket and drawer studies."""
    return read_rows(TABLES[policy])


def published_rows(policy, task):
    with PUBLISHED[policy, task].open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if task == 'drawer':
            study, suffix = DRAWER_SOURCE[policy]
            row['source'] = f'{study}_{suffix[row["inference.mode"]]}'
        if int(row['episodes']) != 200:
            raise ValueError(f'{PUBLISHED[policy, task]}: incomplete point {row}')
    if (policy, task) == ('GR00T N1.7', 'drawer'):
        # The GR00T drawer table was assembled from the rerun's point records in this order.
        rows.sort(key=lambda r: (r['inference.mode'], int(r['inference.latency_ms']), float(r['trajectory.speed_mps'])))
    return rows


def point_of(policy, task, row, speed, mode, latency):
    if policy == 'pi0.5':
        return point_dir(row['source'], speed, mode, latency)
    matches = list(GR00T_ROOTS[task].glob(f'*/points/{speed:g}_{mode}_{latency}/runs/{row["run_id"]}/experiment.yaml'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one rerun point for {task} {speed} {mode} {latency}: {matches}')
    return str(matches[0].parents[2])


def build(policy):
    """Join per-episode outcomes with their schedule entries for basket and drawer."""
    import glob
    import yaml
    rows = []
    for task in ('basket', 'drawer'):
        for r in published_rows(policy, task):
            speed, mode, lat = float(r['trajectory.speed_mps']), r['inference.mode'], int(r['inference.latency_ms'])
            d = point_of(policy, task, r, speed, mode, lat)
            ep_files = glob.glob(os.path.join(d, 'artifacts', '*', 'episodes.jsonl'))
            exp_files = glob.glob(os.path.join(d, 'runs', '*', 'experiment.yaml'))
            if policy != 'pi0.5':
                ep_files = [f for f in ep_files if Path(f).parent.name == r['run_id']]
                exp_files = [f for f in exp_files if Path(f).parent.name == r['run_id']]
            assert len(ep_files) == 1 and len(exp_files) == 1, d
            assert os.path.basename(os.path.dirname(ep_files[0])) == r['run_id'], (d, r['run_id'])
            exp = yaml.safe_load(open(exp_files[0]))
            sched = exp['trajectory']['episode_schedule']
            isi = exp['benchmark']['initial_state_indices']
            eps = [json.loads(line) for line in open(ep_files[0])]
            assert len(eps) == 200 == len(sched), d
            n_ok = sum(e['success'] for e in eps)
            assert n_ok == int(r['successes']), (d, n_ok, r['successes'])
            for e in eps:
                i = e['episode_id']
                trace = os.path.join(d, 'artifacts', os.path.basename(os.path.dirname(e['trace'])),
                                     os.path.basename(e['trace']))
                rows.append(dict(task=task, speed=speed, mode=mode, latency=lat, source=r['source'], episode_id=i,
                                 base_state=isi[i % len(isi)], success=int(e['success']), steps=e['steps'],
                                 control_steps=e['control_steps'], wait_steps=e['wait_steps'],
                                 policy_requests=e['policy_requests'], initial_phase=sched[i]['initial_phase'],
                                 direction=sched[i]['initial_direction'], error=e['error'] or '',
                                 trace_local=int(os.path.exists(trace))))
        # The reruns at the latencies this study ran; the 50/150/250 ms grid has its own table.
        latencies = {int(r['inference.latency_ms']) for r in published_rows(policy, task)} - {0}
        for (speed, lat), successes in sorted(sync_hold.successes(policy, task).items()):
            if lat not in latencies:
                continue
            run, config, eps = sync_hold.rerun(policy, task, speed, lat)
            d = sync_hold.point_dir(policy, task, speed, lat)
            sched = config['trajectory']['episode_schedule']
            isi = config['benchmark']['initial_state_indices']
            for e in eps:
                i = e['episode_id']
                trace = d / 'artifacts' / run / os.path.basename(e['trace'])
                rows.append(dict(task=task, speed=speed, mode='synchronous', latency=lat, source=sync_hold.SOURCE,
                                 episode_id=i, base_state=isi[i % len(isi)], success=int(e['success']),
                                 steps=e['steps'], control_steps=e['control_steps'], wait_steps=e['wait_steps'],
                                 policy_requests=e['policy_requests'], initial_phase=sched[i]['initial_phase'],
                                 direction=sched[i]['initial_direction'], error=e['error'] or '',
                                 trace_local=int(trace.exists())))
    return rows


def ensure(*, workers=None, force=False):
    FOLDER.mkdir(parents=True, exist_ok=True)
    for policy, name in (('pi0.5', 'pi05'), ('GR00T N1.7', 'gr00t')):
        rows = build(policy)
        with (FOLDER / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f'libero-episodes: {len(rows)} {policy} episodes written to {FOLDER / f"{name}.csv"}', flush=True)
    return FOLDER
