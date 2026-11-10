"""Success tables for the latency model: pi0.5 and MolmoAct2 on the four LIBERO tasks, every speed x latency point.

For pi0.5, fitted lines (0/100/200 ms) come from the published result tables, blind lines (50/150/250 ms) from the
published October 14 latency grid (results/libero/*/pi05-latency-grid). MolmoAct2's published table per task holds
both. Each point carries its execution mode, latency, latency in control steps L, target speed v, and successes k out
of n episodes. Asynchronous 0 ms runs equal synchronous 0 ms by construction, so only the synchronous 0 ms line is
kept as the reference. Synchronous points at a nonzero latency come from the hold_pose reruns' published tables
(analysis/sync_hold.py).
"""
import csv
from pathlib import Path

import numpy as np

from analysis.paths import RESULTS
from analysis.latency_model.speed_latency_model import H, HZ, latency_steps, schedule_features

# Task keys follow figures/tasks.py: release, set-down, grasping, articulated.
TASKS = ('basket', 'plate', 'cabinet', 'drawer')
FITTED_TABLES = {
    'basket': RESULTS / 'libero/20260908_moving-basket/pi05/plot_data.csv',
    'plate': RESULTS / 'libero/20261001_task9-moving-plate/pi05-h100-b32/plot_data.csv',
    'cabinet': RESULTS / 'libero/20261010_task9-moving-cabinet-bowl-centered/pi05/plot_data.csv',
    'drawer': RESULTS / 'libero/20260926_moving-drawer-fixed/pi05/plot_data.csv',
}
MOLMOACT2_TABLES = {task: path.parent.parent / 'molmoact2/plot_data.csv' for task, path in FITTED_TABLES.items()}
# Every task uses 0.001 and 0.025-0.2 m/s. The bowl's tables above hold 0.001 and 0.05-0.4 m/s; the October 21 speed
# fill adds 0.025, 0.075, 0.125, and 0.175 m/s at every delay of both policies (same configuration otherwise), and the
# bowl's 0.25-0.4 m/s points are left out. The published speed-fill tables (*-speed-fill) hold the bowl's complete
# 0.001-0.2 m/s grid at every delay: the fill's runs and the bowl-centered and latency-grid runs at the other speeds.
SPEEDS = (.001, .025, .05, .075, .1, .125, .15, .175, .2)
BOWL = RESULTS / 'libero/20261010_task9-moving-cabinet-bowl-centered'
FILL_TAGS = {'pi0.5': 'pi05', 'MolmoAct2': 'molmoact2'}


def fill_table(policy):
    """The bowl's published 0.001-0.2 m/s table of one policy, every execution mode and delay."""
    return BOWL / f'{FILL_TAGS[policy]}-speed-fill' / 'plot_data.csv'


POLICIES = ('pi0.5', 'MolmoAct2')
MODES = {'synchronous': 'sync', 'lazy': 'lazy', 'continuous': 'cont'}
FITTED_LATENCIES = (0, 100, 200)
BLIND_LATENCIES = (50, 150, 250)


def grid_table(task):
    return FITTED_TABLES[task].parent.parent / 'pi05-latency-grid/plot_data.csv'


class Points:
    """Equal-length columns: task, mode (sync, lazy, cont), lat (ms), L (steps), v (m/s), k, n, source
    ('old' for the fitted 0/100/200 ms lines, 'new' for the blind 50/150/250 ms lines), and line ('lazy150')."""

    FIELDS = ('task', 'mode', 'lat', 'L', 'v', 'k', 'n', 'source', 'line')

    def __init__(self, **columns):
        self.columns = {name: np.asarray(columns[name]) for name in self.FIELDS}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.columns[key]
        return Points(**{name: values[key] for name, values in self.columns.items()})

    def __len__(self):
        return len(self.columns['v'])

    def __getattr__(self, name):
        try:
            return self.__dict__['columns'][name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def p(self):
        return self.k / self.n

    def features(self, horizon=H, hz=HZ):
        """Extra mean observation age (steps) and idle steps per acting step of each point's line."""
        return np.array([schedule_features(m, lat, horizon, hz) for m, lat in zip(self.mode, self.lat)], float).reshape(-1, 2)

    def where(self, **conditions):
        mask = np.ones(len(self), bool)
        for name, value in conditions.items():
            mask &= self.columns[name] == value
        return self[mask]

    def lines(self):
        """Line names in a stable order: mode, then latency."""
        order = {'sync': 0, 'lazy': 1, 'cont': 2}
        return sorted(set(self.line), key=lambda line: (order[line[:4]], int(line[4:])))

    def rows(self):
        return [dict(zip(self.FIELDS, values)) for values in zip(*(self.columns[f] for f in self.FIELDS))]


def _read(path, task, source, latencies):
    with Path(path).open() as stream:
        table = list(csv.DictReader(stream))
    rows = []
    for r in table:
        lat = int(r['inference.latency_ms'])
        mode = MODES[r['inference.mode']]
        if lat not in latencies or (lat == 0 and mode != 'sync'):
            continue
        for key, expected in (('action_chunk_steps', '10'), ('execution_horizon_steps', str(H))):
            if r.get(key, expected) != expected:
                raise ValueError(f'{path}: {key} {r[key]} differs from the scheduled setting (K = 10, H = 5)')
        if int(r['episodes']) != 200:
            raise ValueError(f'{path}: {r["episodes"]} episodes at {r["trajectory.speed_mps"]} m/s')
        rows.append(dict(task=task, mode=mode, lat=lat, L=latency_steps(lat), v=float(r['trajectory.speed_mps']),
                         k=int(r['successes']), n=int(r['episodes']), source=source, line=f'{mode}{lat}'))
    return rows


def _hold(policy, task, source, latencies):
    """The synchronous points at the given nonzero latencies from the policy's hold_pose reruns of the task."""
    from analysis.sync_hold import successes
    return [dict(task=task, mode='sync', lat=lat, L=latency_steps(lat), v=v, k=k, n=200, source=source,
                 line=f'sync{lat}') for (v, lat), k in sorted(successes(policy, task).items()) if lat in latencies]


def _grid(rows, task):
    """Keep the task's grid speeds; each (line, speed) must appear once."""
    kept = [r for r in rows if r['v'] in SPEEDS]
    keys = [(r['line'], r['v']) for r in kept]
    if len(set(keys)) != len(keys):
        raise ValueError(f'{task}: a setting appears in more than one table')
    return kept


def from_rows(rows):
    return Points(**{name: [r[name] for r in rows] for name in Points.FIELDS})


def output_folder(policy='pi0.5'):
    """Where a policy's latency-model outputs go: pi0.5 at the top, MolmoAct2 in its own subfolder."""
    from analysis.paths import output
    folder = output('latency_model')
    if policy != 'pi0.5':
        folder = folder / 'molmoact2'
        folder.mkdir(exist_ok=True)
    return folder


def load(tasks=TASKS, within_grid=True, policy='pi0.5'):
    """Every point of the given tasks. within_grid keeps only the speeds the 50/150/250 ms grid covers
    (it drops pi0.5's moving-basket 0.4-1.5 m/s extension), as every fit and blind test does."""
    if policy == 'MolmoAct2':
        rows = []
        for task in tasks:
            table = fill_table(policy) if task == 'cabinet' else MOLMOACT2_TABLES[task]
            part = _read(table, task, 'old', FITTED_LATENCIES) + _read(table, task, 'new', BLIND_LATENCIES)
            part += _hold(policy, task, 'old', FITTED_LATENCIES) + _hold(policy, task, 'new', BLIND_LATENCIES)
            rows += _grid(part, task)
        rows.sort(key=lambda r: (r['task'], r['mode'], r['lat'], r['v']))
        return from_rows(rows)
    if policy != 'pi0.5':
        raise ValueError(f'No latency grid for {policy}')
    rows = []
    for task in tasks:
        if task == 'cabinet':
            new = _grid(_read(fill_table(policy), task, 'new', BLIND_LATENCIES), task)
            old = _grid(_read(fill_table(policy), task, 'old', FITTED_LATENCIES), task)
        else:
            new = _read(grid_table(task), task, 'new', BLIND_LATENCIES)
            old = _read(FITTED_TABLES[task], task, 'old', FITTED_LATENCIES)
        new += _grid(_hold(policy, task, 'new', BLIND_LATENCIES), task)
        old += _grid(_hold(policy, task, 'old', FITTED_LATENCIES), task)
        speeds = {r['v'] for r in new}
        if within_grid:
            old = [r for r in old if r['v'] in speeds]
        expected_new = {(v, MODES[m], lat) for v in speeds for m in MODES for lat in BLIND_LATENCIES}
        if {(r['v'], r['mode'], r['lat']) for r in new} != expected_new or len(new) != len(expected_new):
            raise ValueError(f'{task}: incomplete 50/150/250 ms grid')
        rows += old + new
    rows.sort(key=lambda r: (r['task'], r['mode'], r['lat'], r['v']))
    return from_rows(rows)


def from_counts(cells, task, *, horizon=H, hz=HZ):
    """Points from a analysis.figures.tasks count table {(speed, mode, latency): (successes, episodes)}, for the policies and
    simulators outside the pi0.5 grid (GR00T N1.7, DynamicVLA). Asynchronous 0 ms runs are dropped."""
    rows = []
    for (v, mode, lat), (k, n) in sorted(cells.items()):
        short = MODES[mode]
        if lat == 0 and short != 'sync':
            continue
        rows.append(dict(task=task, mode=short, lat=int(lat), L=latency_steps(lat, hz), v=float(v), k=int(k), n=int(n),
                         source='old', line=f'{short}{lat}'))
    return from_rows(rows)


if __name__ == '__main__':
    points = load(within_grid=False)
    for task in TASKS:
        part = points.where(task=task)
        print(f'{task:8s} {len(part):3d} points  old {np.sum(part.source == "old"):3d}  new {np.sum(part.source == "new"):3d}  '
              f'lines {" ".join(part.lines())}')
