"""The model fitted to the figures' success counts (figures/tasks.py), for the figures and numbers that use it.

Figures read success through the episode tables in analysis/data/, which aggregate to the published result tables
that data.py reads. Each task is fitted on its own: S0 is the task's measured synchronous 0 ms curve
(speed_latency_model.Reference), and the two sensitivities a and b are fitted by binomial maximum likelihood without
a prior on the delayed lines. For pi0.5 and MolmoAct2 the 50, 150, and 250 ms lines can be held out, which is the
blind test of refined.py (M1).
"""
import numpy as np

from analysis.latency_model.data import BLIND_LATENCIES, from_counts
from analysis.latency_model.kappa_ci import kappas
from analysis.latency_model.speed_latency_model import Reference, fit_ab, kappa, predict_from

# Execution horizon H and control rate per simulator: LIBERO 5 actions at 20 Hz, DynamicVLA 14 actions at 25 Hz.
SCHEDULE = {'LIBERO': (5, 20), 'DynamicVLA': (14, 25)}


def schedule_of(task):
    from analysis.figures.tasks import TASKS
    return SCHEDULE[TASKS[task].simulator]


def task_points(policy, task, *, grid=False):
    """Points of one task from the figures' success counts; with grid, the 50/150/250 ms lines are marked new."""
    from analysis.figures.tasks import counts
    horizon, hz = schedule_of(task)
    points = from_counts(counts(policy, task, grid=grid), task, horizon=horizon, hz=hz)
    if grid:
        new = [lat in BLIND_LATENCIES for lat in points.lat]
        points.columns['source'] = points.source.astype(object)
        points.columns['source'][new] = 'new'
    return points


def fit_task(points, task, *, on=('old', 'new')):
    """M1 fit of one task on the lines whose source is in `on`: the measured sync-0 reference and (a, b) fitted on
    the delayed lines."""
    horizon, hz = schedule_of(task)
    used = points[np.isin(points.source, on)]
    zero = used.line == 'sync0'
    reference = Reference(zip(used.v[zero], used.k[zero], used.n[zero]))
    delayed = used[~zero]
    a, b = fit_ab(reference, (0.0, 0.0), delayed.features(horizon, hz), delayed.v, delayed.k, delayed.n)
    return dict(reference=reference, a=a, b=b)


def curve(result, task, mode, latency, speeds):
    """The model's success at the given speeds for one setting."""
    horizon, hz = schedule_of(task)
    return predict_from(result['reference'], speeds, mode, latency, result['a'], result['b'], horizon, hz)


def speedup(result, task, mode, latency):
    horizon, hz = schedule_of(task)
    return kappa(mode, latency, result['a'], result['b'], horizon, hz)


def line_speedups(points, result, task):
    """Each delayed line's own speedup with its 95% profile interval, S0 fixed at the task's fit."""
    horizon, hz = schedule_of(task)
    return kappas(points, {task: result['reference']}, horizon=horizon, hz=hz)
