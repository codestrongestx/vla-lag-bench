"""Execution horizons no calibration includes: pi0.5's asynchronous execution at 50 ms with P = 2, 3, and 4.

Each task's model is fitted on every measured line at the default horizon H = 5 (sync 0 ms and every mode at
50-250 ms; fits.fit_task, M1). It predicts these schedules from their staleness and idle time alone, relative to
synchronous 0 ms at H = 5, by reading the measured synchronous 0 ms curve at the implied speed. The runs (horizon
validation of October 25, nine speeds x 200 episodes) are published in results/libero/<task>/pi05-horizon-validation.
"""
import csv

import numpy as np

from analysis.latency_model.data import FITTED_TABLES
from analysis.latency_model.fits import fit_task, task_points
from analysis.latency_model.speed_latency_model import schedule_features

TASKS = ('plate', 'basket')
REF_HORIZON = 5


def table(task):
    return FITTED_TABLES[task].parent.parent / 'pi05-horizon-validation/plot_data.csv'


def measured(task):
    """(mode, latency ms, horizon, speed) -> (successes, episodes) of the task's horizon runs."""
    with table(task).open() as stream:
        return {(r['inference.mode'], int(r['inference.latency_ms']), int(r['inference.execution_horizon_steps']),
                 float(r['trajectory.speed_mps'])): (int(r['successes']), int(r['episodes'])) for r in csv.DictReader(stream)}


def errors(task):
    """Prediction minus measured success of every horizon run of the task."""
    model = fit_task(task_points('pi0.5', task, grid=True), task)
    out = []
    for (mode, latency, horizon, speed), (k, n) in sorted(measured(task).items()):
        ds, pause = schedule_features(mode, latency, horizon, ref_horizon=REF_HORIZON)
        kappa = max(1 + model['a'] * ds + model['b'] * pause, 1e-3)
        out.append(float(model['reference'](speed * kappa)) - k / n)
    return np.array(out)
