"""Noise floor of the held-out test: the RMS error an exact model would make from sampling noise alone.

The held-out test (fits.fit_task on the 0/100/200 ms lines, predicting the 50/150/250 ms lines) compares each
prediction with a measured success rate, and each prediction reads the task's measured synchronous 0 ms reference
curve. Both carry binomial noise. Each draw resamples the reference's success counts, rebuilds the Reference (which
refits its log-logistic tail), and predicts every held-out point with the fitted a and b; the variance of those
predictions is added to each held-out point's own p(1 - p)/n. Noise in the fitted a and b is left out, so the floor
is a lower bound. Prints, per policy, the RMS of each part and of their sum in percentage points (the model appendix).
"""
import numpy as np

from analysis.latency_model.data import load
from analysis.latency_model.fits import fit_task
from analysis.latency_model.speed_latency_model import Reference, predict_from

TASKS = ('basket', 'plate', 'cabinet', 'drawer')
DRAWS = 400


def floor(policy, draws=DRAWS, seed=0):
    """(held-out point noise, reference noise, combined) RMS in percentage points."""
    rng = np.random.default_rng(seed)
    points = load(policy=policy)
    rows = [r if isinstance(r, dict) else dict(zip(points.FIELDS, r)) for r in points.rows()]
    own, reference = [], []
    for task in TASKS:
        fit = fit_task(points.where(task=task), task, on=('old',))
        curve = sorted((r['v'], r['k'], r['n']) for r in rows if r['task'] == task and r['line'] == 'sync0')
        held = [r for r in rows if r['task'] == task and r['source'] == 'new']
        predicted = np.empty((draws, len(held)))
        for draw in range(draws):
            resampled = Reference([(v, int(rng.binomial(n, k / n)), n) for v, k, n in curve])
            predicted[draw] = [predict_from(resampled, np.array([r['v']]), r['mode'], r['lat'], fit['a'], fit['b'])[0]
                               for r in held]
        reference += list(predicted.var(axis=0))
        own += [(r['k'] / r['n']) * (1 - r['k'] / r['n']) / r['n'] for r in held]
    own, reference = np.array(own), np.array(reference)
    return tuple(float(100 * np.sqrt(part.mean())) for part in (own, reference, own + reference))


if __name__ == '__main__':
    for policy in ('pi0.5', 'MolmoAct2'):
        held, ref, both = floor(policy)
        print(f'{policy}: held-out points {held:.2f}, reference curve {ref:.2f}, combined floor {both:.2f} points')
