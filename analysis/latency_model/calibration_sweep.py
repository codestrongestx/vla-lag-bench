"""Calibration study (Section 5.1 and the model appendix): one complete curve in one mode, plus one point in each of the other two modes.

Every reference line X (any mode, any delay, sync 0 ms included for comparison) is measured completely; one point on
a line of each of the other two modes gives the two ratios kappa_Y / kappa_X that fix a and b. Every other measured
point is then predicted from the measured curve S_X (transfer_sweep.Base). Both probes run at the grid speed nearest
the reference's midpoint speed / 1.4 (speed_latency_model.Reference.midpoint_speed), chosen before probing; `any speed` also scores every speed pair. Calibrations whose three lines cannot
separate age from pause (no synchronous line at a nonzero delay) are left out: a synchronous probe or reference is
required, and the asynchronous one may be lazy or continuous.

Scores per calibration: RMS error over every remaining point, over the lines at lower delays than the reference
(a slow curve predicting fast systems), and over the sync-0 line. pi0.5 and MolmoAct2 use the 0-250 ms grid
(data.load); GR00T N1.7 has only the 0/100/200 ms lines of the figures' success counts (fits.task_points). Also prints each
policy's sensitivities fitted on sync 0 and the 100/200 ms lines (fits.fit_task), in the paper's units: a per
100 ms of staleness (two control steps) and b per unit of idle fraction.
"""
import json

import numpy as np

from analysis.latency_model.data import TASKS, load, output_folder
from analysis.latency_model.fits import task_points, fit_task
from analysis.latency_model.speed_latency_model import PROBE_FACTOR, parse_line, separates_age_and_pause
from analysis.latency_model.transfer_sweep import Base

POLICIES = ('pi0.5', 'MolmoAct2', 'GR00T N1.7')
MODES = ('sync', 'lazy', 'cont')


def points_of(policy):
    if policy == 'GR00T N1.7':
        parts = [task_points(policy, t) for t in TASKS]
        rows = [r for p in parts for r in p.rows()]
        from analysis.latency_model.data import from_rows
        return from_rows(rows)
    return load(policy=policy)


def calibrations(r, base):
    """Index pairs (i, j): one point on a line of each mode other than the base line's, valid designs only."""
    mode = parse_line(base)[0]
    other = [m for m in MODES if m != mode]
    modes = np.array([parse_line(x)[0] for x in r.line])
    i, j = np.meshgrid(np.flatnonzero(modes == other[0]), np.flatnonzero(modes == other[1]), indexing='ij')
    pairs = np.stack([i.ravel(), j.ravel()], 1)
    ok = [separates_age_and_pause((base, r.line[a], r.line[b])) for a, b in pairs]
    return pairs[np.array(ok, bool)] if len(pairs) else pairs


def scores(r, base, pairs):
    """RMS (points) over remaining points, lower-delay lines, and the sync-0 line, per calibration."""
    g, err, err0 = r.error(pairs)
    lat = np.array([parse_line(x)[1] for x in r.line])
    lower = lat < parse_line(base)[1]
    out = []
    for gg, e, e0, (a, b) in zip(g, err, err0, pairs):
        keep = lower.copy()
        keep[[a, b]] = False
        low = float(np.sqrt(r.se[gg, keep].mean()) * 100) if keep.any() else np.nan
        out.append((float(e), low, float(e0), float(r.a[gg]), float(r.b[gg])))
    return out


def sweep(policy):
    points = points_of(policy)
    rows = []
    for t in TASKS:
        part = points.where(task=t)
        for base in part.lines():
            r = Base(part, base)
            pairs = calibrations(r, base)
            if not len(pairs):
                continue
            target = r.probe_speed(1 / PROBE_FACTOR, np.unique(r.v))
            for (a, b), (e, low, e0, fa, fb) in zip(pairs, scores(r, base, pairs)):
                rows.append(dict(task=t, base=base, probe1=r.line[a], v1=float(r.v[a]), probe2=r.line[b],
                                 v2=float(r.v[b]), fixed=bool(r.v[a] == target and r.v[b] == target),
                                 rms=e, rms_lower=low, rms_sync0=e0, a=fa, b=fb))
        print(f'{policy} {t}: {sum(x["task"] == t for x in rows)} calibrations', flush=True)
    return rows


def spread(x):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if not len(x):
        return None
    return dict(n=len(x), median=round(float(np.median(x)), 2), p10=round(float(np.percentile(x, 10)), 2),
                p90=round(float(np.percentile(x, 90)), 2), min=round(float(x.min()), 2), max=round(float(x.max()), 2))


def summarize(rows):
    fixed = [x for x in rows if x['fixed']]
    nonzero = [x for x in fixed if x['base'] != 'sync0']
    s = {'fixed speed, nonzero reference': spread(x['rms'] for x in nonzero),
         'fixed speed, sync-0 reference': spread(x['rms'] for x in fixed if x['base'] == 'sync0'),
         'any speed, nonzero reference': spread(x['rms'] for x in rows if x['base'] != 'sync0'),
         'sync-0 line from a nonzero reference': spread(x['rms_sync0'] for x in nonzero),
         'lower delays from the reference': spread(x['rms_lower'] for x in nonzero)}
    s['by task'] = {t: spread(x['rms'] for x in nonzero if x['task'] == t) for t in TASKS}
    s['by reference mode'] = {m: spread(x['rms'] for x in nonzero if parse_line(x['base'])[0] == m) for m in MODES}
    lats = sorted({parse_line(x['base'])[1] for x in nonzero})
    s['by reference delay'] = {str(l): spread(x['rms'] for x in nonzero if parse_line(x['base'])[1] == l) for l in lats}
    s['lower delays by reference delay'] = {str(l): spread(x['rms_lower'] for x in nonzero
                                                           if parse_line(x['base'])[1] == l) for l in lats}
    return s


def sensitivities(policy):
    """a (per 100 ms) and b per task, fitted on sync 0 and the 100/200 ms lines."""
    out = {}
    for t in TASKS:
        p = task_points(policy, t)
        keep = np.isin(p.lat, (0, 100, 200))
        from analysis.latency_model.data import from_rows
        res = fit_task(from_rows([r for r, k in zip(p.rows(), keep) if k]), t)
        out[t] = dict(a=round(2 * res['a'], 2), b=round(res['b'], 2))
    return out


def main():
    report = {}
    for policy in POLICIES:
        rows = sweep(policy)
        folder = output_folder(policy if policy != 'GR00T N1.7' else 'pi0.5') / 'calibration'
        folder.mkdir(parents=True, exist_ok=True)
        name = policy.replace(' ', '_').replace('.', '')
        (folder / f'{name}_calibrations.json').write_text(json.dumps(rows) + '\n')
        report[policy] = dict(summary=summarize(rows), sensitivities=sensitivities(policy))
        print(json.dumps(report[policy], indent=1), flush=True)
    out = output_folder('pi0.5') / 'calibration' / 'summary.json'
    out.write_text(json.dumps(report, indent=1) + '\n')
    print(f'Written to {out}')


if __name__ == '__main__':
    main()
