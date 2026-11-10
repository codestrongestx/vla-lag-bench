"""The law fitted over the four tasks and tested blind: fit on 0/100/200 ms, predict 50/150/250 ms.

  S_line(v) = rho_line * S0_task(kappa * v)
  kappa     = 1 + a_task * dage + b_task * pause                    (scheduler quantities)
  rho_line  = exp(-c * max(0, mean executed chunk index - 4))      (one policy constant c)

S0_task is the task's measured synchronous 0 ms curve among the fitted points (speed_latency_model.Reference), so
only a and b are fitted per task. M1 is the plain law (c = 0), M2 adds the shared plan-tail constant c, and M4 adds,
per task, the interception term gamma of the synchronous lines: a failure survives each paused sweep of the target
with probability exp(-gamma * v * L * dt).
"""
import csv
import json

import numpy as np
from scipy import optimize

from analysis.latency_model.data import POLICIES, TASKS, load, output_folder
from analysis.latency_model.fitcore import nll_binom, noise_pp, rmse_pp
from analysis.latency_model.speed_latency_model import AB_BOUNDS, HZ, TAIL_START, Reference, fit_ab, schedule

VARIANTS = ('M1', 'M2', 'M4')
NAMES = {'M1': 'law (a, b per task)', 'M2': '+ plan-tail cost c', 'M4': '+ interception gamma per task'}


class Fitted:
    """Parameters x (a, b per task, then c and the per-task gammas), their negative log likelihood, and the
    per-task references they read."""

    def __init__(self, x, fun, references):
        self.x, self.fun, self.references = np.asarray(x, float), float(fun), references


class JointLaw:
    """(a, b) per task on each task's measured S0, then c and the per-task gammas, over one set of points."""

    def __init__(self, points, tasks=TASKS):
        self.points, self.tasks = points, tasks
        self.feats = points.features()
        midx = np.array([schedule(m, L)['idx'] for m, L in zip(points.mode, points.L)])
        self.over = np.maximum(0.0, midx - TAIL_START)
        self.ti = np.array([tasks.index(t) for t in points.task])
        self.lsync = np.where(points.mode == 'sync', points.L, 0).astype(float)
        self.v, self.k, self.n = points.v.astype(float), points.k.astype(float), points.n.astype(float)
        self.zero = np.asarray(points.line) == 'sync0'

    @property
    def nt(self):
        return len(self.tasks)

    def references(self, mask):
        """Each task's measured synchronous 0 ms curve among the masked points."""
        refs = []
        for j in range(self.nt):
            m = mask & self.zero & (self.ti == j)
            refs.append(Reference(zip(self.v[m], self.k[m], self.n[m])))
        return refs

    def predict(self, x, variant, references):
        nt, p = self.nt, np.zeros(len(self.v))
        c = x[nt * 2] if variant in ('M2', 'M4') else 0.0
        for j in range(nt):
            m = self.ti == j
            a, b = x[j * 2], x[j * 2 + 1]
            kap = np.maximum(1 + a * self.feats[m, 0] + b * self.feats[m, 1], 1e-3)
            base = references[j](self.v[m] * kap)
            if variant == 'M4':
                gamma = x[nt * 2 + 1 + j]
                base = 1 - (1 - base) * np.exp(-gamma * self.v[m] * self.lsync[m] / HZ)
            p[m] = np.exp(-c * self.over[m]) * base
        return np.clip(p, 1e-9, 1 - 1e-9)

    def fit(self, mask, variant):
        nt = self.nt
        refs = self.references(mask)
        x0 = []
        for j in range(nt):
            m = mask & ~self.zero & (self.ti == j)
            x0 += list(fit_ab(refs[j], (0.0, 0.0), self.feats[m], self.v[m], self.k[m], self.n[m]))

        def fun(x):
            return nll_binom(self.k[mask], self.n[mask], self.predict(x, variant, refs)[mask]).sum()

        if variant == 'M1':
            return Fitted(x0, fun(np.array(x0)), refs)
        bounds = list(AB_BOUNDS) * nt + [(0.0, 0.5)] + ([(0.0, 200.0)] * nt if variant == 'M4' else [])
        # Start from M1 itself (c = 0, gamma = 0), which the variant contains, and from a typical c and gamma; the
        # likelihood is not smooth, so keep the better end point.
        best = None
        for c0, g0 in ((0.0, 0.0), (0.02, 5.0)):
            start = x0 + [c0] + ([g0] * nt if variant == 'M4' else [])
            r = optimize.minimize(fun, start, method='Powell', bounds=bounds,
                                  options=dict(maxiter=200000, xtol=1e-8, ftol=1e-12))
            if best is None or r.fun < best.fun:
                best = r
        return Fitted(best.x, best.fun, refs)

    def task_params(self, fitted, variant, j):
        nt, x = self.nt, fitted.x
        return dict(a=float(x[j * 2]), b=float(x[j * 2 + 1]), midpoint=fitted.references[j].midpoint_speed(),
                    c=float(x[nt * 2]) if variant != 'M1' else 0.0,
                    gamma=float(x[nt * 2 + 1 + j]) if variant == 'M4' else 0.0)


def blind_test(points=None, variants=VARIANTS):
    """Fit every variant on the old lines and on all lines; score both on the new lines."""
    points = load() if points is None else points
    law = JointLaw(points)
    k, n = law.k, law.n
    old = points.source == 'old'
    new = ~old
    summary, predictions = {}, {}
    for variant in variants:
        fo, fa = law.fit(old, variant), law.fit(np.ones(len(points), bool), variant)
        po, pa = law.predict(fo.x, variant, fo.references), law.predict(fa.x, variant, fa.references)
        res = dict(blind=rmse_pp(k[new], n[new], po[new]), insample_new=rmse_pp(k[new], n[new], pa[new]),
                   insample_all=rmse_pp(k, n, pa), nll_all=float(fa.fun), npar=len(fa.x),
                   x_old=[float(x) for x in fo.x], x_all=[float(x) for x in fa.x])
        res['tasks'] = {}
        for j, t in enumerate(TASKS):
            m = new & (law.ti == j)
            res['tasks'][t] = dict(blind=rmse_pp(k[m], n[m], po[m]), old=law.task_params(fo, variant, j),
                                   all=law.task_params(fa, variant, j))
        res['by_latency'] = {int(lat): rmse_pp(k[new & (points.lat == lat)], n[new & (points.lat == lat)],
                                               po[new & (points.lat == lat)]) for lat in sorted(set(points.lat[new]))}
        summary[variant] = res
        predictions[variant] = (po, pa)
    summary['noise_new'] = noise_pp(k[new], n[new])
    return summary, predictions


def by_line(points, predictions):
    """Blind RMSE of every new line under each variant, with its sampling-noise floor."""
    rows = []
    new = points.source == 'new'
    for task in TASKS:
        for line in points.where(task=task, source='new').lines():
            m = new & (points.task == task) & (points.line == line)
            row = dict(task=task, mode=line[:4], lat=int(line[4:]))
            row.update({variant: rmse_pp(points.k[m], points.n[m], po[m]) for variant, (po, _) in predictions.items()})
            row['noise'] = noise_pp(points.k[m], points.n[m])
            rows.append(row)
    return rows


def report(summary):
    for variant in (v for v in VARIANTS if v in summary):
        res = summary[variant]
        gamma = [round(res['tasks'][t]['all']['gamma'], 1) for t in TASKS] if variant == 'M4' else []
        c_old, c_all = res['tasks'][TASKS[0]]['old']['c'], res['tasks'][TASKS[0]]['all']['c']
        print(f"{variant} {NAMES[variant]}: blind RMSE on new lines {res['blind']:.2f} pp | in-sample on new "
              f"{res['insample_new']:.2f} | all {res['insample_all']:.2f} | nll {res['nll_all']:.1f} npar {res['npar']}"
              + (f" | c_old={c_old:.3f} c_all={c_all:.3f}" if variant != 'M1' else '')
              + (f" gamma_all={gamma}" if gamma else ''))
        for t in TASKS:
            d = res['tasks'][t]
            print(f"     {t:7s} blind {d['blind']:.2f}  a {d['old']['a']:.3f}->{d['all']['a']:.3f}  "
                  f"b {d['old']['b']:.3f}->{d['all']['b']:.3f}")
        print('     by latency: ' + '  '.join(f'{lat} ms {e:.2f}' for lat, e in res['by_latency'].items()))
    print(f"noise floor on new lines: {summary['noise_new']:.2f} pp")


def main(variants=VARIANTS, policy='pi0.5'):
    folder = output_folder(policy)
    points = load(policy=policy)
    summary, predictions = blind_test(points, variants)
    report(summary)
    (folder / 'refined_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    with (folder / 'refined_predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        names = [f'p_{v}_{kind}' for v in predictions for kind in ('blind', 'all')]
        writer.writerow(list(points.FIELDS) + names)
        values = [x for v in predictions for x in predictions[v]]
        for i, row in enumerate(points.rows()):
            writer.writerow([row[f] for f in points.FIELDS] + [f'{p[i]:.6f}' for p in values])
    rows = by_line(points, predictions)
    with (folder / 'blind_by_line.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Written to {folder}')
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=POLICIES, default='pi0.5')
    main(policy=parser.parse_args().policy)
