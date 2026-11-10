"""Per-line kappa (the equivalent speedup) with 95% profile intervals: S_line = r * S0(kappa v).

S0 is fixed at each task's measured synchronous 0 ms curve (speed_latency_model.Reference) and every delayed line
gets its own kappa and a free static factor r, so a constant offset cannot leak into kappa. Plotted against the line's mean observation age, lazy and continuous lines
fall on one curve, 1 + a * dage, and synchronous lines add b * L/H.
"""
import csv

import numpy as np
from scipy import optimize

from analysis.latency_model.data import POLICIES, TASKS, load, output_folder
from analysis.latency_model.fitcore import nll_binom
from analysis.latency_model.speed_latency_model import H, HZ, Reference, schedule


def line_kappa(points, reference):
    """Profile-likelihood kappa of one delayed line, with its 95% interval, given the task's S0 reference."""
    v, k, n = points.v.astype(float), points.k.astype(float), points.n.astype(float)
    top = float(reference.p.max())

    def nll(lk, lr):
        return nll_binom(k, n, np.clip(np.exp(lr) * reference(v * np.exp(lk)), 1e-9, 1 - 1e-9)).sum()

    def profile(lk):
        return optimize.minimize_scalar(lambda lr: nll(lk, lr), bounds=(np.log(0.5), -np.log(top)), method='bounded').fun

    best = optimize.minimize_scalar(profile, bounds=(np.log(0.3), np.log(5.0)), method='bounded')
    grid = np.linspace(np.log(0.3), np.log(5.0), 241)
    inside = grid[np.array([profile(g) for g in grid]) - best.fun <= 1.92]
    return float(np.exp(best.x)), float(np.exp(inside.min())), float(np.exp(inside.max()))


def kappas(points, reference_by_task, horizon=H, hz=HZ):
    rows = []
    for task, s0 in reference_by_task.items():
        part = points.where(task=task)
        for line in part.lines():
            if line == 'sync0':
                continue
            g = part.where(line=line)
            mode, L = str(g.mode[0]), int(g.L[0])
            sc = schedule(mode, L, horizon)
            kap, lo, hi = line_kappa(g, s0)
            rows.append(dict(task=task, line=line, mode=mode, lat=int(g.lat[0]), L=L, age=sc['age'], pause=sc['pause'],
                             kappa=kap, lo=lo, hi=hi))
    return rows


def main(policy='pi0.5'):
    folder = output_folder(policy)
    points = load(policy=policy)
    zero = {t: points.where(task=t, line='sync0') for t in TASKS}
    rows = kappas(points, {t: Reference(zip(z.v, z.k, z.n)) for t, z in zero.items()})
    with (folder / 'kappa_ci.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['task']:8s} {r['line']:8s} age {r['age']:4.1f} pause {r['pause']:.1f}  "
              f"kappa {r['kappa']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]")
    print(f'Written to {folder / "kappa_ci.csv"}')
    return rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=POLICIES, default='pi0.5')
    main(policy=parser.parse_args().policy)
