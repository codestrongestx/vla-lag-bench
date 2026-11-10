"""Strategy 2 on the four pi0.5 tasks: one complete line at a nonzero delay + one point on each of two other lines.

Strategy 1 (refined.py) fits S0 and (a, b) on sync 0 and the six 100/200 ms lines and predicts the other delays.
Strategy 2 measures one line completely at a nonzero delay, in any mode, plus one point on each of two other
lines at nonzero delays, and predicts every other line, the never-measured synchronous 0 ms curve included: the line's own
measured curve S_X (speed_latency_model.Reference) carries the shape, and each point gives one ratio
kappa_Y / kappa_X (speed_latency_model.transfer).

Both strategies solve the same two unknowns and need the same lines: at least one synchronous and one
asynchronous line at a nonzero delay (speed_latency_model.separates_age_and_pause). Synchronous lines alone fix
only a + b/H; asynchronous lines alone never fix b. The sweep scores those designs too, to show they fail.

Every task x every complete line at a nonzero delay (15) x every pair of other lines x every speed pair is scored
on every remaining point of the non-base lines (RMS error in points, probe points excluded), and separately on
the sync-0 line. Errors combine as RMS over the four tasks for the same base line, probe lines, and speed rule;
a speed rule probes both lines at the grid speed nearest c times the base line's midpoint speed (where its measured
curve first falls halfway between its highest and lowest success), chosen before probing.
Against selecting on the scored data: leave one task out (choose the lines on three tasks, score the fourth),
and the oracle (a, b) per base line, the floor no choice of probes beats. Also: the number of probes and sync 0 as the
base line for comparison.
"""
import csv
import gzip
import itertools
import json

import numpy as np

from analysis.latency_model.data import POLICIES, TASKS, load, output_folder
from analysis.latency_model.speed_latency_model import (PROBE_FACTOR, Reference, _ll, parse_line,
                                               separates_age_and_pause, transfer_curves, transfer_grid)

RULES = {'0.5 mid': 0.5, '0.6 mid': 0.6, f'mid/{PROBE_FACTOR:g}': 1 / PROBE_FACTOR, '0.85 mid': 0.85, 'mid': 1.0,
         '1.4 mid': 1.4}
RULE = f'mid/{PROBE_FACTOR:g}'     # the speed rule fixed in advance
PROBE_COUNTS = (1, 2, 3, 4, 14)
EXAMPLE = ('cont100', 'sync200', 'cont200')   # the paper's fixed example: lines from the 100/200 ms runs only
ORDER = {'sync': 0, 'lazy': 1, 'cont': 2}
CHUNK = 256


def line_key(line):
    return ORDER[line[:4]], int(line[4:])


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def design(base, l1, l2):
    """Which of age and pause the base line and the two probe lines can separate."""
    if separates_age_and_pause((base, l1, l2)):
        return 'synchronous and asynchronous'
    return 'no synchronous line' if not any(parse_line(x)[0] == 'sync' and parse_line(x)[1] > 0
                                            for x in (base, l1, l2)) else 'only synchronous lines'


class Base:
    """One complete line of one task, with every other point's success on the (a, b) grid."""

    def __init__(self, part, base):
        g = part.where(line=base)
        self.reference = Reference(zip(g.v, g.k, g.n))
        self.a, self.b, self.log_prior = transfer_grid()
        other = part[part.line != base]
        self.line, self.v = np.asarray(other.line), other.v.astype(float)
        k, n = other.k.astype(float), other.n.astype(float)
        p, ok = transfer_curves(self.reference, g.features()[0], other.features(), self.v, self.a, self.b)
        self.se = (p - (k / n)[None, :]) ** 2
        self.sse = self.se.sum(1)
        self.zero = self.line == 'sync0'
        self.sse0 = self.se[:, self.zero].sum(1)
        self.ll = np.where(ok, _ll(k[None, :], n[None, :], p), -1e9)
        self.nothing = float(np.sqrt(self.sse[np.argmin(np.abs(self.a) + np.abs(self.b))] / len(self.v)) * 100)
        self.oracle = float(np.sqrt(self.sse.min() / len(self.v)) * 100)

    def error(self, idx):
        """Fitted grid index, RMS error over the remaining points, and over the sync-0 line, for probes idx."""
        idx = np.atleast_2d(idx)
        g = np.concatenate([np.argmax(self.ll[:, c].sum(-1) + self.log_prior[:, None], axis=0)
                            for c in np.array_split(idx, max(1, len(idx) // CHUNK))])
        left = self.sse[g] - self.se[g[:, None], idx].sum(1)
        err = np.sqrt(left / (len(self.v) - idx.shape[1])) * 100
        n0 = self.zero.sum()
        if not n0:
            return g, err, np.full(len(g), np.nan)
        left0 = self.sse0[g] - (self.se[g[:, None], idx] * self.zero[idx]).sum(1)
        return g, err, np.sqrt(left0 / (n0 - self.zero[idx].sum(1))) * 100

    def probe_speed(self, factor, speeds):
        return self.reference.probe_speed(speeds, factor)


def sweep(points):
    """Every base line x line pair x speed pair. Returns rows and per-base baselines."""
    rows, bases = [], []
    for t in TASKS:
        part = points.where(task=t)
        for base in part.lines():
            r = Base(part, base)
            i, j = np.triu_indices(len(r.v), 1)
            keep = r.line[i] != r.line[j]
            i, j = i[keep], j[keep]
            g, err, err0 = r.error(np.stack([i, j], 1))
            bases.append(dict(task=t, base=base, midpoint=r.reference.midpoint_speed(), nothing=r.nothing,
                              oracle=r.oracle))
            for a, b, gg, e, e0 in zip(i, j, g, err, err0):
                l1, l2, v1, v2 = r.line[a], r.line[b], r.v[a], r.v[b]
                if line_key(l1) > line_key(l2):
                    l1, l2, v1, v2 = l2, l1, v2, v1
                rows.append((t, base, l1, float(v1), l2, float(v2), float(r.a[gg]), float(r.b[gg]), float(e), float(e0)))
        print(f'{t}: swept', flush=True)
    return rows, bases


def protocol_table(rows, bases, points):
    """Per (base line, probe lines): error per task under each speed rule, the best speeds, the median speeds,
    and the sync-0 line's error under the fixed rule."""
    mid = {(r['task'], r['base']): r['midpoint'] for r in bases}
    speeds = {t: np.unique(points.where(task=t).v) for t in TASKS}
    cells = {}
    for t, base, l1, v1, l2, v2, a, b, e, e0 in rows:
        cells.setdefault((base, l1, l2), {}).setdefault(t, {})[(v1, v2)] = (e, e0)
    table = []
    for (base, l1, l2), per in cells.items():
        row = dict(base=base, line1=l1, line2=l2, design=design(base, l1, l2))
        for rule, c in RULES.items():
            at = [per[t][(vs, vs)] for t in TASKS
                  for vs in [float(speeds[t][np.argmin(np.abs(speeds[t] - c * mid[(t, base)]))])]]
            row[rule] = [e for e, _ in at]
            if rule == RULE:
                row['sync0 line'] = [e0 for _, e0 in at]
        row['best speeds'] = [min(e for e, _ in per[t].values()) for t in TASKS]
        row['median speeds'] = [float(np.median([e for e, _ in per[t].values()])) for t in TASKS]
        table.append(row)
    return table


def leave_one_task_out(candidates, rules):
    held = {}
    for i, t in enumerate(TASKS):
        rest = [j for j in range(len(TASKS)) if j != i]
        x, r = min(((x, r) for x in candidates for r in rules), key=lambda xr: rms([xr[0][xr[1]][j] for j in rest]))
        held[t] = dict(protocol=f"{x['base']} + {x['line1']} + {x['line2']}", rule=r, error=x[r][i],
                       sync0_line=x['sync0 line'][i] if r == RULE else None)
    out = dict(per_task=held, rmse=rms([h['error'] for h in held.values()]))
    if all(h['sync0_line'] is not None for h in held.values()):
        out['sync0_line_rmse'] = rms([h['sync0_line'] for h in held.values()])
    return out


def summarize(table, bases):
    nonzero = [x for x in table if 'sync0' not in (x['base'], x['line1'], x['line2'])]
    valid = [x for x in nonzero if x['design'] == 'synchronous and asynchronous']
    out = {}
    out['leave one task out'] = {
        f'probe lines and base line, speed {RULE}': leave_one_task_out(valid, [RULE]),
        'probe lines, base line, and speed rule': leave_one_task_out(valid, list(RULES)),
    }
    e = np.array([rms(x[RULE]) for x in valid])
    e0 = np.array([rms(x['sync0 line']) for x in valid])
    best = min(valid, key=lambda x: rms(x[RULE]))
    out['valid protocols'] = dict(n=len(valid), median=float(np.median(e)), p10=float(np.percentile(e, 10)),
                                  best=rms(best[RULE]), best_protocol=f"{best['base']} + {best['line1']} + {best['line2']}",
                                  sync0_line_median=float(np.median(e0)), sync0_line_p10=float(np.percentile(e0, 10)),
                                  sync0_line_of_best=rms(best['sync0 line']))
    per_task = np.array([x[RULE] for x in valid])
    out['valid combinations, quartiles'] = {
        **{t: [float(q) for q in np.percentile(per_task[:, i], [25, 50, 75])] for i, t in enumerate(TASKS)},
        'all': [float(q) for q in np.percentile(e, [25, 50, 75])]}
    ex = next(x for x in valid if x['base'] == EXAMPLE[0] and {x['line1'], x['line2']} == set(EXAMPLE[1:]))
    out['example'] = dict(protocol=' + '.join(EXAMPLE), per_task=dict(zip(TASKS, ex[RULE])), all=rms(ex[RULE]),
                          sync0_line=dict(zip(TASKS, ex['sync0 line'])), sync0_all=rms(ex['sync0 line']))
    out['by design'] = {d: dict(n=len(v), median=float(np.median(v)), best=float(min(v)))
                        for d in ('synchronous and asynchronous', 'only synchronous lines', 'no synchronous line')
                        for v in [[rms(x[RULE]) for x in nonzero if x['design'] == d]]}
    out['speed rule medians'] = {r: float(np.median([rms(x[r]) for x in valid]))
                                 for r in list(RULES) + ['best speeds', 'median speeds']}
    lever = {}
    for x in valid:
        lever.setdefault(min(parse_line(l)[1] for l in (x['line1'], x['line2'])), []).append(rms(x[RULE]))
    out['by shorter probe latency'] = {f'{lat:g} ms': float(np.median(v)) for lat, v in sorted(lever.items())}
    by_base = {}
    for base in sorted({x['base'] for x in table}, key=line_key):
        rr = [x for x in table if x['base'] == base and x['design'] == 'synchronous and asynchronous'
              and 'sync0' not in (x['line1'], x['line2'])]
        top = min(rr, key=lambda x: rms(x[RULE]))
        by_base[base] = dict(oracle=rms([r['oracle'] for r in bases if r['base'] == base]), best=rms(top[RULE]),
                             median=float(np.median([rms(x[RULE]) for x in rr])),
                             nothing=rms([r['nothing'] for r in bases if r['base'] == base]),
                             best_probes=f"{top['line1']} + {top['line2']}")
    out['by base line'] = by_base
    zero = [x for x in table if x['base'] == 'sync0' and x['design'] == 'synchronous and asynchronous']
    fixed = next(x for x in zero if {x['line1'], x['line2']} == {'lazy200', 'sync200'})
    out['sync 0 as the base line'] = dict(
        lazy200_sync200=rms(fixed[RULE]), leave_one_task_out=leave_one_task_out(zero, [RULE])['rmse'])
    return out


def probe_count(points, counts=PROBE_COUNTS, factor=1 / PROBE_FACTOR):
    """One complete line at a nonzero delay + k single points on k other lines, all at the rule speed."""
    errs = {}
    for t in TASKS:
        part = points.where(task=t)
        speeds = np.unique(part.v)
        for base in part.lines():
            if base == 'sync0':
                continue
            r = Base(part, base)
            vs = r.probe_speed(factor, speeds)
            col = {l: int(np.where((r.line == l) & (r.v == vs))[0][0]) for l in part.lines() if l not in (base, 'sync0')}
            for k in counts:
                combos = list(itertools.combinations(sorted(col, key=line_key), k))
                _, e, _ = r.error(np.array([[col[l] for l in c] for c in combos]))
                for c, ee in zip(combos, e):
                    errs.setdefault((k, base, c), []).append(float(ee))
    out = {}
    for k in counts:
        vals = {key: rms(e) for key, e in errs.items() if key[0] == k}
        best = min(vals, key=vals.get)
        out[str(k)] = dict(protocols=len(vals), median=float(np.median(list(vals.values()))), best=vals[best],
                           best_protocol=' + '.join((best[1],) + best[2]))
    return out


def main(policy='pi0.5'):
    folder = output_folder(policy) / 'transfer'
    folder.mkdir(parents=True, exist_ok=True)
    points = load(policy=policy)
    rows, bases = sweep(points)
    with gzip.open(folder / 'protocols.csv.gz', 'wt', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['task', 'base', 'line1', 'v1', 'line2', 'v2', 'a', 'b', 'rmse_pp', 'sync0_line_rmse_pp'])
        writer.writerows((t, base, l1, v1, l2, v2, round(a, 4), round(b, 4), round(e, 4), round(e0, 4))
                         for t, base, l1, v1, l2, v2, a, b, e, e0 in rows)
    with (folder / 'base_lines.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bases[0]))
        writer.writeheader()
        writer.writerows(bases)
    table = protocol_table(rows, bases, points)
    with (folder / 'protocol_table.csv').open('w', newline='') as stream:
        cols = list(RULES) + ['best speeds', 'median speeds', 'sync0 line']
        writer = csv.writer(stream)
        writer.writerow(['base', 'line1', 'line2', 'design'] + cols + [f'{c} {t}' for c in cols for t in TASKS])
        for x in sorted(table, key=lambda x: rms(x[RULE])):
            writer.writerow([x['base'], x['line1'], x['line2'], x['design']] + [round(rms(x[c]), 3) for c in cols]
                            + [round(e, 3) for c in cols for e in x[c]])
    summary = summarize(table, bases)
    summary['probe count'] = probe_count(points)
    (folder / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    report(summary)
    print(f'Written to {folder}')
    return summary


def report(s):
    print()
    for name, lo in s['leave one task out'].items():
        extra = f", sync-0 line {lo['sync0_line_rmse']:.2f}" if 'sync0_line_rmse' in lo else ''
        print(f"leave one task out, choosing {name}: {lo['rmse']:.2f} pp{extra}  " + '  '.join(
            f"{t} {h['error']:.1f} ({h['protocol']} at {h['rule']})" for t, h in lo['per_task'].items()))
    v = s['valid protocols']
    print(f"valid protocols at {RULE} (n={v['n']}): median {v['median']:.2f}, best 10% {v['p10']:.2f}, best {v['best']:.2f} "
          f"({v['best_protocol']}); sync-0 line median {v['sync0_line_median']:.2f}, best 10% {v['sync0_line_p10']:.2f}")
    q = s['valid combinations, quartiles']
    print('valid combinations, quartiles: ' + '  '.join(f"{t} {a:.1f}/{m:.1f}/{b:.1f}" for t, (a, m, b) in q.items()))
    ex = s['example']
    print(f"example {ex['protocol']}: {ex['all']:.2f} (" + ' '.join(f'{e:.1f}' for e in ex['per_task'].values())
          + f"); sync-0 line {ex['sync0_all']:.2f} (" + ' '.join(f'{e:.1f}' for e in ex['sync0_line'].values()) + ')')
    print('by design: ' + '; '.join(f"{d}: median {x['median']:.2f}, best {x['best']:.2f} (n={x['n']})"
                                    for d, x in s['by design'].items()))
    print('speed rule medians: ' + '  '.join(f'{r} {e:.2f}' for r, e in s['speed rule medians'].items()))
    print('by shorter probe latency: ' + '  '.join(f'{k} {e:.2f}' for k, e in s['by shorter probe latency'].items()))
    print('by base line: oracle  best  median  none')
    for base, r in s['by base line'].items():
        print(f"  {base:8s} {r['oracle']:5.2f} {r['best']:5.2f} {r['median']:5.2f} {r['nothing']:5.2f}  ({r['best_probes']})")
    z = s['sync 0 as the base line']
    print(f"sync 0 as the base line: + lazy200 + sync200 {z['lazy200_sync200']:.2f}, leave one task out {z['leave_one_task_out']:.2f}")
    print('probes: ' + '  '.join(f"k={k}: best {x['best']:.2f} median {x['median']:.2f}" for k, x in s['probe count'].items()))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=POLICIES, default='pi0.5')
    main(policy=parser.parse_args().policy)
