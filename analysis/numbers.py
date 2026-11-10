"""The paper's analysis-derived numbers, recomputed from the published results and the shipped derived tables.

    python -m analysis.numbers [SECTION ...]

Sections (all by default):
  crossings     Finding 1 and Appendix B: 50%-crossing target speeds, near-static and fastest-target success
                (pi0.5, DynamicVLA)
  tolerance     Section 4.2: tolerance zones of release, set-down, and grasping attempts (Figure tolerance-zone)
  hand-speed    Section 4.2 and Appendix B: median alignment hand speed per setting (Figure pace)
  model         Tables tab:sensitivities and tab:blind, the held-out error by latency and its noise floor, where
                speed scaling departs, the 50 ms horizons H = 2-4, and the DynamicVLA fits
  calibration   Table tab:two-point: one complete curve and two probe points, no zero-latency data
  deployment    Section 5 and the deployment appendix: Tables tab:latency, tab:deployment-error, and
                tab:deployment-systems, and the fact check against the measured grid
"""
import argparse
import math

import numpy as np

from analysis.figures.numerics import crossing
from analysis.figures.tasks import DYNAMICVLA, LIBERO, TASKS, conditions, counts

NAMES = {t: TASKS[t].name for t in TASKS}
MODES = ('synchronous', 'lazy', 'continuous')
SHORT = {'synchronous': 'sync', 'lazy': 'lazy', 'continuous': 'cont'}
MODE_OF = {short: mode for mode, short in SHORT.items()}


def heading(text):
    print(f'\n== {text}')


def crossings():
    """Model-free: the first downward crossing of 50% success, linearly interpolated between grid speeds."""
    heading('Finding 1 and Appendix B: target speed where success crosses 50% (m/s), near-static and fastest-target '
            'success')
    for policy, tasks, grid, (short, long) in (('pi0.5', LIBERO, True, (0, 200)), ('DynamicVLA', DYNAMICVLA, False, (0, 240))):
        for task in tasks:
            spec, cells = TASKS[task], counts(policy, task, grid=grid)
            curve = {}
            for mode, delay in conditions(policy, task, grid=grid):
                success = [100 * cells[(s, mode, delay)][0] / cells[(s, mode, delay)][1] for s in spec.speeds]
                curve[(mode, delay)] = (crossing(list(spec.speeds), [x / 100 for x in success], .5), success)
            (v0, s0), (v1, s1) = curve[('synchronous', short)], curve[('synchronous', long)]
            print(f'  {policy:10s} {NAMES[task]:11s} sync {short} -> {long} ms: v50 {v0:.3f} -> {v1:.3f}; near static '
                  f'{s0[0]:.1f}% -> {s1[0]:.1f}%; at {spec.speeds[-1]:g} m/s {s0[-1]:.1f}% -> {s1[-1]:.1f}%'
                  + ''.join(f'; lazy {d} ms v50 {curve[("lazy", d)][0]:.3f}' for d in TASKS[task].delays))
    heading('Finding 1: pi0.5 synchronous success at 0.1 m/s, zero latency -> 100 ms (share of zero-latency success kept)')
    for task in LIBERO:
        cells = counts('pi0.5', task, grid=True)
        zero, delayed = (100 * cells[(.1, 'synchronous', d)][0] / cells[(.1, 'synchronous', d)][1] for d in (0, 100))
        print(f'  {NAMES[task]:11s} {zero:.1f}% -> {delayed:.1f}% ({100 * delayed / zero:.0f}%)')


def tolerance():
    """Zones as the tolerance-zone figure draws them: 0.5 cm bins with >= 30 attempts and >= 50% hits."""
    from analysis.figures.numerics import wilson_percent
    from analysis.figures.tasks import attempt_rows
    from analysis.figures.tolerance_zone import MINIMUM, PANELS, binned, edges
    heading('Section 4.2: tolerance zones (pi0.5, every condition pooled)')
    for task in ('basket', 'plate', 'cabinet'):
        bins = edges(task)
        k, n = binned(attempt_rows('pi0.5', task), bins)
        keep = n >= MINIMUM
        x = ((bins[:-1] + bins[1:]) / 2)[keep]
        rate = np.array([wilson_percent(a, b)[0] for a, b in zip(k[keep], n[keep])])
        inside = x[rate >= 50]
        low, high = (0. if PANELS[task][5] else inside.min() - .25), inside.max() + .25
        marks = ', '.join(f'{label.lower()} {position:g} cm' for position, label, *_ in PANELS[task][3])
        print(f'  {NAMES[task]:9s} zone {low:.1f}-{high:.1f} cm from {int(n.sum())} attempts (best bin {rate.max():.0f}%); '
              f'reference marks: {marks}')
    print('  release reference: the containment region is 2 x 6.1 = 12.2 cm across; set-down: success radius 3.0 cm')


def hand_speed():
    """Median over target speeds of the pace figure's curve (median episode-average alignment hand speed)."""
    from analysis.figures.tasks import alignment_speeds
    heading('Section 4.2: median alignment hand speed over target speeds (m/s), pi0.5')
    shown = (('synchronous', 0), ('synchronous', 100), ('synchronous', 200), ('lazy', 200), ('continuous', 200))
    print('  ' + ' ' * 11 + '  '.join(f'{SHORT[m]} {d:>3}' for m, d in shown))
    for task in LIBERO:
        speeds = alignment_speeds('pi0.5', task)
        value = {c: float(np.median([np.median(speeds[(s, *c)]) for s in TASKS[task].speeds])) for c in shown}
        drop = 100 * (1 - value[('synchronous', 200)] / value[('synchronous', 0)])
        print(f'  {NAMES[task]:11s}' + '  '.join(f'{value[c]:8.3f}' for c in shown)
              + f'   sync 200 ms {drop:.0f}% below zero delay')


def model():
    from analysis.latency_model import horizons, noise_floor, refined
    from analysis.latency_model.data import TASKS as MODEL_TASKS, load
    from analysis.latency_model.fitcore import noise_pp, rmse_pp
    from analysis.latency_model.fits import curve, fit_task, schedule_of, task_points
    blind = {}
    for policy in ('pi0.5', 'MolmoAct2'):
        points = load(policy=policy)
        summary, predictions = refined.blind_test(points, ('M1',))
        blind[policy] = (points, summary, predictions['M1'][0])
    # Staleness is relative to the reference's mean staleness, (H - 1)/2 control steps, so a is per step times that.
    fits = {}
    for policy, tasks in (('GR00T N1.7', MODEL_TASKS), ('DynamicVLA', DYNAMICVLA)):
        for task in tasks:
            points = task_points(policy, task)
            result, (horizon, _) = fit_task(points, task), schedule_of(task)
            delayed = points.lat > 0
            predicted = np.array([curve(result, task, MODE_OF[m], lat, np.array([v]))[0]
                                  for m, lat, v in zip(points.mode, points.lat, points.v)])
            fits[policy, task] = dict(a=result['a'] * (horizon - 1) / 2, b=result['b'],
                                      error=rmse_pp(points.k[delayed], points.n[delayed], predicted[delayed]),
                                      noise=noise_pp(points.k[delayed], points.n[delayed]))
    heading('Table tab:sensitivities: M1 fitted on every line (a per unit relative staleness, b per unit idle fraction)')
    print('  ' + ' ' * 12 + '  '.join(f'{p:>17s}' for p in (*blind, 'GR00T N1.7')))
    for task in MODEL_TASKS:
        cells = [blind[p][1]['M1']['tasks'][task]['all'] for p in blind]
        print(f'  {NAMES[task]:12s}' + '  '.join(f"a {2 * c['a']:5.2f} b {c['b']:5.2f}" for c in cells)
              + f"  a {fits['GR00T N1.7', task]['a']:5.2f} b {fits['GR00T N1.7', task]['b']:5.2f}")
    heading('Table tab:blind: fitted on 0/100/200 ms, RMS error (points) on the held-out 50/150/250 ms lines')
    for policy, (points, summary, predicted) in blind.items():
        new = points.source == 'new'
        k, n = points.k, points.n
        print(f'  {policy}: every task and mode {summary["M1"]["blind"]:.2f} (point sampling noise {summary["noise_new"]:.2f}); '
              'by latency ' + '  '.join(f'{lat} ms {e:.1f}' for lat, e in summary['M1']['by_latency'].items()))
        for mode in ('sync', 'lazy', 'cont'):
            rows = [rmse_pp(k[m], n[m], predicted[m]) for task in MODEL_TASKS
                    for m in [new & (points.task == task) & (points.mode == mode)]]
            m = new & (points.mode == mode)
            print(f'    {mode:5s} ' + '  '.join(f'{NAMES[t]} {e:.1f}' for t, e in zip(MODEL_TASKS, rows))
                  + f'  all {rmse_pp(k[m], n[m], predicted[m]):.1f}')
        print('    every ' + '  '.join(f"{NAMES[t]} {summary['M1']['tasks'][t]['blind']:.1f}" for t in MODEL_TASKS))
        print('    noise ' + '  '.join(f'{NAMES[t]} {noise_pp(k[m], n[m]):.1f}' for t in MODEL_TASKS
                                       for m in [new & (points.task == t)]))
    heading('Model appendix: the noise floor of the held-out test (an exact model, sampling noise alone; RMS points)')
    for policy in blind:
        held, reference, both = noise_floor.floor(policy)
        print(f'  {policy}: held-out points {held:.1f}, reference curve {reference:.1f}, combined {both:.1f}')
    heading('Model appendix: where speed scaling departs (held-out fit on 0/100/200 ms)')
    for policy, (points, summary, predicted) in blind.items():
        law, new = refined.JointLaw(points), points.source == 'new'
        x = summary['M1']['x_old']
        kappa = np.array([1 + x[2 * j] * law.feats[i, 0] + x[2 * j + 1] * law.feats[i, 1] for i, j in enumerate(law.ti)])
        error = (points.k / points.n - predicted) ** 2
        beyond, fast = new & (points.v * kappa > max(points.v) + 1e-12), new & (points.v >= .15 - 1e-9)
        print(f'  {policy}: {100 * beyond.sum() / new.sum():.0f}% of held-out points read the reference beyond '
              f'{max(points.v):g} m/s and carry {100 * error[beyond].sum() / error[new].sum():.0f}% of the squared error; '
              f'0.15-0.2 m/s carry {100 * error[fast].sum() / error[new].sum():.0f}%')
        if policy != 'pi0.5':
            continue

        def at(task, mode, latency, speed):
            return 100 * predicted[(points.task == task) & (points.mode == mode) & (points.lat == latency)
                                   & np.isclose(points.v, speed)][0]
        below = [at(t, 'lazy', lat, v) - at(t, 'sync', lat, v) for t, lats, speeds in
                 (('cabinet', (100, 150, 200, 250), (.2,)), ('plate', (200, 250), (.175, .2))) for lat in lats for v in speeds]
        moved = [abs(at(t, m, 250, .001) - at(t, 'sync', 0, .001)) for t in ('basket', 'plate') for m in ('lazy', 'cont')]
        print(f'    synchronous predicted {min(below):.1f}-{max(below):.1f} points below lazy on grasping at 0.2 m/s '
              f'(100-250 ms) and set-down at 0.175-0.2 m/s (200, 250 ms); near-static 250 ms asynchronous predictions on '
              f'release and set-down move at most {max(moved):.2f} points from zero latency')
    heading('Section 5.1: pi0.5 asynchronous execution at 50 ms with horizons H = 2, 3, 4 (in no calibration)')
    for task in horizons.TASKS:
        error = horizons.errors(task)
        print(f'  {NAMES[task]:9s} RMS {100 * np.sqrt(np.mean(error ** 2)):.2f} points over {len(error)} runs')
    heading('DynamicVLA (and GR00T N1.7): M1 fitted on every line, in-sample RMS over the delayed runs')
    for (policy, task), fit in fits.items():
        print(f'  {policy:10s} {NAMES[task]:11s} a {fit["a"]:5.2f}  b {fit["b"]:5.2f}  RMS {fit["error"]:.1f} '
              f'(sampling noise {fit["noise"]:.1f})')


def calibration():
    import contextlib
    import io
    from analysis.latency_model import calibration_sweep as sweep
    from analysis.latency_model.data import TASKS as MODEL_TASKS
    heading('Table tab:two-point: one complete curve plus one point on each of two other curves, no zero-latency data')
    print('  median RMS (points) over every remaining point, 10th-90th percentile of calibrations, and the unseen '
          'zero-latency curve')
    for policy in sweep.POLICIES:
        with contextlib.redirect_stdout(io.StringIO()):
            rows = sweep.sweep(policy)
        nonzero = [x for x in rows if x['fixed'] and x['base'] != 'sync0']
        groups = [(NAMES[t], [x for x in nonzero if x['task'] == t]) for t in MODEL_TASKS] + [('All', nonzero)]

        def spread(values):
            return f'{np.median(values):.1f} ({np.percentile(values, 10):.1f}-{np.percentile(values, 90):.1f})'
        print(f'  {policy}:')
        print('    ' + '  '.join(f"{name} {spread([x['rms'] for x in group])}" for name, group in groups))
        print('    zero-latency curve ' + '  '.join(f"{name} {np.median([x['rms_sync0'] for x in group]):.1f}"
                                                   for name, group in groups))
        print(f"    both points at any pair of grid speeds: median "
              f"{np.median([x['rms'] for x in rows if x['base'] != 'sync0']):.1f}")


def deployment():
    from analysis.figures import deployment as figure
    heading('Section 5: Tables tab:latency, tab:deployment-error, and tab:deployment-systems')
    cells, model = figure.summary()
    measured = figure.measured_rate(cells)
    matches, misses = 0, []
    for delay in figure.DELAYS:
        for speed in TASKS[figure.TASK].speeds:
            def predicted(mode):
                return 100 * figure.success(model, mode, delay, speed)[0]
            chosen = max(MODES, key=predicted)
            best = max(MODES, key=lambda m: measured(m, delay, speed))
            # A tie in the measured grid matches, and so do lazy and continuous execution once the latency reaches the
            # execution horizon, where they run the same schedule.
            same = {chosen, best} == {'lazy', 'continuous'} and delay * figure.HZ / 1000 >= figure.HORIZON
            if measured(chosen, delay, speed) == measured(best, delay, speed) or same:
                matches += 1
            else:
                misses.append((delay, speed, predicted(chosen) - predicted(best),
                               measured(best, delay, speed) - measured(chosen, delay, speed)))
    print(f'counting ties: the predicted best mode is the measured best in {matches} of {matches + len(misses)} '
          f'latency-speed cells; misses at {min(m[0] for m in misses)}-{max(m[0] for m in misses)} ms, each ranked '
          f'at most {max(m[2] for m in misses):.1f} points above the measured best, which beats it by '
          + ', '.join(f'{gap:.1f} ({delay} ms, {speed:g} m/s)' for delay, speed, _, gap in misses))


SECTIONS = {'crossings': crossings, 'tolerance': tolerance, 'hand-speed': hand_speed, 'model': model,
            'calibration': calibration, 'deployment': deployment}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sections', nargs='*', metavar='SECTION', help=', '.join(SECTIONS))
    args = parser.parse_args(argv)
    for name in args.sections or SECTIONS:
        SECTIONS[name]()


if __name__ == '__main__':
    main()
