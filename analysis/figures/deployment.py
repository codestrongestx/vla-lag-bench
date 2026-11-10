"""Predicted success of pi0.5 served by FlashRT bf16 on set-down, for choosing an execution mode on given hardware
(Section 5, with the predicted grids in the appendix).

The delay model rounds a request's end-to-end latency (FlashRT's 99th-percentile latency plus network delay) up to
whole 50 ms control steps (L = ceil(f tau), Section 3.4), so success depends on that rounded latency alone. The model is
calibrated on openpi JAX's set-down grid (analysis.figures.tasks, synchronous points from the hold_pose reruns) from one complete
curve and one point on each of two other curves (the fixed example of Appendix A.3, analysis.latency_model.transfer_sweep.EXAMPLE: continuous
100 ms complete, and synchronous and continuous 200 ms at the grid speed nearest that curve's midpoint speed / 1.4). Every prediction
reads the continuous 100 ms curve as measured (analysis.latency_model.speed_latency_model.Reference) at the speed its
setting's equivalent speedup implies. Both figures use the
change-from-zero-delay scale of the recovery heatmap for the color, while each cell prints the predicted success
itself with a percent sign, so it cannot be read as a latency. Two one-column figures. deployment-modes (appendix,
predicted success at each latency) covers every execution mode and latency at every target speed, under synchronous
zero delay. deployment-rate (appendix, cited from Section 5) gives, for each latency the hardware holds (rows) and
each inference rate a shared server grants a robot (columns), the best schedule that rate allows at 0.075 m/s.
Synchronous execution needs one request per execution horizon plus its wait and lazy execution one per execution
horizon (4 Hz). Continuous execution needs one per latency; under a lower rate it requests once per granted period of P
control steps and switches to each chunk on arrival, which is lazy execution with execution horizon P
(capped_continuous). FlashRT serves only the latency: success comes from openpi JAX throughout.

The run prints the model's error on every grid point outside the calibration, the 99th-percentile batch-1 latency
per GPU for openpi PyTorch and FlashRT at bf16 (Table 3, to one decimal), a fact check of the model's choices against
the measured grid, the latency each GPU holds on each network, rounded up to whole control steps, and Section 5's
four systems with every schedule each
allows, predicted and, where run, measured. Network delays are VLA-Perf's estimates for a pi0 request (Table 8 latency
on a B100 minus its 3.18 ms compute): about 5 ms for WiFi 7 and 25 ms for 5G.
"""
import csv
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.optimize import brentq
from analysis.paths import PI05_STACKS, figure_path
from analysis.figures.figure_kit import MODES, SPEED_LABEL, note, panel_title, save, style
from analysis.figures.style import CHANGE_LIMIT, DIVERGING, INK, INK2, MODE_COLORS, MODE_SHORT
from analysis.figures.tasks import TASKS, counts
from analysis.figures.typography import COLUMN_WIDTH, PAPER, RC
from analysis.latency_model.data import from_counts
from analysis.latency_model.fits import schedule_of
from analysis.latency_model.speed_latency_model import Reference, parse_line, predict_from, speed_ratio, transfer
from analysis.latency_model.transfer_sweep import EXAMPLE

TASK = 'plate'               # set-down, on openpi JAX's measured grid
DELAYS = (50, 100, 150, 200, 250)
STEP = 50                    # one control step at 20 Hz
REQUIRED = 60                # the success requirement the text reads the fastest target at, in percent
SPEED = .075                 # the target speed Section 5 and the rate figure read success at
MODE_OF = {'sync': 'synchronous', 'lazy': 'lazy', 'cont': 'continuous'}
# Latency tables, most authoritative first.
TABLES = ('latency-batch1-h100-quiet/latency.csv', 'latency-batch1-additions/latency.csv', 'latency-batch1/latency.csv')
GPUS = (('GeForce RTX 3090', 'RTX 3090'), ('GeForce RTX 4090', 'RTX 4090'), ('GeForce RTX 5090', 'RTX 5090'),
        ('H100 80GB HBM3', 'H100'))
STACKS = (('openpi-pytorch', 'bf16', 'PyTorch bf16'), ('flashrt', 'bf16', 'FlashRT bf16'))
NETWORK = {'WiFi 7': 5, '5G': 25}
DIV = LinearSegmentedColormap.from_list('change', DIVERGING)
CELL_HEIGHT = .13
RATE_CELL_HEIGHT = .17   # the rate heatmap's cells hold a mode and its success
RATES = (4, 5, 10, 20)       # granted inference rates (Hz); 4 Hz is lazy execution's one chunk per execution horizon
# Section 5's four systems: (name, latency the hardware holds in ms, granted rate in Hz, or None for a GPU dedicated
# to one robot). A holds 100 ms: pi0.5 under FlashRT FP16 on a Jetson AGX Thor, 72.8 ms median (VLA-HWBench).
# B and C hold 50 ms: an RTX 4090 over WiFi 7 and an H100 over 5G (Table 3). B's edge server grants 20 Hz, exactly
# continuous execution's rate at 50 ms; C's shared server grants 5 Hz. D holds 250 ms: an H100 reached over a wired
# link and VLA-Perf's slow cloud (100 ms per direction, Table 7), 217.9 ms end to end (17.9 + 200), with no rate limit.
SYSTEMS = (('A onboard GPU', 100, None), ('B edge server', 50, 20), ('C shared server', 50, 5),
           ('D remote cloud', 250, None))
# Rate-capped continuous runs at SPEED are openpi JAX's asynchronous execution with the period as its horizon; the
# H = 4 run at 50 ms comes from the horizon validation (analysis/latency_model/horizons.py).
HORIZON, HZ = 5, 20          # execution horizon and control rate
CHUNK = 10                   # pi0.5's action chunk K: a chunk must hold the L skipped actions plus those executed
STACK = 'FlashRT bf16'       # the stack Section 5 deploys


def setups():
    """(GPU, stack) -> 99th-percentile in-process batch-1 request latency (ms) for the selected GPUs and stacks."""
    found = {}
    for table in TABLES:
        with (PI05_STACKS / table).open() as stream:
            for row in csv.DictReader(stream):
                if row['model'] != 'pi05' or row['transport'] != 'in-process':
                    continue
                gpu = dict(GPUS).get(row['gpu'])
                stack = {(s, p): name for s, p, name in STACKS}.get((row['stack'], row['precision']))
                if gpu and stack and (gpu, stack) not in found:
                    found[(gpu, stack)] = float(row['p99'])
    return found


def capped_run(delay, period, speed=SPEED):
    """(successes, episodes, table) of the measured rate-capped run, or None when it was not run."""
    from analysis.latency_model.horizons import measured, table
    run = measured(TASK).get(('lazy', delay, period, speed))
    return run and (*run, table(TASK).parent.name)


def calibrate(cells, task=TASK):
    """The model from one complete curve and one point on each of two other curves of the given success grid, and
    the calibration points."""
    horizon, hz = schedule_of(task)
    points = list(from_counts(cells, task, horizon=horizon, hz=hz).rows())
    complete, *others = EXAMPLE
    line = [(r['v'], r['k'], r['n']) for r in points if r['line'] == complete]
    speed = Reference(line).probe_speed(sorted({r['v'] for r in points}))
    probes = []
    for other in others:
        row = next(r for r in points if r['line'] == other and r['v'] == speed)
        probes.append((other, row['v'], row['k'], row['n']))
    model = transfer(line, complete, probes, horizon=horizon, hz=hz, draws=0)
    used = {(complete, v) for v, _, _ in line} | {(other, v) for other, v, _, _ in probes}
    return model, points, used


def success(model, mode, delay, speeds):
    horizon, hz = schedule_of(TASK)
    return predict_from(model['reference'], speeds, mode, delay, model['a'], model['b'], horizon, hz,
                        ref_feat=model['ref_feat'])


def fastest(model, mode, delay):
    """The fastest target speed at which the mode keeps the required success, or 0 if it never does."""
    def gap(v):
        return success(model, mode, delay, v)[0] - REQUIRED / 100
    return brentq(gap, 1e-3, 1.) if gap(1e-3) > 0 else 0.


def request_rate(mode, delay):
    """Inference requests per second each execution mode needs at a latency (ms), at 20 Hz with H = 5:
    synchronous one per chunk plus its wait, lazy one per H actions, continuous one per latency."""
    chunk = 1000 * HORIZON / HZ
    return {'synchronous': 1000 / (chunk + delay), 'lazy': 1000 / chunk, 'continuous': 1000 / delay}[mode]


def fact_check(model, cells):
    """Check the model's choices against the measured grid: the best mode at each latency and speed, and the
    fastest grid speed each mode keeps at the required success."""
    speeds = TASKS[TASK].speeds

    def measured(mode, delay, speed):
        k, n = cells[(speed, mode, delay)][:2]
        return 100 * k / n

    gaps = []
    for delay in DELAYS:
        for speed in speeds:
            chosen = max(MODES, key=lambda m: success(model, m, delay, speed)[0])
            best = max(MODES, key=lambda m: measured(m, delay, speed))
            if chosen != best:
                gaps.append((delay, speed, MODE_SHORT[chosen], MODE_SHORT[best],
                             measured(best, delay, speed) - measured(chosen, delay, speed)))
    print(f'fact check: predicted best mode is the measured best in {len(DELAYS) * len(speeds) - len(gaps)} of '
          f'{len(DELAYS) * len(speeds)} latency-speed cells; misses: '
          + '  '.join(f'{d} ms {v:g} m/s {c}->{b} {g:+.1f}' for d, v, c, b, g in gaps))
    for delay in DELAYS:
        chosen = max(MODES, key=lambda m: success(model, m, delay, SPEED)[0])
        best = max(MODES, key=lambda m: measured(m, delay, SPEED))
        fastest_pred = max((v for v in speeds if 100 * success(model, 'continuous', delay, v)[0] >= REQUIRED), default=0)
        fastest_meas = max((v for v in speeds if measured('continuous', delay, v) >= REQUIRED), default=0)
        print(f'  {delay} ms at {SPEED:g} m/s: predicted {MODE_SHORT[chosen]} {100 * success(model, chosen, delay, SPEED)[0]:.0f}%, '
              f'measured best {MODE_SHORT[best]} {measured(best, delay, SPEED):.1f}%; continuous fastest at {REQUIRED}%: '
              f'predicted {fastest_pred:g}, measured {fastest_meas:g} m/s')


def measured_rate(cells):
    """Measured success (percent) of a mode at a latency and speed in a success grid, keyed as the model is."""
    def measured(mode, delay, speed):
        k, n = cells[(speed, mode, delay)][:2]
        return 100 * k / n
    return measured


def capped_continuous(model, delay, cap, speed=SPEED):
    """Continuous execution held to a granted rate below one request per latency: it requests once per period P
    (the rate's interval, in whole control steps) and switches to each chunk on arrival, so every chunk executes
    actions L through L+P-1, as lazy execution would with an execution horizon of P. The schedule's mean observation
    staleness is L + (P - 1)/2 steps with no idle time, so the model predicts it from its staleness alone. Returns
    (P, staleness in ms, predicted success), or None when the rate allows plain continuous execution or the chunk is
    too short to bridge the period (L + P > K)."""
    steps = math.ceil(delay * HZ / 1000 - 1e-9)
    period = math.ceil(HZ / cap - 1e-9)
    if period <= steps or steps + period > CHUNK:
        return None
    staleness = steps + (period - 1) / 2
    ratio = speed_ratio([(staleness - (HORIZON - 1) / 2, 0.0)], model['ref_feat'], model['a'], model['b'])[0]
    return period, 1000 * staleness / HZ, float(model['reference'](speed * ratio))


def options(model, delay, cap, speed=SPEED):
    """Every schedule a granted rate (Hz, None for a dedicated GPU) allows at a latency, as (mode, period of a
    rate-capped continuous execution in steps or None, predicted success)."""
    limit = math.inf if cap is None else cap
    out = [(m, None, success(model, m, delay, speed)[0]) for m in ('synchronous', 'lazy')
           if request_rate(m, delay) <= limit + 1e-9]
    if request_rate('continuous', delay) <= limit + 1e-9:
        out.append(('continuous', None, success(model, 'continuous', delay, speed)[0]))
    elif capped := capped_continuous(model, delay, cap, speed):
        out.append(('continuous', capped[0], capped[2]))
    return out


def systems(model, cells):
    """Section 5's systems: every schedule the system's rate allows, predicted and, where run, measured."""
    measured = measured_rate(cells)
    for name, delay, cap in SYSTEMS:
        parts = []
        for mode, period, predicted in options(model, delay, cap):
            if period is None:
                parts.append(f'{MODE_SHORT[mode]} {100 * predicted:.0f}% ({100 * predicted:.1f}; '
                             f'measured {measured(mode, delay, SPEED):.1f}%)')
            else:
                staleness = capped_continuous(model, delay, cap)[1]
                run = capped_run(delay, period)
                parts.append(f'{MODE_SHORT[mode]} once per {period * STEP} ms = lazy H={period} '
                             f'({f"measured {100 * run[0] / run[1]:.1f}%, {run[0]}/{run[1]}, {run[2]}" if run else "not measured"}): '
                             f'staleness {staleness:.0f} ms, predicted {100 * predicted:.0f}% ({100 * predicted:.1f})')
        best = max(options(model, delay, cap), key=lambda o: o[2])
        print(f'  {name} ({delay} ms, ' + ('dedicated' if cap is None else f'{cap:.2g} Hz') + '): '
              + '  '.join(parts) + f'; best {MODE_SHORT[best[0]]}' + (f' (H={best[1]})' if best[1] else ''))


def report(model, points, used):
    errors, by_mode = [], {m: [] for m in MODES}
    for r in points:
        if (r['line'], r['v']) not in used:
            mode, delay = parse_line(r['line'])
            p = r['k'] / r['n']
            error = success(model, MODE_OF[mode], delay, r['v'])[0] - p
            errors.append(error)
            by_mode[MODE_OF[mode]].append((error, p * (1 - p) / r['n']))
    print(f'{TASKS[TASK].name}: a {model["a"]:.3f} b {model["b"]:.2f}; RMS on {len(errors)} uncalibrated points '
          f'{100 * math.sqrt(np.mean(np.square(errors))):.1f}')
    # Table tab:deployment-error: prediction minus measured success on every uncalibrated point, by execution mode,
    # beside the RMS binomial standard error of the measured rates.
    print('error by mode (points; RMS error, mean error, RMS sampling standard error):')
    for name, rows in [*by_mode.items(), ('all', [x for rows in by_mode.values() for x in rows])]:
        e, var = np.array([x[0] for x in rows]), np.array([x[1] for x in rows])
        print(f'  {name}: {len(rows)} points, RMS {100 * math.sqrt(np.mean(e ** 2)):.1f}, '
              f'mean {100 * e.mean():+.1f}, noise {100 * math.sqrt(var.mean()):.1f}')
    print(f'zero delay: {100 * success(model, "synchronous", 0, SPEED)[0]:.0f}% at {SPEED:g} m/s, '
          f'{fastest(model, "synchronous", 0):.3f} m/s at {REQUIRED}%')
    for delay in DELAYS:
        print(f'  {delay} ms ' + '  '.join(f'{MODE_SHORT[m]} {100 * success(model, m, delay, SPEED)[0]:.0f}% '
                                           f'{fastest(model, m, delay):.3f} m/s' for m in MODES))


def scenarios(latency):
    """The latency each GPU holds with the deployed stack on each network, rounded up to whole control steps."""
    for _, gpu in GPUS:
        if (gpu, STACK) in latency:
            print(f'  {STACK} {gpu}: ' + '  '.join(f'{n} {held_latency(latency, gpu, STACK, n)} ms' for n in NETWORK))


def held_latency(latency, gpu, stack, network):
    """End-to-end latency (99th-percentile inference plus network delay) rounded up to whole control steps."""
    return STEP * math.ceil((latency[(gpu, stack)] + NETWORK[network]) / STEP)


def heat(ax, M, xlabels, ylabels, index, title, labels=None, groups=(), rotate=False):
    """A heatmap colored by change from zero delay in points, as in the recovery figure, with the signed change or
    the given label in each cell."""
    norm = TwoSlopeNorm(vmin=-CHANGE_LIMIT, vcenter=0, vmax=CHANGE_LIMIT)
    im = ax.imshow(M, cmap=DIV, norm=norm, aspect='auto')
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=90 if rotate else 0)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        panel_title(ax, index, title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            rgba = DIV(norm(M[i, j]))
            dark = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] < 0.55
            text = labels[i][j] if labels else f'{M[i, j]:+.0f}'.replace('-', '\u2212')
            note(ax, j, i, text, ha='center', va='center', color=PAPER if dark else INK)
    ax.set_xticks(np.arange(-.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, M.shape[0], 1), minor=True)
    ax.grid(True, which='minor', color=PAPER, lw=1)
    ax.tick_params(which='minor', length=0)
    for gap in groups:
        ax.axhline(gap - .5, color=PAPER, lw=2.2)
    return im


def summary():
    """Print Table 3's latencies, the model's errors and fact check, the held latencies, and Section 5's systems;
    return the success grid and the calibrated model."""
    latency = setups()
    for _, gpu in GPUS:
        print(gpu, '  '.join(f'{name} {latency[(gpu, name)]:.1f} ms' if (gpu, name) in latency else f'{name} -'
                             for _, _, name in STACKS))
    cells = counts('pi0.5', TASK, grid=True)
    model, points, used = calibrate(cells)
    report(model, points, used)
    fact_check(model, cells)
    scenarios(latency)
    print(f'Section 5 systems at {SPEED:g} m/s:')
    systems(model, cells)
    return cells, model


def main():
    style(plt)
    cells, model = summary()
    speeds = TASKS[TASK].speeds
    zero = 100 * success(model, 'synchronous', 0, speeds)
    rows = [('synchronous', 0)] + [(m, d) for m in MODES for d in DELAYS]
    M = np.array([100 * success(model, m, d, speeds) - zero for m, d in rows])
    # Colors show the change from zero delay; each cell prints the predicted success itself, to check a requirement,
    # with a percent sign so it cannot be read as a latency.
    labels_modes = [[f'{v + z:.0f}%' for v, z in zip(row, zero)] for row in M]
    reference = 100 * success(model, 'synchronous', 0, SPEED)[0]
    B = np.full((len(DELAYS), len(RATES)), np.nan)
    labels = [[''] * len(RATES) for _ in DELAYS]
    for i, delay in enumerate(DELAYS):
        for j, cap in enumerate(RATES):
            mode, _, predicted = max(options(model, delay, cap), key=lambda o: o[2])
            B[i, j] = 100 * predicted - reference
            # The percent sign keeps the success from being read as a latency.
            labels[i][j] = f'{MODE_SHORT[mode]} {100 * predicted:.0f}%'
    # Two one-column figures: every mode and latency, and the best schedule each granted rate allows. Both use the
    # change-from-zero-delay scale, drawn below each heatmap.
    fig = plt.figure(figsize=(COLUMN_WIDTH, CELL_HEIGHT * len(rows) + 1.55), layout='constrained')
    a = fig.add_subplot()
    # Rows give only the latency; each mode names its block once, at the left, so the cells keep the width their
    # success and percent sign need at 7 pt.
    im = heat(a, M, [f'{s:g}' for s in speeds], ['0 ms' if d == 0 else f'{d} ms' for _, d in rows], None, None,
              labels=labels_modes, groups=(1, 1 + len(DELAYS), 1 + 2 * len(DELAYS)), rotate=True)
    for mode in MODES:
        block = [i for i, (m, d) in enumerate(rows) if m == mode and d > 0]
        a.annotate(MODE_SHORT[mode], xy=(0, np.mean(block)), xycoords=('axes fraction', 'data'), xytext=(-43, 0),
                   textcoords='offset points', rotation=90, ha='center', va='center', color=MODE_COLORS[mode],
                   fontweight='bold', fontsize=RC['axes.labelsize'])
    a.set_xlabel(SPEED_LABEL)
    a.set_ylabel('Latency', color=INK2, labelpad=24)
    scale(fig, im, a)
    save(fig, figure_path('deployment-modes'))
    plt.close(fig)
    fig = plt.figure(figsize=(COLUMN_WIDTH, RATE_CELL_HEIGHT * len(DELAYS) + 1.25), layout='constrained')
    b = fig.add_subplot()
    # Units go in the axis labels, leaving the cells the width their mode and success need.
    im = heat(b, B, [f'{r:.2g}' for r in RATES], [f'{d}' for d in DELAYS], None, None, labels=labels)
    b.set_xlabel('Granted inference rate (Hz)', color=INK2)
    b.set_ylabel('Latency (ms)', color=INK2)
    scale(fig, im, b)
    save(fig, figure_path('deployment-rate'))
    plt.close(fig)


def scale(fig, im, ax):
    """The change-from-zero-delay color scale, horizontal below the heatmap so the cells keep the column's width."""
    cb = fig.colorbar(im, ax=ax, location='bottom', fraction=.06, pad=.02, aspect=40, shrink=.8)
    cb.set_ticks(range(-CHANGE_LIMIT, CHANGE_LIMIT + 1, 20))
    cb.set_label('Predicted change from zero latency (points)', color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2)


if __name__ == '__main__':
    main()
