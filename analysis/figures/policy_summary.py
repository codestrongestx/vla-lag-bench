"""Main-text policy comparison: success against speed for pi0.5, GR00T N1.7, and MolmoAct2 at 200 ms, synchronous
beside lazy execution, on release and set-down.

One paper column: release above set-down, synchronous left of lazy. These are panels (b,c,e,f) of the appendix
policy figure (policy_comparison.py), which keeps the zero-delay curves and the other two tasks.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analysis.paths import figure_path
from analysis.figures.figure_kit import (MODE_SHORT, SPEED_LABEL, SUCCESS_LABEL, interval_bars, panel_title, percent_axis,
                              policy_handles, policy_style, save, shared_xlabel, shared_ylabel, speed_axis, style,
                              top_legend)
from analysis.figures.numerics import wilson_bounds
from analysis.figures.style import POLICY_COLORS
from analysis.figures.tasks import TASKS, counts
from analysis.figures.typography import COLUMN_WIDTH

POLICIES = ('pi0.5', 'GR00T N1.7', 'MolmoAct2')
TASK_ROWS = ('basket', 'plate')
DELAY = 200
MODE_COLUMNS = ('synchronous', 'lazy')

style(plt)
cells = {(p, t): counts(p, t) for p in POLICIES for t in TASK_ROWS}
fig, axes = plt.subplots(2, 2, figsize=(COLUMN_WIDTH, 2.9), sharey=True, layout='constrained')
for row, task in enumerate(TASK_ROWS):
    spec = TASKS[task]
    x = np.array(spec.speeds)
    for column, mode in enumerate(MODE_COLUMNS):
        ax = axes[row, column]
        for policy in POLICIES:
            k, n = zip(*(cells[policy, task][(s, mode, DELAY)] for s in spec.speeds))
            lo, hi = zip(*(wilson_bounds(a, b) for a, b in zip(k, n)))
            y = 100 * np.array(k) / np.array(n)
            interval_bars(ax, x, y, 100 * np.array(lo), 100 * np.array(hi), color=POLICY_COLORS[policy])
            ax.plot(x, y, **policy_style(policy))
        panel_title(ax, 2 * row + column, f'{spec.name}\n{MODE_SHORT[mode]} {DELAY} ms')
        speed_axis(ax, spec.speeds, narrow=True)
        percent_axis(ax)
shared_xlabel(fig, SPEED_LABEL)
shared_ylabel(fig, SUCCESS_LABEL)
top_legend(fig, policy_handles(POLICIES))
save(fig, figure_path('policy-summary'))
plt.close(fig)
print('written policy-summary.png')
