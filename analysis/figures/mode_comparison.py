"""Compare execution modes at each delay, with synchronous 0 ms as reference: LIBERO tasks, then DynamicVLA tasks.

Rows are the delays, columns the tasks: pi0.5 and MolmoAct2 run 50 to 250 ms, DynamicVLA 120 and 240 ms. The DynamicVLA figure
keeps the LIBERO panel size.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.paths import figure_path
from analysis.figures.figure_kit import (MODES, SPEED_LABEL, SUCCESS_LABEL, column_legend, panel_title, percent_axis, rate_line,
                              save, setting_legend, shared_xlabel, shared_ylabel, single_delay_handles,
                              single_delay_panel, speed_axis, style, top_legend, width_for)
from analysis.figures.typography import COLUMN_WIDTH
from analysis.figures.tasks import DYNAMICVLA, LIBERO, TASKS, counts, delays


ROW_HEIGHT = .92
# One delay's block in the one-column summary: its key and two rows of panels, tall enough that the three modes'
# curves read apart.
COLUMN_BLOCK_HEIGHT = 2.35


def draw(policy, tasks, filename, grid, selected_delays=None):
    cells = {task: counts(policy, task, grid=grid) for task in tasks}
    shown = selected_delays or delays(policy, tasks[0], grid=grid)
    fig, axes = plt.subplots(len(shown), len(tasks), figsize=(width_for(len(tasks)), ROW_HEIGHT * len(shown) + .7),
                             sharey=True, sharex='col', squeeze=False, layout='constrained')
    for row, delay in enumerate(shown):
        for column, task in enumerate(tasks):
            ax, spec = axes[row, column], TASKS[task]
            rate_line(ax, spec.speeds, [cells[task][(s, 'synchronous', 0)] for s in spec.speeds], 'synchronous', 0)
            for mode in MODES:
                rate_line(ax, spec.speeds, [cells[task][(s, mode, delay)] for s in spec.speeds], mode, delay)
            panel_title(ax, row * len(tasks) + column, spec.name)
            speed_axis(ax, spec.speeds, narrow=True)
            percent_axis(ax)
            if column == 0:
                ax.set_ylabel(f'{delay} ms\nlatency')
            if row < len(shown) - 1:
                ax.tick_params(labelbottom=False)
    shared_xlabel(fig, SPEED_LABEL)
    shared_ylabel(fig, SUCCESS_LABEL)
    setting_legend(fig, shown)
    save(fig, figure_path(filename))
    plt.close(fig)


def column_blocks(policy, tasks, filename, selected_delays):
    """One paper column: a block per selected delay, its tasks in a 2x2 grid in catalog order under a key that names
    the delay and draws each mode exactly as plotted there."""
    cells = {task: counts(policy, task, grid=True) for task in tasks}
    fig = plt.figure(figsize=(COLUMN_WIDTH, COLUMN_BLOCK_HEIGHT * len(selected_delays) + .35), layout='constrained')
    subfigures = fig.subfigures(len(selected_delays), 1, hspace=0)
    for block, (sub, delay) in enumerate(zip(subfigures, selected_delays)):
        axes = sub.subplots(2, 2, sharex=False, sharey=True)
        for index, (ax, task) in enumerate(zip(axes.flat, tasks)):
            spec = TASKS[task]
            single_delay_panel(ax, delay)
            rate_line(ax, spec.speeds, [cells[task][(s, 'synchronous', 0)] for s in spec.speeds], 'synchronous', 0)
            for mode in MODES:
                rate_line(ax, spec.speeds, [cells[task][(s, mode, delay)] for s in spec.speeds], mode, delay)
            panel_title(ax, block * len(tasks) + index, spec.name)
            speed_axis(ax, spec.speeds, narrow=True)
            percent_axis(ax)
        column_legend(sub, single_delay_handles(delay, named=True), ncol=4)
    shared_xlabel(fig, SPEED_LABEL)
    shared_ylabel(fig, SUCCESS_LABEL)
    save(fig, figure_path(filename))
    plt.close(fig)


style(plt)
draw('pi0.5', LIBERO, 'mode-comparison', grid=True)
column_blocks('pi0.5', LIBERO, 'mode-summary', selected_delays=(50, 200))
draw('MolmoAct2', LIBERO, 'molmoact2-mode-comparison', grid=True)
draw('DynamicVLA', DYNAMICVLA, 'dynamicvla-mode-comparison', grid=False)
print('Mode comparisons rendered for the LIBERO and DynamicVLA tasks with the synchronous zero-delay reference.')
