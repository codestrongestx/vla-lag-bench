"""Success at the delays the model is fitted to, and its predictions at the delays it never saw.

Grid (figR1_grid_*), one figure per policy: tasks in columns, execution modes in rows. Each panel shows
synchronous 0 ms success with its 95% Wilson intervals and the row's mode at the task's two fitted delays (100 and
200 ms; DynamicVLA 120 and 240 ms) as data lines in their shade and marker. No model is drawn.

Synchronous row (figR1_sync_pi05): the grid's synchronous row alone, for the main text's view of how each task's
success falls with target speed and delay.

Examples (figR1_examples_pi05, figR1_examples_molmoact2): three task/mode columns and one held-out delay per row.
The main-text summary (figR1_summary_pi05) is one paper column wide and keeps one held-out delay per column. Each panel shows synchronous
0 ms success, measured points with capped intervals, and one black dashed prediction (latency_model/fits.py).
The model is fitted to 0, 100, and 200 ms only, so every
dashed line predicts runs it never saw; the paper's error table covers every task and mode.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from analysis.paths import figure_path
from analysis.figures.figure_kit import (LINE, MODES, SPEED_LABEL, SUCCESS_LABEL, delay_point_style, panel_title, percent_axis,
                              prediction_style, rate_line, row_label, save, setting_legend, shared_xlabel, shared_ylabel, speed_axis,
                              style, top_legend, width_for, zero_delay_handle)
from analysis.figures.typography import COLUMN_WIDTH
from analysis.figures.style import INK, MODE_NAMES, MODE_SHORT, PREDICTION
from analysis.figures.tasks import DYNAMICVLA, LIBERO, TASKS, counts
from analysis.latency_model.data import BLIND_LATENCIES
from analysis.latency_model.fitcore import rmse_pp
from analysis.latency_model.fits import task_points, curve, fit_task

ROW_HEIGHT = 1.1
# A panel row in a one-column figure.
COLUMN_ROW_HEIGHT = .92
# One task per execution mode: release under synchronous delay (idle time), set-down under lazy execution (observation
# age), and set-down under continuous execution (fresher observations); each has a typical held-out error.
EXAMPLES = (('basket', 'synchronous'), ('plate', 'lazy'), ('plate', 'continuous'))
GRID_NAMES = {'pi0.5': 'success-grid', 'GR00T N1.7': 'gr00t-success-grid', 'DynamicVLA': 'dynamicvla-success-grid'}
EXAMPLE_NAMES = {'pi0.5': 'prediction-examples', 'MolmoAct2': 'molmoact2-prediction-examples'}


def grid(policy, tasks, with_grid):
    fig, axes = plt.subplots(3, len(tasks), figsize=(width_for(len(tasks)), ROW_HEIGHT * 3 + .6), sharey=True,
                             sharex='col', squeeze=False, layout='constrained')
    for col, task in enumerate(tasks):
        spec, cells = TASKS[task], counts(policy, task, grid=with_grid)
        for row, mode in enumerate(MODES):
            ax = axes[row, col]
            rate_line(ax, spec.speeds, [cells[(s, 'synchronous', 0)] for s in spec.speeds], 'synchronous', 0)
            for delay in spec.delays:
                rate_line(ax, spec.speeds, [cells[(s, mode, delay)] for s in spec.speeds], mode, delay)
            speed_axis(ax, spec.speeds, narrow=True)
            percent_axis(ax)
            panel_title(ax, row * len(tasks) + col, spec.name)
            if col == 0:
                row_label(ax, mode)
            if row < 2:
                ax.tick_params(labelbottom=False)
    shared_xlabel(fig, SPEED_LABEL)
    shared_ylabel(fig, SUCCESS_LABEL)
    setting_legend(fig, TASKS[tasks[0]].delays)
    save(fig, figure_path(GRID_NAMES[policy]))
    plt.close(fig)


def sync_row(policy, tasks):
    """One column of the paper: the tasks in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN_WIDTH, 2 * COLUMN_ROW_HEIGHT + .7), sharey=True,
                             squeeze=False, layout='constrained')
    for col, task in enumerate(tasks):
        spec, cells = TASKS[task], counts(policy, task, grid=True)
        ax = axes.flat[col]
        rate_line(ax, spec.speeds, [cells[(s, 'synchronous', 0)] for s in spec.speeds], 'synchronous', 0)
        for delay in spec.delays:
            rate_line(ax, spec.speeds, [cells[(s, 'synchronous', delay)] for s in spec.speeds], 'synchronous', delay)
        speed_axis(ax, spec.speeds, narrow=True)
        percent_axis(ax)
        panel_title(ax, col, spec.name)
    shared_xlabel(fig, SPEED_LABEL)
    shared_ylabel(fig, SUCCESS_LABEL)
    setting_legend(fig, TASKS[tasks[0]].delays, modes=('synchronous',))
    save(fig, figure_path('task-success'))
    plt.close(fig)


def held_out_errors(policy='pi0.5'):
    """The model's RMS error over the held-out 50/150/250 ms runs of every task and mode, for the paper's table."""
    for task in LIBERO:
        spec, cells = TASKS[task], counts(policy, task, grid=True)
        model = fit_task(task_points(policy, task, grid=True), task, on=('old',))
        for mode in MODES:
            k, n, p = [], [], []
            for delay in BLIND_LATENCIES:
                k += [cells[(s, mode, delay)][0] for s in spec.speeds]
                n += [cells[(s, mode, delay)][1] for s in spec.speeds]
                p += list(curve(model, task, mode, delay, np.array(spec.speeds)))
            print(f'{task:8s} {mode:12s} held-out error {rmse_pp(k, n, p):.2f} points')


def examples(policy='pi0.5', summary=False):
    rows = (None,) if summary else BLIND_LATENCIES
    shown = (50, 150) if summary else BLIND_LATENCIES
    # The main-text summary is one paper column wide; the appendix examples span both columns.
    size = (COLUMN_WIDTH, 2.0) if summary else (width_for(4), 3.9)
    fig, axes = plt.subplots(len(rows), len(EXAMPLES), figsize=size, sharey=True,
                             sharex='col', squeeze=False,
                             layout='constrained')
    for col, (task, mode) in enumerate(EXAMPLES):
        spec, cells = TASKS[task], counts(policy, task, grid=True)
        model = fit_task(task_points(policy, task, grid=True), task, on=('old',))
        fine = np.linspace(spec.speeds[0], spec.speeds[-1], 160)
        for row, delay in enumerate(rows):
            if summary:
                delay = (150, 150, 50)[col]
            ax = axes[row, col]
            rate_line(ax, spec.speeds, [cells[(s, 'synchronous', 0)] for s in spec.speeds], 'synchronous', 0)
            rate_line(ax, spec.speeds, [cells[(s, mode, delay)] for s in spec.speeds], mode, delay, points=True)
            ax.plot(fine, 100 * curve(model, task, mode, delay, fine), color=INK, ls=PREDICTION,
                    lw=LINE['lw'], zorder=2)
            speed_axis(ax, spec.speeds, narrow=True)
            percent_axis(ax)
            title = f'{spec.name}\n{MODE_SHORT[mode]} {delay} ms' if summary else f'{spec.name}, {MODE_SHORT[mode].lower()}'
            panel_title(ax, row * len(EXAMPLES) + col, title)
            if col == 0 and not summary:
                ax.set_ylabel(f'{delay} ms', fontweight='bold')
    shared_xlabel(fig, SPEED_LABEL)
    shared_ylabel(fig, SUCCESS_LABEL)
    prediction = Line2D([], [], color=INK, lw=LINE['lw'], ls=PREDICTION, label='Prediction')
    if summary:
        # Each panel shows one mode at one delay; key each exactly as drawn, in panel order on the first row, then
        # 0 ms and the prediction. Matplotlib fills legend columns first, so interleave the two rows.
        measured = [Line2D([], [], label=f'{MODE_SHORT[mode]} {delay} ms', **delay_point_style(mode, delay))
                    for (_, mode), delay in zip(EXAMPLES, (150, 150, 50))]
        handles = [m for pair in zip(measured, [zero_delay_handle(), prediction]) for m in pair]
        handles.append(measured[-1])
        top_legend(fig, handles, ncol=3)
    else:
        setting_legend(fig, shown, modes=tuple(mode for _, mode in EXAMPLES), points=True, extra=[prediction])
    save(fig, figure_path('prediction-summary' if summary else EXAMPLE_NAMES[policy]))
    plt.close(fig)


def main():
    style(plt)
    grid('pi0.5', LIBERO, True)
    sync_row('pi0.5', LIBERO)
    grid('GR00T N1.7', LIBERO, False)
    grid('DynamicVLA', DYNAMICVLA, False)
    examples()
    examples(summary=True)
    examples('MolmoAct2')
    held_out_errors()
    print('Success grids and the prediction examples written')


if __name__ == '__main__':
    main()
