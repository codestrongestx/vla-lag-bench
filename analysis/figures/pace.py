"""pi0.5's hand speed during alignment against target speed, per task: the four LIBERO tasks in one row.

Each curve is the median episode-average alignment hand speed of the qualifying episodes at each target speed;
the dotted diagonal marks a hand as fast as the target. Lazy 0 ms repeats the synchronous zero-delay
schedule, so the grey zero-delay line stands for both.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analysis.paths import figure_path
from analysis.figures.figure_kit import (HAND_SPEED_LABEL, REFERENCE, SPEED_LABEL, note, numeric_ticks, panel_title, save,
                              series_style, setting_legend, shared_xlabel, speed_axis, style, top_legend, width_for)
from analysis.figures.tasks import DYNAMICVLA, LIBERO, TASKS, alignment_speeds

# Hand-speed axis (top, tick step) in m/s, shared by the LIBERO panels.
HAND_AXIS = {'pi0.5': (.3, .1)}
HEIGHT = {4: 1.75}


def draw(policy, tasks, filename):
    shared = policy == 'pi0.5'
    fig, axes = plt.subplots(1, len(tasks), figsize=(width_for(len(tasks)), HEIGHT[len(tasks)]), squeeze=False,
                             sharey=shared, layout='constrained')
    for index, (ax, task) in enumerate(zip(axes.flat, tasks)):
        spec, speeds = TASKS[task], alignment_speeds(policy, task)
        # Lazy 0 ms runs the synchronous zero-delay schedule; the reference line stands for both.
        for mode, delay in [c for c in spec.conditions if c != ('lazy', 0)]:
            y = [float(np.median(speeds[(s, mode, delay)])) for s in spec.speeds]
            ax.plot(spec.speeds, y, **series_style(mode, delay))
        speed_axis(ax, spec.speeds, narrow=True)
        top, step = HAND_AXIS['pi0.5' if shared else task]
        ax.set_ylim(0, top)
        numeric_ticks(ax, 'y', [round(t, 3) for t in np.arange(0, top + 1e-9, step)])
        edge = ax.get_xlim()[1]
        ax.plot([0, edge], [0, edge], **REFERENCE)
        if index == 0:
            note(ax, .05, .95, 'Hand > target', transform=ax.transAxes, ha='left', va='top')
            note(ax, .95, .05, 'Hand < target', transform=ax.transAxes, ha='right', va='bottom')
        panel_title(ax, index, spec.name)
    axes[0, 0].set_ylabel(HAND_SPEED_LABEL)
    shared_xlabel(fig, SPEED_LABEL)
    delays = sorted({d for task in tasks for _, d in TASKS[task].conditions if d})
    setting_legend(fig, delays)
    save(fig, figure_path(filename))
    plt.close(fig)


def main():
    style(plt)
    draw('pi0.5', LIBERO, 'pace')
    print('Alignment hand-speed curves written for the LIBERO tasks.')


if __name__ == '__main__':
    main()
