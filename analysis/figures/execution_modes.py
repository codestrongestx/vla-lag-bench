"""Execution timelines generated from the benchmark scheduler.

The schedule-quantities figure is one paper column wide. Each execution mode takes one panel on a shared time axis
that opens when one chunk, B, is requested, so the axis reads the staleness of B's actions: the inference row, and the
actions the robot executes, each named by its chunk and index from 0, with idle steps hatched. A pointer runs from B's
observation to the mean start of B's executed actions, which a bracket spans, and gives the mean staleness. Action
chunking itself is drawn in Background (plots/action_chunking.py), so this figure only uses it.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml

from analysis.paths import CONFIGS, figure_path, use_benchmark_src
from analysis.figures.figure_kit import MODES, REFERENCE, note, panel_title, save, shared_xlabel, style
from analysis.figures.style import IDLE_FILL, MODE_COLORS, MODE_DARK, MODE_NAMES, MUTED
from analysis.figures.typography import COLUMN_WIDTH, PAPER

use_benchmark_src()
from vlalagbench.timing import ChunkScheduler, TimingConfig, effective_steps

CONFIG = CONFIGS / 'experiments/libero/pi05_libero_goal_task0_2x_lazy_100ms.yaml'
settings = yaml.safe_load(CONFIG.read_text())
inference = settings['inference']
K = inference['action_chunk_steps']
H = inference['execution_horizon_steps']
HZ = settings['benchmark']['control_hz']
STEP_MS = 1000 / HZ
L = effective_steps(inference['latency_ms'], HZ)
assert (K, H, L) == (10, 5, 2)
WARMUP = 14  # chunk B is the first chunk requested once startup has settled into the repeating schedule
SPAN = 2 * L + H + 1  # synchronous execution runs B, waits for C, and starts C within the window
CELL = .5
ROWS = {'inference': 1.62, 'pointer': 1.12, 'bracket': .66, 'executed': 0}
TOP = 2.0
BOTTOM = -.12


def schedule(mode, steps):
    scheduler = ChunkScheduler(TimingConfig(mode, K, H, L))
    requests, actions = [], []
    for step in range(steps):
        request = scheduler.request(step)
        if request:
            requests.append(request)
            scheduler.deliver(request, list(range(K)))
        actions.append(scheduler.action(step))
    return requests, actions


def cell(ax, x, text, mode, idle):
    """One action step: outlined in the mode color and named by chunk and index, or hatched while the robot idles."""
    style = dict(facecolor=IDLE_FILL, edgecolor=MUTED, hatch='///') if idle else \
        dict(facecolor=PAPER, edgecolor=MODE_COLORS[mode])
    ax.add_patch(Rectangle((x, ROWS['executed']), STEP_MS, CELL, lw=.7, **style))
    if text:
        note(ax, x + STEP_MS / 2, ROWS['executed'] + CELL / 2, text, color=MODE_DARK[mode], ha='center',
             va='center')


def draw_quantities():
    """One panel per mode on one shared time axis that opens when chunk B is requested and its observation taken."""
    fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH, 2.95), layout='constrained')
    for index, (ax, mode) in enumerate(zip(axes, MODES)):
        requests, actions = schedule(mode, steps=WARMUP + 4 * (H + L) + K)
        chunk = next(r for r in requests if r.sent_step >= WARMUP)
        first, last = chunk.sent_step, chunk.sent_step + SPAN

        def letter(chunk_id):
            return chr(ord('B') + chunk_id - chunk.id)

        for request in requests:
            if request.available_step <= first or request.sent_step >= last:
                continue
            start, end = max(request.sent_step, first), min(request.available_step, last)
            x, width = (start - first) * STEP_MS, (end - start) * STEP_MS
            ax.add_patch(Rectangle((x, ROWS['inference']), width, .36, facecolor=PAPER, edgecolor=MODE_COLORS[mode],
                                   lw=.7))
            note(ax, x + width / 2, ROWS['inference'] + .18, letter(request.id), color=MODE_DARK[mode], ha='center',
                 va='center')
        executed = []
        idle_start = None
        for step in range(first, last + 1):
            action = actions[step] if step < last else None
            if action is not None:
                assert not action.active or action.chunk_id >= chunk.id - 1
                name = (f'${letter(action.chunk_id).lower()}_{{{action.index_in_chunk}}}$'
                        if action.active else '')
                cell(ax, (step - first) * STEP_MS, name, mode, idle=not action.active)
                if action.active and action.chunk_id == chunk.id:
                    executed.append(step)
            if action is not None and not action.active:
                idle_start = step if idle_start is None else idle_start
            elif idle_start is not None:
                note(ax, ((idle_start + step) / 2 - first) * STEP_MS, ROWS['executed'] + CELL / 2, 'Idle',
                     ha='center', va='center', bbox=dict(facecolor=IDLE_FILL, edgecolor='none', pad=.2))
                idle_start = None
        # Staleness is measured from B's observation, taken at its request (x = 0), to the step each action starts.
        ages = [(step - first) * STEP_MS for step in executed]
        mean = sum(ages) / len(ages)
        x0, x1 = (executed[0] - first) * STEP_MS, (executed[-1] + 1 - first) * STEP_MS
        ax.annotate('', (x1 - 2, ROWS['bracket']), (x0 + 2, ROWS['bracket']),
                    arrowprops=dict(arrowstyle='|-|,widthA=.2,widthB=.2', color=MODE_COLORS[mode], lw=.7, shrinkA=0,
                                    shrinkB=0))
        ax.plot([0, 0], [ROWS['pointer'], ROWS['inference']], **REFERENCE)
        ax.annotate('', (mean, ROWS['bracket'] + .03), (0, ROWS['pointer']),
                    arrowprops=dict(arrowstyle='-|>', color=MODE_DARK[mode], lw=.7, shrinkA=0, shrinkB=0,
                                    mutation_scale=6, connectionstyle='angle,angleA=0,angleB=90,rad=0'))
        note(ax, mean + 10, ROWS['pointer'] - .02, f'Mean staleness {mean:g} ms', color=MODE_DARK[mode], ha='left',
             va='center')
        idle = L * STEP_MS if mode == 'synchronous' else 0
        note(ax, SPAN * STEP_MS, TOP + .08, f'Idle {idle:g} ms per {H} actions', ha='right', va='bottom',
             clip_on=False)
        for label, y in (('Inference', ROWS['inference'] + .18), ('Executed', ROWS['executed'] + CELL / 2)):
            note(ax, -10, y, label, ha='right', va='center')
        # The reviewer asked this figure to mark lazy and continuous execution as asynchronous (2026-10-25).
        panel_title(ax, index, MODE_NAMES[mode] + ('' if mode == 'synchronous' else ' async'))
        ax.set_xlim(0, SPAN * STEP_MS)
        ax.set_ylim(BOTTOM, TOP)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Every panel ticks each 50 ms step along an arrow; the bottom panel alone numbers them.
        bottom = index == len(MODES) - 1
        ax.set_xticks([i * STEP_MS for i in range(SPAN + 1)],
                      [f'{i * STEP_MS:g}' for i in range(SPAN + 1)] if bottom else [])
        ax.tick_params(axis='x', length=3)
        ax.annotate('', xy=(SPAN * STEP_MS, BOTTOM), xytext=(0, BOTTOM),
                    arrowprops=dict(arrowstyle='->', color=MUTED, lw=.6), annotation_clip=False)
    shared_xlabel(fig, "Time since chunk B's observation (ms)")
    fig.get_layout_engine().set(h_pad=.02, hspace=.1)
    save(fig, figure_path('schedule-quantities'))
    plt.close(fig)


def main():
    style(plt, **{'axes.grid': False})
    draw_quantities()
    print('Execution timelines rendered from vlalagbench.timing.ChunkScheduler.')


if __name__ == '__main__':
    main()
