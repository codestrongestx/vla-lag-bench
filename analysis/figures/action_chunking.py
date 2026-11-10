"""Two Background figures: a VLA policy's architecture, and action chunking with K and H.

The architecture figure shows the structure VLA policies share: a VLM backbone (a vision encoder feeding a
language model) and an action expert that predicts a chunk of K actions. The chunking
figure defines K and H alone, on the 50 ms control-step axis.

A vision-language model encodes the camera images and the instruction (src/vlalagbench/benchmarks/libero/evaluator.py), and an action decoder returns K actions (pi05_libero's action horizon). The robot state the benchmark
also sends enters different policies at different places, and pi05_libero does not read it, so the diagram leaves it
out. The robot applies one action per 20 Hz control step and executes the first H before the next observation's chunk
replaces the rest. K, H, and the control rate come from the checked-in configuration, and which actions run comes from
the benchmark scheduler at zero delay. One paper column wide.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
import yaml

from analysis.paths import CONFIGS, figure_path, use_benchmark_src
from analysis.figures.figure_kit import note, save, shared_xlabel, style, top_legend
from analysis.figures.style import GRID, INK, INK2, MUTED, TARGET_STAGE_FILL
from analysis.figures.typography import COLUMN_WIDTH, PAPER

use_benchmark_src()
from vlalagbench.timing import ChunkScheduler, TimingConfig

CONFIG = CONFIGS / 'experiments/libero/pi05_libero_goal_task0_2x_lazy_100ms.yaml'
settings = yaml.safe_load(CONFIG.read_text())
inference = settings['inference']
K = inference['action_chunk_steps']
H = inference['execution_horizon_steps']
HZ = settings['benchmark']['control_hz']
STEP_MS = 1000 / HZ
assert (K, H, HZ) == (10, 5, 20)
CHUNKS = 2  # two consecutive chunks show one chunk replacing the rest of the previous one
SPAN = (CHUNKS - 1) * H + K  # control steps until the last drawn chunk's final action
CELL = .5
ROWS = {'A': 1.0, 'B': .3}
BLOCK_Y, BLOCK_H = 0, .66
ARROW = dict(arrowstyle='-|>', color=INK2, lw=.7, shrinkA=0, shrinkB=0, mutation_scale=7)
KINDS = {
    'executed': dict(facecolor=PAPER, edgecolor=INK2, lw=.7),
    'replaced': dict(facecolor=PAPER, edgecolor=GRID, lw=.7),
}


def executed_actions():
    """Chunk index -> action indices the robot executes, from the scheduler at zero delay."""
    scheduler = ChunkScheduler(TimingConfig('synchronous', K, H, 0))
    executed, starts = {}, {}
    for step in range(CHUNKS * H):
        request = scheduler.request(step)
        if request:
            starts[request.id] = step
            scheduler.deliver(request, list(range(K)))
        action = scheduler.action(step)
        assert action.active and step == starts[action.chunk_id] + action.index_in_chunk
        executed.setdefault(action.chunk_id, []).append(action.index_in_chunk)
    return starts, executed


def bracket(ax, x0, x1, y, text):
    """A dimension line across [x0, x1] with end ticks and its label above the middle."""
    ax.annotate('', (x1 - 2, y), (x0 + 2, y), arrowprops=dict(arrowstyle='|-|,widthA=.2,widthB=.2', color=INK2,
                                                               lw=.6, shrinkA=0, shrinkB=0))
    note(ax, (x0 + x1) / 2, y + .05, text, ha='center', va='bottom')


def architecture():
    """The structure VLA policies share: a VLM backbone, whose vision encoder embeds the images for a language model
    that also reads the instruction, and an action expert that turns the language model's output into a chunk of K
    actions (a flow-matching or diffusion transformer, or a decoder of action tokens).

    Drawn in inches on a 1:1 axis so box corners stay round."""
    height = 1.0
    fig = plt.figure(figsize=(COLUMN_WIDTH, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, COLUMN_WIDTH)
    ax.set_ylim(0, height)
    ax.set_axis_off()
    vlm_style = dict(facecolor=TARGET_STAGE_FILL, edgecolor=INK2, lw=.7)
    head_style = dict(facecolor=PAPER, edgecolor=INK2, lw=.7)
    upper, lower, row_h, box_w = .6, .3, .22, .58

    def block(x, y0, y1, text, style_):
        ax.add_patch(FancyBboxPatch((x, y0), box_w, y1 - y0, boxstyle='round,pad=0,rounding_size=.05', **style_))
        note(ax, x + box_w / 2, (y0 + y1) / 2, text, ha='center', va='center', color=INK, linespacing=1.15)

    def arrow(x0, x1, y):
        ax.annotate('', (x1, y), (x0, y), arrowprops=ARROW)

    # The images pass through the vision encoder; the instruction goes straight to the language model.
    inputs = .6
    encoder, model = inputs + .1, inputs + .1 + box_w + .12
    note(ax, inputs - .03, upper, 'Instruction', ha='right', va='center')
    note(ax, inputs - .03, lower, 'Images', ha='right', va='center')
    block(encoder, lower - row_h / 2, lower + row_h / 2, 'Vision\nencoder', vlm_style)
    arrow(inputs, model, upper)
    arrow(inputs, encoder, lower)
    arrow(encoder + box_w, model, lower)
    bottom, top = lower - row_h / 2, upper + row_h / 2
    block(model, bottom, top, 'Language\nmodel', vlm_style)
    # The vision encoder and the language model form the VLM backbone.
    group = (encoder - .05, bottom - .06, model + box_w + .05, top + .06)
    ax.add_patch(FancyBboxPatch(group[:2], group[2] - group[0], group[3] - group[1],
                                boxstyle='round,pad=0,rounding_size=.07', facecolor='none', edgecolor=MUTED, lw=.6,
                                ls=(0, (3, 2))))
    note(ax, (group[0] + group[2]) / 2, group[3] + .03, 'VLM backbone', ha='center', va='bottom', color=INK,
         fontweight='bold')
    middle = (upper + lower) / 2
    head = group[2] + .12
    arrow(model + box_w, head, middle)
    block(head, middle - .17, middle + .17, 'Action\nexpert', head_style)
    arrow(head + box_w, head + box_w + .12, middle)
    note(ax, head + box_w + .15, middle, 'Action\nchunk', ha='left', va='center', linespacing=1.15)
    save(fig, figure_path('vla-architecture'))
    plt.close(fig)


def chunks():
    """Two consecutive chunks of K actions, of which the robot executes the first H."""
    starts, executed = executed_actions()
    assert all(indices == list(range(H)) for indices in executed.values())
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.45), layout='constrained')
    for chunk, letter in zip(sorted(starts), 'AB'):
        y = ROWS[letter]
        origin = starts[chunk] * STEP_MS
        for i in range(K):
            kind = 'executed' if i in executed[chunk] else 'replaced'
            ax.add_patch(Rectangle((origin + i * STEP_MS, y), STEP_MS, CELL, **KINDS[kind]))
            note(ax, origin + (i + .5) * STEP_MS, y + CELL / 2, f'${letter.lower()}_{{{i}}}$',
                 color=INK if kind == 'executed' else MUTED, ha='center', va='center')
        note(ax, -12, y + CELL / 2, f'Chunk {letter}', ha='right', va='center')
    a = starts[min(starts)] * STEP_MS
    bracket(ax, a, a + H * STEP_MS, ROWS['A'] + CELL + .1, f'Execution horizon $H$ = {H}')
    bracket(ax, a, a + K * STEP_MS, ROWS['A'] + CELL + .65, f'Action chunk size $K$ = {K}')
    ax.set_xlim(-115, SPAN * STEP_MS)
    ax.set_ylim(0, ROWS['A'] + CELL + 1.0)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Ticks mark every 50 ms control step up to the arrowhead, which ends the axis; labels every 100 ms keep 7 pt
    # numbers apart at one column. The chunk-size bracket sits high enough that its end ticks clear the horizon label.
    ax.set_xticks([i * STEP_MS for i in range(SPAN)],
                  [f'{i * STEP_MS:g}' if i % 2 == 0 else '' for i in range(SPAN)])
    ax.tick_params(axis='x', length=3)
    ax.annotate('', xy=(SPAN * STEP_MS, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=MUTED, lw=.6),
                annotation_clip=False)
    handles = [Patch(**KINDS['executed'], label='Executed'), Patch(**KINDS['replaced'], label='Replaced by next chunk')]
    top_legend(fig, handles)
    shared_xlabel(fig, 'Elapsed time (ms)')
    save(fig, figure_path('action-chunking'))
    plt.close(fig)


def main():
    style(plt, **{'axes.grid': False})
    architecture()
    chunks()
    print('VLA architecture and action chunking figures rendered; chunks from vlalagbench.timing.ChunkScheduler.')


if __name__ == '__main__':
    main()
