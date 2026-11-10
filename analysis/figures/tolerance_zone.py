"""Tolerance zones: hit rate of each release, set-down, or grasp attempt against its offset from the target center.

One function draws the LIBERO panels (pi0.5: moving basket, moving plate, moving bowl) and the DynamicVLA panels
(moving box, moving can). A zone is the offset range whose 0.5 cm bins hold at least 30 attempts and at least
half of them hit. Beside each LIBERO panel, side views (stage_views.py) show one hit inside the zone and one miss
outside it at the moment of the attempt, the hit above the miss, in one column.
"""
from analysis.paths import figure_path, output  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
from analysis.figures.figure_kit import (LINE, REFERENCE, interval_bars, note, panel_title, percent_axis, save, shared_ylabel,  # noqa: E402
                              style, top_legend)
from analysis.figures.typography import COLUMN_WIDTH, PAPER  # noqa: E402
from analysis.figures.numerics import wilson_percent  # noqa: E402
from analysis.figures.style import INK, INK2, MUTED, TARGET  # noqa: E402
from analysis.figures.tasks import POLICY_LABEL, TASKS, attempt_rows  # noqa: E402
from analysis.figures import stage_views  # noqa: E402

MINIMUM = 30
# Offsets are binned out to 15 cm, or 30 cm for the larger DynamicVLA box.
REACH = {'box': 30}
# The zone is neutral; target travel takes the moving-target color.
ZONE, ZONE_ALPHA = MUTED, .16
# Per task: x-axis label, x extent and ticks, reference marks (position cm, label, side of the line, label top in %
# hits; a negative value instead gives the label's bottom), zone-label position (x, y, alignment), and whether the
# zone starts at zero offset; a zone label may end in True to break after "≥50% hits:". Panels stack in one
# column, so every label runs horizontally. Reference dimensions are
# measured from the simulator models (meshes, the containment site, and the recorded success predicate's box size).
PANELS = {
    'basket': ('Cheese–basket offset (cm)', 12, (0, 3, 6, 9, 12), [(6.1, 'Containment edge', 'right', 97)],
               (3.1, 35, 'center'), True),
    'plate': ('Bowl–plate offset (cm)', 12, (0, 3, 6, 9, 12),
              [(3.0, 'Success\nradius', 'right', 97), (6.9, 'Plate rim', 'right', 74)], (3.6, 45, 'left', True), True),
    'cabinet': ('Gripper–bowl offset (cm)', 12, (0, 3, 6, 9, 12), [(5.6, 'Bowl rim', 'left', 44)],
                (11.8, 78, 'right', True), False),
    'box': ('Can–box offset (cm)', 24, (0, 6, 12, 18, 24),
            [(8.4, 'Half-width', 'left', -8), (11.8, 'Half-length', 'right', 97)], (23.6, 55, 'right'), True),
    'can': ('Gripper–can offset (cm)', 12, (0, 3, 6, 9, 12), [(3.4, 'Can radius', 'right', 97)], (11.8, 62, 'right'),
            True),
}


def edges(task):
    return np.arange(0, REACH.get(task, 15) + .01, .5)


def binned(rows, EDGES):
    k = np.zeros(len(EDGES) - 1); n = np.zeros(len(EDGES) - 1)
    offsets = np.array([100 * float(r['offset_m']) for r in rows])
    hits = np.array([int(r['hit']) for r in rows])
    index = np.searchsorted(EDGES, offsets, side='right') - 1
    keep = (index >= 0) & (index < len(k))
    np.add.at(k, index[keep], hits[keep]); np.add.at(n, index[keep], 1)
    return k, n


def side_views(tasks):
    """Render one hit and one miss per task; (image, offset in cm) pairs per task, hit first."""
    folder = output('tolerance_zone/stage_views')
    views = {}
    for task in tasks:
        traces = stage_views.trace_rows(task)
        views[task] = [stage_views.render(task, hit, stage_views.VIEWS[task, hit], traces, folder) for hit in (1, 0)]
    return {task: [(plt.imread(folder / r['image']), r['offset_cm']) for r in records] for task, records in views.items()}


# One-column layout with side views, in inches: the legend rows, each panel's row (title, axes, tick labels and
# x-label), the plot's left margin and axes width, and the side views' width and the gap between hit and miss.
LEGEND_H, ROW_H, TITLE_H, AXES_H = .6, 1.18, .17, .69
PLOT_LEFT, PLOT_W, PHOTO_W, PHOTO_GAP = .42, 2.02, .7, .02
# Each side view keeps the middle of the frame, where the gripper meets the target, at the frame's 4:3 shape.
CROP = (.1, 0, .9, .8)


def beside_views(tasks, views):
    """Plots on the left and, beside each panel, a hit inside the zone above a miss outside it."""
    height = LEGEND_H + ROW_H * len(tasks)
    fig = plt.figure(figsize=(COLUMN_WIDTH, height))
    photo_h = PHOTO_W * 3 / 4
    left = COLUMN_WIDTH - PHOTO_W - .01
    axes = []
    for row, task in enumerate(tasks):
        top = height - LEGEND_H - row * ROW_H
        axes.append([fig.add_axes([PLOT_LEFT / COLUMN_WIDTH, (top - TITLE_H - AXES_H) / height,
                                   PLOT_W / COLUMN_WIDTH, AXES_H / height])])
        upper = top - (ROW_H - 2 * photo_h - PHOTO_GAP) / 2
        for index, (pixels, offset) in enumerate(views[task]):
            h, w = pixels.shape[:2]
            crop = pixels[int(CROP[1] * h):int(CROP[3] * h), int(CROP[0] * w):int(CROP[2] * w)]
            bottom = upper - (index + 1) * photo_h - index * PHOTO_GAP
            photo = fig.add_axes([left / COLUMN_WIDTH, bottom / height, PHOTO_W / COLUMN_WIDTH, photo_h / height])
            photo.imshow(crop, interpolation='none')
            photo.set_axis_off()
            note(photo, .03, .95, 'Hit' if index == 0 else 'Miss', color=INK, transform=photo.transAxes,
                 ha='left', va='top', bbox=dict(boxstyle='square,pad=.15', facecolor=PAPER, edgecolor='none', alpha=.85))
    return fig, np.array(axes)


def draw(policy, tasks, filename, *, delay_ms, views=None):
    travel = 100 * .1 * delay_ms / 1000
    if views:
        fig, axes = beside_views(tasks, views)
    else:
        fig, axes = plt.subplots(len(tasks), 1, figsize=(COLUMN_WIDTH, .25 + 1.15 * len(tasks)), squeeze=False,
                                 layout='constrained')
    summary = {}
    for index, (ax, task) in enumerate(zip(axes[:, 0], tasks)):
        xlabel, extent, ticks, marks, (tx, ty, ha, *wrap), from_zero = PANELS[task]
        bins = edges(task)
        k, n = binned(attempt_rows(policy, task), bins)
        keep = n >= MINIMUM
        x = ((bins[:-1] + bins[1:]) / 2)[keep]
        estimate = np.array([wilson_percent(a, b) for a, b in zip(k[keep], n[keep])]).T
        inside = x[estimate[0] >= 50]
        low, high = inside.min() - .25, inside.max() + .25
        if from_zero:
            low = 0.
        summary[task] = dict(low=low, high=high, peak=float(estimate[0].max()), attempts=int(n.sum()))
        ax.axvspan(low, high, color=ZONE, alpha=ZONE_ALPHA, lw=0)
        interval_bars(ax, x, *estimate, color=INK)
        ax.plot(x, estimate[0], color=INK, lw=LINE['lw'])
        for position, label, side, top in marks:
            ax.axvline(position, **REFERENCE)
            left = side == 'left'
            pad = .15 * extent / 12
            note(ax, position - pad if left else position + pad, abs(top), label,
                 ha='right' if left else 'left', va='top' if top > 0 else 'bottom')
        ax.annotate('', xy=(low, 50), xytext=(high, 50), arrowprops=dict(arrowstyle='<->', color=INK2, lw=.9, shrinkA=0, shrinkB=0))
        zone = f'0–{high:.1f} cm' if from_zero else f'{low:.1f}–{high:.1f} cm'
        extra = '' if from_zero else f'\nbest {summary[task]["peak"]:.0f}%'
        note(ax, tx, ty, f"≥50% hits:{chr(10) if wrap else ' '}{zone}{extra}", ha=ha, va='center', linespacing=1.15)
        # Target travel starts at the zone's inner edge, clear of the reference lines near its outer edge.
        ax.plot([low, low + travel], [12, 12], color=TARGET, lw=2.2, solid_capstyle='butt')
        note(ax, low + travel / 2, 15, f'{travel:g} cm', ha='center', color=TARGET)
        ax.set_xlim(0, extent); percent_axis(ax); ax.set_xticks(ticks)
        panel_title(ax, index, TASKS[task].name); ax.set_xlabel(xlabel)
    shared_ylabel(fig, 'Hits per attempt (%)')
    handles = [Patch(color=ZONE, alpha=ZONE_ALPHA, label='Tolerance zone'),
               Line2D([], [], color=INK, lw=LINE['lw'], label=f'Hit rate ({POLICY_LABEL[policy]}, all conditions)'),
               Line2D([], [], color=TARGET, lw=2.2, label=f'Target travel, {delay_ms} ms at 0.1 m/s')]
    if views:
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1), ncol=1, borderaxespad=.1)
    else:
        top_legend(fig, handles, ncol=1)
    save(fig, figure_path(filename))
    plt.close(fig)
    return summary


def main():
    style(plt)
    tasks = ('basket', 'plate', 'cabinet')
    libero = draw('pi0.5', tasks, 'tolerance-zone', delay_ms=200, views=side_views(tasks))
    dynamicvla = draw('DynamicVLA', ('box', 'can'), 'dynamicvla-tolerance-zone', delay_ms=240)
    print('Tolerance zones:', {t: {k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()}
                               for t, s in {**libero, **dynamicvla}.items()})


if __name__ == '__main__':
    main()
