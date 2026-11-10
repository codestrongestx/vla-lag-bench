"""Shared figure components; every paper renderer draws through them.

Colors, markers, and hatches come from style.py and text sizes from typography.py. save() writes a figure and then
checks it with style_check.py; a figure that breaks a rule is deleted and the render fails. A PNG figure also gets a
vector PDF beside it: the paper prints the PDF.
"""
from pathlib import Path
import numpy as np
from analysis.figures.numerics import wilson
from analysis.figures.style import (INTERVAL_CAP, INTERVAL_WIDTH, DELAY_GREYS, DELAY_MARKERS, DELAY_RANK, GRID, INK, INK2, MODE_COLORS, MODE_NAMES,
                         MODE_SHADES, MODE_SHORT, MUTED, FRAME, POLICY_COLORS, POLICY_MARKERS, POLICY_NAMES, PREDICTION,
                         ZERO_DELAY, ZERO_DELAY_WIDTH)
from analysis.figures.typography import DPI, MIN_SIZE, PAPER, RC, TEXT_WIDTH

MODES = ('synchronous', 'lazy', 'continuous')
MODE_TITLES = MODE_NAMES
SPEED_LABEL = 'Target speed (m/s)'
SUCCESS_LABEL = 'Success rate (%)'
HAND_SPEED_LABEL = 'Hand speed (m/s)'
RATIO_LABEL = 'Target-to-hand speed ratio'
PERCENT_TICKS = (0, 25, 50, 75, 100)
PERCENT_TOP = 103
# A data series: a line with markers. POINT: markers without a line.
LINE = dict(lw=1.1, ms=2.8, mec=PAPER, mew=.4)
# A delayed execution setting: larger markers, so the delay's shape reads at print size.
DELAYED = dict(LINE, ms=4.2)
POINT = dict(ms=4, mec=PAPER, mew=.4)
# Reference marks (equal speeds, object dimensions) and the dividers between execution-mode groups.
REFERENCE = dict(color=INK2, lw=.6, ls=(0, (2, 2)))
DIVIDER = dict(color=MUTED, lw=.6, ls=(0, (3, 2)))
# In-plot notes and cell values; a note may take the color of the series it labels.
NOTE = dict(fontsize=MIN_SIZE, color=INK2)


def style(plt, **extra):
    """Printed-size text, a thin full frame, and a light grid behind the data."""
    plt.rcParams.update(RC)
    plt.rcParams.update({'axes.edgecolor': FRAME, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': True,
                         'axes.spines.right': True, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .5,
                         'axes.axisbelow': True, 'axes.titlelocation': 'left', **extra})


# Fixed metadata so the same figure always writes the same PDF bytes.
PDF_METADATA = {'Creator': 'plots/figure_kit.py', 'Producer': None, 'CreationDate': None, 'ModDate': None}


def save(fig, path, **keywords):
    """Write the figure at print resolution, then check it; a figure that breaks a style rule is deleted.

    A PNG also gets a vector PDF of the same size beside it, from the same layout."""
    from analysis.figures.style_check import FigureStyleError, problems
    path = Path(path)
    paths = [path]
    fig.savefig(path, dpi=DPI, **keywords)
    if path.suffix == '.png':
        save_vector(fig, path.with_suffix('.pdf'), **keywords)
        paths.append(path.with_suffix('.pdf'))
    found = problems(fig)
    if found:
        for written in paths:
            written.unlink(missing_ok=True)
        raise FigureStyleError(path.name, found)


def save_vector(fig, path, **keywords):
    """Write the PDF with the layout and crop the PNG just used, so the two differ only in rendering.

    A layout engine or a tight crop would otherwise be recomputed from the PDF backend's font metrics and shift
    panels by a few pixels."""
    import matplotlib.pyplot as plt
    # PNG text fields such as Source have no PDF key; the PDF carries them as its Subject.
    notes = keywords.pop('metadata', {})
    metadata = {**PDF_METADATA, **({'Subject': '; '.join(f'{k}: {v}' for k, v in notes.items())} if notes else {})}
    engine, dpi = fig.get_layout_engine(), fig.dpi
    fig.set_layout_engine('none')
    try:
        if keywords.get('bbox_inches', plt.rcParams['savefig.bbox']) == 'tight':
            fig.set_dpi(DPI)
            renderer = fig.canvas.get_renderer()
            pad = keywords.pop('pad_inches', plt.rcParams['savefig.pad_inches'])
            keywords['bbox_inches'] = fig.get_tightbbox(renderer, bbox_extra_artists=keywords.pop(
                'bbox_extra_artists', None)).padded(pad)
            fig.set_dpi(dpi)
        fig.savefig(path, dpi=DPI, metadata=metadata, **keywords)
    finally:
        fig.set_dpi(dpi)
        fig.set_layout_engine(engine)


def panel_title(ax, index, name):
    ax.set_title(f'({chr(97 + index)}) {name}', loc='left')


def subpanel_title(ax, name, **keywords):
    """An unlettered title for an axes inside a lettered panel, such as the path beside a task's frames."""
    ax.set_title(name, loc='left', **keywords)
    ax.style_subpanel = True


def note(ax, x, y, text, **keywords):
    """An in-plot note or cell value at the smallest text size, in the secondary ink unless a color is given."""
    return ax.text(x, y, text, **{**NOTE, **keywords})


def row_label(ax, mode):
    """Name a row of panels by its execution mode, at the left in bold mode color; the shared y label names the
    quantity."""
    ax.set_ylabel(MODE_NAMES[mode], color=MODE_COLORS[mode], fontweight='bold')


def shared_xlabel(fig, text):
    return fig.supxlabel(text, fontsize=RC['axes.labelsize'], color=INK2)


def shared_ylabel(fig, text):
    return fig.supylabel(text, fontsize=RC['axes.labelsize'], color=INK2)


def rates(cells, minimum=1):
    """Success percentages with 95% Wilson bounds for (successes, episodes) pairs; NaN below `minimum` episodes."""
    out = np.array([wilson(k, n) if n >= minimum else (np.nan,) * 3 for k, n in cells], dtype=float) * 100
    return out[:, 0], out[:, 1], out[:, 2]


def delay_color(mode, delay):
    """The mode color at the delay's place on the light-to-dark ramp; the middle delay takes the mode color."""
    return MODE_SHADES[mode][DELAY_RANK[delay]]


def series_style(mode, delay):
    """Line keywords for one execution setting: zero delay is the grey reference, otherwise the mode's color at the
    delay's shade, with the delay's marker on a solid line."""
    if delay == 0:
        return dict(color=ZERO_DELAY, ls='-', lw=ZERO_DELAY_WIDTH, marker=None, zorder=1, solid_capstyle='round')
    return dict(color=delay_color(mode, delay), ls='-', marker=DELAY_MARKERS[delay], zorder=3, **DELAYED)


def single_delay_style(mode, delay):
    """A delayed setting in a panel that shows one delay, named in its title: the mode color itself, so a mode looks
    the same in every panel, with the delay's marker on a solid line."""
    return dict(color=MODE_COLORS[mode], ls='-', marker=DELAY_MARKERS[delay], zorder=3, **DELAYED)


def single_delay_panel(ax, delay):
    """Mark a panel whose delayed series all share one delay, named in its title (see single_delay_style)."""
    ax.style_delay = delay


def prediction_style(mode, delay, *, fitted=False):
    """A model curve for one execution setting: the setting's shade without markers, dashed where it predicts a
    delay the model was not fitted to."""
    if delay == 0:
        return dict(color=ZERO_DELAY, ls='-', lw=ZERO_DELAY_WIDTH, zorder=1, solid_capstyle='round')
    return dict(color=delay_color(mode, delay), ls='-' if fitted else PREDICTION, lw=LINE['lw'], zorder=2)


def delay_point_style(mode, delay):
    """Markers without a line for one execution setting, in its shade and shape."""
    return dict(color=delay_color(mode, delay), ls='', marker=DELAY_MARKERS[delay], ms=DELAYED['ms'], mec=PAPER,
                mew=DELAYED['mew'], zorder=3)


def policy_style(policy, *, line=True):
    """Keywords for one policy's series: its color and marker, as a line or as points."""
    marks = dict(color=POLICY_COLORS[policy], marker=POLICY_MARKERS[policy])
    return {**marks, **LINE} if line else {**marks, 'ls': '', **POINT}


def interval_bars(ax, x, value, low, high, *, color, zorder=2):
    """Capped intervals at observations only; preserve asymmetric bounds, including rates of 0 and 100%."""
    value, low, high = (np.asarray(v, dtype=float) for v in (value, low, high))
    # Floating-point roundoff at the binomial endpoints can put a bound just past the observed value.
    return ax.errorbar(x, value, yerr=np.maximum([value - low, high - value], 0), fmt='none', color=color, ecolor=color,
                       elinewidth=INTERVAL_WIDTH, capsize=INTERVAL_CAP, capthick=INTERVAL_WIDTH, zorder=zorder)


def rate_line(ax, x, cells, mode, delay, *, points=False, minimum=1):
    """Plot a success or pass rate with pointwise Wilson intervals; optional unconnected measured markers. In a
    single_delay_panel a delayed setting takes the mode color itself."""
    value, low, high = rates(cells, minimum)
    if delay and getattr(ax, 'style_delay', None) == delay and not points:
        keywords = single_delay_style(mode, delay)
    else:
        keywords = delay_point_style(mode, delay) if points else series_style(mode, delay)
    interval_bars(ax, x, value, low, high, color=keywords['color'], zorder=keywords['zorder'] - .5)
    ax.plot(x, value, **keywords)
    return value


def zero_delay_handle(label='0 ms'):
    from matplotlib.lines import Line2D
    return Line2D([], [], color=ZERO_DELAY, lw=ZERO_DELAY_WIDTH, label=label, solid_capstyle='round')


def setting_columns(delays, *, modes=MODES, points=False):
    """Every delayed setting keyed exactly as drawn, one list per mode: its delays from short to long, each in its
    shade and marker, named with the mode ("Sync. 100 ms"; a single mode takes its full name). points keys
    unconnected measured markers."""
    from matplotlib.lines import Line2D
    names = MODE_NAMES if len(modes) == 1 else MODE_SHORT
    look = delay_point_style if points else series_style
    return [[Line2D([], [], label=f'{names[m]} {d} ms', **look(m, d)) for d in delays] for m in modes]


def setting_handles(delays, *, modes=MODES, zero=True, points=False):
    """The key's entries in reading order: 0 ms, then every setting exactly as drawn, mode by mode."""
    return [zero_delay_handle()] * zero + [h for column in setting_columns(delays, modes=modes, points=points)
                                           for h in column]


def setting_legend(fig, delays, *, modes=MODES, zero=True, points=False, extra=()):
    """The standard key, every entry drawn exactly as a series in the figure: one column per mode with its delays
    from top to bottom, then a column with 0 ms and any extra entries (a prediction, a stage fill). A narrow figure
    turns the grid: one row per mode, then 0 ms and the extras. One mode keys its delays in a single row after
    0 ms."""
    from matplotlib.lines import Line2D
    columns = setting_columns(delays, modes=modes, points=points)
    last = [zero_delay_handle()] * zero + list(extra)
    if len(modes) == 1:
        handles = last[:1] + columns[0] + last[1:]
        return top_legend(fig, handles, ncol=len(handles))
    narrow = fig.get_figwidth() < .6 * TEXT_WIDTH
    # A wide key holds 0 ms and the extras in one column, as tall as the longer of it and the delays; a narrow key
    # gives each its own row.
    size = len(delays) if narrow else max(len(delays), len(last))
    grid = columns + ([[h] for h in last] if narrow else [last[i:i + size] for i in range(0, len(last), size)])
    # Matplotlib fills legend columns first; pad short lines with blank entries so each mode keeps its own line.
    grid = [line + [Line2D([], [], color=PAPER, ls='', marker='', label=' ')] * (size - len(line)) for line in grid]
    if narrow:
        return top_legend(fig, [line[c] for c in range(size) for line in grid], ncol=size)
    return top_legend(fig, [h for line in grid for h in line], ncol=len(grid))


def single_delay_handles(delay, *, modes=MODES, zero=True, named=False):
    """The key of a row or column of single-delay panels: each mode drawn exactly as plotted there, then 0 ms. named
    puts the delay in each entry ("Sync. 50 ms") where no row label names it."""
    from matplotlib.lines import Line2D
    labels = {m: f'{MODE_SHORT[m]} {delay} ms' for m in modes} if named else MODE_NAMES
    handles = [Line2D([], [], label=labels[m], **single_delay_style(m, delay)) for m in modes]
    if zero:
        handles.append(Line2D([], [], color=ZERO_DELAY, lw=ZERO_DELAY_WIDTH, label='0 ms', solid_capstyle='round'))
    return handles


def policy_handles(policies):
    """One key entry per policy, in the given order."""
    from matplotlib.lines import Line2D
    return [Line2D([], [], label=POLICY_NAMES[p], **policy_style(p)) for p in policies]


def top_legend(fig, handles, ncol=None, **extra):
    """One legend for the figure, centered above the panels (needs a constrained layout)."""
    if fig.get_figwidth() < .6 * TEXT_WIDTH:
        extra = dict(columnspacing=.7, handlelength=1.5, handletextpad=.4, **extra)
    return fig.legend(handles=handles, loc='outside upper center', ncol=ncol or len(handles), **extra)


def column_legend(subfig, handles, ncol=2, **extra):
    """The legend of one row or column subfigure, centered above its panels: a figure whose rows or columns differ in
    what their series look like, such as one delay per row, keys each exactly as it is drawn. Entries fill rows
    left to right, so the key reads in the given order in a narrow column."""
    rows = -(-len(handles) // ncol)
    handles = [handles[r * ncol + c] for c in range(ncol) for r in range(rows) if r * ncol + c < len(handles)]
    extra = dict(columnspacing=.6, handlelength=1.3, handletextpad=.35, **extra)
    return subfig.legend(handles=handles, loc='outside upper center', ncol=ncol, **extra)


def setting_label(mode, delay, *, wrap=False):
    """An execution setting on a categorical axis: 'Sync. 100 ms', or on two lines in a narrow panel."""
    return f'{MODE_SHORT[mode]}{chr(10) if wrap else " "}{delay} ms'


def setting_ticks(ax, axis, positions, conditions, *, colored=False, wrap=False, rotation=0):
    """Label a categorical axis with execution settings; grouped bar charts color them bold by mode, and zero
    delay takes the secondary ink."""
    labels = [setting_label(m, d, wrap=wrap) for m, d in conditions]
    getattr(ax, f'set_{axis}ticks')(positions, labels, rotation=rotation)
    if colored:
        for tick, (mode, delay) in zip(getattr(ax, f'get_{axis}ticklabels')(), conditions):
            tick.set_color(INK2 if delay == 0 else MODE_COLORS[mode])
            tick.set_fontweight('bold')


def grouped_positions(conditions, *, gap=.6):
    """Bar positions in condition order, with extra space between execution modes, and the dividers in those gaps."""
    positions, dividers, at = [], [], 0.
    for i, (mode, _) in enumerate(conditions):
        if i and mode != conditions[i - 1][0]:
            dividers.append(at - .2)
            at += gap
        positions.append(at)
        at += 1
    return positions, dividers


def mode_dividers(ax, dividers, *, axis):
    """Dashed dividers between execution-mode groups of bars; `axis` is the bars' category axis."""
    line = ax.axhline if axis == 'y' else ax.axvline
    for at in dividers:
        line(at, zorder=1, **DIVIDER)


def numeric_ticks(ax, axis, values):
    """Ticks labelled in their shortest form: 0, 0.05, 0.1."""
    getattr(ax, f'set_{axis}ticks')(values, [f'{v:g}' for v in values])


def speed_ticks(speeds, *, narrow=False):
    """Ticks from zero to the grid's top speed (0.2 or 0.4 m/s): five, or three in a narrow panel."""
    top = .2 if speeds[-1] <= .2 else .4
    ticks = np.linspace(0, top, 3 if narrow else 5)
    return tuple(ticks), [f'{t:g}' for t in ticks]


def speed_axis(ax, speeds, *, narrow=False, margin=.04):
    ax.set_xticks(*speed_ticks(speeds, narrow=narrow))
    high = speeds[-1]
    ax.set_xlim(-margin * high, (1 + margin) * high)
    ax.style_speed = True


def measured_speed_axis(ax, speeds, *, static=False, margin=.08):
    """A speed axis ticked at each measured speed only, for an experiment that tests too few speeds for the 0.05-step
    grid (the real robot); static=True labels speed 0 as 'Static'."""
    labels = ['Static' if static and s == 0 else f'{s:g}' for s in speeds]
    ax.set_xticks(speeds, labels)
    span = speeds[-1] - speeds[0]
    ax.set_xlim(speeds[0] - margin * span, speeds[-1] + margin * span)
    ax.style_speed = 'measured'


def count_axis(ax, attempts):
    """Successes out of a small number of attempts, one tick and grid line per success."""
    ax.set_ylim(0, attempts * PERCENT_TOP / 100)
    ax.set_yticks(range(attempts + 1))
    ax.style_count = attempts


def percent_axis(ax):
    ax.set_ylim(0, PERCENT_TOP)
    ax.set_yticks(PERCENT_TICKS)
    ax.style_percent = True


def width_for(columns, full_columns=4):
    """Printed width that keeps a panel the same size as in the four-task LIBERO figure."""
    return TEXT_WIDTH * columns / full_columns
