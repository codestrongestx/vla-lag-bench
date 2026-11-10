"""Save-time check that a paper figure follows the shared figure style (style.py, typography.py).

figure_kit.save() calls problems() on every figure it writes. Each problem names the rule it breaks; any problem
deletes the output and fails the render, so an off-style figure never reaches the paper.
"""
import re
import matplotlib
from matplotlib.colors import to_hex, to_rgba
from matplotlib.container import ErrorbarContainer
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from analysis.figures import style as S
from analysis.figures.typography import FULL_WIDTH, MIN_SIZE, PAPER, RC, SIZES, TEXT_WIDTH

TAG = re.compile(r'^\(([a-z])\) ')
TRAILING_ZERO = re.compile(r'^[−+-]?\d+\.\d*0$')
SPEED_LABEL, PERCENT_TICKS, PERCENT_TOP = 'Target speed (m/s)', (0, 25, 50, 75, 100), 103
LINE_WIDTH = 1.1
SPEED_STEPS = (.05, .1, .2)


class FigureStyleError(ValueError):
    def __init__(self, name, problems):
        super().__init__(f'{name} breaks the figure style:\n  ' + '\n  '.join(problems))
        self.problems = problems


def palette():
    """Every color a board or paper figure may draw outside the change-from-zero-delay color scale."""
    colors = {S.INK, S.INK2, S.MUTED, S.GRID, S.FRAME, PAPER, S.ZERO_DELAY, S.TARGET_STAGE_FILL, S.IDLE_FILL, S.TARGET, S.TARGET_BOX}
    for table in (S.MODE_COLORS, S.MODE_DARK, S.POLICY_COLORS, S.OUTCOME_COLORS):
        colors |= set(table.values())
    for shades in (*S.MODE_SHADES.values(), S.DELAY_GREYS):
        colors |= set(shades)
    return {to_hex(c) for c in colors}


def problems(fig):
    renderer = fig.canvas.get_renderer()
    axes = [ax for ax in fig.axes if ax.get_visible()]
    found = []
    if fig.get_figwidth() > FULL_WIDTH + 1e-6:
        found.append(f'width {fig.get_figwidth():.2f} in exceeds the {FULL_WIDTH} in text width')
    if fig._suptitle is not None and fig._suptitle.get_text():
        found.append('figure-level title; captions carry titles')
    return found + texts(fig, axes) + panels(fig, axes, renderer) + legends(fig, axes, renderer) + \
        colors(fig, axes) + series(axes) + uncertainty(axes) + frames(axes) + ticks(fig, axes)


# ---------------- helpers ----------------
def colorbar(ax):
    return hasattr(ax, '_colorbar')


def framed(ax):
    return ax.axison and not colorbar(ax) and all(s.get_visible() for s in ax.spines.values())


def drawn_tick_labels(ax):
    if not ax.axison:
        return []
    labels = []
    for axis in (ax.xaxis, ax.yaxis):
        for tick in axis._update_ticks():
            labels += [t for t in (tick.label1, tick.label2) if t.get_visible() and t.get_text().strip()]
    return labels


def all_texts(fig, axes):
    found = [t for t in fig.texts]
    for label in (fig._supxlabel, fig._supylabel):
        if label is not None:
            found.append(label)
    for ax in axes:
        found += [ax._left_title, ax.title, ax._right_title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
        found += drawn_tick_labels(ax)
    for legend in fig.legends + [ax.get_legend() for ax in axes if ax.get_legend()]:
        found += legend.get_texts()
    return [t for t in found if t.get_visible() and t.get_text().strip()]


def name(ax):
    return ax._left_title.get_text() or ax.title.get_text() or ax.get_ylabel() or 'untitled axes'


# ---------------- rules ----------------
def texts(fig, axes):
    found = []
    family = RC['font.family']
    for text in all_texts(fig, axes):
        content = text.get_text()
        size = round(text.get_fontsize() * 4) / 4
        if size < MIN_SIZE:
            found.append(f'text "{content[:30]}" is {size:g} pt, below the {MIN_SIZE:g} pt floor beside 10 pt body text')
        elif size not in SIZES:
            found.append(f'text "{content[:30]}" is {size:g} pt; sizes are {", ".join(f"{s:g}" for s in SIZES)} pt')
        if text.get_fontfamily()[0] != family:
            found.append(f'text "{content[:30]}" is not set in {family}')
        body = TAG.sub('', content)
        if body and 'a' <= body[0] <= 'z':
            found.append(f'text "{content[:30]}" starts with a lowercase letter')
    return found


def panels(fig, axes, renderer):
    found, tagged, candidates = [], [], 0
    for ax in axes:
        if colorbar(ax):
            continue
        left, center, right = ax._left_title.get_text(), ax.title.get_text(), ax._right_title.get_text()
        if center or right:
            found.append(f'title "{center or right}" is not left-aligned; use panel_title')
        subpanel = getattr(ax, 'style_subpanel', False)
        if TAG.match(left):
            tagged.append(ax._left_title)
            if subpanel:
                found.append(f'subpanel title "{left}" carries a panel letter')
        elif left and not subpanel:
            found.append(f'panel title "{left}" lacks its "(a)" letter')
        if framed(ax) or left:
            candidates += 1
    tagged += [t for t in fig.texts if TAG.match(t.get_text())]
    if candidates > 1:
        for ax in axes:
            if framed(ax) and not ax._left_title.get_text():
                found.append(f'panel "{name(ax)}" has no "(a)" title')
    # Letters run a, b, c, ... in reading order: rows from the top, then left to right.
    boxes = [(t.get_window_extent(renderer), TAG.match(t.get_text()).group(1)) for t in tagged]
    boxes.sort(key=lambda item: -item[0].y1)
    rows = []
    for box, letter in boxes:
        if rows and abs(rows[-1][0][0].y1 - box.y1) <= box.height / 2:
            rows[-1].append((box, letter))
        else:
            rows.append([(box, letter)])
    order = ''.join(letter for row in rows for _, letter in sorted(row, key=lambda item: item[0].x0))
    if order != ''.join(chr(97 + i) for i in range(len(order))):
        found.append(f'panel letters read "{order}" instead of running a, b, c in reading order')
    return found


def legends(fig, axes, renderer):
    found = [f'panel "{name(ax)}" has its own legend; use one top_legend' for ax in axes if ax.get_legend()]
    # One legend for the figure, or one per row or column subfigure, each centered above its own panels.
    groups = [(fig, fig.legends, axes)]
    for sub in getattr(fig, 'subfigs', []):
        groups.append((sub, sub.legends, [ax for ax in sub.axes if ax.get_visible()]))
    if len(fig.legends) > 1:
        found.append(f'{len(fig.legends)} legends; a figure has at most one')
    if fig.legends and any(sub.legends for sub, _, _ in groups[1:]):
        found.append('a figure legend and column legends; a figure has at most one')
    for owner, owned, members in groups:
        if owner is not fig and len(owned) > 1:
            found.append(f'{len(owned)} legends in one column; a column has at most one')
        for legend in owned:
            box = legend.get_window_extent(renderer)
            frame = owner.bbox
            if legend.get_frame_on():
                found.append('the legend has a frame')
            if abs((box.x0 + box.x1) / 2 - (frame.x0 + frame.x1) / 2) > .02 * fig.bbox.width:
                found.append('the legend is not centered')
            if box.x0 < frame.x0 - 1 or box.x1 > frame.x1 + 1:
                found.append('the legend is wider than the figure; give it more rows')
            tops = [max(ax.get_window_extent(renderer).y1, ax._left_title.get_window_extent(renderer).y1)
                    for ax in members if not colorbar(ax)]
            if tops and box.y0 < max(tops) - 1:
                found.append('the legend is not above the panels')
    return found


def artist_colors(artist):
    """The colors an artist draws: (color, what) pairs, skipping fully transparent and colormapped ones."""
    out = []

    def add(color, what):
        if color is None or (isinstance(color, str) and color.lower() == 'none'):
            return
        rgba = to_rgba(color)
        if rgba[3] > 0 and (artist.get_alpha() is None or artist.get_alpha() > 0):
            out.append((to_hex(rgba), what))

    if isinstance(artist, Line2D):
        if artist.get_linestyle() not in ('None', '', ' ') and artist.get_linewidth() > 0:
            add(artist.get_color(), 'line')
        if artist.get_marker() not in (None, 'None', '', ' '):
            add(artist.get_markerfacecolor(), 'marker')
            add(artist.get_markeredgecolor(), 'marker edge')
    elif isinstance(artist, Patch):
        if artist.get_fill():
            add(artist.get_facecolor(), 'fill')
        if artist.get_linewidth() > 0 or artist.get_hatch():
            add(artist.get_edgecolor(), 'edge')
    elif isinstance(artist, Text):
        add(artist.get_color(), 'text')
        if artist.get_bbox_patch() is not None:
            out += artist_colors(artist.get_bbox_patch())
    elif hasattr(artist, 'get_facecolors') and artist.get_array() is None:
        for color in artist.get_facecolors():
            add(color, 'fill')
        if len(artist.get_linewidths()) and max(artist.get_linewidths()) > 0:
            for color in artist.get_edgecolors():
                add(color, 'edge')
    return out


def colors(fig, axes):
    allowed, found = palette(), set()
    artists = [fig.patch] + list(fig.texts) + all_texts(fig, axes)
    for legend in fig.legends:
        artists += list(legend.legend_handles)
    for ax in axes:
        if ax.axison:
            artists += [ax.patch] + [s for s in ax.spines.values() if s.get_visible()]
            for axis in (ax.xaxis, ax.yaxis):
                artists += [line for line in axis.get_gridlines() if line.get_visible()]
        artists += list(ax.lines) + list(ax.patches) + list(ax.collections) + list(ax.artists)
        artists += [t.arrow_patch for t in ax.texts if getattr(t, 'arrow_patch', None) is not None]
        for image in ax.images:
            if image.get_array().ndim == 2:
                ends = [to_hex(image.get_cmap()(v)) for v in (0., 1.)]
                if ends != [to_hex(S.DIVERGING[0]), to_hex(S.DIVERGING[-1])]:
                    found.add(f'heatmap in "{name(ax)}" does not use the change-from-zero-delay scale')
    used = set()
    for artist in artists:
        if not artist.get_visible():
            continue
        for color, what in artist_colors(artist):
            used.add(color)
            if color not in allowed:
                found.add(f'{type(artist).__name__} {what} color {color} is not in plots/style.py')
    # A figure encodes execution modes or policies by color, never both.
    modes = {to_hex(c) for table in (S.MODE_COLORS, S.MODE_DARK) for c in table.values()}
    modes |= {to_hex(c) for shades in S.MODE_SHADES.values() for c in shades}
    if used & modes and used & {to_hex(c) for c in S.POLICY_COLORS.values()}:
        found.add('mode and policy colors share a figure; a figure encodes one of them')
    return sorted(found)


def dashes(line):
    offset, pattern = line._unscaled_dash_pattern
    return None if pattern is None else tuple(round(float(v), 3) for v in pattern)


def series(axes):
    """Data lines: a mode shade carries the marker of its delay on a solid line (dashed for a model prediction), a
    policy color its marker, the zero-delay reference is a thick solid grey line, and reference marks and dividers
    use their shared styles."""
    found = set()
    shades = {to_hex(c): (m, rank) for m, colors in S.MODE_SHADES.items() for rank, c in enumerate(colors)}
    policies = {to_hex(c): p for p, c in S.POLICY_COLORS.items()}
    dashed = tuple(round(float(v), 3) for v in matplotlib.rcParams['lines.dashed_pattern'])
    reference, divider = (to_hex(S.INK2), (2.0, 2.0), .6), (to_hex(S.MUTED), (3.0, 2.0), .6)
    for ax in axes:
        if not framed(ax):
            continue
        caps = {cap for container in ax.containers if isinstance(container, ErrorbarContainer)
                for cap in container.lines[1]}
        for line in ax.lines:
            if line in caps:
                continue  # Interval caps are checked below, not as categorical data markers.
            color, marker = to_hex(line.get_color()), line.get_marker()
            marked = marker not in (None, 'None', '', ' ')
            drawn = line.get_linestyle() not in ('None', '', ' ')
            width, pattern = round(line.get_linewidth(), 3), dashes(line)
            where = f'in "{name(ax)}"'
            if color in shades:
                mode, rank = shades[color]
                expected = S.DELAY_MARKS[rank]
                # A panel showing one delay draws each mode in its own color with that delay's marker.
                single = getattr(ax, 'style_delay', None)
                if single is not None and color == to_hex(S.MODE_COLORS[mode]):
                    expected = S.DELAY_MARKERS[single]
                if marked and marker != expected:
                    found.add(f'{mode} series {where} uses marker {marker!r}, not {expected!r} for its shade')
                if drawn and (pattern not in (None, dashed) or width != LINE_WIDTH):
                    found.add(f'{mode} line {where} is not a solid or dashed {LINE_WIDTH} pt delay line')
                if drawn and marked and pattern is not None:
                    found.add(f'{mode} line {where} is dashed with markers; dashed lines are model predictions')
            elif color in policies:
                policy = policies[color]
                if marked and marker != S.POLICY_MARKERS[policy]:
                    found.add(f'{policy} line {where} uses marker {marker!r}, not {S.POLICY_MARKERS[policy]!r}')
                if drawn and (pattern is not None or width != LINE_WIDTH):
                    found.add(f'{policy} line {where} is not a solid {LINE_WIDTH} pt line')
            elif color == to_hex(S.ZERO_DELAY):
                if marked or pattern is not None or width != S.ZERO_DELAY_WIDTH:
                    found.add(f'zero-delay line {where} is not the thick solid grey reference')
            elif color == to_hex(S.INK) and drawn:
                if pattern not in (None, dashed) or width != LINE_WIDTH or (pattern and marked):
                    found.add(f'ink line {where} is not a solid data line or dashed unmarked prediction')
            elif color in (reference[0], divider[0]) and drawn:
                if (color, pattern, width) not in (reference, divider):
                    found.add(f'line {where} is neither a reference mark nor a mode divider')
    return sorted(found)


def uncertainty(axes):
    """Intervals stay local to measured points and use one visible, capped style."""
    found = []
    for ax in axes:
        if any(isinstance(c, PolyCollection) for c in ax.collections):
            found.append(f'filled uncertainty ribbon in "{name(ax)}"; use interval_bars')
        for container in ax.containers:
            if not isinstance(container, ErrorbarContainer):
                continue
            _, caps, bars = container.lines
            if not caps or any(abs(cap.get_markersize() - 2 * S.INTERVAL_CAP) > 1e-6 or
                               abs(cap.get_markeredgewidth() - S.INTERVAL_WIDTH) > 1e-6 for cap in caps):
                found.append(f'interval caps in "{name(ax)}" differ from the shared style')
            if any(abs(w - S.INTERVAL_WIDTH) > 1e-6 for bar in bars for w in bar.get_linewidths()):
                found.append(f'interval width in "{name(ax)}" differs from the shared style')
    return found


def frames(axes):
    found = []
    for ax in axes:
        if not ax.axison or colorbar(ax):
            continue
        shown = [s.get_visible() for s in ax.spines.values()]
        grid = any(line.get_visible() for axis in (ax.xaxis, ax.yaxis) for line in axis.get_gridlines())
        if all(shown):
            if ax.images:
                found.append(f'heatmap "{name(ax)}" has a frame')
            if not grid:
                found.append(f'panel "{name(ax)}" has no grid')
            for spine in ax.spines.values():
                if to_hex(spine.get_edgecolor()) != to_hex(S.FRAME) or spine.get_linewidth() != RC['axes.linewidth']:
                    found.append(f'panel "{name(ax)}" frame differs from the thin shared frame')
                    break
        elif any(shown):
            found.append(f'panel "{name(ax)}" has a partial frame')
        elif grid:
            found.append(f'frameless panel "{name(ax)}" has a grid')
    return found


def ticks(fig, axes):
    found = []
    for ax in axes:
        for label in drawn_tick_labels(ax):
            if TRAILING_ZERO.match(label.get_text()):
                found.append(f'tick label "{label.get_text()}" in "{name(ax)}" has a trailing zero')
        if ax.images or not ax.axison or colorbar(ax):
            continue
        if getattr(ax, 'style_percent', False) or ax.get_ylabel().endswith('(%)'):
            if not getattr(ax, 'style_percent', False):
                found.append(f'percent axis in "{name(ax)}" is not drawn with percent_axis')
            elif ax.get_ylim() != (0, PERCENT_TOP) or tuple(ax.get_yticks()) != PERCENT_TICKS:
                found.append(f'percent axis in "{name(ax)}" is not 0-100 in steps of 25')
        if getattr(ax, 'style_speed', False) == 'measured':
            # measured_speed_axis: ticks sit at the measured speeds, which need not form an equal-step grid.
            if not ax.get_xticks().size:
                found.append(f'measured speed axis in "{name(ax)}" has no ticks')
        elif getattr(ax, 'style_speed', False) or ax.get_xlabel() == SPEED_LABEL:
            locations = [t.get_loc() for t in ax.xaxis._update_ticks() if t in ax.xaxis.majorTicks]
            steps = {round(b - a, 6) for a, b in zip(locations, locations[1:])}
            if (not locations or abs(locations[0]) > 1e-9 or len(steps) != 1 or steps.pop() not in SPEED_STEPS
                    or not 3 <= len(locations) <= 6):
                found.append(f'speed axis in "{name(ax)}" does not tick from 0 in equal steps of 0.05, 0.1, or 0.2')
    supx = fig._supxlabel.get_text() if fig._supxlabel is not None else ''
    if supx == SPEED_LABEL:
        bottom = min(ax.get_position().y0 for ax in axes if framed(ax))
        for ax in axes:
            if framed(ax) and abs(ax.get_position().y0 - bottom) < .01 and not getattr(ax, 'style_speed', False):
                found.append(f'panel "{name(ax)}" sits above the shared speed label without speed_axis')
    return found
