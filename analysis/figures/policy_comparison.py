"""One clean policy comparison: success against speed for pi0.5, GR00T N1.7, and MolmoAct2, three conditions, four tasks."""
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.paths import figure_path
POLICIES = ("pi0.5", "GR00T N1.7", "MolmoAct2")
SP = [0.001, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
PANELS = [("Zero latency", [("synchronous", 0), ("lazy", 0)]), ("Synchronous 200 ms", [("synchronous", 200)]), ("Lazy 200 ms", [("lazy", 200)])]
from analysis.figures.figure_kit import (SPEED_LABEL, SUCCESS_LABEL, interval_bars, note, panel_title, percent_axis, policy_handles,
                              policy_style, save, shared_xlabel, speed_axis, style, top_legend)
from analysis.figures.style import POLICY_COLORS
from analysis.figures.tasks import LIBERO, TASKS
from analysis.figures.typography import TEXT_WIDTH
style(plt)

from analysis.figures.numerics import wilson_bounds as wilson

# Per-episode rows by policy and task; synchronous points at a nonzero latency take the hold_pose reruns.
from analysis.sync_hold import replace_rows
ROWS = collections.defaultdict(list)
from analysis.cohorts.libero import episode_rows as libero_rows
for m in ("GR00T N1.7", "pi0.5"):
    for r in libero_rows(m):
        if float(r["speed"]) in SP:
            ROWS[(m, r["task"])].append(r)
from analysis.cohorts.plate_cabinet.data import SPEEDS as PLATE_CABINET_SPEEDS, episode_rows as plate_cabinet_rows
for r in plate_cabinet_rows():
    ROWS[(r["model"], r["task"])].append(r)
from analysis.cohorts.molmoact2.data import episode_rows as molmoact2_rows
for r in molmoact2_rows():
    ROWS[("MolmoAct2", r["task"])].append(r)
S = {m: collections.defaultdict(lambda: [0, 0]) for m in POLICIES}
for (m, task), rows in ROWS.items():
    for r in replace_rows(m, task, rows):
        k = (r["task"], float(r["speed"]), r["mode"], int(r["latency"])); S[m][k][0] += int(r["success"]); S[m][k][1] += 1
GRID_SPEEDS = {"basket": SP, "drawer": SP, **PLATE_CABINET_SPEEDS}
GAP = 11   # minimum vertical distance between two value labels, in percentage points

fig, axes = plt.subplots(4, 3, figsize=(TEXT_WIDTH, 4.7), sharey=True, layout="constrained")
for i, task in enumerate(LIBERO):
    speeds = GRID_SPEEDS[task]
    x = np.array(speeds)
    high = speeds[-1]
    pad = .045 * high   # gap between a curve end and its value label
    for j, (name, conds) in enumerate(PANELS):
        ax = axes[i, j]
        ends = {}
        for m in POLICIES:
            ks = [sum(S[m][(task, s, mo, l)][0] for mo, l in conds) for s in speeds]
            ns = [sum(S[m][(task, s, mo, l)][1] for mo, l in conds) for s in speeds]
            y = np.array([100 * k / n for k, n in zip(ks, ns)])
            lo, hi = zip(*[wilson(k, n) for k, n in zip(ks, ns)])
            interval_bars(ax, x, y, 100 * np.array(lo), 100 * np.array(hi), color=POLICY_COLORS[m])
            ax.plot(x, y, **policy_style(m))
            ends[m] = y[0], y[-1]
        # Start and end labels: push apart upward where curves meet, then back down below the top.
        for side, position, ha in ((0, x[0] - pad, "right"), (1, x[-1] + pad, "left")):
            order = sorted(ends, key=lambda m: ends[m][side])
            ys = [max(ends[order[0]][side], 3)]
            for m in order[1:]:
                ys.append(max(ends[m][side], ys[-1] + GAP))
            excess = max(0, ys[-1] - 97)
            for m, yy in zip(order, ys):
                note(ax, position, yy - excess, f"{ends[m][side]:.0f}", color=POLICY_COLORS[m], va="center", ha=ha)
        panel_title(ax, 3 * i + j, name)
        # The wider margin leaves room for the start and end values.
        speed_axis(ax, speeds, narrow=True, margin=.2)
        percent_axis(ax)
        if j == 0: ax.set_ylabel(f"{TASKS[task].name}\n{SUCCESS_LABEL}")
shared_xlabel(fig, SPEED_LABEL)
top_legend(fig, policy_handles(POLICIES))
save(fig, figure_path("policy")); plt.close(fig)
print("written policy.png")
