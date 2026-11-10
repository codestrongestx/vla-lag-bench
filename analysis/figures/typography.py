"""Printed-size typography for the paper figures.

Figures are drawn at the width they print in the paper, so these point sizes
print as written beside the 10 pt body text: 5.5 in for a figure spanning both
MLSys columns (centered in the 6.75 in text block) and 3.25 in for a figure in
one column. Captions carry titles, cohort notes, and interval
definitions; figures carry no figure-level header. Backgrounds are white so a
figure does not print as a tinted panel.
"""
TEXT_WIDTH = 5.5
COLUMN_WIDTH = 3.25
# The whole MLSys text block; only a figure that needs more room than 5.5 in, such as the task overview, uses it.
FULL_WIDTH = 6.75
TEXT_HEIGHT = 9.0
DPI = 300
PAPER = '#ffffff'
RC = {
    'figure.facecolor': PAPER,
    'axes.facecolor': PAPER,
    'savefig.facecolor': PAPER,
    'font.family': 'DejaVu Sans',
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.titlepad': 4,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,
    'axes.labelpad': 2.5,
    'lines.linewidth': 1.2,
    'lines.markersize': 3,
    'legend.frameon': False,
    'legend.handlelength': 1.8,
    'legend.columnspacing': 1.2,
    # Embed TrueType fonts so text in the vector PDFs stays selectable; venues reject Type 3 fonts.
    'pdf.fonttype': 42,
}
# The smallest text any figure may use (cell values, bar counts, path labels, ticks, legends): 7 pt. DejaVu Sans at
# 7 pt is about as tall as the 10 pt Times body text, so figure text reads beside the body without printing larger.
MIN_SIZE = 7
# Every size a figure may use: notes, cell values, ticks, and legends; axis labels; panel titles.
SIZES = (MIN_SIZE, RC['axes.labelsize'], RC['axes.titlesize'])
