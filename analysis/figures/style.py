"""Figure style for the paper.

Every color, marker, and hatch a paper figure may draw is named here. style_check.py rejects any other color at save
time, and ENCODINGS keeps each color and marker to a single meaning.
"""
# Neutrals: text, frame, grid, and reference marks.
INK, INK2, MUTED, GRID, FRAME = '#0b0b0b', '#52514e', '#8a8985', '#e6e5e1', '#6b6a66'

# Execution mode: color, the same in every figure, named by its label (legend entry, row label, or bar label).
# No other encoding uses these colors.
MODE_COLORS = {'synchronous': '#D55E00', 'lazy': '#0072B2', 'continuous': '#009E73'}
MODE_NAMES = {'synchronous': 'Synchronous', 'lazy': 'Lazy', 'continuous': 'Continuous'}
MODE_SHORT = {'synchronous': 'Sync.', 'lazy': 'Lazy', 'continuous': 'Cont.'}
# The execution-mode timeline alternates each mode color with this darker shade so neighbouring chunks stay apart.
MODE_DARK = {'synchronous': '#843a00', 'lazy': '#00476e', 'continuous': '#006247'}

# Delay: marker plus shade of the mode color, the same in every mode, so one delay keeps one shape and one place on
# the light-to-dark ramp whatever its mode. Delays rank from short to long: 50 ms, 100 or 120 ms, 150 ms, 200 or
# 240 ms, 250 ms; the middle rank is the mode color itself. Data lines are solid; a dashed line in a mode shade is a
# model prediction. The zero-delay reference is a thick light-grey line (or bar) without markers.
ZERO_DELAY = '#b4b3af'
ZERO_DELAY_WIDTH = 2.4
# The real robot's 267, 367, and 467 ms (8, 11, and 14 steps at 30 Hz) exceed every simulated delay and take the three
# longest ranks.
DELAY_RANK = {50: 0, 100: 1, 120: 1, 150: 2, 200: 3, 240: 3, 250: 4, 267: 2, 367: 3, 467: 4}
DELAY_MARKS = {0: 'v', 1: 'o', 2: 's', 3: '^', 4: 'X'}
DELAY_MARKERS = {d: DELAY_MARKS[r] for d, r in DELAY_RANK.items()}
# The mode color mixed with white (shorter delays) or black (longer delays): +50%, +25%, 0, -25%, -45%.
MODE_SHADES = {'synchronous': ('#eaae80', '#e08640', '#d55e00', '#a04600', '#753400'),
               'lazy': ('#80b8d8', '#4095c5', '#0072b2', '#005686', '#003f62'),
               'continuous': ('#80ceb9', '#40b696', '#009e73', '#007656', '#00573f')}
# Legend keys for the delays: the same ramp in grey.
DELAY_GREYS = ('#c4c3c0', '#9a9995', '#767571', '#4f4e4a', '#2e2d2a')
PREDICTION = '--'

# Policies: color and marker, never a mode color or delay marker.
POLICY_COLORS = {'pi0.5': '#AA4499', 'GR00T N1.7': '#DDAA33', 'MolmoAct2': '#117733', 'DynamicVLA': '#882255'}
POLICY_MARKERS = {'pi0.5': 'D', 'GR00T N1.7': 'p', 'MolmoAct2': '*', 'DynamicVLA': 'h'}
POLICY_NAMES = {'pi0.5': r'$\pi_{0.5}$', 'GR00T N1.7': 'GR00T N1.7', 'MolmoAct2': 'MolmoAct2', 'DynamicVLA': 'DynamicVLA'}
# Outcomes: success, then failures by the last stage reached, each with its own hatch.
OUTCOME_COLORS = {'success': '#4d4d4d', 'late': '#CC6677', 'early': '#DDCC77', 'never': '#d9d8d4'}
OUTCOME_HATCHES = {'success': '', 'late': 'xx', 'early': '//', 'never': '..'}
# The moving target: magenta for its path and travel (paths on frames and path panels), and its own color for the
# box that marks the moving object on frames, so a box never reads as a path.
TARGET = '#E7298A'
TARGET_BOX = '#17BECF'
# Background fills: stages at the moving target, and idle steps in the timeline.
TARGET_STAGE_FILL = '#f1f0ed'
IDLE_FILL = '#e9e9e6'
# Change from zero delay, in points: the only continuous scale, diverging (loss blue, gain red) around zero over
# a fixed range, with the value printed in each cell.
DIVERGING = ['#104281', '#2a78d6', '#86b6ef', '#f0efec', '#f0a3a2', '#e34948', '#8f1d1c']
CHANGE_LIMIT = 60
# Uncertainty: capped 95% Wilson intervals at the measured x values, in the series color.
INTERVAL_WIDTH = .7
INTERVAL_CAP = 1.8

# The categorical encodings: each color, marker, and hatch belongs to exactly one meaning.
ENCODINGS = {
    'mode': {'colors': MODE_COLORS, 'marks': {}},
    'delay': {'colors': {f'{mode} {r}': shades[r] for mode, shades in MODE_SHADES.items() for r in DELAY_MARKS if r != 2}
              | {f'key {r}': grey for r, grey in enumerate(DELAY_GREYS)},
              'marks': {f'{r}': mark for r, mark in DELAY_MARKS.items()}},
    'policy': {'colors': POLICY_COLORS, 'marks': POLICY_MARKERS},
    'outcome': {'colors': OUTCOME_COLORS, 'marks': OUTCOME_HATCHES},
    'target': {'colors': {'path': TARGET, 'object box': TARGET_BOX}, 'marks': {}},
}

