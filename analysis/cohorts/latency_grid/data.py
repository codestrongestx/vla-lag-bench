"""The pi0.5 latency grid: the four LIBERO tasks at 50, 150, and 250 ms in every execution mode.

Twelve studies (one per task and mode) in benchmark storage, 9 speeds x 3 latencies x 200 episodes each, with the
same protocol as the 0/100/200 ms cohorts (H100 batched serving, K = 10, H = 5, 600 action steps). The derived
episode table ships in analysis/data/.
"""
from analysis.paths import DERIVED, STUDIES, data_rows

TASKS = ('basket', 'plate', 'cabinet', 'drawer')
MODES = ('synchronous', 'lazy', 'continuous')
LATENCIES = (50, 150, 250)
# Every task uses 0.001 and 0.025-0.2 m/s. The bowl's grid study ran 0.001 and 0.05-0.4 m/s; the October 21 speed
# fill supplies its 0.025, 0.075, 0.125, and 0.175 m/s settings with otherwise identical configurations.
SPEEDS = {task: (.001, .025, .05, .075, .1, .125, .15, .175, .2) for task in TASKS}
GRID = STUDIES / 'libero_latency_grid_20261014'
BOWL_FILL = STUDIES / 'bowl_speed_fill_20261021'
STUDY_NAME = {'basket': 'basket', 'plate': 'plate', 'cabinet': 'bowl', 'drawer': 'drawer'}
TARGET = {'basket': 'basket_1_main', 'plate': 'plate_1_main', 'cabinet': 'wooden_cabinet_1_main', 'drawer': 'wooden_cabinet_1_main'}
CHECKPOINT = 'gs://openpi-assets/checkpoints/pi05_libero'
FOLDER = DERIVED / 'latency_grid_episodes'


def study(task, mode):
    return GRID / f'pi05_{STUDY_NAME[task]}_{mode}_b32_20261014'


def studies(task, mode):
    """The studies that hold a task's settings in one mode: the grid study, and for the bowl the speed fill."""
    extra = (BOWL_FILL / f'pi05_bowl_speed_fill_{mode}_b32_20261021',) if task == 'cabinet' else ()
    return (study(task, mode), *extra)


def episode_rows():
    """Per-episode success table of the grid; every setting keeps all 200 episodes."""
    return data_rows('latency_grid_episodes/episodes')
