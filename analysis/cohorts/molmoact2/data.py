"""MolmoAct2-LIBERO on the four LIBERO tasks: synchronous 0 ms and every mode from 50 to 250 ms.

Twenty-eight studies (seven per task) in benchmark storage, 9 speeds x 200 episodes per setting, with the protocol of
the pi0.5 latency grid (H100 batched serving, K = 10, H = 5, 600 action steps). The derived episode table ships
in analysis/data/.
"""
from analysis.paths import DERIVED, STUDIES as ARTIFACT_STUDIES, data_rows
from analysis.cohorts.latency_grid.data import SPEEDS, TARGET, TASKS  # noqa: F401

MODES = ('synchronous', 'lazy', 'continuous')
LATENCIES = (50, 100, 150, 200, 250)
ROOT_STUDIES = ARTIFACT_STUDIES / 'molmoact2_libero_20261015'
STUDY_NAME = {'basket': 'basket', 'plate': 'plate', 'cabinet': 'bowl', 'drawer': 'drawer'}
# Each task's settings are split over seven studies: synchronous 0 ms, then 100/200 ms and 50/150/250 ms ('_ext')
# for each mode.
STUDIES = (('synchronous0', 'synchronous', (0,)),
           *((f'{mode}', mode, (100, 200)) for mode in MODES),
           *((f'{mode}_ext', mode, (50, 150, 250)) for mode in MODES))
CHECKPOINT = 'allenai/MolmoAct2-LIBERO'
FOLDER = DERIVED / 'molmoact2_episodes'


# The bowl's studies ran 0.001 and 0.05-0.4 m/s; the October 21 speed fill supplies 0.025, 0.075, 0.125, and 0.175 m/s
# at every setting with otherwise identical configurations, giving every task the 0.001-0.2 m/s grid (SPEEDS).
BOWL_FILL = ARTIFACT_STUDIES / 'bowl_speed_fill_20261021'


def study(task, part):
    return ROOT_STUDIES / f'molmoact2_{STUDY_NAME[task]}_{part}_b32_20261015'


def studies(task, mode):
    """(folder, latencies) of every study holding the task's settings in one mode."""
    found = [(study(task, part), latencies) for part, m, latencies in STUDIES if m == mode]
    if task == 'cabinet':
        found.append((BOWL_FILL / f'molmoact2_bowl_speed_fill_{mode}_b32_20261021',
                      ((0,) if mode == 'synchronous' else ()) + LATENCIES))
    return found


def episode_rows():
    """Per-episode success table; every setting keeps all 200 episodes."""
    return data_rows('molmoact2_episodes/episodes')
