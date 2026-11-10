"""Moving-plate and moving-cabinet cohorts, for pi0.5 and GR00T N1.7.

Both tasks use the LIBERO-Spatial instruction "pick up the black bowl on the wooden cabinet and
place it on the plate". The moving plate moves the plate; the moving cabinet moves the cabinet,
which carries the bowl until it is grasped.
Studies are read in place from benchmark artifact storage; the derived tables ship in analysis/data/.
"""
from analysis.paths import DERIVED, STUDIES, data_rows

TASKS = ('plate', 'cabinet')
MODELS = ('pi0.5', 'GR00T N1.7')
SPEEDS = {'plate': (.001, .025, .05, .075, .1, .125, .15, .175, .2),
          'cabinet': (.001, .025, .05, .075, .1, .125, .15, .175, .2)}
CONDITIONS = (('synchronous', 0), ('synchronous', 100), ('synchronous', 200),
              ('lazy', 100), ('lazy', 200), ('continuous', 100), ('continuous', 200))
# Moving plate. pi0.5: the H100 rerun with batched serving (same settings as the single-request run, which it
# reproduces). GR00T N1.7: the October 1 study with dynamic batching on RTX 3090s.
# Moving bowl: the October 10 H100 rerun of both policies with the bowl pinned at the center of the cabinet top
# in all 50 layouts, otherwise the same settings as the plate studies. It ran 0.001 and 0.05-0.4 m/s; the October 21
# speed fill ran 0.025, 0.075, 0.125, and 0.175 m/s at every delay with otherwise identical configurations, so the
# bowl uses the other tasks' 0.001-0.2 m/s grid: 0.001, 0.05, 0.1, 0.15, 0.2 from the first, the rest from the fill.
BOWL_CENTERED = STUDIES / 'libero_spatial_task9_bowl_centered_20261010'
BOWL_FILL = STUDIES / 'bowl_speed_fill_20261021'
STUDY = {
    'pi0.5': lambda task, mode: ((BOWL_CENTERED / f'pi05_task9_cabinet_bowl_centered_{mode}_b32_20261010',
                                  BOWL_FILL / f'pi05_bowl_speed_fill_{mode}_b32_20261021') if task == 'cabinet'
                                 else (STUDIES / 'libero_spatial_task9_pi05_b32_h100_20261003' / f'pi05_task9_{task}_{mode}_b32_20261003',)),
    'GR00T N1.7': lambda task, mode: ((BOWL_CENTERED / f'gr00t_task9_cabinet_bowl_centered_{mode}_20261010',
                                       BOWL_FILL / f'gr00t_bowl_speed_fill_{mode}_b32_20261021') if task == 'cabinet'
                                      else (STUDIES / 'libero_spatial_task9_20261001' / f'gr00t_task9_{task}_{mode}_20261001',)),
}
CHECKPOINT = {'pi0.5': 'gs://openpi-assets/checkpoints/pi05_libero', 'GR00T N1.7': 'nvidia/GR00T-N1.7-LIBERO'}
CHUNK = {'pi0.5': 10, 'GR00T N1.7': 16}
TARGET_BODY = {'plate': 'plate_1_main', 'cabinet': 'wooden_cabinet_1_main'}
EPISODE_FOLDER = DERIVED / 'plate_cabinet_episodes'
REPLAY_FOLDER = DERIVED / 'plate_cabinet_replay'

OUTCOMES = ('success', 'placed_failed', 'picked_up_never_placed', 'never_picked_up')
DEFINITION = {
    'version': 1,
    'pickup': 'Three consecutive samples with both fingers in nonpositive-distance contact with the bowl and the bowl '
              'at least 2 cm above its post-warm-up height within that run',
    'placement': 'Nonpositive-distance contact between bowl and plate geoms',
    'outcome': 'Success first; then plate contact after pickup (placed_failed), pickup without plate contact, or no pickup',
    'alignment_plate': 'Both endpoints: bowl held by both fingers, bowl origin within 15 cm horizontally of the plate '
                       'origin, grip site at or above the plate origin; warm-up excluded, waits included',
    'alignment_cabinet': 'Both endpoints: grip site within 10 cm (3D) of the bowl origin and no finger-bowl contact; '
                         'warm-up excluded, waits included',
    'speed': 'Sum of selected 3D grip-site displacements / selected elapsed simulated time',
    'attempt_plate': 'Each onset of bowl-plate contact after warm-up; gaps shorter than 3 samples join one contact. '
                     'Offset: horizontal bowl-origin to plate-origin distance at onset. Hit: success during that contact',
    'attempt_cabinet': 'Each first sample with the lowest finger point below the highest bowl point (collision-geom '
                       'bounding boxes) after >= 3 samples above it, with the grip site within 15 cm horizontally of the '
                       'bowl, before the first confirmed pickup and before the bowl falls 5 cm. Offset: horizontal '
                       'grip-site to bowl-origin distance. Hit: the first pickup run starts before the next attempt and '
                       'within 40 samples (2 s)',
    'age': 'Control steps between the observation that produced an executed action and its execution step',
}


def episode_rows(model=None):
    """Per-episode success table; every condition keeps all 200 episodes."""
    rows = data_rows('plate_cabinet_episodes/episodes')
    return [r for r in rows if model is None or r['model'] == model]


def replay_rows(name, model=None):
    """Replay products: alignment or attempts."""
    return [r for r in data_rows(f'plate_cabinet_replay/{name}') if model is None or r['model'] == model]
