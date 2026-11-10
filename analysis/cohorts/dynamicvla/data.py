"""DynamicVLA moving-box and moving-can cohorts (DOM checkpoint, Isaac Sim), for the cross-simulator check.

Moving box: a food can on the table is placed into a box that moves on a closed path (release into a moving
receptacle). Moving can: a food can carried by a ribbed carrier on a closed path is grasped and placed into a
static box (moving-object grasping). Each setting pools 200 conditions from the runs listed in its published
results folder (sources.json); the run records and traces are read in place from benchmark artifact storage, and
the derived episode and attempt tables ship in analysis/data/.
"""
from analysis.paths import DERIVED, RESULTS as PUBLISHED, STUDIES, data_rows

MODEL = 'DynamicVLA'
TASKS = ('box', 'can')
RESULTS = {'box': PUBLISHED / 'dynamicvla/fcan04_native200_20261001',
           'can': PUBLISHED / 'dynamicvla/fcan04_moving_can200_20261008'}
# Collected studies that hold each cohort's runs (the October 1 plot snapshot only copies records).
STUDY_ROOTS = {'box': ('fcan04_native_pilot20_20260928', 'fcan04_native_remaining180_20260928',
                       'fcan04_native200_speed_extension_20261001'),
               'can': ('fcan04_moving_can_pilot20_20261008_xtra3090', 'fcan04_moving_can_remaining180_20261008')}
# Synchronous waits re-sent the last absolute pose target, so the arm kept moving toward it while it waited. Both tasks'
# synchronous 120 and 240 ms settings were rerun with the hand held where it was at the first wait step
# (inference.idle_control hold_pose) on the same 200 conditions, and the reruns replace those settings. The moving can's
# study tables list its runs, which keep the original order. The moving box's study ran on the speed-extension base;
# the benchmark lists its runs beside the cohort (sources.yaml, with plot_data.csv to check them). At the seven box
# speeds whose other settings pool the 20 + 180 runs, 186 of each rerun setting's 200 episodes run their condition at
# another replica index, so their instruction wording and policy noise seed differ from the other settings' (README).
HOLD_POSE_STUDIES = {'box': 'dvl_box_hold_pose_20261029', 'can': 'dvl_hold_pose_20261027'}
HOLD_POSE_RESULTS = {'box': RESULTS['box'] / 'dynamicvla-sync-hold-pose'}
SPEEDS = {'box': (.001, .025, .05, .1, .15, .2, .25, .3, .35, .4),
          'can': (.001, .025, .05, .075, .1, .125, .15, .175, .2)}
CONDITIONS = (('synchronous', 0), ('synchronous', 120), ('synchronous', 240), ('lazy', 120), ('lazy', 240),
              ('continuous', 120), ('continuous', 240))
TARGET = {'box': 'container', 'can': 'carrier'}
CHECKPOINT_SHA256 = 'e3da8b1e0326d7a3ee6873f7fa17cdf7015201251901be5107377cb851ce3af8'
CHUNK, HORIZON, HZ, BUDGET = 20, 14, 25, 750
# Box (box14) size from the recorded success predicate; its origin is the box center.
BOX_SIZE_M = (0.23622047901153564, 0.16732284426689148, 0.1012696847319603)
# Can (fcan04): a cylinder lying on its side about its local z axis, origin at its center. Collider radius from the
# moving-can build; half length from the native mesh footprint. Half extents along the can's local axes.
CAN_HALF_M = (0.034, 0.034, 0.05)
# Franka fingers from the robot asset the environment records (IsaacLab 4.5 panda_instanceable.usd, SHA-256
# 7f5a0c0a…, collision meshes Props/instanceable_collision_meshes.usd, e2d053fd…). The recorded end effector is
# panda_hand offset 0.1034 m along its z axis with the hand's orientation. Each finger body sits 0.0584 m along the
# hand z axis and slides along the hand y axis by its joint value (left +y, right -y); its collision-mesh bounding
# box, in finger coordinates, is (low, high) per axis.
EE_OFFSET_M, FINGER_BASE_M = 0.1034, 0.0584
FINGER_BOX_M = {'left': ((-0.010479, -0.000133, 0.000132), (0.010495, 0.026403, 0.053849)),
                'right': ((-0.010495, -0.026403, 0.000132), (0.010479, 0.000133, 0.053849))}
EPISODE_FOLDER = DERIVED / 'dynamicvla_episodes'
REPLAY_FOLDER = DERIVED / 'dynamicvla_replay'

DEFINITION = {
    'version': 3,
    'frame': 'Robot-base frame: recorded end-effector (grip point) position and orientation; can, box, and carrier '
             'root poses transformed by the recorded robot root pose. Samples are the recorded states at 25 Hz; warm-up '
             '(the leading policy_warmup records) is excluded.',
    'closed': 'Gripper commanded closed: the action executed in the previous step has a negative gripper value.',
    'held': 'Gripper commanded closed, finger opening (sum of both finger joints) at least 2 cm, so the fingers are '
            'blocked by an object, and the can origin within 6 cm of the grip point.',
    'lifted': 'Can origin at least 2 cm above its height at the first post-warm-up sample.',
    'pickup': 'At least three consecutive held samples with the can lifted at least once within that run; the '
              'pickup step is the start of the first such run.',
    'release': 'At or after pickup, the gripper command switches from closed to open while the can is held and '
               'lifted in the previous sample.',
    'outcome_box': 'Success first; then released_failed (a release after pickup), picked_up_never_released, or '
                   'never_picked_up.',
    'outcome_can': 'Success first; then picked_up_failed (pickup confirmed) or never_picked_up.',
    'alignment_box': 'Both endpoints: can held, can origin within 15 cm horizontally of the box origin, grip point '
                     'at or above the box top (box origin plus half the box height); waits included.',
    'alignment_can': 'Both endpoints: grip point within 10 cm (3D) of the can origin with the gripper commanded '
                     'open; waits included.',
    'speed': 'Sum of selected 3D grip-point displacements / selected elapsed time (0.04 s per interval).',
    'attempt_box': 'Each release. Offset: horizontal can-origin to box-origin distance in the last held sample. '
                   'Hit: the episode succeeds with no later release and no finger-can contact after the first run '
                   'of >= 2 samples without contact (contact: a finger collision-box grid point at or inside the '
                   'can cylinder).',
    'attempt_can': 'Each first sample with the lowest finger point below the highest can point (collision bounding '
                   'boxes in the recorded pose: finger boxes from the hand pose and finger joints, the can as a '
                   'lying cylinder) after >= 3 samples above it, with the grip point within 15 cm horizontally of '
                   'the can, no later than the pickup step and before the can falls 5 cm below its initial height. '
                   'Offset: horizontal grip-point to can-origin distance. Hit: the pickup run starts at or after the '
                   'attempt, before the next attempt, and within 50 samples (2 s).',
    'age': 'Control steps between the observation that produced an executed policy action and its execution step.',
    'continued': 'Lazy and continuous only. For each chunk, the steps between its observation and its release '
                 'that execute a policy action from an older chunk; the fresh prediction for step s is action '
                 's - observation of the new chunk. Relative difference: norm(older - fresh) / '
                 '(0.5 * (norm(older - grip) + norm(fresh - grip)) + 1e-6) on the commanded positions, with grip the '
                 'grip point before the step; per-episode median.',
}


def episode_rows(task=None):
    """Per-episode success table; every setting keeps all 200 conditions."""
    return [r for r in data_rows('dynamicvla_episodes/episodes') if task is None or r['task'] == task]


def replay_rows(name, task=None):
    """Trace measurements of every episode: attempts (analysis/data/dynamicvla_replay/attempts.csv.gz)."""
    return [r for r in data_rows(f'dynamicvla_replay/{name}') if task is None or r['task'] == task]
