"""One catalog of evaluated tasks, so every figure type draws LIBERO and DynamicVLA tasks with the same code.

The four LIBERO tasks form the benchmark; the DynamicVLA moving box and moving can (Isaac Sim) are the
cross-simulator check. Figures and text name each task by its interaction type: release (moving basket; DynamicVLA
moving box), set-down (moving plate), grasping (moving bowl; DynamicVLA moving can), and articulated-object
manipulation (moving drawer). Each task names its speed grid, delays, interaction type, success source,
tolerance-zone attempts, and alignment-speed measurement. Loaders read a table only when a renderer asks for it.
"""
import collections
from dataclasses import dataclass

from analysis.figures.style import POLICY_NAMES

LIBERO_SPEEDS = (.001, .025, .05, .075, .1, .125, .15, .175, .2)
BOX_SPEEDS = (.001, .025, .05, .1, .15, .2, .25, .3, .35, .4)


@dataclass(frozen=True)
class Task:
    key: str
    name: str
    simulator: str
    interaction: str
    speeds: tuple
    delays: tuple
    lazy_zero: bool = False      # separate lazy 0 ms runs exist (moving basket and moving drawer)

    @property
    def conditions(self):
        """Execution settings in display order; synchronous 0 ms first."""
        short, long = self.delays
        lazy = ((('lazy', 0),) if self.lazy_zero else ()) + (('lazy', short), ('lazy', long))
        return (('synchronous', 0), ('synchronous', short), ('synchronous', long), *lazy,
                ('continuous', short), ('continuous', long))


TASKS = {
    'basket': Task('basket', 'Release', 'LIBERO', 'release', LIBERO_SPEEDS, (100, 200), True),
    'plate': Task('plate', 'Set-down', 'LIBERO', 'set-down', LIBERO_SPEEDS, (100, 200)),
    'cabinet': Task('cabinet', 'Grasping', 'LIBERO', 'grasping', LIBERO_SPEEDS, (100, 200)),
    'drawer': Task('drawer', 'Articulated', 'LIBERO', 'articulated', LIBERO_SPEEDS, (100, 200), True),
    'box': Task('box', 'Release', 'DynamicVLA', 'release', BOX_SPEEDS, (120, 240)),
    'can': Task('can', 'Grasping', 'DynamicVLA', 'grasping', LIBERO_SPEEDS, (120, 240)),
}
# pi0.5 also runs the four LIBERO tasks at 50, 150, and 250 ms in every mode (cohorts/latency_grid/), and MolmoAct2
# runs the same delays (cohorts/molmoact2/). MolmoAct2 has success tables only, no trace measurements.
GRID_DELAYS = (50, 150, 250)
GRID_POLICIES = ('pi0.5', 'MolmoAct2')
# Every figure and paper passage presents tasks in this order: release, set-down, grasping,
# articulated; the DynamicVLA moving box before the moving can.
LIBERO = ('basket', 'plate', 'cabinet', 'drawer')
DYNAMICVLA = ('box', 'can')
POLICY_TAG = {'pi0.5': 'pi05', 'GR00T N1.7': 'gr00t', 'MolmoAct2': 'molmoact2', 'DynamicVLA': 'dynamicvla'}
POLICY_LABEL = POLICY_NAMES



def delays(policy, task, *, grid=False):
    """The task's delays for a policy; with grid, pi0.5's and MolmoAct2's LIBERO tasks add 50/150/250 ms."""
    spec = TASKS[task]
    if grid and policy in GRID_POLICIES and task in LIBERO:
        return tuple(sorted(spec.delays + GRID_DELAYS))
    return spec.delays


def lazy_zero(policy, task):
    """Whether separate lazy 0 ms runs exist; MolmoAct2 runs synchronous 0 ms only."""
    return TASKS[task].lazy_zero and policy != 'MolmoAct2'


def conditions(policy, task, *, grid=False):
    """Execution settings in display order, synchronous 0 ms first; with grid, every grid delay as well."""
    spec = TASKS[task]
    if not grid:
        return tuple(c for c in spec.conditions if c != ('lazy', 0) or lazy_zero(policy, task))
    shown = delays(policy, task, grid=True)
    lazy_zero_runs = (('lazy', 0),) if lazy_zero(policy, task) else ()
    return (('synchronous', 0), *(('synchronous', d) for d in shown), *lazy_zero_runs,
            *(('lazy', d) for d in shown), *(('continuous', d) for d in shown))


# ---------------- success ----------------
def _libero_summary(policy):
    from analysis.cohorts.libero import episode_rows
    return [r for r in episode_rows(policy) if float(r['speed']) <= .2]


def success_rows(policy, task, *, grid=False):
    """Per-episode rows with task, speed, mode, latency, episode_id, direction, and success, as strings. With grid,
    pi0.5's LIBERO rows include the 50/150/250 ms latency grid. MolmoAct2 rows come from one table; without grid they
    keep the 0/100/200 ms runs. Synchronous points at a nonzero latency on the LIBERO tasks take their success from
    the hold_pose reruns (analysis/sync_hold.py)."""
    from analysis.sync_hold import replace_rows
    return replace_rows(policy, task, _success_rows(policy, task, grid=grid))


def _success_rows(policy, task, *, grid=False):
    if policy == 'MolmoAct2':
        from analysis.cohorts.molmoact2.data import episode_rows
        shown = {0, *delays(policy, task, grid=grid)}
        return [r for r in episode_rows() if r['task'] == task and int(r['latency']) in shown]
    extra = []
    if grid and policy == 'pi0.5' and task in LIBERO:
        from analysis.cohorts.latency_grid.data import episode_rows
        extra = [r for r in episode_rows() if r['task'] == task]
    if task in ('basket', 'drawer'):
        return [r for r in _libero_summary(policy) if r['task'] == task] + extra
    if task in ('plate', 'cabinet'):
        from analysis.cohorts.plate_cabinet.data import episode_rows
        return [r for r in episode_rows(policy) if r['task'] == task] + extra
    from analysis.cohorts.dynamicvla.data import episode_rows
    assert policy == 'DynamicVLA'
    return episode_rows(task)


def tally(rows, *keys):
    """(speed, mode, latency, *keys) -> [successes, episodes]."""
    out = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        cell = out[(float(r['speed']), r['mode'], int(r['latency']), *(int(r[k]) for k in keys))]
        cell[0] += int(r['success'])
        cell[1] += 1
    return dict(out)


def counts(policy, task, *, grid=False):
    """(speed, mode, latency) -> [successes, episodes]; every setting of the task's grid has 200 episodes."""
    out = tally(success_rows(policy, task, grid=grid))
    spec = TASKS[task]
    expected = {(s, m, d) for s in spec.speeds for m, d in conditions(policy, task, grid=grid)}
    if set(out) != expected or any(n != 200 for _, n in out.values()):
        raise ValueError(f'{policy} {task}: incomplete success grid')
    return out


def zero_delay_reference(policy, task, cells):
    """Zero-delay success fraction per speed: pooled synchronous/lazy runs where both exist, else synchronous."""
    def rate(mode, speed):
        k, n = cells[(speed, mode, 0)]
        return k / n
    if lazy_zero(policy, task):
        return {s: (rate('synchronous', s) + rate('lazy', s)) / 2 for s in TASKS[task].speeds}
    return {s: rate('synchronous', s) for s in TASKS[task].speeds}


# ---------------- alignment hand speed ----------------
def alignment_speeds(policy, task):
    """(speed, mode, latency) -> pi0.5's episode-average alignment hand speeds of qualifying episodes (m/s)."""
    if policy != 'pi0.5':
        raise ValueError(f'No alignment measurement for {policy}')
    out = collections.defaultdict(list)
    if task in ('basket', 'drawer'):
        if task == 'basket':
            from analysis.cohorts.basket_alignment.data import load_rows
        else:
            from analysis.cohorts.drawer_alignment.data import load_rows
        for r in load_rows():
            if int(r['qualifying_steps']) and float(r['target_speed_mps']) <= .2:
                out[(float(r['target_speed_mps']), r['mode'], int(r['delay_ms']))].append(float(r['average_hand_speed_mps']))
        return dict(out)
    from analysis.cohorts.plate_cabinet.data import replay_rows
    for r in replay_rows('alignment', policy):
        if r['task'] == task and r['average_hand_speed_mps'] != '':
            out[(float(r['speed']), r['mode'], int(r['latency']))].append(float(r['average_hand_speed_mps']))
    return dict(out)


# ---------------- tolerance-zone attempts ----------------
def attempt_rows(policy, task):
    """Release, set-down, or grasp attempts with offset_m and hit."""
    if task == 'basket':
        from analysis.cohorts.basket_releases import basket_releases
        return [r for r in basket_releases() if r['model'] == policy]
    if task in ('plate', 'cabinet'):
        from analysis.cohorts.plate_cabinet.data import replay_rows
        return [r for r in replay_rows('attempts', policy) if r['task'] == task]
    if task in DYNAMICVLA:
        from analysis.cohorts.dynamicvla.data import replay_rows
        return replay_rows('attempts', task)
    raise ValueError(f'No tolerance-zone attempts for {task}')
