"""Definition, interval reduction, and loading for basket hand speed."""
import statistics

import numpy as np

from analysis.paths import data_rows

SPEEDS = (.001, .025, .05, .075, .1, .125, .15, .175, .2)
DEFINITION = {
    'version': 1,
    'horizontal_radius_m': .15,
    'held': 'Nonpositive-distance cheese contact with both fingers',
    'height': 'Gripper site at or above highest world-space basket wall corner',
    'selection': 'Both endpoints qualify; all visits after warm-up, including retries',
    'speed': 'Sum of selected 3D gripper displacements / selected elapsed simulated time',
    'waits': 'Included whenever selection conditions hold; never filter by acting',
    'missing': 'No selected intervals gives missing speed, not zero',
}
OUTPUT_NAMES = ('episodes.csv', 'by_speed.csv', 'pooled.csv')


def reduce_intervals(hand, times, acting, held, distance, rim_height, warmup):
    """Include whole sampled intervals, never bridge excluded portions of a rollout."""
    dt = np.diff(times)
    if len(hand) != len(acting) + 1 or len(times) != len(hand):
        raise ValueError('State/action lengths disagree')
    if not np.all(np.isfinite(hand)) or not np.all(dt > 0):
        raise ValueError('Invalid positions or timestamps')
    eligible = (held & (distance <= DEFINITION['horizontal_radius_m'])
                & (hand[:, 2] >= rim_height) & (np.arange(len(hand)) >= warmup))
    mask = eligible[:-1] & eligible[1:]
    steps = np.linalg.norm(np.diff(hand, axis=0), axis=1)
    duration = float(dt[mask].sum())
    path = float(steps[mask].sum())
    return dict(
        qualifying_steps=int(mask.sum()), qualifying_s=duration,
        qualifying_wait_s=float(dt[mask & ~acting].sum()), hand_path_m=path,
        average_hand_speed_mps=path / duration if duration else '',
        peak_hand_speed_mps=float((steps[mask] / dt[mask]).max()) if duration else '',
        qualifying_bouts=int(np.count_nonzero(mask & ~np.r_[False, mask[:-1]])),
    )


def summarize(rows):
    valid = [r for r in rows if int(r['qualifying_steps']) > 0]
    def median(key):
        return statistics.median(float(r[key]) for r in valid) if valid else ''
    duration = sum(float(r['qualifying_s']) for r in valid)
    return dict(
        episodes=len(rows), qualifying_episodes=len(valid),
        median_average_speed_mps=median('average_hand_speed_mps'),
        median_qualifying_s=median('qualifying_s'),
        median_wait_fraction=statistics.median(float(r['qualifying_wait_s']) / float(r['qualifying_s']) for r in valid) if valid else '',
        pooled_path_over_time_mps=sum(float(r['hand_path_m']) for r in valid) / duration if duration else '',
        multiple_bouts=sum(int(r['qualifying_bouts']) > 1 for r in valid),
        shorter_than_025s=sum(float(r['qualifying_s']) < .25 - 1e-7 for r in valid),
    )


def load_rows():
    """The shipped per-episode table (analysis/data/basket_alignment_speed/episodes.csv.gz)."""
    return data_rows('basket_alignment_speed/episodes')
