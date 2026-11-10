"""Shared numerical definitions. Probabilities are fractions unless named percent."""
import math
import numpy as np
from collections import defaultdict


def wilson(k, n, z=1.96, *, clip=True):
    """Point estimate and Wilson bounds; empty samples yield three NaNs."""
    if n == 0:
        return math.nan, math.nan, math.nan
    if n < 0 or not 0 <= k <= n:
        raise ValueError('Wilson counts must satisfy 0 <= successes <= trials')
    p = k / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half_width = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    low, high = center-half_width, center+half_width
    return p, max(low,0) if clip else low, min(high,1) if clip else high


def wilson_bounds(k, n, z=1.96):
    return wilson(k, n, z, clip=False)[1:]


def wilson_percent(k, n):
    return tuple(100 * value for value in wilson(k, n, clip=False))


def crossing(xs, ys, level, *, require_initial_above=True, above_range=math.inf):
    """First downward linear crossing. Explicitly distinguish below/above range."""
    if len(xs) != len(ys) or not len(xs):
        raise ValueError('Crossing requires nonempty, equally sized coordinates')
    if require_initial_above and ys[0] < level:
        return math.nan
    for i in range(len(xs)-1):
        if ys[i] >= level > ys[i+1]:
            return xs[i] + (ys[i]-level)/(ys[i]-ys[i+1]) * (xs[i+1]-xs[i])
    return above_range


def v50(xs, ys):
    """Historical fraction-valued crossing, allowing a later upward recovery."""
    return crossing(xs, ys, .5, require_initial_above=False, above_range=math.nan)


def select(data, task, mode, latency, speed=None):
    return [r for r in data if r['task'] == task
            and (mode is None or r['mode'] == mode)
            and int(r['latency']) == latency
            and (speed is None or float(r['speed']) == speed)]


def success_counts(rows):
    return sum(int(r['success']) for r in rows), len(rows)


def success_series(rows, task, mode, latency, speeds):
    return np.array([wilson_percent(*success_counts(select(rows,task,mode,latency,s)))
                     for s in speeds]).T


def episode_counts(rows):
    """Counts, direction counts, and paired episode outcomes from one cohort."""
    counts = defaultdict(lambda: [0,0])
    direction = defaultdict(lambda: [0,0])
    paired = defaultdict(dict)
    for row in rows:
        key = row['task'], float(row['speed']), row['mode'], int(row['latency'])
        k = int(row['success'])
        counts[key][0] += k; counts[key][1] += 1
        dkey = key + (int(row['direction']),)
        direction[dkey][0] += k; direction[dkey][1] += 1
        paired[key][int(row['episode_id'])] = k
    return counts, direction, paired
