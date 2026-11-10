"""Measure every DynamicVLA moving-box and moving-can trace: alignment speed, outcomes, attempts, and continued actions.

Recorded states are read directly (no simulation); definitions are in data.DEFINITION and enter the provenance.
Traces are hash-verified first and checked against the episode table and their run configuration.
"""
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

from .data import (BOX_SIZE_M, CAN_HALF_M, CHUNK, DEFINITION, EE_OFFSET_M, EPISODE_FOLDER, FINGER_BASE_M, FINGER_BOX_M,
                   HORIZON, HZ, REPLAY_FOLDER, TARGET)
from analysis.paths import ROOT, portable, read_rows, sha, trace_path

OUTPUTS = ('alignment.csv', 'outcomes.csv', 'attempts.csv', 'continued.csv')
DT = 1 / HZ
LIFT, HELD_OPENING, HELD_DISTANCE, FALL = .02, .02, .06, .05
ALIGN_BOX, ALIGN_CAN, ATTEMPT_RADIUS, ATTEMPT_WINDOW = .15, .10, .15, 50


def code_hashes():
    return {portable(p): sha(p) for p in (Path(__file__), Path(__file__).with_name('data.py'),
                                          ROOT / 'analysis/paths.py')}


def environment():
    return dict(python=sys.version.split()[0], numpy=np.__version__)


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def rotation(q):
    """Rotation matrix of a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def lowest_finger(grip, quat, joints):
    """Lowest z of both fingers' collision bounding boxes in each recorded pose (cf. the moving bowl's extent)."""
    low = np.full(len(grip), np.inf)
    for i, (position, q, opening) in enumerate(zip(grip, quat, joints)):
        R = rotation(q)
        hand = position - R[:, 2] * EE_OFFSET_M
        for side, sign, slide in (('left', 1, opening[0]), ('right', -1, opening[1])):
            lo, hi = (np.array(v) for v in FINGER_BOX_M[side])
            center = (lo + hi) / 2 + np.array([0, sign * slide, FINGER_BASE_M])
            z = (hand + R @ center)[2] - np.abs(R[2]) @ ((hi - lo) / 2)
            low[i] = min(low[i], z)
    return low


def finger_points(grip, quat, joints):
    """Grid points (0.5-0.7 cm spacing) filling both fingers' collision bounding boxes, per recorded pose: [T, P, 3]."""
    out = []
    for side, sign, slide in (('left', 1, joints[:, 0]), ('right', -1, joints[:, 1])):
        lo, hi = (np.array(v) for v in FINGER_BOX_M[side])
        grid = np.stack(np.meshgrid(*(np.linspace(a, b, k) for a, b, k in zip(lo, hi, (4, 5, 9))), indexing='ij'), -1)
        local = grid.reshape(-1, 3) + np.array([0, 0, FINGER_BASE_M])
        for i, (position, q, s) in enumerate(zip(grip, quat, slide)):
            R = rotation(q)
            if len(out) <= i:
                out.append([])
            out[i].append((position - R[:, 2] * EE_OFFSET_M) + (local + np.array([0, sign * s, 0])) @ R.T)
    return np.array([np.concatenate(parts) for parts in out])


def finger_can_distance(grip, quat, joints, can, can_quat, base):
    """Smallest signed distance from the finger grid points to the can cylinder, per recorded pose."""
    points = finger_points(grip, quat, joints)
    distance = np.empty(len(grip))
    for i, (pts, center, q) in enumerate(zip(points, can, can_quat)):
        local = (pts - center) @ (base.T @ rotation(q))
        radial = np.hypot(local[:, 0], local[:, 1]) - CAN_HALF_M[0]
        axial = np.abs(local[:, 2]) - CAN_HALF_M[2]
        outside = np.hypot(np.maximum(radial, 0), np.maximum(axial, 0))
        distance[i] = (outside + np.minimum(np.maximum(radial, axial), 0)).min()
    return distance


def highest_can(can, quat, base):
    """Highest z of the can's bounding box (a lying cylinder) in each recorded pose, in the robot-base frame."""
    rows = np.array([(base.T @ rotation(q))[2] for q in quat])
    return can[:, 2] + np.abs(rows) @ np.array(CAN_HALF_M)


def episode(row):
    path = trace_path(row['trace'])
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != row['trace_sha256']:
        raise ValueError(f'Trace hash mismatch: {path}')
    with np.load(io.BytesIO(blob), allow_pickle=True) as z:
        meta = json.loads(str(z['meta']))
        keys = {'/'.join(m['path']): m['key'] for m in json.loads(str(z['state_manifest']))}
        protocol = json.loads(str(z['protocol']))
        actions = z['actions'][:, 0].astype(float)
        grip = z['observation_0000'][:, 0].astype(float)
        hand_quat = z['observation_0001'][:, 0].astype(float)
        robot = z[keys['scene/articulation/robot/root_pose']][:, 0].astype(float)
        joints = z[keys['scene/articulation/robot/joint_position']][:, 0].astype(float)
        can_pose = z[keys['scene/rigid_object/object/root_pose']][:, 0].astype(float)
        can_w = can_pose[:, :3]
        box_w = z[keys['scene/rigid_object/container/root_pose']][:, 0, :3].astype(float)
    task, mode, latency = row['task'], row['mode'], int(row['latency'])
    n = len(protocol)
    actor = meta['motion']['actors'][0]
    checks = (bool(meta['success']) == bool(int(row['success'])), meta['inference_mode'] == mode,
              int(meta['latency_steps']) == round(latency * HZ / 1000), int(meta['action_chunk_steps']) == CHUNK,
              int(meta['execution_horizon_steps']) == HORIZON, float(meta['control_hz']) == HZ,
              actor['target'] == TARGET[task], abs(float(actor['motion']['speed_mps']) - float(row['speed'])) < 1e-12,
              int(actor['motion']['initial_direction']) == int(row['direction']),
              len(actions) == n, len(grip) == len(hand_quat) == len(robot) == len(can_w) == len(box_w) == n + 1,
              sum(p['wait_reason'] == 'model' for p in protocol) == int(row['steps']),
              np.allclose(robot, robot[0], atol=1e-5))
    if not all(checks):
        raise ValueError(f'Trace disagrees with the episode table or configuration: {path} {checks}')
    R, origin = rotation(robot[0, 3:]), robot[0, :3]
    can, box = (can_w - origin) @ R, (box_w - origin) @ R
    reason = [p['wait_reason'] for p in protocol]
    warm = next(i for i, r in enumerate(reason) if r != 'policy_warmup')
    acting = np.array([r == 'model' for r in reason])
    waiting = np.array([r == 'inference_wait' for r in reason])
    if set(reason) - {'policy_warmup', 'inference_wait', 'model'} or 'policy_warmup' in reason[warm:]:
        raise ValueError(f'Unexpected protocol records: {path}')
    after = np.arange(n + 1) >= warm
    closed = np.r_[False, actions[:, 7] < 0]
    opening = joints[:, 7] + joints[:, 8]
    to_can = np.linalg.norm(can - grip, axis=1)
    held = closed & (opening >= HELD_OPENING) & (to_can <= HELD_DISTANCE)
    rest = can[warm, 2]
    lifted = can[:, 2] >= rest + LIFT
    pickups = [s0 for s0, s1 in runs(held & after) if s1 - s0 >= 3 and lifted[s0:s1].any()]
    pickup = int(pickups[0]) if pickups else -1
    releases = [t for t in range(max(warm, 1) + 1, n + 1)
                if pickup >= 0 and t - 1 >= pickup and closed[t - 1] and not closed[t] and held[t - 1] and lifted[t - 1]]
    success = int(row['success'])
    if task == 'box':
        outcome = ('success' if success else 'released_failed' if releases
                   else 'picked_up_never_released' if pickup >= 0 else 'never_picked_up')
    else:
        outcome = 'success' if success else 'picked_up_failed' if pickup >= 0 else 'never_picked_up'
    base = {k: row[k] for k in ('model', 'task', 'speed', 'mode', 'latency', 'episode_id', 'direction')}

    # Alignment hand speed: intervals whose two endpoints both qualify; waits count, warm-up does not.
    if task == 'box':
        top = box[:, 2] + BOX_SIZE_M[2] / 2
        eligible = held & (np.linalg.norm(can[:, :2] - box[:, :2], axis=1) <= ALIGN_BOX) & (grip[:, 2] >= top) & after
    else:
        eligible = (to_can <= ALIGN_CAN) & ~closed & after
    mask = eligible[:-1] & eligible[1:]
    path_length = float(np.linalg.norm(np.diff(grip, axis=0), axis=1)[mask].sum())
    duration = DT * int(mask.sum())
    observed = np.array([p['observation_step'] if p['wait_reason'] == 'model' else -1 for p in protocol])
    ages = (np.arange(n) - observed)[acting]
    alignment = dict(base, success=success, qualifying_steps=int(mask.sum()), qualifying_s=duration,
                     qualifying_wait_s=DT * int((mask & waiting).sum()), hand_path_m=path_length,
                     average_hand_speed_mps=path_length / duration if duration else '',
                     mean_observation_age_steps=float(ages.mean()) if len(ages) else '')
    last_acting = np.maximum.accumulate(np.where(acting, np.arange(n), -1))

    def age_at(t):
        j = last_acting[t - 1] if t >= 1 else -1
        return int((t - 1) - observed[j]) if j >= warm else -1

    attempts = []
    if task == 'box':
        touching = finger_can_distance(grip, hand_quat, joints[:, 7:9], can, can_pose[:, 3:], R) <= 0
        for k, t in enumerate(releases):
            later = k + 1 < len(releases)
            apart = [a for a, b in runs(~touching[t:]) if b - a >= 2]
            retouched = not apart or touching[t + apart[0]:].any()
            attempts.append(dict(base, success=success, k=k, step=t,
                                 offset_m=float(np.linalg.norm(can[t - 1, :2] - box[t - 1, :2])),
                                 hit=int(success and not later and not retouched), age_steps=age_at(t)))
    else:
        fallen = np.flatnonzero(after & (can[:, 2] < rest - FALL))
        stop = min(pickup if pickup >= 0 else n, int(fallen[0]) if len(fallen) else n)
        horizontal = np.linalg.norm(grip[:, :2] - can[:, :2], axis=1)
        below = (lowest_finger(grip, hand_quat, joints[:, 7:9]) < highest_can(can, can_pose[:, 3:], R)) & after
        onsets = [t for t in range(warm + 3, stop + 1)
                  if below[t] and not below[t - 3:t].any() and horizontal[t] <= ATTEMPT_RADIUS]
        for k, t in enumerate(onsets):
            following = onsets[k + 1] if k + 1 < len(onsets) else n + 1
            hit = int(pickup >= 0 and t <= pickup < following and pickup - t <= ATTEMPT_WINDOW)
            attempts.append(dict(base, success=success, k=k, step=t, offset_m=float(horizontal[t]), hit=hit,
                                 age_steps=age_at(t)))

    # Continued actions: older commands executed while a newer chunk was pending, against its prediction.
    relative, distance = [], []
    if mode != 'synchronous':
        for i, p in enumerate(protocol):
            if not p['released_chunk']:
                continue
            fresh, chunk, observation = p['released_chunk']['actions'], p['chunk_index'], p['release']['observation_step']
            if p['release']['release_step'] != i or p['release']['chunk_index'] != chunk:
                raise ValueError(f'Inconsistent chunk release record: {path}')
            for s in range(max(observation, warm), i):
                if not acting[s] or protocol[s]['chunk_index'] >= chunk or s - observation >= len(fresh):
                    continue
                older, newer = actions[s, :3], np.asarray(fresh[s - observation][0][:3], float)
                relative.append(np.linalg.norm(older - newer) /
                                (.5 * (np.linalg.norm(older - grip[s]) + np.linalg.norm(newer - grip[s])) + 1e-6))
                distance.append(np.linalg.norm(older - newer))
    continued = dict(base, success=success, eligible_steps=len(relative),
                     relative_difference=float(np.median(relative)) if relative else '',
                     position_difference_m=float(np.median(distance)) if distance else '')
    outcome_row = dict(base, success=success, outcome=outcome, pickup_step=pickup,
                       release_step=releases[0] if releases else -1, releases=len(releases),
                       attempts=len(attempts), warmup_steps=warm, trace_sha256=row['trace_sha256'])
    return alignment, outcome_row, attempts, continued


def ensure(*, workers=4, force=False):
    folder = REPLAY_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    provenance_path = folder / 'provenance.json'
    table = EPISODE_FOLDER / 'episodes.csv'
    code, runtime = code_hashes(), environment()
    sources = {portable(table): sha(table)}
    if not force and provenance_path.exists():
        try:
            previous = json.loads(provenance_path.read_text())
            if (previous['code'] == code and previous['environment'] == runtime and previous['definition'] == DEFINITION
                    and previous['sources'] == sources
                    and all(sha(folder / name) == value for name, value in previous['outputs'].items())):
                print('dynamicvla-replay: episode table, code, and outputs verified; using cached measurements', flush=True)
                return folder
        except (OSError, ValueError, KeyError):
            pass
    rows = read_rows(table)
    groups = defaultdict(list)
    for row in rows:
        groups[row['task'], row['speed'], row['mode'], row['latency']].append(row)
    ordered = [r for key in sorted(groups) for r in groups[key]]
    alignment, outcomes, attempts, continued = [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (a, o, t, c) in enumerate(pool.map(episode, ordered, chunksize=20), 1):
            alignment.append(a); outcomes.append(o); attempts.extend(t); continued.append(c)
            if index % 2000 == 0:
                print(f'dynamicvla-replay: {index}/{len(ordered)} episodes', flush=True)
    if sha(table) != sources[portable(table)] or code_hashes() != code:
        raise ValueError('Inputs or calculation changed while measuring; rerun')
    for name, values in (('alignment.csv', alignment), ('outcomes.csv', outcomes), ('attempts.csv', attempts),
                         ('continued.csv', continued)):
        with (folder / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    counts = defaultdict(int)
    for o in outcomes:
        counts[f"{o['task']}|{o['outcome']}"] += 1
    payload = dict(definition=DEFINITION, code=code, environment=runtime, sources=sources, episodes=len(outcomes),
                   attempts=len(attempts), outcome_counts=dict(sorted(counts.items())),
                   outputs={name: sha(folder / name) for name in OUTPUTS})
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f'dynamicvla-replay: {len(outcomes)} episodes, {len(attempts)} attempts', flush=True)
    return folder


if __name__ == '__main__':
    ensure(workers=int(sys.argv[1]) if len(sys.argv) > 1 else 4, force=True)
