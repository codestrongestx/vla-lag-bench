"""Replay every moving-plate and moving-cabinet trace (kinematics and collision only) for alignment speed, outcomes, and attempts.

No simulation steps are run: recorded states are loaded into MuJoCo, positions and nonpositive-distance
contacts are read, and the definitions in data.DEFINITION are applied. Traces are hash-verified first.
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

sys.dont_write_bytecode = True
from .data import CHUNK, DEFINITION, EPISODE_FOLDER, REPLAY_FOLDER, TARGET_BODY
from analysis.paths import ROOT, localize_xml, portable, read_rows, sha, trace_path


OUTPUTS = ('alignment.csv', 'outcomes.csv', 'attempts.csv')
CACHE = {}


def code_hashes():
    return {portable(p): sha(p) for p in (Path(__file__), Path(__file__).with_name('data.py'),
                                          ROOT / 'analysis/paths.py')}


def environment():
    import mujoco
    return dict(python=sys.version.split()[0], numpy=np.__version__, mujoco=mujoco.__version__)


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def geometry(xml):
    import mujoco
    key = hashlib.sha256(xml.encode()).hexdigest()
    if key not in CACHE:
        m = mujoco.MjModel.from_xml_string(localize_xml(xml))
        names = [m.geom(i).name for i in range(m.ngeom)]

        def mask(prefix):
            return np.array([n.startswith(prefix) for n in names], bool)
        ids = dict(grip=m.site('gripper0_grip_site').id, bowl=m.body('akita_black_bowl_1_main').id,
                   plate=m.body('plate_1_main').id, cabinet=m.body('wooden_cabinet_1_main').id,
                   bowl_geoms=mask('akita_black_bowl_1_'), plate_geoms=mask('plate_1_'),
                   left=mask('gripper0_finger1'), right=mask('gripper0_finger2'))
        if not (ids['bowl_geoms'].any() and ids['plate_geoms'].any() and ids['left'].any() and ids['right'].any()):
            raise ValueError('Missing bowl, plate, or finger geoms')
        CACHE[key] = (m, mujoco.MjData(m), ids)
    return CACHE[key]


def extent(m, d, geoms):
    """Lowest and highest world z of the geoms' bounding boxes in the current pose."""
    index = np.flatnonzero(geoms)
    rotation = d.geom_xmat[index].reshape(-1, 3, 3)
    center = d.geom_xpos[index] + np.einsum('gij,gj->gi', rotation, m.geom_aabb[index, :3])
    half = np.einsum('gij,gj->gi', np.abs(rotation), m.geom_aabb[index, 3:])
    return float((center[:, 2] - half[:, 2]).min()), float((center[:, 2] + half[:, 2]).max())


def episode(row):
    import mujoco
    path = trace_path(row['trace'])
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != row['trace_sha256']:
        raise ValueError(f'Trace hash mismatch: {path}')
    with np.load(io.BytesIO(blob), allow_pickle=True) as z:
        states = z['states']
        acting = z['acting'].astype(bool)
        observed = z['observation_steps'].astype(int)
        xml = str(z['model_xml'])
        meta = json.loads(str(z['meta']))
    task, model = row['task'], row['model']
    latency = int(row['latency'])
    warm, hz, n = int(meta['num_steps_wait']), float(meta['control_freq']), len(acting)
    expected_latency = math.ceil(round(latency * hz / 1000, 9))
    checks = (warm == 10, hz == 20, len(states) == n + 1, not acting[:warm].any(),
              int(meta['episode_id']) == int(row['episode_id']), bool(meta['success']) == bool(int(row['success'])),
              int(meta['latency_steps']) == expected_latency, int(meta['execution_horizon_steps']) == 5,
              int(meta['action_chunk_steps']) == CHUNK[model], meta['trajectory']['target'] == TARGET_BODY[task],
              abs(float(meta['trajectory']['speed_mps']) - float(row['speed'])) < 1e-12,
              n - warm == int(row['control_steps']), int(acting.sum()) == int(row['steps']),
              np.allclose(np.diff(states[:, 0]), 1 / hz, atol=1e-7))
    if not all(checks):
        raise ValueError(f'Trace metadata disagrees with the episode table: {path} {checks}')
    m, d, ids = geometry(xml)
    hand = np.zeros((n + 1, 3)); bowl = hand.copy(); plate = hand.copy()
    left = np.zeros(n + 1, bool); right = left.copy(); on_plate = left.copy()
    fingertips = np.full(n + 1, np.inf); rim = np.full(n + 1, -np.inf)
    fingers = ids['left'] | ids['right']
    for i in range(warm, n + 1):
        d.qpos[:] = states[i, 1:1 + m.nq]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        hand[i] = d.site_xpos[ids['grip']]
        bowl[i] = d.xpos[ids['bowl']]
        plate[i] = d.xpos[ids['plate']]
        if task == 'cabinet':
            fingertips[i] = extent(m, d, fingers)[0]
            rim[i] = extent(m, d, ids['bowl_geoms'])[1]
        if d.ncon:
            contact = d.contact
            touching = contact.dist[:d.ncon] <= 0
            a, b = contact.geom1[:d.ncon][touching], contact.geom2[:d.ncon][touching]
            with_bowl = np.r_[b[ids['bowl_geoms'][a]], a[ids['bowl_geoms'][b]]]
            left[i] = ids['left'][with_bowl].any()
            right[i] = ids['right'][with_bowl].any()
            on_plate[i] = ids['plate_geoms'][with_bowl].any()
    after = np.arange(n + 1) >= warm
    bilateral, finger = left & right, left | right
    lifted = bowl[:, 2] >= bowl[warm, 2] + .02
    pickups = [s0 for s0, s1 in runs(bilateral & after) if s1 - s0 >= 3 and lifted[s0:s1].any()]
    pickup = pickups[0] if pickups else -1
    # A pickup is confirmed on the third bilateral sample of a lifted run.
    confirmed = [s0 + 2 for s0 in pickups]
    contacts = [s0 for s0, _ in runs(on_plate & after)]
    placed = next((s for s in contacts if pickup >= 0 and s >= pickup), -1)
    success = int(row['success'])
    outcome = ('success' if success else 'placed_failed' if placed >= 0
               else 'picked_up_never_placed' if pickup >= 0 else 'never_picked_up')
    base = {k: row[k] for k in ('model', 'task', 'speed', 'mode', 'latency', 'episode_id')}
    times = states[:, 0]
    if task == 'plate':
        near = np.linalg.norm(bowl[:, :2] - plate[:, :2], axis=1) <= .15
        eligible = bilateral & near & (hand[:, 2] >= plate[:, 2]) & after
    else:
        eligible = (np.linalg.norm(hand - bowl, axis=1) <= .10) & ~finger & after
    mask = eligible[:-1] & eligible[1:]
    dt = np.diff(times)
    path_length = float(np.linalg.norm(np.diff(hand, axis=0), axis=1)[mask].sum())
    duration = float(dt[mask].sum())
    ages = (np.arange(n) - warm) - observed
    ages = ages[acting]
    alignment = dict(base, success=success, qualifying_steps=int(mask.sum()), qualifying_s=duration,
                     qualifying_wait_s=float(dt[mask & ~acting].sum()), hand_path_m=path_length,
                     average_hand_speed_mps=path_length / duration if duration else '',
                     mean_observation_age_steps=float(ages.mean()) if len(ages) else '')
    last_acting = np.maximum.accumulate(np.where(acting, np.arange(n), -1))

    def age_at(t):
        j = last_acting[t - 1] if t >= 1 else -1
        return int((t - 1 - warm) - observed[j]) if j >= warm else -1
    attempts = []
    if task == 'plate':
        joined = []
        for s0, s1 in runs(on_plate & after):
            if joined and s0 - joined[-1][1] < 3:
                joined[-1][1] = s1
            else:
                joined.append([s0, s1])
        for k, (s0, s1) in enumerate(joined):
            attempts.append(dict(base, success=success, k=k, step=s0, held=int(bilateral[s0]),
                                 offset_m=float(np.linalg.norm(bowl[s0, :2] - plate[s0, :2])),
                                 hit=int(success and s1 >= n + 1), age_steps=age_at(s0)))
    else:
        knocked = np.flatnonzero(after & (bowl[:, 2] < bowl[warm, 2] - .05))
        stop = min(confirmed[0] - 2 if confirmed else n, int(knocked[0]) if len(knocked) else n)
        below = fingertips < rim
        horizontal = np.linalg.norm(hand[:, :2] - bowl[:, :2], axis=1)
        onsets = [i for i in range(warm + 3, stop + 1)
                  if below[i] and not below[i - 3:i].any() and horizontal[i] <= .15]
        for k, i in enumerate(onsets):
            following = onsets[k + 1] if k + 1 < len(onsets) else n + 1
            hit = int(pickup >= 0 and i <= pickup < following and pickup - i <= 40)
            attempts.append(dict(base, success=success, k=k, step=i, held=0, offset_m=float(horizontal[i]),
                                 hit=hit, age_steps=age_at(i)))
    outcome_row = dict(base, success=success, outcome=outcome, pickup_step=pickup, placed_step=placed,
                       attempts=len(attempts), trace_sha256=row['trace_sha256'])
    rim_above = float(rim[warm] - bowl[warm, 2]) if task == 'cabinet' else math.nan
    return alignment, outcome_row, attempts, round(rim_above, 3)


def ensure(*, workers=4, force=False):
    folder = (REPLAY_FOLDER)
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
                print('plate-cabinet-replay: episode table, code, and outputs verified; using cached replay', flush=True)
                return folder
        except (OSError, ValueError, KeyError):
            pass
    rows = read_rows(table)
    groups = defaultdict(list)
    for row in rows:
        groups[row['model'], row['task'], row['speed'], row['mode'], row['latency']].append(row)
    ordered = [r for key in sorted(groups) for r in groups[key]]
    alignment, outcomes, attempts, rims = [], [], [], set()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (a, o, t, rim) in enumerate(pool.map(episode, ordered, chunksize=25), 1):
            alignment.append(a); outcomes.append(o); attempts.extend(t)
            if not math.isnan(rim):
                rims.add(rim)
            if index % 2000 == 0:
                print(f'plate-cabinet-replay: {index}/{len(ordered)} episodes', flush=True)
    if sha(table) != sources[portable(table)] or code_hashes() != code:
        raise ValueError('Inputs or calculation changed while replaying; rerun')
    for name, values in (('alignment.csv', alignment), ('outcomes.csv', outcomes), ('attempts.csv', attempts)):
        with (folder / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    counts = defaultdict(int)
    for o in outcomes:
        counts[f"{o['model']}|{o['task']}|{o['outcome']}"] += 1
    payload = dict(definition=DEFINITION, code=code, environment=runtime, sources=sources, episodes=len(outcomes),
                   attempts=len(attempts), bowl_rim_above_origin_m=sorted(rims), outcome_counts=dict(sorted(counts.items())),
                   outputs={name: sha(folder / name) for name in OUTPUTS})
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f'plate-cabinet-replay: {len(outcomes)} episodes, {len(attempts)} attempts', flush=True)
    return folder
