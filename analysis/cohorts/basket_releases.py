"""Every cheese release in the pi0.5 basket cohort, with its offset from the basket's containment region.

Read-only replay of recorded states (kinematics and collision, no simulation steps). The cohort and
trace hashes come from the basket-alignment measurement's full table, which covers all 14,400 episodes.
basket_releases() loads the shipped table for the tolerance-zone figure.
"""
import csv
import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

from analysis.paths import DERIVED, ROOT, data_rows, localize_xml, portable, sha, trace_path

SOURCE = DERIVED / 'basket_alignment_speed/episodes.csv'
FOLDER = DERIVED / 'basket_releases'
DEFINITION = {
    'version': 3,
    'release': 'First sample of a run of >= 2 samples without finger-cheese contact, with both fingers in contact '
               'within the 3 preceding samples, the cheese >= 2 cm above its post-warm-up height, and the gripper '
               'commanded open (the action executed in the previous step has a negative gripper value)',
    'offset': 'Horizontal distance from the cheese origin to the basket containment-region center in the last sample '
              'with finger contact',
    'hit': 'The episode succeeds with no later release and no finger-cheese contact after the release',
}
CACHE = {}


def code_hashes():
    return {portable(p): sha(p) for p in (Path(__file__), ROOT / 'analysis/paths.py')}


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def geometry(xml):
    import mujoco
    key = hashlib.sha256(xml.encode()).hexdigest()
    if key not in CACHE:
        m = mujoco.MjModel.from_xml_string(localize_xml(xml))
        names = [m.geom(i).name for i in range(m.ngeom)]
        ids = dict(cheese=m.body('cream_cheese_1_main').id, region=m.site('basket_1_contain_region').id,
                   cheese_geoms=np.array([n.startswith('cream_cheese_1_') for n in names]),
                   left=np.array([n.startswith('gripper0_finger1') for n in names]),
                   right=np.array([n.startswith('gripper0_finger2') for n in names]))
        CACHE[key] = (m, mujoco.MjData(m), ids)
    return CACHE[key]


def episode(row):
    import mujoco
    path = trace_path(row['trace'])
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != row['trace_sha256']:
        raise ValueError(f'Trace hash mismatch: {path}')
    with np.load(io.BytesIO(blob), allow_pickle=True) as z:
        states = z['states']
        gripper = z['actions'][:, -1].astype(float)
        acting = z['acting'].astype(bool)
        observed = z['observation_steps'].astype(int)
        xml = str(z['model_xml'])
        meta = json.loads(str(z['meta']))
    warm, n = int(meta['num_steps_wait']), len(acting)
    if (warm != 10 or len(states) != n + 1 or len(gripper) != n
            or bool(meta['success']) != bool(int(row['success']))):
        raise ValueError(f'Trace metadata disagrees with the cohort table: {path}')
    m, d, ids = geometry(xml)
    cheese = np.zeros((n + 1, 3)); center = cheese.copy()
    left = np.zeros(n + 1, bool); right = left.copy()
    for i in range(warm, n + 1):
        d.qpos[:] = states[i, 1:1 + m.nq]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        cheese[i] = d.xpos[ids['cheese']]
        center[i] = d.site_xpos[ids['region']]
        if d.ncon:
            touching = d.contact.dist[:d.ncon] <= 0
            a, b = d.contact.geom1[:d.ncon][touching], d.contact.geom2[:d.ncon][touching]
            other = np.r_[b[ids['cheese_geoms'][a]], a[ids['cheese_geoms'][b]]]
            left[i] = ids['left'][other].any()
            right[i] = ids['right'][other].any()
    after = np.arange(n + 1) >= warm
    finger, bilateral = left | right, left & right
    raised = cheese[:, 2] >= cheese[warm, 2] + .02
    opened = np.r_[False, gripper < 0]
    last_acting = np.maximum.accumulate(np.where(acting, np.arange(n), -1))
    success = int(row['success'])
    out = []
    for f0, f1 in runs(~finger & after):
        if (f0 <= warm or f1 - f0 < 2 or not bilateral[max(f0 - 3, warm):f0].any() or not raised[f0]
                or not opened[f0]):
            continue
        j, last_end = last_acting[f0 - 1], f1
        out.append(dict(task='basket', model='pi0.5', speed=row['target_speed_mps'], mode=row['mode'],
                        latency=row['delay_ms'], episode_id=row['episode_id'], success=success, k=len(out), step=int(f0),
                        offset_m=float(np.linalg.norm(cheese[f0 - 1, :2] - center[f0 - 1, :2])),
                        hit=0, age_steps=int((f0 - 1 - warm) - observed[j]) if j >= warm else -1))
    if out:
        out[-1]['hit'] = int(success and last_end == n + 1)
    return out


def ensure(*, workers=4, force=False):
    import mujoco
    folder = (FOLDER)
    folder.mkdir(parents=True, exist_ok=True)
    provenance_path = folder / 'provenance.json'
    code = code_hashes()
    environment = dict(python=sys.version.split()[0], numpy=np.__version__, mujoco=mujoco.__version__)
    sources = {portable(SOURCE): sha(SOURCE)}
    if not force and provenance_path.exists():
        try:
            previous = json.loads(provenance_path.read_text())
            if (previous['code'] == code and previous['environment'] == environment and previous['definition'] == DEFINITION
                    and previous['sources'] == sources
                    and previous['outputs']['attempts.csv'] == sha(folder / 'attempts.csv')):
                print('basket-releases: cohort table, code, and output verified; using cached releases', flush=True)
                return folder
        except (OSError, ValueError, KeyError):
            pass
    with SOURCE.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 14400 or len({r['trace'] for r in rows}) != 14400:
        raise ValueError('Expected the complete 14,400-episode pi0.5 basket cohort')
    releases = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, values in enumerate(pool.map(episode, rows, chunksize=25), 1):
            releases.extend(values)
            if index % 2000 == 0:
                print(f'basket-releases: {index}/{len(rows)} episodes', flush=True)
    if sha(SOURCE) != sources[portable(SOURCE)] or code_hashes() != code:
        raise ValueError('Inputs or calculation changed while replaying; rerun')
    with (folder / 'attempts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(releases[0]))
        writer.writeheader()
        writer.writerows(releases)
    payload = dict(definition=DEFINITION, code=code, environment=environment, sources=sources, episodes=len(rows),
                   releases=len(releases), outputs={'attempts.csv': sha(folder / 'attempts.csv')})
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f'basket-releases: {len(releases)} releases from {len(rows)} episodes', flush=True)
    return folder


def basket_releases():
    """The shipped release attempts (analysis/data/basket_releases/attempts.csv.gz)."""
    return data_rows('basket_releases/attempts')
