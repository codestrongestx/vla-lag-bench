"""Reconstruct held/near/above basket intervals from read-only pi0.5 traces."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

sys.dont_write_bytecode = True
from .data import DEFINITION, SPEEDS, OUTPUT_NAMES, reduce_intervals, summarize
from analysis.paths import ROOT, ARTIFACTS, DERIVED, localize_xml, point_dir, portable, resolve, sha
from analysis import sync_hold
CACHE = {}


def code_hashes():
    return {portable(p): sha(p) for p in (
        Path(__file__), Path(__file__).with_name('data.py'),
        ROOT/'analysis/paths.py', ROOT/'analysis/sync_hold.py',
    )}


def environment():
    import mujoco
    import yaml
    return dict(python=sys.version.split()[0], numpy=np.__version__, mujoco=mujoco.__version__, pyyaml=yaml.__version__)


def point(job):
    """Measure one recorded run; reconstruct contacts without stepping simulation."""
    import mujoco, yaml
    (key, rows) = job
    (speed, mode, latency, source) = key
    # Synchronous points at a nonzero latency come from the hold_pose reruns (analysis/sync_hold.py).
    p = sync_hold.point_dir('pi0.5', 'basket', speed, latency) if source == sync_hold.SOURCE else Path(point_dir(source, speed, mode, latency))
    run = yaml.safe_load((p / 'point.yaml').read_text())['run_id']
    raw = yaml.safe_load((p / 'runs' / run / 'experiment.yaml').read_text())
    assert raw['model']['checkpoint'] == 'gs://openpi-assets/checkpoints/pi05_libero'
    files = {int(f.stem.rsplit('_ep', 1)[1]): f for f in (p / 'artifacts' / run).glob('*.npz')}
    assert len(files) == 200
    out = []
    dependencies = {portable(p / 'point.yaml'): sha(p / 'point.yaml'), portable(p / 'runs' / run / 'experiment.yaml'): sha(p / 'runs' / run / 'experiment.yaml')}
    if source == sync_hold.SOURCE:
        jsonl = p / 'artifacts' / run / 'episodes.jsonl'
        dependencies[portable(jsonl)] = sha(jsonl)
    for row in rows:
        f = files[int(row['episode_id'])]
        blob = f.read_bytes()
        with np.load(io.BytesIO(blob), allow_pickle=True) as z:
            st = z['states']
            acting = z['acting'].astype(bool)
            xml = str(z['model_xml'])
            meta = json.loads(str(z['meta']))
        warm = int(meta['num_steps_wait'])
        hz = float(meta['control_freq'])
        N = len(acting)
        assert warm == 10 and hz == 20 and (not acting[:warm].any()) and (len(st) == N + 1)
        dt = np.diff(st[:, 0])
        assert np.allclose(dt, 1 / hz, atol=1e-07)
        assert N - warm == int(row['control_steps']) and int(acting.sum()) == int(row['steps'])
        dependencies[portable(f)] = hashlib.sha256(blob).hexdigest()
        h = hashlib.sha256(xml.encode()).hexdigest()
        if h not in CACHE:
            localized = localize_xml(xml)
            import xml.etree.ElementTree as ET
            assets = {}
            for element in ET.fromstring(localized).iter():
                if 'file' in element.attrib:
                    asset = Path(element.attrib['file'])
                    if not asset.is_absolute():
                        raise ValueError(f'Unresolved model asset: {asset}')
                    assets[portable(asset)] = sha(asset)
            m = mujoco.MjModel.from_xml_string(localized)
            d = mujoco.MjData(m)
            ids = {kind: {i for i in range(m.ngeom) if m.geom(i).name.startswith(prefix)} for (kind, prefix) in [('cheese', 'cream_cheese_1_'), ('left', 'gripper0_finger1'), ('right', 'gripper0_finger2')]}
            ids.update(grip=m.site('gripper0_grip_site').id, region=m.site('basket_1_contain_region').id, cheese_body=m.body('cream_cheese_1_main').id, basket_body=m.body('basket_1_main').id, walls=np.array([m.geom(f'basket_1_g{i}').id for i in (2, 3, 4, 5)]))
            assert all((m.geom_type[i] == 6 for i in ids['walls']))
            CACHE[h] = (m, d, ids, assets)
        (m, d, ids, assets) = CACHE[h]
        dependencies.update(assets)
        ee = np.zeros((N + 1, 3))
        distance = np.zeros(N + 1)
        rim = np.zeros(N + 1)
        rimlocal = np.zeros(N + 1)
        bilateral = np.zeros(N + 1, bool)
        height_ok = np.zeros(N + 1, bool)
        for i in range(warm, N + 1):
            d.qpos[:] = st[i, 1:1 + m.nq]
            mujoco.mj_kinematics(m, d)
            ee[i] = d.site_xpos[ids['grip']]
            c = d.xpos[ids['cheese_body']]
            center = d.site_xpos[ids['region']]
            distance[i] = np.linalg.norm(c[:2] - center[:2])
            w = ids['walls']
            tops = d.geom_xpos[w, 2] + np.einsum('ij,ij->i', np.abs(d.geom_xmat[w].reshape(-1, 3, 3)[:, 2, :]), m.geom_size[w])
            rim[i] = tops.max()
            rimlocal[i] = rim[i] - d.xpos[ids['basket_body'], 2]
            height_ok[i] = ee[i, 2] >= rim[i]
            if distance[i] > DEFINITION['horizontal_radius_m'] or not height_ok[i]:
                continue
            mujoco.mj_collision(m, d)
            left = False
            right = False
            for contact in d.contact:
                if contact.dist > 0:
                    continue
                (a, b) = (int(contact.geom1), int(contact.geom2))
                other = b if a in ids['cheese'] else a if b in ids['cheese'] else -1
                left |= other in ids['left']
                right |= other in ids['right']
            bilateral[i] = left and right
        result = dict(task='basket', target_speed_mps=speed, mode=mode, delay_ms=latency, episode_id=int(row['episode_id']), success=int(row['success']), **reduce_intervals(ee, st[:, 0], acting, bilateral, distance, rim, warm), geometry_near_above_steps=int(((distance[:-1] <= DEFINITION['horizontal_radius_m']) & height_ok[:-1] & (np.arange(N) >= warm)).sum()), rim_height_world_min_m=float(rim[warm:].min()), rim_height_world_max_m=float(rim[warm:].max()), rim_above_body_min_m=float(rimlocal[warm:].min()), rim_above_body_max_m=float(rimlocal[warm:].max()), trace='artifacts/' + str(f.relative_to(ARTIFACTS)), trace_sha256=hashlib.sha256(blob).hexdigest())
        out.append(result)
    return (key, out, dependencies)


def ensure(*, workers=4, force=False):
    """Validate all upstream hashes; rebuild when calculation, sources, or outputs change."""
    if workers < 1:
        raise ValueError('workers must be positive')
    destination = DERIVED / 'basket_alignment_speed'
    destination.mkdir(parents=True, exist_ok=True)
    provenance_path = destination / 'provenance.json'
    code, runtime = code_hashes(), environment()
    if not force and provenance_path.exists():
        try:
            previous = json.loads(provenance_path.read_text())
            current = (previous['definition'] == DEFINITION and previous['code'] == code
                       and previous['environment'] == runtime
                       and previous['episodes'] == 14400
                       and set(previous['outputs']) == set(OUTPUT_NAMES))
            if current:
                current = all(sha(destination / name) == value for name, value in previous['outputs'].items())
            if current:
                current = all(sha(resolve(name)) == value for name, value in previous['sources'].items())
            if current:
                print('basket-alignment: source/code/output hashes verified; using cached measurements', flush=True)
                return destination
        except (OSError, ValueError, KeyError):
            pass
    from analysis.cohorts.libero import TABLES, episode_rows
    source = TABLES['pi0.5']
    sources = {portable(source): sha(source)}
    rows = sync_hold.trace_rows('pi0.5', 'basket', [r for r in episode_rows('pi0.5') if r['task'] == 'basket' and float(r['speed']) in SPEEDS])
    groups = defaultdict(list)
    for row in rows:
        if row['error'] or row['trace_local'] != '1':
            raise ValueError('Incomplete basket trace cohort')
        groups[float(row['speed']), row['mode'], int(row['latency']), row['source']].append(row)
    if len(rows) != 14400 or len(groups) != 72 or any(len(g) != 200 for g in groups.values()):
        raise ValueError('Expected 72 basket conditions with 200 episodes each')
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(point, job) for job in groups.items()]
        for index, future in enumerate(as_completed(pending), 1):
            _, values, dependencies = future.result()
            results.extend(values)
            for name, checksum in dependencies.items():
                if sources.setdefault(name, checksum) != checksum:
                    raise ValueError(f'Source changed during measurement: {name}')
            if index % 8 == 0 or index == len(pending):
                print(f'basket-alignment: measured {index}/{len(pending)} conditions', flush=True)
    results.sort(key=lambda r: (r['target_speed_mps'], r['mode'], r['delay_ms'], r['episode_id']))
    if len({(r['target_speed_mps'], r['mode'], r['delay_ms'], r['episode_id']) for r in results}) != 14400:
        raise ValueError('Duplicate or missing episode identity')
    tables = {'episodes.csv': results}
    for filename, keys in [('by_speed.csv', ('target_speed_mps','mode','delay_ms')),
                           ('pooled.csv', ('mode','delay_ms'))]:
        grouped = defaultdict(list)
        for row in results:
            grouped[tuple(row[k] for k in keys)].append(row)
        tables[filename] = [dict(zip(keys, key), **summarize(values)) for key, values in sorted(grouped.items())]
    # Check inputs again before publishing a result from a long trace read.
    if any(sha(resolve(name)) != value for name, value in sources.items()) or code_hashes() != code:
        raise ValueError('Inputs or calculation changed while measuring; rerun')
    import tempfile
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        staging = Path(temporary)
        for name, table in tables.items():
            with (staging / name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
        provenance = dict(definition=DEFINITION, episodes=len(results), code=code,
                          environment=runtime, sources=sources,
                          outputs={name: sha(staging/name) for name in OUTPUT_NAMES})
        (staging/'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True)+'\n')
        for name in (*OUTPUT_NAMES, 'provenance.json'):
            (staging/name).replace(destination/name)
    print(f'basket-alignment: wrote {len(results)} episodes to {destination}', flush=True)
    return destination


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    ensure(workers=args.workers, force=args.force)
