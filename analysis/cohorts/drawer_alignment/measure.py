"""Reconstruct near-handle, non-contact drawer intervals from read-only pi0.5 traces."""
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
from analysis.paths import (ROOT, ARTIFACTS, DERIVED, localize_xml, point_dir, portable, resolve, sha,
                            use_benchmark_src)
from analysis import sync_hold
BENCHMARK_SRC = Path(use_benchmark_src())
CACHE = {}


def code_hashes():
    return {portable(p): sha(p) for p in (
        Path(__file__), Path(__file__).with_name('data.py'),
        ROOT/'analysis/paths.py', ROOT/'analysis/sync_hold.py',
        *(BENCHMARK_SRC/'vlalagbench'/name for name in (
            'simulators/mujoco/dynamics.py', 'config.py', 'trajectory.py',
            'timing.py', 'files.py', 'benchmarks/libero/evaluator.py',
        )),
    )}


def environment():
    import mujoco
    import yaml
    import pydantic
    return dict(python=sys.version.split()[0], numpy=np.__version__, mujoco=mujoco.__version__, pyyaml=yaml.__version__, pydantic=pydantic.__version__)


def point(job):
    """Reconstruct every state, including later realignment, without stepping physics."""
    import mujoco
    import yaml
    import xml.etree.ElementTree as ET
    from vlalagbench.config import DynamicActor
    from vlalagbench.trajectory import motion_path as _motion_path, offset_and_velocity

    key, rows = job
    speed, mode, latency, source = key
    # Synchronous points at a nonzero latency come from the hold_pose reruns (analysis/sync_hold.py).
    p = sync_hold.point_dir('pi0.5', 'drawer', speed, latency) if source == sync_hold.SOURCE else Path(point_dir(source, speed, mode, latency))
    run = yaml.safe_load((p / 'point.yaml').read_text())['run_id']
    experiment = p / 'runs' / run / 'experiment.yaml'
    raw = yaml.safe_load(experiment.read_text())
    assert raw['model']['checkpoint'] == 'gs://openpi-assets/checkpoints/pi05_libero'
    files = {int(f.stem.rsplit('_ep', 1)[1]): f for f in (p / 'artifacts' / run).glob('*.npz')}
    assert len(files) == 200
    out = []
    dependencies = {portable(f): sha(f) for f in (p / 'point.yaml', experiment)}
    if source == sync_hold.SOURCE:
        jsonl = p / 'artifacts' / run / 'episodes.jsonl'
        dependencies[portable(jsonl)] = sha(jsonl)
    for row in rows:
        f = files[int(row['episode_id'])]
        blob = f.read_bytes()
        trace_hash = hashlib.sha256(blob).hexdigest()
        with np.load(io.BytesIO(blob), allow_pickle=True) as z:
            st, acting = z['states'], z['acting'].astype(bool)
            xml, meta = str(z['model_xml']), json.loads(str(z['meta']))
        warm, hz, n = int(meta['num_steps_wait']), float(meta['control_freq']), len(acting)
        assert warm == 10 and hz == 20 and not acting[:warm].any() and len(st) == n + 1
        assert np.allclose(np.diff(st[:, 0]), 1 / hz, atol=1e-7)
        assert n - warm == int(row['control_steps']) and int(acting.sum()) == int(row['steps'])
        assert meta['episode_id'] == int(row['episode_id'])
        assert bool(meta['success']) == bool(int(row['success'])) and not meta['error']
        assert meta['checkpoint'] == raw['model']['checkpoint']
        assert meta['dynamic_step_zero_prepared_before_warmup']
        assert len(meta['dynamics']) == 1
        actor = DynamicActor.model_validate(meta['dynamics'][0])
        assert actor.target == 'wooden_cabinet_1_main' and actor.motion.speed_mps == speed
        assert actor.motion.initial_phase == float(row['initial_phase'])
        assert actor.motion.initial_direction == int(row['direction'])
        path = _motion_path(actor)
        dependencies[portable(f)] = trace_hash
        h = hashlib.sha256(xml.encode()).hexdigest()
        if h not in CACHE:
            localized = localize_xml(xml)
            assets = {}
            for element in ET.fromstring(localized).iter():
                if 'file' in element.attrib:
                    asset = Path(element.attrib['file'])
                    if not asset.is_absolute():
                        raise ValueError(f'Unresolved model asset: {asset}')
                    assets[portable(asset)] = sha(asset)
            m = mujoco.MjModel.from_xml_string(localized)
            d = mujoco.MjData(m)
            main = m.body('wooden_cabinet_1_main').id
            middle = m.body('wooden_cabinet_1_cabinet_middle').id
            # Same bar geometry criterion as the benchmark contact probe: a
            # middle-body collision box with 4.4 cm half-width, foremost in -y.
            bars = [i for i in range(m.ngeom) if m.geom_bodyid[i] == middle
                    and m.geom_type[i] == 6 and .04 < m.geom_size[i].max() < .05]
            if not bars:
                raise ValueError('Missing middle-drawer handle collision bar')
            bar = min(bars, key=lambda i: m.geom_pos[i, 1])
            fingers = {i for i in range(m.ngeom)
                       if m.geom(i).name.startswith(('gripper0_finger1', 'gripper0_finger2'))}
            assert fingers
            grip = m.site('gripper0_grip_site').id
            CACHE[h] = m, d, main, bar, fingers, grip, assets
        m, d, main, bar, fingers, grip, assets = CACHE[h]
        dependencies.update(assets)
        override = meta['initial_state_override']
        assert override['initial_state_index'] == int(row['base_state'])
        assert not override.get('object_yaws_rad')
        base = np.asarray(override['object_positions_m']['wooden_cabinet_1_main'])
        ee = np.zeros((n + 1, 3))
        distance = np.zeros(n + 1)
        contact = np.zeros(n + 1, dtype=bool)
        for i in range(warm, n + 1):
            # Cabinet motion changes model.body_pos, which is not stored in
            # qpos. Recover its exact dynamic time from the saved actor.
            offset, _ = offset_and_velocity(actor, i - warm, hz, path)
            m.body_pos[main] = base + offset * np.array([1, 1, 0])
            d.qpos[:] = st[i, 1:1 + m.nq]
            mujoco.mj_kinematics(m, d)
            ee[i] = d.site_xpos[grip]
            distance[i] = np.linalg.norm(ee[i] - d.geom_xpos[bar])
            if distance[i] > DEFINITION['radius_m']:
                continue
            mujoco.mj_collision(m, d)
            contact[i] = any(
                c.dist <= 0 and ((c.geom1 == bar and c.geom2 in fingers)
                                or (c.geom2 == bar and c.geom1 in fingers))
                for c in d.contact
            )
        out.append(dict(
            task='drawer', target_speed_mps=speed, mode=mode, delay_ms=latency,
            episode_id=int(row['episode_id']), success=int(row['success']),
            **reduce_intervals(ee, st[:, 0], acting, contact, distance, warm),
            handle_geom=m.geom(bar).name, trace='artifacts/' + str(f.relative_to(ARTIFACTS)), trace_sha256=trace_hash,
        ))
    return key, out, dependencies


def ensure(*, workers=4, force=False):
    """Validate all upstream hashes; rebuild when calculation, sources, or outputs change."""
    if workers < 1:
        raise ValueError('workers must be positive')
    destination = DERIVED / 'drawer_alignment_speed'
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
                print('drawer-alignment: source/code/output hashes verified; using cached measurements', flush=True)
                return destination
        except (OSError, ValueError, KeyError):
            pass
    from analysis.cohorts.libero import TABLES, episode_rows
    source = TABLES['pi0.5']
    sources = {portable(source): sha(source)}
    rows = sync_hold.trace_rows('pi0.5', 'drawer', [r for r in episode_rows('pi0.5') if r['task'] == 'drawer' and float(r['speed']) in SPEEDS])
    groups = defaultdict(list)
    for row in rows:
        if row['error'] or row['trace_local'] != '1':
            raise ValueError('Incomplete drawer trace cohort')
        groups[float(row['speed']), row['mode'], int(row['latency']), row['source']].append(row)
    if len(rows) != 14400 or len(groups) != 72 or any(len(g) != 200 for g in groups.values()):
        raise ValueError('Expected 72 drawer conditions with 200 episodes each')
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
                print(f'drawer-alignment: measured {index}/{len(pending)} conditions', flush=True)
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
    print(f'drawer-alignment: wrote {len(results)} episodes to {destination}', flush=True)
    return destination


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    ensure(workers=args.workers, force=args.force)
