"""Side views of recorded release, set-down, and grasp attempts at the sample where the tolerance-zone offset is taken.

A release is shown one sample after it, once the fingers have visibly opened, from the camera framed at the measured sample.

Each view restores one recorded MuJoCo state (positions only, no simulation steps) from a hash-verified pi0.5 trace
and renders it from a level free camera (zero pitch) at the grip site's height, looking perpendicular to the measured
offset so that the offset runs across the image. The offset is recomputed from the restored state and must equal
the measurement's attempts table. --orthographic renders a level parallel projection instead, in which offsets read
to scale across the image.

Traces are read from benchmark artifact storage (VLALAGBENCH_ARTIFACTS) and located through the full measurement
outputs under ANALYSIS_DERIVED (basket-alignment, plate-cabinet-episodes); the LIBERO assets come from
VLALAGBENCH_CHECKOUT.

    python -m analysis.figures.stage_views --candidates
    python -m analysis.figures.stage_views [--orthographic] [--output DIR]
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from analysis.cohorts.plate_cabinet.data import EPISODE_FOLDER
from analysis.figures.tasks import attempt_rows
from analysis.paths import DERIVED, localize_xml, output, read_rows, trace_path

# Candidates come from one condition, the one the figure's target-travel bar describes.
CONDITION = dict(speed='0.1', mode='synchronous', latency='200')
# Offset (cm) a representative attempt should have: mid-zone for a hit, just outside the zone for a miss.
TYPICAL = {('basket', 1): 3.5, ('basket', 0): 8.5, ('plate', 1): 1.5, ('plate', 0): 4.5,
           ('cabinet', 1): 4.0, ('cabinet', 0): 7.5}
# Rendered attempts, chosen from --candidates by looking at the renders: (task, hit) -> episode and attempt index k.
# "side" -1 moves the camera to the robot's side of the scene (default: the far side, looking toward the robot).
# A view with its own "trace" (and sha256) is measured from that trace instead of the attempts table. The release and
# set-down views keep the pre-rerun synchronous 200 ms episodes, whose renders face the offset better; the zero-action
# waits the reruns fixed change how the arm reaches the target, not where an attempt lands against the zone.
PRE_RERUN = {
    'basket': 'artifacts/studies/xtra_startfix_20260908/native_ellipse_600_xtra_20260908_sync/points/0.1_synchronous_200/'
              'artifacts/20260909T161016Z-e46c8b56-0.1_synchronous_200/libero_object_task01_ep{:03d}.npz',
    'plate': 'artifacts/studies/libero_spatial_task9_pi05_b32_h100_20261003/pi05_task9_plate_synchronous_b32_20261003/'
             'points/0.1_synchronous_200/artifacts/20261003T122004Z-4e2c53f3-0.1_synchronous_200/libero_spatial_task09_ep{:03d}.npz',
}


def pre_rerun(task, episode_id, k, sha256):
    return dict(episode_id=episode_id, k=k, trace=PRE_RERUN[task].format(episode_id), sha256=sha256)


VIEWS = {
    ('basket', 1): pre_rerun('basket', 61, 0, '5f4f9f8649e23ac84a07b6114137953056b065331c64dacbcd1d7f22fdd3cc96'),
    ('basket', 0): pre_rerun('basket', 114, 0, '4ab2dd217e3a3c9c82b250d419569e431fbfb1f647dfe26c12d80a384b767879'),
    ('plate', 1): pre_rerun('plate', 9, 0, '0702e1aced37d125cfc1863998e72dcf94d7ff50f466b73bd4e86f9a8fed63d1'),
    ('plate', 0): pre_rerun('plate', 56, 8, '685c55a2b470052c57d540e9c60930290e3e84ee9e53d1e0cd0ccb53e255afd5'),
    ('cabinet', 1): dict(episode_id=82, k=0),
    ('cabinet', 0): dict(episode_id=94, k=0),
}
# Per task, shared by its hit and miss: camera distance (m); for the level perspective camera, the image row (fraction
# from the top) where its horizon, the grip site's height, falls, set by rendering a taller frame and cropping it, as
# a shifted lens would; for the orthographic view, the image height (m) and how far its center sits above the midpoint
# of the grip site and target center (m). The orthographic view sits far enough back to keep the table unclipped.
ORTHOGRAPHIC_DISTANCE = 1.2
FRAMING = {'basket': dict(distance=.6, horizon=.3, height=.32, lift=.04),
           'plate': dict(distance=.5, horizon=.45, height=.3, lift=.04),
           'cabinet': dict(distance=.6, horizon=.4, height=.3)}
WIDTH, HEIGHT, FOVY = 960, 720, 30.
# Visual geoms only; group 0 holds collision geoms, which would be drawn over the meshes.
GROUPS = (0, 1, 0, 0, 0, 0)


def trace_rows(task):
    """Episode -> (trace, sha256) from the verified cohort tables the measurements replayed."""
    if task == 'basket':
        return {(r['target_speed_mps'], r['mode'], r['delay_ms'], r['episode_id']): (r['trace'], r['trace_sha256'])
                for r in read_rows(DERIVED / 'basket_alignment_speed/episodes.csv') if r['task'] == 'basket'}
    return {(r['speed'], r['mode'], r['latency'], r['episode_id']): (r['trace'], r['trace_sha256'])
            for r in read_rows(EPISODE_FOLDER / 'episodes.csv') if r['model'] == 'pi0.5' and r['task'] == task}


def attempts(task):
    return [r for r in attempt_rows('pi0.5', task) if all(r[k] == v for k, v in CONDITION.items())]


def own_attempts(task, view):
    """Attempts measured from a view's own trace by the code that builds the attempts tables."""
    trace, digest = view['trace'], view['sha256']
    with np.load(trace_path(trace), allow_pickle=True) as z:
        meta, acting = json.loads(str(z['meta'])), z['acting'].astype(bool)
    if abs(float(meta['trajectory']['speed_mps']) - float(CONDITION['speed'])) > 1e-12:
        raise ValueError(f'{trace} is not at {CONDITION["speed"]} m/s')
    row = dict(task=task, model='pi0.5', speed=CONDITION['speed'], mode=CONDITION['mode'],
               latency=CONDITION['latency'], episode_id=str(meta['episode_id']), success=str(int(meta['success'])),
               trace=trace, trace_sha256=digest)
    if task == 'basket':
        from analysis.cohorts.basket_releases import episode
        rows = episode(dict(row, target_speed_mps=row['speed'], delay_ms=row['latency']))
    else:
        from analysis.cohorts.plate_cabinet.replay import episode
        rows = episode(dict(row, control_steps=len(acting) - int(meta['num_steps_wait']), steps=int(acting.sum())))[2]
    return [{k: str(v) for k, v in r.items()} for r in rows]


def candidates(count=6):
    for (task, hit), typical in TYPICAL.items():
        rows = [r for r in attempts(task) if int(r['hit']) == hit]
        rows.sort(key=lambda r: abs(100 * float(r['offset_m']) - typical))
        print(f'{task} hit={hit} (near {typical} cm):')
        for r in rows[:count]:
            print(f"  episode {r['episode_id']} k={r['k']} step={r['step']} offset={100 * float(r['offset_m']):.2f} cm "
                  f"success={r['success']}")


def load(trace, digest):
    blob = trace_path(trace).read_bytes()
    if hashlib.sha256(blob).hexdigest() != digest:
        raise ValueError(f'Trace hash mismatch: {trace}')
    with np.load(io.BytesIO(blob), allow_pickle=True) as z:
        return z['states'], str(z['model_xml']), json.loads(str(z['meta']))


def points(task, d):
    """Offset endpoints (object point, target center) and the grip site in the current pose, as the measurements define them."""
    grip = d.site('gripper0_grip_site').xpos.copy()
    if task == 'basket':
        return d.body('cream_cheese_1_main').xpos.copy(), d.site('basket_1_contain_region').xpos.copy(), grip
    if task == 'plate':
        return d.body('akita_black_bowl_1_main').xpos.copy(), d.body('plate_1_main').xpos.copy(), grip
    return grip, d.body('akita_black_bowl_1_main').xpos.copy(), grip


def rounded(values, digits=4):
    return [round(float(v), digits) for v in values]


def render(task, hit, view, traces, folder, projection='perspective'):
    import mujoco
    from PIL import Image
    table = own_attempts(task, view) if 'trace' in view else attempts(task)
    rows = [r for r in table if r['episode_id'] == str(view['episode_id']) and r['k'] == str(view['k'])]
    if len(rows) != 1 or int(rows[0]['hit']) != hit:
        raise ValueError(f'No unique {task} attempt {view} with hit={hit}')
    row = rows[0]
    trace, digest = ((view['trace'], view['sha256']) if 'trace' in view
                     else traces[CONDITION['speed'], CONDITION['mode'], CONDITION['latency'], row['episode_id']])
    states, xml, meta = load(trace, digest)
    if int(meta['episode_id']) != int(row['episode_id']) or bool(meta['success']) != bool(int(row['success'])):
        raise ValueError(f'Trace disagrees with the attempts table: {trace}')
    # The basket offset is taken in the last sample with finger contact, one before the release; the release is
    # shown one sample after it, when the fingers have visibly opened.
    sample = int(row['step']) - (task == 'basket')
    shown = int(row['step']) + 1 if task == 'basket' else sample
    m = mujoco.MjModel.from_xml_string(localize_xml(xml))
    d = mujoco.MjData(m)
    d.qpos[:] = states[sample, 1:1 + m.nq]
    mujoco.mj_forward(m, d)
    obj, center, grip = points(task, d)
    offset = obj[:2] - center[:2]
    if abs(np.linalg.norm(offset) - float(row['offset_m'])) > 1e-9:
        raise ValueError(f'Replayed offset {np.linalg.norm(offset)} differs from the table {row["offset_m"]}')
    across = offset / np.linalg.norm(offset)
    look = np.array([-across[1], across[0]])
    middle = (obj[:2] + center[:2]) / 2
    if np.dot(look, d.body('robot0_base').xpos[:2] - middle) * view.get('side', 1) < 0:
        look = -look
    frame = FRAMING[task]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = frame['distance']
    camera.azimuth = math.degrees(math.atan2(look[1], look[0]))
    camera.elevation = 0.
    if projection == 'perspective':
        # Render a taller frame about the horizon and crop it, which keeps the camera level at the grip site's height.
        camera.lookat[:] = [*middle, grip[2]]
        above = round(frame['horizon'] * HEIGHT)
        tall = 2 * max(above, HEIGHT - above)
        focal = HEIGHT / 2 / math.tan(math.radians(FOVY) / 2)
        m.vis.global_.fovy = 2 * math.degrees(math.atan(tall / 2 / focal))
        top = tall // 2 - above
    else:
        # A parallel projection is level at every height; frame the grip site and the target center together.
        camera.lookat[:] = [*middle, (grip[2] + center[2]) / 2 + frame.get('lift', 0)]
        camera.distance = ORTHOGRAPHIC_DISTANCE
        m.vis.global_.orthographic = 1
        m.vis.global_.fovy = frame['height']
        tall, top = HEIGHT, 0
    m.vis.global_.offwidth, m.vis.global_.offheight = max(WIDTH, m.vis.global_.offwidth), max(tall, m.vis.global_.offheight)
    option = mujoco.MjvOption()
    option.geomgroup[:] = GROUPS
    d.qpos[:] = states[shown, 1:1 + m.nq]
    mujoco.mj_forward(m, d)
    with mujoco.Renderer(m, tall, WIDTH) as renderer:
        renderer.update_scene(d, camera, option)
        pixels = renderer.render()[top:top + HEIGHT]
    name = f"{task}-{'hit' if hit else 'miss'}{'-orthographic' if projection == 'orthographic' else ''}.png"
    Image.fromarray(pixels).save(folder / name)
    forward = np.array([look[0], look[1], 0.])
    lens = (dict(vertical_fov_deg=FOVY, horizon_row_fraction=frame['horizon']) if projection == 'perspective'
            else dict(image_height_m=frame['height']))
    return dict(image=name, task=task, outcome='hit' if hit else 'miss', policy='pi0.5', **CONDITION,
                episode_id=int(row['episode_id']), attempt=int(row['k']), attempt_step=int(row['step']),
                measured_sample=sample, rendered_sample=shown, episode_success=int(row['success']), offset_cm=round(100 * float(row['offset_m']), 3),
                offset_direction_xy=rounded(across), object_xyz=rounded(obj), target_center_xyz=rounded(center),
                grip_site_xyz=rounded(grip), trace=trace, trace_sha256=digest,
                camera=dict(projection=projection, position=rounded(np.array(camera.lookat) - camera.distance * forward),
                            lookat=rounded(camera.lookat), forward=rounded(forward), distance=camera.distance,
                            azimuth_deg=round(camera.azimuth, 2), elevation_deg=0., width=WIDTH, height=HEIGHT, **lens))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--candidates', action='store_true', help='List representative attempts per task and outcome')
    parser.add_argument('--orthographic', action='store_true', help='Level parallel projection instead of a camera')
    parser.add_argument('--output', type=Path, help='Folder for the PNGs and views.json')
    args = parser.parse_args()
    if args.candidates:
        candidates()
        return
    folder = args.output or output('tolerance_zone/stage_views')
    folder.mkdir(parents=True, exist_ok=True)
    projection = 'orthographic' if args.orthographic else 'perspective'
    records = []
    for (task, hit), view in VIEWS.items():
        records.append(render(task, hit, view, trace_rows(task), folder, projection))
        print(f"{records[-1]['image']}: episode {view['episode_id']} attempt {view['k']}, "
              f"{records[-1]['offset_cm']:.2f} cm", flush=True)
    (folder / f'views-{projection}.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
