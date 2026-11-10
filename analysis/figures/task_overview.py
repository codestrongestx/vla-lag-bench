"""Show recorded task scenes beside the configured planar target trajectories.

Reads, for each task, one recorded episode's video, trace, and run configuration from benchmark artifact storage
(VLALAGBENCH_ARTIFACTS; the files are listed in SOURCES and DYNAMICVLA_SOURCES with their SHA-256), and the LIBERO
assets its MuJoCo model references (VLALAGBENCH_CHECKOUT/third_party). Target paths come from this repository's
vlalagbench trajectory code.
"""
from analysis.paths import ARTIFACTS, CONFIGS, figure_path, localize_xml, use_benchmark_src

use_benchmark_src()
import argparse
import hashlib
import io
import json
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mujoco
import numpy as np
import yaml
from vlalagbench.config import ClosedPathTrajectory, DynamicActor, Experiment, load_experiment
from vlalagbench.trajectory import sample_trajectory
from vlalagbench.trajectory import (motion_path as _motion_path, _constant_speed_offset_and_velocity,
                                  motion_path_length, offset_and_velocity)
from analysis.cohorts.dynamicvla.replay import rotation
from analysis.figures.figure_kit import save, style, subpanel_title
from analysis.figures.style import INK2, TARGET, TARGET_BOX
from analysis.figures.typography import FULL_WIDTH, PAPER, RC, TEXT_WIDTH



# "video" is relative to the point directory.
# "boxed": bodies whose visual meshes the target box encloses in both recorded frames.
SOURCES = {
    "basket": {
        "title": "(a) Release into a moving receptacle",
        "point": "studies/xtra_startfix_20260908/native_ellipse_600_xtra_20260908_sync/points/0.1_synchronous_0",
        "run": "20260909T155239Z-5152031f-0.1_synchronous_0",
        "video": "videos/task01_ep026_success_agentview.mp4",
        "sha256": "5e2f28b0366252a9c76c32dcd9d297a23a3ac5b421763918874dd81313d56b3d",
        "trace": "libero_object_task01_ep026.npz",
        "trace_sha256": "9f3925e71f2dba09fdc417e1f12be3d09f7de4518aec3febf4d727887ee6b634",
        "boxed": ("basket_1_main",),
        "frames": (10, 133),
        "label": "Target path",
    },
    "plate": {
        "title": "(b) Set-down on a moving surface",
        "point": "studies/libero_spatial_task9_pi05_b32_h100_20261003/pi05_task9_plate_synchronous_b32_20261003/points/0.1_synchronous_0",
        "run": "20261003T120915Z-0dd0f679-0.1_synchronous_0",
        "video": "videos/task09_ep081_success_agentview.mp4",
        "sha256": "965ea307331b64a52ec5f64cc00d50ab593add818cd3ee3438baab526755d6a0",
        "trace": "libero_spatial_task09_ep081.npz",
        "trace_sha256": "bc44807262ae60009f0dfc58cc1dc377d2f553ad29b4a94c477eaa048b390ced",
        "boxed": ("plate_1_main",),
        "frames": (10, 128),
        "label": "Target path",
    },
    "cabinet": {
        "title": "(c) Moving-object grasping",
        "point": "studies/libero_spatial_task9_bowl_centered_20261010/pi05_task9_cabinet_bowl_centered_synchronous_b32_20261010/points/0.1_synchronous_0",
        "run": "20261010T122616Z-dfbb1ef8-0.1_synchronous_0",
        "video": "videos/task09_ep040_success_agentview.mp4",
        "sha256": "ea2bcdfa931add65feadf6fc2d24265583ea1393f4b42ee5ca2541272d2b94e1",
        "trace": "libero_spatial_task09_ep040.npz",
        "trace_sha256": "9380371b6bb73dde6367dfa8d2b9a9c1fd1e3a178bb787b72c6e87947855c932",
        # The cabinet carries the bowl, placed at the center of its top, until pickup.
        "boxed": ("wooden_cabinet_1_main", "akita_black_bowl_1_main"),
        "frames": (10, 80),
        "label": "Support path",
    },
    "drawer": {
        "title": "(d) Articulated-object manipulation",
        "point": "studies/pi05_drawer_fixed_20260926/pi05_drawer_fixed_20260926_sync/points/0.05_synchronous_0",
        "run": "20260926T194236Z-7caabe2b-0.05_synchronous_0",
        "video": "videos/task00_ep009_success_agentview.mp4",
        "sha256": "962e9b0d1d86717f9cc5dd7a85e71687fa42c21feb82e0c935e4734c3bf58554",
        "trace": "libero_goal_task00_ep009.npz",
        "trace_sha256": "21463bd5fc4b57a3db3cb07bba76c444ebb4d23646faf08d6bfca7dbde4c9094",
        "boxed": ("wooden_cabinet_1_main",),
        "frames": (10, 123),
        "label": "Target path",
    },
}

# Both rows show the opst_cam view and draw robot-base paths at one scale, so path sizes compare
# directly. The moving target takes the same highlight color as in the LIBERO rows.
DYNAMICVLA_PATH = {"control_hz": 25, "frame": "Base", "limits": ((-21, 21), (-12.5, 12.5)),
                   "ticks": ((-20, -10, 0, 10, 20), (-10, 0, 10))}
DYNAMICVLA_SOURCES = {
    "food_can": {
        "title": "(a) Release into a moving receptacle",
        "point": "studies/fcan04_native_remaining180_20260928/fcan04-native-remaining180-750steps-synchronous/points/0.1_0",
        "run": "20260930T072358Z-206ac05f-0.1_0",
        "video": "videos/task00_ep000_success_opst_cam.mp4",
        "sha256": "f8e107b672c42b5b87175bd1677ed4e839aff23a08d15eb10be1a4b4dec2c683",
        "frames": (5, 188),
        # Pixel extent (x0, x1, y0, y1) of the moving box in each frame, read from the frames.
        "boxes": ((66, 170, 205, 294), (85, 170, 182, 248)),
        # The episode's trace; its boxed object's recorded path places the loop drawn on the frames.
        "trace": "dynamicvla_task00_ep000.npz",
        "trace_sha256": "934d399e1e9839c013da11e89f08510a06a36addbb035c31f85fdbe54fcd4a91",
        "body": "container",
        "label": "Target path",
        **DYNAMICVLA_PATH,
    },
    "moving_can": {
        "title": "(b) Moving-object grasping",
        "point": "studies/fcan04_moving_can_pilot20_20261008_xtra3090/fcan04-moving-can-ribbed-random20-750steps-synchronous/points/0.1_0",
        "run": "20261010T132303Z-08fb3233-0.1_0",
        # Episode 2, recorded live; the invisible carrier holds the can until pickup.
        "video": "artifacts/20261010T132303Z-08fb3233-0.1_0/evaluation/dynamic-vla-DOM/0000/task00_env000_replica002.mp4",
        "sha256": "cde4f8ba8ef96a946057b70c1d5b42ecf44b7aa02a9a22fd8641eb161b70020c",
        # Live recordings tile the cameras left to right in the run's recorded order.
        "cameras": ("wrist_cam", "opst_cam", "side_cam"),
        "camera": "opst_cam",
        "frames": (5, 64),
        # Pixel extent of the moving can in each frame; the carrier is not rendered.
        "boxes": ((214, 252, 231, 261), (249, 280, 204, 228)),
        "trace": "dynamicvla_task00_ep002.npz",
        "trace_sha256": "43fdace4fce5c1c94d31082b0f739b1b5b8cecbf0a293ece6e171979b33567b6",
        "body": "object",
        "label": "Support path",
        **DYNAMICVLA_PATH,
    },
}

# Every DynamicVLA run renders through DOM's opst_cam, whose pose relative to the robot base and pinhole optics are
# recorded in the moving-can environment configuration; the native moving-box scene uses the same camera, which the
# frame check in dynamicvla_loop confirms against the boxes read from its frames.
OPST_CONFIG = CONFIGS / "experiments/dynamicvla/fcan04_moving_can_20261006/fcan04-moving-can-ribbed.json"

# Path sample pair (of 721 around the loop) drawn as the path panel's one arrow: episodes travel both ways, and
# the arrow shows that the target moves along the loop. Frames draw the loop without arrows.
ARROW = (95, 120)

# Printed layout, in inches.
GAP, YLABEL, ROW_TITLE, ROW_GAP, PATH_BELOW, PATH_TITLE = .06, .45, .2, .08, .3, .17


def frame(video, index):
    capture = cv2.VideoCapture(str(video))
    try:
        for _ in range(index + 1):
            ok, pixels = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {index}: {video}")
        return cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def camera_view(pixels, spec):
    """The source camera's tile of a frame; a single-camera video is returned whole."""
    if "camera" not in spec:
        return pixels
    count = len(spec["cameras"])
    if pixels.shape[1] % count:
        raise ValueError(f"Frame width {pixels.shape[1]} does not split into {count} cameras")
    width = pixels.shape[1] // count
    left = spec["cameras"].index(spec["camera"]) * width
    return pixels[:, left:left + width]


def experiment(path):
    """Load a run configuration; keys added after the pinned revision do not affect the path."""
    data = yaml.safe_load(path.read_text())
    if data.get("benchmark", {}).get("kind") == "dynamicvla":
        # The pinned revision predates the DOM schema; the path uses the shared trajectory model.
        # carried_objects only moves the can onto the carrier's start pose before the first step.
        data["trajectory"].pop("carried_objects", None)
        return ClosedPathTrajectory.model_validate(data["trajectory"])
    if data["trajectory"].get("target") in ("plate_1_main",) or "carried_objects" in data["trajectory"] \
            or "slide_joints" in data["trajectory"]:
        for key in ("slide_joints", "carried_objects"):
            data["trajectory"].pop(key, None)
        return Experiment.model_validate(data)
    return load_experiment(path)


def target_boxes(spec, point, shape):
    """Image extent of the boxed bodies' visual meshes in each recorded frame, from the episode trace.

    Video frame i is trace state i rendered through the agentview camera and flipped upright, so the
    meshes posed by that state and projected through the same camera cover the target's pixels.
    """
    return target_views(spec, point, shape, trajectory=False)[0]


def target_views(spec, point, shape, *, trajectory=True):
    """The boxed bodies' image extent in each recorded frame, and the driven target's closed path in the image.

    The path is the one the path panel draws (same parameterization, so the same arrow positions), placed in the
    world where the episode's trace puts the target's center at the interaction frame, and projected through the
    agentview camera.
    """
    blob = (point / "artifacts" / spec["run"] / spec["trace"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != spec["trace_sha256"]:
        raise ValueError(f"Recorded trace changed: {spec['trace']}")
    with np.load(io.BytesIO(blob), allow_pickle=True) as z:
        states, xml, meta = z["states"], str(z["model_xml"]), json.loads(str(z["meta"]))
    model = mujoco.MjModel.from_xml_string(localize_xml(xml))
    data = mujoco.MjData(model)
    (actor,) = meta["dynamics"]
    driven = model.body(actor["target"]).id
    jointless = model.body_jntnum[driven] == 0
    # Keys added after the pinned revision (such as carried objects) do not affect the target's own path.
    actor = DynamicActor.model_validate({k: v for k, v in actor.items() if k in DynamicActor.model_fields})
    path = _motion_path(actor)
    hz, wait = float(meta["control_freq"]), int(meta["num_steps_wait"])
    if jointless:
        # Motion of a jointless body lives in model.body_pos, which the flattened state omits;
        # recover it from the saved actor at the frame's dynamic time.
        origin = np.asarray(meta["initial_state_override"]["object_positions_m"][actor.target])

    def subtree(roots):
        def inside(body):
            while body:
                if body in roots:
                    return True
                body = model.body_parentid[body]
            return False
        return [g for g in range(model.ngeom) if model.geom_group[g] == 1 and inside(model.geom_bodyid[g])]
    geoms = subtree({model.body(name).id for name in spec["boxed"]})
    if not geoms or any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH for g in geoms):
        raise ValueError(f"Expected visual meshes for {spec['boxed']}")
    target_geoms = subtree({driven})
    camera = model.cam("agentview").id
    height, width = shape
    focal = .5 * height / np.tan(np.deg2rad(model.cam_fovy[camera]) / 2)

    def pose(index):
        if jointless:
            offset, _ = offset_and_velocity(actor, max(0, index - wait), hz, path)
            model.body_pos[driven] = origin + offset * np.array([1, 1, 0])
        data.qpos[:] = states[index, 1:1 + model.nq]
        mujoco.mj_forward(model, data)

    def vertices(selected):
        points = []
        for g in selected:
            mesh = model.geom_dataid[g]
            local = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            points.append(data.geom_xpos[g] + local @ data.geom_xmat[g].reshape(3, 3).T)
        return np.concatenate(points)

    def project(points):
        # MuJoCo cameras look along -z with +y up; image rows grow downward.
        local = (np.asarray(points) - data.cam_xpos[camera]) @ data.cam_xmat[camera].reshape(3, 3)
        u = width / 2 + focal * local[:, 0] / -local[:, 2] - .5
        v = height / 2 - focal * local[:, 1] / -local[:, 2] - .5
        return np.stack([u, v], axis=1)

    boxes = []
    for index in spec["frames"]:
        pose(index)
        u, v = project(vertices(geoms)).T
        boxes.append((u.min(), u.max(), v.min(), v.max()))
    if not trajectory:
        return boxes, None
    # The path is fixed in the world: the target's center at the interaction frame, less its configured offset
    # at that frame, plus the offsets around the loop.
    pose(spec["frames"][1])
    points = vertices(target_geoms)
    anchor = .5 * (points.min(0) + points.max(0)) \
        - offset_and_velocity(actor, spec["frames"][1] - wait, hz, path)[0] * np.array([1, 1, 0])
    loop = actor.model_copy(update={"motion": actor.motion.model_copy(
        update={"initial_phase": 0., "initial_direction": 1})})
    period = float(path[2][-1]) / actor.motion.speed_mps
    offsets = np.array([_constant_speed_offset_and_velocity(loop, t * hz, hz, path)[0]
                        for t in np.linspace(0, period, 721)])
    return boxes, project(anchor + offsets * np.array([1, 1, 0]))


def opst_project(points):
    """Pixel positions of robot-base points in opst_cam (OpenGL convention: the camera looks along -z, +y up)."""
    camera = json.loads(OPST_CONFIG.read_text())["scene"]["opst_cam"]
    offset, optics = camera["offset"], camera["spawn"]
    if offset["convention"] != "opengl" or optics["projection_type"] != "pinhole":
        raise ValueError("opst_cam is no longer an OpenGL pinhole camera")
    width, height = camera["width"], camera["height"]
    focal = optics["focal_length"] / optics["horizontal_aperture"] * width
    local = (np.asarray(points) - offset["pos"]) @ rotation(offset["rot"])
    u = width / 2 + focal * local[:, 0] / -local[:, 2] - .5
    v = height / 2 - focal * local[:, 1] / -local[:, 2] - .5
    return np.stack([u, v], axis=1)


def dynamicvla_loop(spec, point):
    """The boxed object's closed path in the opst_cam frames, from the episode trace.

    Video frame i shows trace state i. The driven body's recorded robot-base positions, less the actor's configured
    offsets, place the loop; a body driven by velocity trails its written path by a step, so the lag whose
    placement is most consistent is used. The loop is drawn at the boxed object's start height and position on its
    support, and the boxed object's projected center must fall inside the box read from each frame.
    """
    blob = (point / "artifacts" / spec["run"] / spec["trace"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != spec["trace_sha256"]:
        raise ValueError(f"Recorded trace changed: {spec['trace']}")
    with np.load(io.BytesIO(blob), allow_pickle=True) as z:
        meta = json.loads(str(z["meta"]))
        keys = {"/".join(m["path"]): m["key"] for m in json.loads(str(z["state_manifest"]))}
        robot = z[keys["scene/articulation/robot/root_pose"]][:, 0].astype(float)
        (actor,) = meta["motion"]["actors"]
        driven, boxed = (z[keys[f"scene/rigid_object/{name}/root_pose"]][:, 0, :3].astype(float)
                         for name in (actor["target"], spec["body"]))
    rotate, origin = rotation(robot[0, 3:]), robot[0, :3]
    driven, boxed = (driven - origin) @ rotate, (boxed - origin) @ rotate
    actor = DynamicActor.model_validate({k: v for k, v in actor.items() if k in DynamicActor.model_fields})
    path = _motion_path(actor)
    hz = float(meta["control_hz"])
    states = range(spec["frames"][1] + 1)
    spread, anchor = min(
        ((float(np.abs(placed - np.median(placed, 0)).max()), np.median(placed, 0))
         for placed in (np.array([driven[s, :2] - offset_and_velocity(actor, max(0, s - lag), hz, path)[0][:2]
                                  for s in states]) for lag in (0, 1, 2))),
        key=lambda pair: pair[0])
    if spread > .005:
        raise ValueError(f"{spec['trace']}: the configured path does not place the recorded motion ({spread:.4f} m)")
    for index, (x0, x1, y0, y1) in zip(spec["frames"], spec["boxes"]):
        (u, v), = opst_project(boxed[[index]])
        if not (x0 - 3 <= u <= x1 + 3 and y0 - 3 <= v <= y1 + 3):
            raise ValueError(f"{spec['trace']}: frame {index} projects the object outside its box")
    loop = actor.model_copy(update={"motion": actor.motion.model_copy(
        update={"initial_phase": 0., "initial_direction": 1})})
    period = float(path[2][-1]) / actor.motion.speed_mps
    offsets = np.array([_constant_speed_offset_and_velocity(loop, t * hz, hz, path)[0][:2]
                        for t in np.linspace(0, period, 721)])
    start = anchor + boxed[0, :2] - driven[0, :2]
    return opst_project(np.c_[start + offsets, np.full(len(offsets), boxed[0, 2])])


def draw_trajectory(ax, points):
    """The target's closed path on a frame, without arrows: an arrow on the small projected loop misleads."""
    ax.plot(*points.T, color=TARGET, lw=1.0)


def draw_box(ax, box, color, shape):
    """Outline a target box, padded slightly and kept inside the frame."""
    height, width = shape
    pad, edge = 2, .5
    x0, x1 = max(box[0] - pad, -.5 + edge), min(box[1] + pad, width - .5 - edge)
    y0, y1 = max(box[2] - pad, -.5 + edge), min(box[3] + pad, height - .5 - edge)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, lw=1.0))


def path_points(spec, exp):
    if isinstance(exp, ClosedPathTrajectory):
        actor = sample_trajectory(exp, base_seed=0, task_id=0, episode_id=0)
        hz = spec["control_hz"]
    else:
        actor = sample_trajectory(exp.trajectory, base_seed=exp.seeds.dynamics,
                                  task_id=exp.benchmark.task_ids[0], episode_id=0)
        hz = exp.benchmark.control_hz
    actor = actor.model_copy(update={"motion": actor.motion.model_copy(
        update={"initial_phase": 0., "initial_direction": 1})})
    path = _motion_path(actor)
    period = float(path[2][-1]) / actor.motion.speed_mps
    points = np.array([_constant_speed_offset_and_velocity(
        actor, t * hz, hz, path)[0][:2]
        for t in np.linspace(0, period, 721)])
    return (points - .5 * (points.min(0) + points.max(0))) * 100


def draw_path(ax, spec, points, compact=False):
    ax.plot(*points.T, color=TARGET, lw=1.4)
    first, last = ARROW
    ax.annotate("", xy=points[last], xytext=points[first],
                arrowprops={"arrowstyle": "-|>", "color": TARGET, "lw": 1.1, "mutation_scale": 8})
    ax.scatter([0], [0], s=10, marker="+", color=INK2, lw=.8)
    ax.set_aspect("equal")
    (x0, x1), (y0, y1) = spec.get("limits", ((-13, 13), (-8, 8)))
    xticks, yticks = spec.get("ticks", ((-10, 0, 10), (-5, 0, 5)))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    frame_name = spec.get("frame", "Table")
    if compact:
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)", labelpad=1)
    else:
        ax.set_xlabel(f"{frame_name} x offset (cm)")
        ax.set_ylabel(f"{frame_name} y offset (cm)")
    subpanel_title(ax, spec["label"], pad=3)


def label_frame(ax, text):
    ax.text(.035, .955, text, transform=ax.transAxes, ha="left", va="top",
            bbox={"boxstyle": "round,pad=.22", "facecolor": PAPER, "edgecolor": "none", "alpha": .85})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamicvla", action="store_true",
                        help="Render the separate DynamicVLA moving-box and moving-can illustration")
    args = parser.parse_args()
    sources = DYNAMICVLA_SOURCES if args.dynamicvla else SOURCES
    style(plt)
    if not args.dynamicvla:
        return cards(sources)
    # Frames are 4:3 for Isaac Sim and square for LIBERO; the path panel takes the remaining width.
    image_w = 1.42
    image_h = image_w * .75
    # The path panel is no wider than its frames need, centered in the remaining width.
    room = TEXT_WIDTH - .06 - (.02 + 2 * image_w + GAP + YLABEL)
    path_w = min(1.4, room)
    path_left = .02 + 2 * image_w + GAP + YLABEL + (room - path_w) / 2
    (x0, x1), (y0, y1) = next(iter(sources.values())).get("limits", ((-13, 13), (-8, 8)))
    path_h = path_w * (y1 - y0) / (x1 - x0)
    # Each row band holds the frames, or the path axes with its title above and x-axis labels below.
    band = max(image_h, PATH_BELOW + PATH_TITLE + path_h)
    height = len(sources) * (ROW_TITLE + band) + (len(sources) - 1) * ROW_GAP + .04
    fig = plt.figure(figsize=(TEXT_WIDTH, height))
    for row, spec in enumerate(sources.values()):
        top = height - .02 - row * (ROW_TITLE + band + ROW_GAP) - ROW_TITLE
        point = ARTIFACTS / spec["point"]
        video = point / spec["video"]
        if hashlib.sha256(video.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"Recorded video changed: {video}")
        exp = experiment(point / "runs" / spec["run"] / "experiment.yaml")
        images = [camera_view(frame(video, index), spec) for index in spec["frames"]]
        boxes, loop = spec["boxes"], dynamicvla_loop(spec, point)
        image_top = top - (band - image_h) / 2
        for col, (pixels, box, heading) in enumerate(zip(images, boxes, ("Start", "Interaction"))):
            ax = fig.add_axes([(.02 + col * (image_w + GAP)) / TEXT_WIDTH, (image_top - image_h) / height,
                               image_w / TEXT_WIDTH, image_h / height])
            ax.imshow(pixels, interpolation="none")
            ax.axis("off")
            draw_box(ax, box, TARGET_BOX, pixels.shape[:2])
            draw_trajectory(ax, loop)
            ax.set_xlim(-.5, pixels.shape[1] - .5)
            ax.set_ylim(pixels.shape[0] - .5, -.5)
            label_frame(ax, heading)
        path_bottom = top - band + PATH_BELOW + (band - PATH_BELOW - PATH_TITLE - path_h) / 2
        ax = fig.add_axes([path_left / TEXT_WIDTH, path_bottom / height, path_w / TEXT_WIDTH, path_h / height])
        draw_path(ax, spec, path_points(spec, exp))
        fig.text(.02 / TEXT_WIDTH, (top + .05) / height, spec["title"],
                 fontsize=RC["axes.titlesize"], va="bottom")
    filename = "dynamicvla-task-overview.png" if args.dynamicvla else "task-overview.png"
    save(fig, figure_path(filename[:-4]),
         metadata={"Source": "Recorded simulator videos, traces, and benchmark trajectories"})
    plt.close(fig)


def cards(sources):
    """The four LIBERO tasks as two cards per row across the full text block: start and interaction frames,
    then the target path in table coordinates."""
    width, columns = FULL_WIDTH, 2
    card_w = width / columns
    image_w = image_h = .86
    ylabel = .4
    # Tighter bands than the DynamicVLA layout: 7 pt tick labels need less room below the path.
    row_title, row_gap, path_below, path_title = .17, .07, .3, .16
    (x0, x1), (y0, y1) = next(iter(sources.values())).get("limits", ((-13, 13), (-8, 8)))
    path_w = min(.9, card_w - 2 * image_w - GAP - ylabel - .1)
    path_h = path_w * (y1 - y0) / (x1 - x0)
    band = max(image_h, path_below + path_title + path_h)
    rows = -(-len(sources) // columns)
    height = rows * (row_title + band) + (rows - 1) * row_gap + .04
    fig = plt.figure(figsize=(width, height))
    for index, spec in enumerate(sources.values()):
        row, col = divmod(index, columns)
        left = col * card_w + .02
        top = height - .02 - row * (row_title + band + row_gap) - row_title
        point = ARTIFACTS / spec["point"]
        video = point / spec["video"]
        if hashlib.sha256(video.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"Recorded video changed: {video}")
        exp = experiment(point / "runs" / spec["run"] / "experiment.yaml")
        images = [camera_view(frame(video, i), spec) for i in spec["frames"]]
        boxes, loop = target_views(spec, point, images[0].shape[:2])
        image_top = top - (band - image_h) / 2
        for k, (pixels, box, heading) in enumerate(zip(images, boxes, ("Start", "Interaction"))):
            ax = fig.add_axes([(left + k * (image_w + GAP)) / width, (image_top - image_h) / height,
                               image_w / width, image_h / height])
            ax.imshow(pixels, interpolation="none")
            ax.axis("off")
            draw_box(ax, box, TARGET_BOX, pixels.shape[:2])
            draw_trajectory(ax, loop)
            ax.set_xlim(-.5, pixels.shape[1] - .5)
            ax.set_ylim(pixels.shape[0] - .5, -.5)
            label_frame(ax, heading)
        path_left = left + 2 * image_w + GAP + ylabel + (card_w - 2 * image_w - GAP - ylabel - .1 - path_w) / 2
        path_bottom = top - band + path_below + (band - path_below - path_title - path_h) / 2
        ax = fig.add_axes([path_left / width, path_bottom / height, path_w / width, path_h / height])
        draw_path(ax, spec, path_points(spec, exp), compact=True)
        fig.text(left / width, (top + .05) / height, spec["title"], fontsize=RC["axes.titlesize"], va="bottom")
    save(fig, figure_path("task-overview"),
         metadata={"Source": "Recorded simulator videos, traces, and benchmark trajectories"})
    plt.close(fig)


if __name__ == "__main__":
    main()
