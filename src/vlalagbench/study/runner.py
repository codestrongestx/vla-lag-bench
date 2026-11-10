"""Evaluate studies with owned processes, immutable inputs, and resumable points."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import http.client
import importlib.metadata
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

import yaml

from vlalagbench.config import DynamicVLAPlacement, load_run_result, write_yaml
from vlalagbench.files import atomic_text, local_path
from vlalagbench.runner import REPO_ROOT, run_experiment
from vlalagbench.study.point import POINT_RECORD, read_point, verify_files, verify_point
from vlalagbench.study.spec import (
    STUDY_RECORD, Plan, Point, check_placement, load_placement, load_study, resolve, shard_runtime,
)

logger = logging.getLogger(__name__)
SERVER_START_TIMEOUT_S = 1800.0  # includes the policy warm-up compile
POLICY_SERVER = REPO_ROOT / "src/vlalagbench/benchmarks/libero/policy_server.py"
POLICY_PYTHONPATH = f"{REPO_ROOT / 'src'}:{REPO_ROOT / 'third_party/openpi/src'}"


@contextlib.contextmanager
def study_lock(directory: Path, lock_name: str = "lock"):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / lock_name).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another command holds {directory / lock_name}") from None
        yield


@contextlib.contextmanager
def gpu_locks(gpus):
    # Locks are shared by our study commands across output directories.
    with contextlib.ExitStack() as stack:
        for gpu in sorted(set(gpus)):
            path = Path("/tmp") / f"vlalagbench-{os.getuid()}-gpu{gpu}.lock"
            lock = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"GPU {gpu} is owned by another vlalagbench command") from None
        yield


def execution_identity() -> dict:
    """Freeze evaluator sources and its installed numerical stack across resume."""
    root = REPO_ROOT / "src/vlalagbench"
    paths = [*sorted((root / "benchmarks/libero").glob("*.py")),
             *sorted((root / "simulators/mujoco").glob("*.py")),
             *(root / name for name in ("config.py", "timing.py", "trajectory.py"))]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "mujoco", "robosuite")}
    revisions = {}
    for name, path in (("openpi", REPO_ROOT / "third_party/openpi"),
                       ("libero", REPO_ROOT / "third_party/openpi/third_party/libero")):
        revisions[name] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True).stdout.strip()
    return {"evaluator_sha256": digest.hexdigest(), "python": sys.version.split()[0], "packages": versions, "vendors": revisions}


def plan_execution_identity(plan: Plan) -> dict:
    if isinstance(plan.placement, DynamicVLAPlacement):
        from vlalagbench.benchmarks.dynamicvla.study import execution_identity as dynamicvla_identity

        return dynamicvla_identity(plan.points[0].experiment, shard_runtime(plan, 0), REPO_ROOT)
    return execution_identity()


def compatible_execution(recorded: dict | None, current: dict) -> bool:
    """Recognize unchanged vendored trees when Git resolves to the parent repo."""
    if recorded == current:
        return True
    if not recorded or {k: v for k, v in recorded.items() if k != "vendors"} != {
        k: v for k, v in current.items() if k != "vendors"
    }:
        return False
    old, new = recorded.get("vendors", {}), current.get("vendors", {})
    if old.keys() != new.keys():
        return False
    paths = {"openpi": "third_party/openpi", "libero": "third_party/openpi/third_party/libero"}
    for name in old:
        if old[name] == new[name]:
            continue
        if name not in paths:
            return False
        relative = paths[name]
        def git(*args, cwd=REPO_ROOT):
            return subprocess.run(["git", *args], cwd=cwd, check=True,
                                  capture_output=True, text=True).stdout.strip()
        try:
            if Path(git("rev-parse", "--show-toplevel", cwd=REPO_ROOT / relative)).resolve() != REPO_ROOT.resolve():
                return False
            before = git("rev-parse", f"{old[name]}:{relative}")
            after = git("rev-parse", f"{new[name]}:{relative}")
            if before != after or git("status", "--porcelain", "--", relative):
                return False
        except (OSError, subprocess.CalledProcessError):
            return False
    return True


def prepare(plan: Plan) -> None:
    """Called under the study lock, before any shard can write output."""
    check_placement(plan)
    study_dir = plan.study_dir
    record_path = study_dir / STUDY_RECORD
    execution = plan_execution_identity(plan)
    if record_path.exists():
        record = yaml.safe_load(record_path.read_text())
        if not compatible_execution(record.get("execution"), execution):
            raise ValueError("evaluation code or dependencies changed; use the original environment or a new study name")
    else:
        foreign = [path.name for path in study_dir.iterdir() if path.name != "lock"] if study_dir.exists() else []
        if foreign:
            raise ValueError(f"{study_dir} has no study record but contains {foreign}; choose an empty directory")
        atomic_text(record_path, yaml.safe_dump({"name": plan.spec.name, "study_hash": plan.study_hash,
                    "spec": plan.spec.model_dump(mode="python"), "execution": execution}, sort_keys=False))
    (study_dir / "queue").mkdir(exist_ok=True)
    for point in plan.points:
        config = study_dir / "configs" / f"{point.name}.yaml"
        expected = yaml.safe_dump(point.experiment.model_dump(mode="python"), sort_keys=False)
        if config.exists() and config.read_text() != expected:
            raise ValueError(f"{config}: expanded experiment changed")
        if not config.exists():
            atomic_text(config, expected)
        if state(study_dir, point.name) is None:
            (study_dir / "queue" / f"{point.name}.todo").touch()


def state(study_dir: Path, point: str) -> tuple[str, str | None] | None:
    queue = study_dir / "queue"
    for kind in ("done", "failed", "todo"):
        if (queue / f"{point}.{kind}").exists():
            return kind, None
    claims = list(queue.glob(f"{point}.claimed.*"))
    return ("claimed", claims[0].name.rsplit(".", 1)[1]) if claims else None


def claim(plan: Plan, shard: int) -> Point | None:
    queue = plan.study_dir / "queue"
    for point in plan.queue:
        target = queue / f"{point.name}.claimed.{shard:02d}"
        try:
            (queue / f"{point.name}.todo").rename(target)
        except FileNotFoundError:
            continue
        target.touch()  # rename preserves mtime; status needs the actual claim time
        return point
    return None


def _exit_on_sigterm() -> None:
    def handler(signum, frame):
        for child in multiprocessing.active_children():
            child.terminate()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, handler)


def _stop(process: subprocess.Popen, *, group: bool = False) -> None:
    if group:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if group:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()
    if group:
        # Also reap GPU/worker descendants if their shard died before cleanup.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


def run(plan: Plan, study_path: Path, placement_path: Path, selects: list[str]) -> int:
    if plan.spec.schema_version != 2:
        raise ValueError("schema 1 studies are historical; use a schema 2 study with a new name for evaluation")
    for variable in ("VLALAGBENCH_PROMOTION_OFFSET_STEPS", "VLALAGBENCH_LIBERO_ROOT"):
        if os.environ.get(variable):
            raise ValueError(f"unset {variable}; studies do not accept hidden scientific overrides")
    _exit_on_sigterm()
    placement = plan.placement
    with study_lock(plan.study_dir):
        prepare(plan)
        for point in plan.queue:
            previous = state(plan.study_dir, point.name)
            if previous and previous[0] in {"claimed", "failed"}:
                suffix = f"claimed.{previous[1]}" if previous[0] == "claimed" else "failed"
                (plan.study_dir / "queue" / f"{point.name}.{suffix}").rename(plan.study_dir / "queue" / f"{point.name}.todo")
            point_dir = plan.study_dir / "points" / point.name
            if previous and previous[0] == "done":
                verify_files(point_dir, read_point(point_dir, plan.study_hash))
            elif _completed_result(point_dir):
                # Evaluation finished before its process/stage failed. Reverify it
                # without loading a policy server or recomputing an episode.
                _verify(plan, point)
                (plan.study_dir / "queue" / f"{point.name}.todo").rename(plan.study_dir / "queue" / f"{point.name}.done")
        if all(state(plan.study_dir, point.name) == ("done", None) for point in plan.queue):
            return 0
        logs = plan.study_dir / "logs"
        logs.mkdir(exist_ok=True)
        env = os.environ.copy()
        if isinstance(placement, DynamicVLAPlacement):
            gpus = [gpu for slot in placement.shards for gpu in (slot.policy_gpu, slot.simulator_gpu)]
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                       MUJOCO_EGL_DEVICE_ID=str(placement.render_gpu), CUDA_VISIBLE_DEVICES=str(placement.render_gpu))
            gpus = [placement.render_gpu, *(shard.policy_gpu for shard in placement.shards)]
        shards = []
        backends = []
        with gpu_locks(gpus):
            try:
                if isinstance(placement, DynamicVLAPlacement):
                    backends = _start_backends(plan, env, logs)
                for index in range(len(placement.shards)):
                    with (logs / f"shard{index:02d}.log").open("a") as log:
                        shards.append(subprocess.Popen(
                            [sys.executable, "-m", "vlalagbench.study.runner", str(study_path), str(placement_path), str(index)]
                            + [f"--select={select}" for select in selects], stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True,
                        ))
                codes = [shard.wait() for shard in shards]
            finally:
                for shard in shards:
                    _stop(shard, group=True)
                for backend in backends:
                    _stop(backend, group=True)
            incomplete = any(state(plan.study_dir, point.name) != ("done", None) for point in plan.queue)
            print(f"shards exited with {codes}; selected points {'incomplete' if incomplete else 'verified'}", flush=True)
            return int(any(codes) or incomplete)


def run_shard(plan: Plan, shard: int) -> int:
    placement = plan.placement
    if not any(state(plan.study_dir, point.name) == ("todo", None) for point in plan.queue):
        return 0
    _exit_on_sigterm()
    os.sched_setaffinity(0, placement.shards[shard].cpu_set)
    runtime_path = write_yaml(shard_runtime(plan, shard), plan.study_dir / "runtimes" / f"shard{shard:02d}.yaml")
    if isinstance(placement, DynamicVLAPlacement):
        from vlalagbench.benchmarks.dynamicvla.shared_backend import ENV_ENDPOINT

        os.environ.pop(ENV_ENDPOINT, None)
        if placement.shards[shard].shared_endpoint is not None:
            os.environ[ENV_ENDPOINT] = placement.shards[shard].shared_endpoint
        failed = False
        while (point := claim(plan, shard)) is not None:
            failed = not _run_point(plan, point, runtime_path, shard) or failed
        return int(failed)
    server = _start_server(plan, shard, runtime_path)
    failed = False
    try:
        _wait_healthy(placement.shards[shard].port, server)
        identity_path = plan.study_dir / "runtimes" / f"shard{shard:02d}-policy.yaml"
        identity = yaml.safe_load(identity_path.read_text())
        with (plan.study_dir / "policy-lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            recorded = plan.study_dir / "policy-identity.yaml"
            if recorded.exists() and yaml.safe_load(recorded.read_text()) != identity:
                raise ValueError("policy environment changed; use the original policy environment or a new study name")
            if not recorded.exists():
                atomic_text(recorded, yaml.safe_dump(identity, sort_keys=True))
        while (point := claim(plan, shard)) is not None:
            failed = not _run_point(plan, point, runtime_path, shard) or failed
    finally:
        _stop(server)
    return int(failed)


def _start_backends(plan: Plan, env: dict[str, str], logs: Path) -> list[subprocess.Popen]:
    """Start one shared DynamicVLA inference backend per endpoint and wait until it serves."""
    placement = plan.placement
    groups: dict[str, list] = {}
    for slot in placement.shards:
        if slot.shared_endpoint is not None:
            groups.setdefault(slot.shared_endpoint, []).append(slot)
    installation = placement.installation
    checkpoint = str(local_path(plan.points[0].experiment.model.checkpoint))
    backends = []
    try:
        for endpoint, slots in groups.items():
            port = endpoint.rsplit(":", 1)[1]
            ready = plan.study_dir / "runtimes" / f"backend-{port}.json"
            ready.parent.mkdir(exist_ok=True)
            ready.unlink(missing_ok=True)
            cpus = ",".join(str(cpu) for cpu in sorted(frozenset().union(*(slot.cpu_set for slot in slots))))
            backend_env = {**env, "CUDA_VISIBLE_DEVICES": str(slots[0].policy_gpu),
                           "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
            with (logs / f"backend-{port}.log").open("a") as log:
                backends.append(subprocess.Popen(
                    ["taskset", "-c", cpus, installation.policy_python, "-m",
                     "vlalagbench.benchmarks.dynamicvla.shared_backend",
                     "--dynamicvla-root", installation.dynamicvla_root, "--weights", checkpoint,
                     "--endpoint", endpoint, "--ready-file", str(ready)],
                    stdout=log, stderr=subprocess.STDOUT, env=backend_env, start_new_session=True))
            deadline = time.monotonic() + installation.startup_timeout_s
            while not ready.exists():
                if backends[-1].poll() is not None:
                    raise RuntimeError(f"shared inference backend {endpoint} exited; see logs/backend-{port}.log")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"shared inference backend {endpoint} did not become ready")
                time.sleep(1)
    except BaseException:
        for backend in backends:
            _stop(backend, group=True)
        raise
    return backends


def _completed_result(point_dir: Path) -> bool:
    results = list(point_dir.glob("runs/*/result.yaml"))
    if len(results) != 1:
        return False
    result = load_run_result(results[0])
    return result.status == "completed" and result.error is None and result.summary.episode_errors == 0


def _verify(plan: Plan, point: Point):
    point_dir = plan.study_dir / "points" / point.name
    identity = plan.study_dir / "policy-identity.yaml"
    if identity.exists():
        from vlalagbench.study.point import sole_run_id

        recorded = point_dir / "runs" / sole_run_id(point_dir) / "policy-identity.yaml"
        if recorded.exists() and recorded.read_bytes() != identity.read_bytes():
            raise ValueError("point was evaluated under a different policy environment")
        if not recorded.exists():
            atomic_text(recorded, identity.read_text())
    return verify_point(point_dir, name=point.name, values=point.values,
                        experiment=point.experiment, study_hash=plan.study_hash, policy=plan.spec.policy)


def _run_point(plan: Plan, point: Point, runtime_path: Path, shard: int) -> bool:
    point_dir = plan.study_dir / "points" / point.name
    marker = plan.study_dir / "queue" / f"{point.name}.claimed.{shard:02d}"
    try:
        if not _completed_result(point_dir):
            recorded = yaml.safe_load((plan.study_dir / STUDY_RECORD).read_text())
            if not compatible_execution(recorded["execution"], plan_execution_identity(plan)):
                raise ValueError("evaluation sources changed during the study; restore them before resuming")
            stamp = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
            if point_dir.exists():
                (plan.study_dir / "failed").mkdir(exist_ok=True)
                point_dir.rename(plan.study_dir / "failed" / f"{point.name}-{stamp}")
            run_experiment(plan.study_dir / "configs" / f"{point.name}.yaml", runtime_path,
                           run_id=f"{stamp}-{point.name}", runs_root=point_dir / "runs",
                           artifact_root=point_dir / "artifacts", study=(plan.spec.name, plan.study_hash))
        _verify(plan, point)
        (point_dir / "error.txt").unlink(missing_ok=True)
        marker.rename(marker.with_name(f"{point.name}.done"))
        return True
    except Exception:
        atomic_text(point_dir / "error.txt", traceback.format_exc())
        marker.rename(marker.with_name(f"{point.name}.failed"))
        logger.exception("shard %02d: %s failed", shard, point.name)
        return False


def _engine_prompt(plan: Plan) -> str:
    """The one LIBERO instruction a FlashRT server is warmed on and serves."""
    from vlalagbench.benchmarks.libero import _vendor

    prompts = set()
    for point in plan.points:
        benchmark = point.experiment.benchmark
        suite = _vendor.libero().benchmark.get_benchmark_dict()[benchmark.suite]()
        task_ids = range(suite.n_tasks) if benchmark.task_ids == "all" else benchmark.task_ids
        prompts.update(suite.get_task(task_id).language for task_id in task_ids)
    if len(prompts) != 1:
        raise ValueError(f"a FlashRT server serves one task instruction; this study has {len(prompts)}")
    return prompts.pop()


def _start_server(plan: Plan, shard: int, runtime_path: Path) -> subprocess.Popen:
    placement = plan.placement
    model = plan.points[0].experiment.model
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(placement.shards[shard].policy_gpu),
           "PYTHONPATH": POLICY_PYTHONPATH, "XLA_PYTHON_CLIENT_MEM_FRACTION": str(placement.policy_memory_fraction or 0.75)}
    if getattr(model, "backend", "openpi") == "gr00t":
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] += f":{REPO_ROOT / 'third_party/Isaac-GR00T'}:{REPO_ROOT / 'third_party/openpi/packages/openpi-client/src'}"
        model_args = ["--experiment", str(plan.study_dir / "configs" / f"{plan.points[0].name}.yaml")]
    elif getattr(model, "backend", "openpi") == "molmoact2":
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] += f":{REPO_ROOT / 'third_party/openpi/packages/openpi-client/src'}"
        model_args = ["--experiment", str(plan.study_dir / "configs" / f"{plan.points[0].name}.yaml")]
    elif getattr(model, "engine", "openpi") != "openpi":
        model_args = ["--experiment", str(plan.study_dir / "configs" / f"{plan.points[0].name}.yaml"),
                      "--prompt", _engine_prompt(plan)]
    else:
        model_args = ["--openpi-config", model.openpi_config, "--checkpoint", model.checkpoint]
    interpreter = [placement.policy_python]
    replicas = sum(item.policy_gpu == placement.shards[shard].policy_gpu for item in placement.shards)
    if getattr(model, "backend", "openpi") in ("gr00t", "molmoact2") and replicas > 1:
        interpreter += ["-m", "vlalagbench.study.policy_process", str(placement.policy_memory_fraction)]
    with (plan.study_dir / "logs" / f"shard{shard:02d}-server.log").open("a") as log:
        return subprocess.Popen([*interpreter, str(POLICY_SERVER), *model_args, "--runtime", str(runtime_path),
                                 "--identity-out", str(runtime_path.with_name(f"shard{shard:02d}-policy.yaml"))],
                                stdout=log, stderr=subprocess.STDOUT, env=env)


def _wait_healthy(port: int, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + SERVER_START_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"policy server exited with {server.returncode}; see its log")
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            connection.request("GET", "/healthz")
            if connection.getresponse().status == 200:
                return
        except OSError:
            pass
        finally:
            connection.close()
        time.sleep(5)
    raise TimeoutError(f"policy server on port {port} was not healthy within {SERVER_START_TIMEOUT_S:.0f} s")


def status(plan: Plan) -> None:
    states = {point.name: state(plan.study_dir, point.name) for point in plan.queue}
    counts = {kind: sum(bool(value and value[0] == kind) for value in states.values()) for kind in ("done", "claimed", "failed", "todo")}
    print(f"{plan.spec.name}: {counts['done']} done, {counts['claimed']} running, {counts['failed']} failed, {counts['todo']} waiting")
    for name, value in states.items():
        point_dir = plan.study_dir / "points" / name
        if value and value[0] == "done":
            record = read_point(point_dir, plan.study_hash)
            print(f"  done    {name}  {record.successes}/{record.episodes}")
        elif value and value[0] == "claimed":
            claimed = plan.study_dir / "queue" / f"{name}.claimed.{value[1]}"
            started = dt.datetime.fromtimestamp(claimed.stat().st_mtime)
            print(f"  running {name}  shard {value[1]}  since {started:%Y-%m-%d %H:%M}")
        error = point_dir / "error.txt"
        if error.exists():
            print(f"  failed  {name}\n" + "\n".join(error.read_text().splitlines()[-20:]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="one study shard")
    parser.add_argument("study")
    parser.add_argument("placement")
    parser.add_argument("shard", type=int)
    parser.add_argument("--select", action="append", default=[])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    plan = resolve(load_study(args.study), load_placement(args.placement), REPO_ROOT, selects=args.select)
    return run_shard(plan, args.shard)


if __name__ == "__main__":
    raise SystemExit(main())
