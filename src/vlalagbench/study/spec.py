"""Expand a study into concrete experiments and check its placement.

Two documents in, one plan out. The study says what is measured and under
which policy semantics; the placement says where on one host. Everything
that can change a trajectory is hashed into ``study_hash``; ``order`` and
``name`` are preferences and are not.
"""

from __future__ import annotations

import dataclasses
import copy
import hashlib
import itertools
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from vlalagbench.config import (
    DynamicVLAExperiment, DynamicVLAPlacement, DynamicVLARuntime, DynamicVLAStudyPolicy,
    ExperimentDocument, RuntimeDocument,
    EnvironmentRuntime,
    Experiment,
    Placement,
    PolicyServer,
    Runtime,
    RUNTIME_SCHEMA_VERSION,
    StudySpec,
    SweepValue,
    load_experiment,
    load_yaml,
)


STUDY_RECORD = "study.yaml"


@dataclasses.dataclass(frozen=True)
class Point:
    name: str
    values: dict[str, SweepValue]
    experiment: ExperimentDocument


@dataclasses.dataclass(frozen=True)
class Plan:
    spec: StudySpec
    placement: Placement | DynamicVLAPlacement | None
    points: list[Point]  # every point, in queue order
    queue: list[Point]  # the points this invocation may claim
    study_hash: str
    study_dir: Path


def load_study(path: str | Path) -> StudySpec:
    return load_yaml(path, StudySpec)


def load_placement(path: str | Path) -> Placement | DynamicVLAPlacement:
    data = yaml.safe_load(Path(path).read_text())
    model = DynamicVLAPlacement if data.get("kind") == "dynamicvla" else Placement
    return model.model_validate(data)


def token(value: SweepValue) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def expand(spec: StudySpec, repo_root: Path) -> list[Point]:
    """One experiment per sweep combination, in the sweep's listed order."""

    base = load_experiment(repo_root / spec.base_experiment).model_dump(mode="python")
    if spec.schema_version == 2 and base["benchmark"].get("task_ids") == "all":
        from vlalagbench.benchmarks.libero import _vendor

        suite = _vendor.libero().benchmark.get_benchmark_dict()[base["benchmark"]["suite"]]()
        base["benchmark"]["task_ids"] = list(range(suite.n_tasks))
    keys = list(spec.sweep)
    points = []
    for combination in itertools.product(*spec.sweep.values()):
        values = dict(zip(keys, combination, strict=True))
        name = "_".join(quote(token(value), safe=".-") for value in combination)
        data = copy.deepcopy(base)
        for path, value in {**values, **spec.override}.items():
            _assign(data, path.split("."), value)
        data["name"] = f"{spec.name}/{name}"
        model = DynamicVLAExperiment if data["benchmark"].get("kind") == "dynamicvla" else Experiment
        points.append(Point(name, values, model.model_validate(data)))
    if len({point.name for point in points}) != len(points):
        raise ValueError("sweep values produce duplicate point names; use distinct values")
    if any(point.experiment.model != points[0].experiment.model for point in points):
        raise ValueError("one study uses one model; put different checkpoints in separate studies")
    return points


def ordered(spec: StudySpec, points: list[Point]) -> list[Point]:
    def rank(point: Point) -> tuple:
        keys = []
        for entry in spec.order:
            (key, rule), = entry.items()
            value = point.values[key]
            if isinstance(rule, list):
                keys.append([token(item) for item in rule].index(token(value)))
            else:
                keys.append(-value if rule == "descending" else value)
        return tuple(keys)

    return sorted(points, key=rank)


def selected(points: list[Point], selects: list[str]) -> list[Point]:
    """Keep points matching every ``KEY=VALUE``; repeated keys are any-of."""

    wanted: dict[str, set[str]] = {}
    for select in selects:
        key, separator, value = select.partition("=")
        if not separator or not points or key not in points[0].values:
            raise ValueError(f"invalid selection {select!r}; use a sweep KEY=VALUE")
        available = {token(point.values[key]) for point in points}
        if not isinstance(points[0].values[key], str):
            try:
                value = token(float(value))
            except ValueError:
                raise ValueError(f"selection {select!r} needs a numeric value") from None
        if value not in available:
            raise ValueError(f"selection {select!r} matches no sweep value; choose from {sorted(available)}")
        wanted.setdefault(key, set()).add(value)
    return [
        point
        for point in points
        if all(token(point.values[key]) in values for key, values in wanted.items())
    ]


def study_hash(spec: StudySpec, points: list[Point]) -> str:
    digest = hashlib.sha256()
    if spec.schema_version == 2:
        digest.update(_canonical({"schema_version": 2, "episode_reset": "fresh"}))
    digest.update(
        _canonical(
            {
                "sweep": spec.sweep,
                "override": spec.override,
                "policy": spec.policy.model_dump(mode="python"),
            }
        )
    )
    for point in sorted(points, key=lambda point: point.name):
        digest.update(_canonical(point.experiment.model_dump(mode="python", exclude={"name"})))
    return digest.hexdigest()


def shard_runtime(plan: Plan, shard: int) -> RuntimeDocument:
    """The runtime one shard evaluates against: study semantics plus its placement."""
    if plan.spec.schema_version != 2:
        raise ValueError("schema 1 studies are historical and read-only; use a schema 2 study for evaluation")

    policy = plan.spec.policy
    if isinstance(plan.placement, DynamicVLAPlacement):
        slot = plan.placement.shards[shard]
        return DynamicVLARuntime(
            schema_version=RUNTIME_SCHEMA_VERSION, kind="dynamicvla",
            name=f"{plan.spec.name}-shard{shard:02d}",
            **plan.placement.installation.model_dump(),
            image_port=slot.image_port, action_port=slot.action_port,
            simulator_gpu=slot.simulator_gpu, policy_gpu=slot.policy_gpu,
            **policy.model_dump(exclude={"kind"}),
        )
    return Runtime(
        schema_version=RUNTIME_SCHEMA_VERSION,
        name=f"{plan.spec.name}-shard{shard:02d}",
        policy_server=PolicyServer(
            host="127.0.0.1",
            port=plan.placement.shards[shard].port,
            implementation=policy.implementation,
            pytorch_compile_mode=policy.pytorch_compile_mode,
            max_batch_size=policy.max_batch_size,
            max_wait_ms=policy.max_wait_ms,
            batch_padding=policy.batch_padding,
            deterministic_ops=policy.deterministic_ops,
        ),
        environment=EnvironmentRuntime(
            workers=plan.placement.workers_per_shard, reuse_task_envs=False,
            render_policy_observations_only=plan.placement.render_policy_observations_only,
            dynamic_episode_queue=plan.placement.dynamic_episode_queue,
        ),
    )


def resolve(
    spec: StudySpec,
    placement: Placement | DynamicVLAPlacement,
    repo_root: Path,
    *,
    selects: tuple[str, ...] | list[str] = (),
) -> Plan:
    """Expand, hash, order, select, and check every invariant before any process starts."""

    points = ordered(spec, expand(spec, repo_root))
    plan = Plan(spec, placement, points, selected(points, selects), study_hash(spec, points), Path(placement.output_root) / spec.name)
    check_placement(plan)
    return plan


def open_study(directory: str | Path, selects: tuple[str, ...] | list[str] = ()) -> Plan:
    """Open recorded experiments on any machine, without the original repo or placement."""

    directory = Path(directory)
    record = yaml.safe_load((directory / STUDY_RECORD).read_text())
    spec = StudySpec.model_validate(record["spec"])
    combinations = itertools.product(*spec.sweep.values())
    points = []
    for combination in combinations:
        values = dict(zip(spec.sweep, combination, strict=True))
        name = "_".join(quote(token(value), safe=".-") for value in combination)
        points.append(Point(name, values, load_experiment(directory / "configs" / f"{name}.yaml")))
    if study_hash(spec, points) != record["study_hash"]:
        raise ValueError(f"{directory}: recorded experiments do not match study_hash")
    points = ordered(spec, points)
    return Plan(spec, None, points, selected(points, selects), record["study_hash"], directory)


def check_placement(plan: Plan) -> None:
    placement = plan.placement
    dynamicvla = isinstance(plan.points[0].experiment, DynamicVLAExperiment)
    if dynamicvla != isinstance(placement, DynamicVLAPlacement):
        raise ValueError("study benchmark and placement kind must match")
    if dynamicvla != isinstance(plan.spec.policy, DynamicVLAStudyPolicy):
        raise ValueError("DynamicVLA studies require policy.kind: dynamicvla; LIBERO uses StudyPolicy")
    if dynamicvla:
        if plan.spec.schema_version != 2:
            raise ValueError("DynamicVLA studies require schema_version: 2")
        from vlalagbench.benchmarks.dynamicvla.adapter import DynamicVLAAdapter
        for point in plan.points:
            DynamicVLAAdapter().validate(point.experiment, shard_runtime(plan, 0))
        slots = placement.shards
        ports = [port for slot in slots for port in (slot.image_port, slot.action_port)]
        if len(set(ports)) != len(ports):
            raise ValueError("DynamicVLA shard ports must all be distinct")
        gpus = [set((slot.policy_gpu, slot.simulator_gpu)) for slot in slots]
        if len(set.union(*gpus)) > placement.gpu_budget:
            raise ValueError("DynamicVLA GPUs exceed gpu_budget")
        for index, left in enumerate(gpus):
            for other_index in range(index + 1, len(gpus)):
                if not left & gpus[other_index]:
                    continue
                if not placement.allow_gpu_sharing:
                    raise ValueError("DynamicVLA shards must not share GPUs unless allow_gpu_sharing is enabled")
                if slots[index].cpu_set & slots[other_index].cpu_set:
                    raise ValueError("DynamicVLA shards sharing GPUs need disjoint CPU sets")
        backends: dict[str, int] = {}
        for slot in slots:
            if slot.shared_endpoint is None:
                continue
            if backends.setdefault(slot.shared_endpoint, slot.policy_gpu) != slot.policy_gpu:
                raise ValueError("shards sharing an inference backend need the same policy_gpu")
            if int(slot.shared_endpoint.rsplit(":", 1)[1]) in ports:
                raise ValueError("shared inference endpoints must not reuse shard ports")
        if len({point.experiment.model.checkpoint for point in plan.points}) > 1 and backends:
            raise ValueError("a shared inference backend serves exactly one checkpoint")
        record = plan.study_dir / STUDY_RECORD
        if record.exists() and yaml.safe_load(record.read_text())["study_hash"] != plan.study_hash:
            raise ValueError("study directory has a different study_hash; use a new study name")
        return
    policy_gpus = [shard.policy_gpu for shard in placement.shards]
    ports = [shard.port for shard in placement.shards]
    record = plan.study_dir / STUDY_RECORD
    if len(set(policy_gpus) | {placement.render_gpu}) > placement.gpu_budget:
        raise ValueError("policy and render GPUs exceed gpu_budget; drop a shard or raise the budget")
    if placement.render_gpu in policy_gpus and placement.policy_memory_fraction is None:
        raise ValueError("rendering on a policy GPU needs policy_memory_fraction so JAX leaves room for EGL")
    replicas = Counter(policy_gpus)
    if len(replicas) != len(policy_gpus):
        fraction = placement.policy_memory_fraction
        if fraction is None or any(count * fraction > 0.9 + 1e-12 for count in replicas.values()):
            raise ValueError("shards with the same policy_gpu require memory fractions totaling at most 0.9 per GPU")
        for index, shard in enumerate(placement.shards):
            for other in placement.shards[index + 1:]:
                if shard.policy_gpu == other.policy_gpu and shard.cpu_set & other.cpu_set:
                    raise ValueError("shards sharing a policy GPU need disjoint CPU sets")
    if len(set(ports)) != len(ports):
        raise ValueError("two shards name the same port")
    if record.exists() and yaml.safe_load(record.read_text())["study_hash"] != plan.study_hash:
        raise ValueError(f"{plan.study_dir} was started under a different study_hash; use a new output_root or the original spec")


def _assign(data: dict[str, Any] | list, path: list[str], value: Any) -> None:
    head, *rest = path
    key = int(head) if isinstance(data, list) else head
    if rest:
        _assign(data[key], rest, value)
    else:
        data[key] = value


def _canonical(data: Any) -> bytes:
    return yaml.safe_dump(data, sort_keys=True).encode()
