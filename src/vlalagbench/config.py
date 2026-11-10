"""The complete, strict schema for experiment workspace YAML files."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Annotated, Any, Final, Literal, TypeVar

try:
    from typing import Self
except ImportError:  # Python 3.10 simulator interpreters
    from typing_extensions import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from vlalagbench.timing import effective_steps
from vlalagbench.files import atomic_text


PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
PositiveFiniteFloat = Annotated[FiniteFloat, Field(gt=0)]
UnitFloat = Annotated[float, Field(ge=0, le=1)]
PhaseFloat = Annotated[float, Field(ge=0, lt=1)]
NonEmptyStr = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitCommit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Position = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
Action = Annotated[list[float], Field(min_length=1)]

EXPERIMENT_SCHEMA_VERSION: Final = 5
RUNTIME_SCHEMA_VERSION: Final = 3
RUN_RESULT_SCHEMA_VERSION: Final = 4
COMPARISON_RESULT_SCHEMA_VERSION: Final = 1
STUDY_SCHEMA_VERSION: Final = 2
PLACEMENT_SCHEMA_VERSION: Final = 1


class _SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelSpec(_SchemaModel):
    openpi_config: NonEmptyStr
    checkpoint: NonEmptyStr
    # The program that runs the network. openpi serves the checkpoint with its
    # own JAX or PyTorch model; flashrt takes the same openpi inputs and noise
    # and runs its own kernels at the stated precision.
    engine: Literal["openpi", "flashrt"] = Field(default="openpi", exclude_if=lambda value: value == "openpi")
    precision: Literal["bf16", "fp8", "nvfp4"] = Field(default="bf16", exclude_if=lambda value: value == "bf16")
    # Recorded policy requests that set FP8 activation scales; held out from
    # evaluation.
    # FlashRT's nvfp4 runs the prefix in NVFP4 and keeps the calibrated FP8
    # action expert, so it takes them too.
    calibration: NonEmptyStr | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_engine(self) -> Self:
        if self.engine == "openpi" and self.precision != "bf16":
            raise ValueError("openpi serves its checkpoints in bf16; use the flashrt engine for fp8 or nvfp4")
        if (self.precision in ("fp8", "nvfp4")) != (self.calibration is not None):
            raise ValueError("calibration frames are required for fp8 and nvfp4 and only used by them")
        return self


class Gr00tModelSpec(_SchemaModel):
    backend: Literal["gr00t"]
    checkpoint: NonEmptyStr
    revision: GitCommit
    checkpoint_subdir: NonEmptyStr
    embodiment: Literal["LIBERO_PANDA"] = "LIBERO_PANDA"

    @model_validator(mode="after")
    def validate_subdir(self) -> Self:
        path = Path(self.checkpoint_subdir)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("checkpoint_subdir must be relative to the checkpoint root")
        return self


class MolmoAct2ModelSpec(_SchemaModel):
    backend: Literal["molmoact2"]
    checkpoint: NonEmptyStr
    revision: GitCommit
    precision: Literal["float32", "tf32", "bfloat16"]


class Benchmark(_SchemaModel):
    kind: Literal["libero"] = "libero"
    suite: NonEmptyStr
    task_ids: Literal["all"] | list[NonNegativeInt]
    episodes_per_task: PositiveInt
    control_hz: PositiveInt
    warmup_steps: NonNegativeInt
    max_episode_steps: PositiveInt
    initial_state_indices: list[NonNegativeInt] | None = None
    # A subset of the episode IDs to run. IDs keep their seeds, initial states,
    # and trajectory phases, so a probe over a few episodes is the same
    # episodes the full condition runs.
    episode_ids: list[NonNegativeInt] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_task_ids(self) -> Self:
        if isinstance(self.task_ids, list):
            if not self.task_ids:
                raise ValueError("task_ids must be 'all' or a non-empty list")
            if len(self.task_ids) != len(set(self.task_ids)):
                raise ValueError("task_ids must not contain duplicates")
        if self.initial_state_indices is not None:
            if not self.initial_state_indices:
                raise ValueError("initial_state_indices must not be empty")
            if len(self.initial_state_indices) != len(set(self.initial_state_indices)):
                raise ValueError("initial_state_indices must not contain duplicates")
        if self.episode_ids is not None:
            if not self.episode_ids:
                raise ValueError("episode_ids must not be empty")
            if len(self.episode_ids) != len(set(self.episode_ids)):
                raise ValueError("episode_ids must not contain duplicates")
            if max(self.episode_ids) >= self.episodes_per_task:
                raise ValueError("episode_ids must be below episodes_per_task")
        return self


class Keyframe(_SchemaModel):
    step: NonNegativeInt | None = Field(default=None, exclude_if=lambda value: value is None)
    position_offset_m: Position


class Motion(_SchemaModel):
    """Distance-based timing for a spatial waypoint path."""

    speed_mps: PositiveFiniteFloat
    initial_phase: PhaseFloat | Literal["random"] = 0.0
    initial_direction: Literal[-1, 1] = 1


class DynamicActor(_SchemaModel):
    target: NonEmptyStr
    start_step: NonNegativeInt
    end_step: NonNegativeInt | None
    loop: bool
    interpolation: Literal["linear", "cubic"]
    motion: Motion | None = None
    release_on_grasp_steps: PositiveInt | None = None
    # Drive a jointless body through added planar slide joints so contacts
    # see its velocity and friction carries objects resting on it.
    slide_joints: bool = Field(default=False, exclude_if=lambda value: not value)
    # Free objects shifted with a carrier at the scheduled start only.
    # LIBERO requires slide joints; DynamicVLA uses a contact-driven carrier.
    carried_objects: list[NonEmptyStr] = Field(default_factory=list, exclude_if=lambda value: not value)
    keyframes: Annotated[list[Keyframe], Field(min_length=2)]

    @model_validator(mode="after")
    def validate_timing(self) -> Self:
        if self.end_step is not None and self.end_step <= self.start_step:
            raise ValueError("end_step must be greater than start_step")
        steps = [keyframe.step for keyframe in self.keyframes]
        if self.motion is None:
            if any(step is None for step in steps):
                raise ValueError(
                    "legacy timing requires step on every keyframe; alternatively set "
                    "motion.speed_mps and omit every keyframe step"
                )
            if steps[0] != 0:
                raise ValueError("the first keyframe step must be 0")
            if any(right <= left for left, right in zip(steps, steps[1:])):
                raise ValueError("keyframe steps must be strictly increasing")
        elif any("step" in keyframe.model_fields_set for keyframe in self.keyframes):
            raise ValueError(
                "motion.speed_mps timing requires spatial waypoints without step"
            )
        elif any(
            left.position_offset_m == right.position_offset_m
            for left, right in zip(self.keyframes, self.keyframes[1:])
        ):
            raise ValueError(
                "motion.speed_mps timing requires consecutive waypoints to differ"
            )
        elif any(keyframe.position_offset_m[2] != 0.0 for keyframe in self.keyframes):
            raise ValueError("motion.speed_mps timing supports planar XY waypoints only")
        if (
            self.loop
            and self.keyframes[0].position_offset_m
            != self.keyframes[-1].position_offset_m
        ):
            raise ValueError("looping dynamics must end at their initial offset")
        return self


class TrajectoryEpisode(_SchemaModel):
    """One explicit trajectory initial condition for an evaluation episode."""

    initial_phase: PhaseFloat
    initial_direction: Literal[-1, 1]


class ReflectedTrajectory(_SchemaModel):
    """One fixed-speed straight path with balanced episode initial conditions."""

    target: NonEmptyStr
    start_step: NonNegativeInt = 0
    path_start_offset_m: Position
    path_end_offset_m: Position
    speed_mps: NonNegativeFloat
    phase_bins: PositiveInt | None = None
    shuffle: bool = False
    episode_schedule: Annotated[
        list[TrajectoryEpisode], Field(min_length=1)
    ] | None = None

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if self.path_start_offset_m == self.path_end_offset_m:
            raise ValueError("trajectory endpoints must differ")
        _validate_trajectory_sampling(self.phase_bins, self.shuffle, self.episode_schedule)
        return self


class ClosedPathTrajectory(_SchemaModel):
    """One closed waypoint path with paired phase/direction episode sampling."""

    kind: Literal["closed_path"]
    target: NonEmptyStr
    start_step: NonNegativeInt = 0
    interpolation: Literal["linear", "cubic"]
    speed_mps: PositiveFiniteFloat
    release_on_grasp_steps: PositiveInt | None = Field(default=None, exclude_if=lambda value: value is None)
    slide_joints: bool = Field(default=False, exclude_if=lambda value: not value)
    carried_objects: list[NonEmptyStr] = Field(default_factory=list, exclude_if=lambda value: not value)
    phase_bins: PositiveInt | None = None
    shuffle: bool = False
    episode_schedule: Annotated[
        list[TrajectoryEpisode], Field(min_length=1)
    ] | None = None
    keyframes: Annotated[list[Keyframe], Field(min_length=3)]

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if any("step" in keyframe.model_fields_set for keyframe in self.keyframes):
            raise ValueError("closed-path trajectory keyframes must omit step")
        if self.keyframes[0].position_offset_m != self.keyframes[-1].position_offset_m:
            raise ValueError("closed-path trajectory must end at its initial offset")
        if any(
            left.position_offset_m == right.position_offset_m
            for left, right in zip(self.keyframes, self.keyframes[1:])
        ):
            raise ValueError("closed-path trajectory waypoints must differ consecutively")
        if any(keyframe.position_offset_m[2] != 0.0 for keyframe in self.keyframes):
            raise ValueError("closed-path trajectory supports planar XY waypoints only")
        _validate_trajectory_sampling(self.phase_bins, self.shuffle, self.episode_schedule)
        return self


def _validate_trajectory_sampling(
    phase_bins: int | None,
    shuffle: bool,
    episode_schedule: list[TrajectoryEpisode] | None,
) -> None:
    if (phase_bins is None) == (episode_schedule is None):
        raise ValueError(
            "trajectory requires exactly one of phase_bins or episode_schedule"
        )
    if episode_schedule is not None and shuffle:
        raise ValueError("trajectory shuffle cannot be used with episode_schedule")


TrajectorySpec = ReflectedTrajectory | ClosedPathTrajectory


class ReflectedTrajectorySample(_SchemaModel):
    """Concrete per-episode actor sampled from a reflected trajectory."""

    kind: Literal["reflected_trajectory"] = "reflected_trajectory"
    target: NonEmptyStr
    start_step: NonNegativeInt
    path_start_offset_m: Position
    path_end_offset_m: Position
    speed_mps: NonNegativeFloat
    phase_bin: NonNegativeInt
    initial_phase: PhaseFloat
    initial_direction: Literal[-1, 1]

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if self.path_start_offset_m == self.path_end_offset_m:
            raise ValueError("trajectory endpoints must differ")
        return self


ResolvedDynamicActor = DynamicActor | ReflectedTrajectorySample


class Inference(_SchemaModel):
    mode: Literal["synchronous", "lazy", "continuous"]
    action_chunk_steps: PositiveInt
    execution_horizon_steps: PositiveInt
    latency_ms: NonNegativeInt
    unavailable_action: Action

    @model_validator(mode="after")
    def validate_horizon(self) -> Self:
        if self.execution_horizon_steps > self.action_chunk_steps:
            raise ValueError("execution_horizon_steps must not exceed action_chunk_steps")
        return self


class Seeds(_SchemaModel):
    environment: NonNegativeInt
    dynamics: NonNegativeInt
    policy: NonNegativeInt


class InitialStateOverride(_SchemaModel):
    """Object poses applied after one official LIBERO initial state."""

    task_id: NonNegativeInt
    initial_state_index: NonNegativeInt
    object_positions_m: dict[NonEmptyStr, Position] = Field(default_factory=dict)
    object_yaws_rad: dict[NonEmptyStr, FiniteFloat] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_pose(self) -> Self:
        if not self.object_positions_m and not self.object_yaws_rad:
            raise ValueError("initial-state object pose override must not be empty")
        return self


class Experiment(_SchemaModel):
    schema_version: Literal[EXPERIMENT_SCHEMA_VERSION]
    name: NonEmptyStr
    model: ModelSpec | Gr00tModelSpec | MolmoAct2ModelSpec
    benchmark: Benchmark
    initial_state_overrides: list[InitialStateOverride] = Field(default_factory=list)
    dynamics: list[DynamicActor]
    dynamics_by_task: dict[int, list[DynamicActor]] = Field(default_factory=dict)
    trajectory: TrajectorySpec | None = None
    inference: Inference
    seeds: Seeds

    @property
    def effective_latency_steps(self) -> int:
        return effective_steps(self.inference.latency_ms, self.benchmark.control_hz)

    @model_validator(mode="after")
    def validate_async_window(self) -> Self:
        if isinstance(self.model, Gr00tModelSpec) and self.inference.action_chunk_steps != 16:
            raise ValueError("GR00T LIBERO checkpoints require action_chunk_steps=16")
        if isinstance(self.model, MolmoAct2ModelSpec) and self.inference.action_chunk_steps != 10:
            raise ValueError("MolmoAct2 LIBERO checkpoints require action_chunk_steps=10")
        if isinstance(self.model, Gr00tModelSpec) and self.model.checkpoint == "nvidia/GR00T-N1.7-LIBERO":
            if self.benchmark.suite in {"libero_spatial", "libero_object", "libero_goal", "libero_10"} and self.model.checkpoint_subdir != self.benchmark.suite:
                raise ValueError("official GR00T baseline checkpoint_subdir must match the LIBERO suite")
        override_keys = [
            (override.task_id, override.initial_state_index)
            for override in self.initial_state_overrides
        ]
        if len(override_keys) != len(set(override_keys)):
            raise ValueError(
                "initial_state_overrides must contain one entry per task and state"
            )
        invalid_task_ids = [task_id for task_id in self.dynamics_by_task if task_id < 0]
        if invalid_task_ids:
            raise ValueError("dynamics_by_task keys must be nonnegative task IDs")
        if isinstance(self.benchmark.task_ids, list):
            unexpected_overrides = {
                override.task_id for override in self.initial_state_overrides
            } - set(self.benchmark.task_ids)
            if unexpected_overrides:
                raise ValueError(
                    "initial_state_overrides contains task IDs outside benchmark.task_ids: "
                    + ", ".join(map(str, sorted(unexpected_overrides)))
                )
            unexpected = set(self.dynamics_by_task) - set(self.benchmark.task_ids)
            if unexpected:
                raise ValueError(
                    "dynamics_by_task contains task IDs outside benchmark.task_ids: "
                    + ", ".join(map(str, sorted(unexpected)))
                )
        # MuJoCo moves resting objects with a fixture only through its slide joints.
        actors = [*self.dynamics, *(actor for group in self.dynamics_by_task.values() for actor in group)]
        if self.trajectory is not None:
            actors.append(self.trajectory)
        if any(
            getattr(actor, "carried_objects", None) and not getattr(actor, "slide_joints", False)
            for actor in actors
        ):
            raise ValueError("carried_objects requires slide_joints")
        if self.trajectory is not None:
            dynamic_targets = {actor.target for actor in self.dynamics}
            dynamic_targets.update(
                actor.target
                for actors in self.dynamics_by_task.values()
                for actor in actors
            )
            if self.trajectory.target in dynamic_targets:
                raise ValueError("trajectory target overlaps deterministic dynamics")
            if self.trajectory.episode_schedule is not None:
                if len(self.trajectory.episode_schedule) != self.benchmark.episodes_per_task:
                    raise ValueError(
                        "trajectory episode_schedule length must equal "
                        "episodes_per_task"
                    )
            else:
                cells = self.trajectory.phase_bins * 2
                if self.benchmark.episodes_per_task % cells:
                    raise ValueError(
                        "episodes_per_task must be divisible by phase_bins * 2"
                    )
        if (
            self.inference.mode != "synchronous"
            and self.effective_latency_steps + self.inference.execution_horizon_steps
            > self.inference.action_chunk_steps
        ):
            raise ValueError("async latency plus execution horizon must fit in the action chunk")
        return self


class DynamicVLAModelSpec(_SchemaModel):
    checkpoint: NonEmptyStr
    checkpoint_sha256: Sha256


class DynamicVLATask(_SchemaModel):
    name: NonEmptyStr
    environments: Annotated[list[NonEmptyStr], Field(min_length=1)]


class DynamicVLABenchmark(_SchemaModel):
    kind: Literal["dynamicvla"]
    upstream_revision: GitCommit
    tasks: Annotated[list[DynamicVLATask], Field(min_length=1)]
    episodes_per_environment: PositiveInt
    # Each DOM environment JSON defines one official state (index 0). Replica
    # r uses state r % initial_states_per_environment; states above 0 exist
    # only through initial_state_overrides.
    initial_states_per_environment: PositiveInt = Field(
        default=1, exclude_if=lambda value: value == 1
    )
    # DOM launches the target with an official velocity. At rest, every rigid
    # object starts static and moves only through dynamics or trajectory.
    start_at_rest: bool = Field(default=False, exclude_if=lambda value: not value)
    # Active policy actions; latency holds do not consume this budget.
    # Omission preserves the native environment deadline.
    max_episode_steps: PositiveInt | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    control_hz: Literal[25] = 25
    driven_friction: Literal["zero", "native"] = "native"

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        names = [task.name for task in self.tasks]
        if len(names) != len(set(names)):
            raise ValueError("DynamicVLA task names must be globally unique")
        environments = [
            environment for task in self.tasks for environment in task.environments
        ]
        normalized_environments = [
            Path(environment).as_posix() for environment in environments
        ]
        if len(normalized_environments) != len(set(normalized_environments)):
            raise ValueError("DynamicVLA environment paths must be globally unique")
        for environment in environments:
            path = Path(environment)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("environments must be relative to the runtime assets_root")
            if path.suffix != ".json":
                raise ValueError("DynamicVLA environments must be JSON files")
        return self

    @property
    def resolved_episode_counts(self) -> dict[str, int]:
        return {
            task.name: len(task.environments) * self.episodes_per_environment
            for task in self.tasks
        }


class DynamicVLAInference(_SchemaModel):
    mode: Literal["synchronous", "lazy", "continuous"]
    action_chunk_steps: PositiveInt
    execution_horizon_steps: PositiveInt
    latency_ms: NonNegativeInt

    @model_validator(mode="after")
    def validate_horizon(self) -> Self:
        if self.execution_horizon_steps > self.action_chunk_steps:
            raise ValueError("execution_horizon_steps must not exceed action_chunk_steps")
        return self


class DynamicVLASeeds(_SchemaModel):
    policy: NonNegativeInt


class DynamicVLAInitialStateOverride(_SchemaModel):
    """Rigid-asset pose applied to one DOM environment initial state.

    Keys are DOM scene asset names such as ``object``, ``object01``, and
    ``container``. Values are in the robot base frame (x forward, y left, z
    up). A yaw replaces the XYZ-Euler yaw and keeps the official roll and
    pitch.
    """

    environment: NonEmptyStr
    initial_state_index: NonNegativeInt
    object_positions_m: dict[NonEmptyStr, Position] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    object_yaws_rad: dict[NonEmptyStr, FiniteFloat] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def validate_pose(self) -> Self:
        if not self.object_positions_m and not self.object_yaws_rad:
            raise ValueError("initial-state object pose override must not be empty")
        if "robot" in {*self.object_positions_m, *self.object_yaws_rad}:
            raise ValueError("initial-state overrides apply to rigid objects, not the robot")
        return self


class DynamicVLAExperiment(_SchemaModel):
    """DOM evaluation with optional authored initial states and motion.

    Dynamics and trajectory offsets are in the robot base frame and are
    relative to the target's reset position in each episode's state.
    """

    schema_version: Literal[EXPERIMENT_SCHEMA_VERSION]
    name: NonEmptyStr
    model: DynamicVLAModelSpec
    benchmark: DynamicVLABenchmark
    initial_state_overrides: list[DynamicVLAInitialStateOverride] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    dynamics: list[DynamicActor] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    trajectory: TrajectorySpec | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    inference: DynamicVLAInference
    seeds: DynamicVLASeeds

    @property
    def has_authored_scene(self) -> bool:
        return bool(
            self.benchmark.initial_states_per_environment != 1
            or self.benchmark.start_at_rest
            or self.initial_state_overrides
            or self.dynamics
            or self.trajectory is not None
        )

    @model_validator(mode="after")
    def validate_authored_scene(self) -> Self:
        environments = {
            Path(environment).as_posix()
            for task in self.benchmark.tasks
            for environment in task.environments
        }
        state_count = self.benchmark.initial_states_per_environment
        states: dict[tuple[str, int], DynamicVLAInitialStateOverride] = {}
        for override in self.initial_state_overrides:
            environment = Path(override.environment).as_posix()
            if environment not in environments:
                raise ValueError(
                    f"initial_state_overrides names an unconfigured environment: {override.environment}"
                )
            if override.initial_state_index >= state_count:
                raise ValueError(
                    "initial_state_index must be below initial_states_per_environment"
                )
            key = (environment, override.initial_state_index)
            if key in states:
                raise ValueError(
                    "initial_state_overrides must contain one entry per environment and state"
                )
            states[key] = override
        # Placements may repeat across state indices. An episode's condition
        # also includes its independently scheduled trajectory phase/direction.
        motion_targets = [actor.target for actor in self.dynamics]
        if self.trajectory is not None:
            motion_targets.append(self.trajectory.target)
        if "robot" in motion_targets:
            raise ValueError("DynamicVLA motion applies to rigid objects, not the robot")
        # Driving the grasped target would fight the gripper; releasing it on
        # grasp needs a DOM grasp detector.
        if "object" in motion_targets:
            raise ValueError("DynamicVLA motion cannot drive the grasped target 'object'")
        carriers = [*self.dynamics, *([self.trajectory] if self.trajectory is not None else [])]
        passengers: set[str] = set()
        for actor in carriers:
            carried = getattr(actor, "carried_objects", [])
            if actor.target in carried or "robot" in carried or set(carried) & set(motion_targets):
                raise ValueError("DynamicVLA carried objects must be free rigid objects")
            if len(set(carried)) != len(carried) or passengers.intersection(carried):
                raise ValueError("DynamicVLA carried objects must have exactly one carrier")
            passengers.update(carried)
        if any(actor.release_on_grasp_steps is not None for actor in self.dynamics) or (
            isinstance(self.trajectory, ClosedPathTrajectory)
            and self.trajectory.release_on_grasp_steps is not None
        ):
            raise ValueError("DynamicVLA motion does not support release_on_grasp_steps")
        if any(getattr(actor, "slide_joints", False) for actor in self.dynamics) or (
            isinstance(self.trajectory, ClosedPathTrajectory) and self.trajectory.slide_joints
        ):
            raise ValueError("DynamicVLA motion does not support slide_joints")
        for actor in self.dynamics:
            if actor.motion is not None and actor.motion.initial_phase == "random":
                raise ValueError("DynamicVLA dynamics require an explicit initial_phase")
        if self.trajectory is not None:
            if self.trajectory.target in {actor.target for actor in self.dynamics}:
                raise ValueError("trajectory target overlaps deterministic dynamics")
            # Phase-bin jitter needs a dynamics seed; DOM schedules are explicit.
            if self.trajectory.episode_schedule is None:
                raise ValueError("DynamicVLA trajectories require an explicit episode_schedule")
            if len(self.trajectory.episode_schedule) != self.benchmark.episodes_per_environment:
                raise ValueError(
                    "trajectory episode_schedule length must equal episodes_per_environment"
                )
        return self

    @property
    def effective_latency_steps(self) -> int:
        return effective_steps(self.inference.latency_ms, self.benchmark.control_hz)

    @property
    def resolved_execution_horizon_steps(self) -> int:
        return self.inference.execution_horizon_steps

    @model_validator(mode="after")
    def validate_latency_schedule(self) -> Self:
        if self.inference.latency_ms * self.benchmark.control_hz % 1000:
            raise ValueError("DynamicVLA latency_ms must align to the 25 Hz control grid")
        if (
            self.inference.mode != "synchronous"
            and self.effective_latency_steps + self.resolved_execution_horizon_steps
            > self.inference.action_chunk_steps
        ):
            raise ValueError("async latency plus execution horizon must fit in the action chunk")
        return self


class PolicyServer(_SchemaModel):
    host: NonEmptyStr
    port: Annotated[int, Field(ge=1, le=65535)]
    implementation: Literal["single", "vanilla", "dynamic_batching"]
    pytorch_compile_mode: NonEmptyStr
    max_batch_size: PositiveInt
    max_wait_ms: NonNegativeFloat
    # Batch shape changes the numerics, so a run records how requests were
    # shaped: "fixed" pads every batch to max_batch_size rows.
    batch_padding: Literal["none", "fixed"] = "none"
    deterministic_ops: bool = False

    @model_validator(mode="after")
    def validate_single_request(self) -> Self:
        if self.implementation == "single" and (self.max_batch_size != 1 or self.batch_padding != "none" or self.max_wait_ms != 0):
            raise ValueError("single-request inference requires max_batch_size=1, batch_padding=none, max_wait_ms=0")
        return self


class EnvironmentRuntime(_SchemaModel):
    workers: PositiveInt
    reuse_task_envs: bool
    initial_state_strategy: Literal["cycle_official", "official_then_seeded"] = "cycle_official"
    process_start_method: Literal["spawn", "fork"] = "spawn"
    # Render cameras only on steps whose observation reaches the policy.
    render_policy_observations_only: bool = Field(default=False, exclude_if=lambda value: not value)
    # Workers take the next unstarted episode instead of fixed strided slices.
    dynamic_episode_queue: bool = Field(default=False, exclude_if=lambda value: not value)


class Runtime(_SchemaModel):
    schema_version: Literal[RUNTIME_SCHEMA_VERSION]
    kind: Literal["libero"] = "libero"
    name: NonEmptyStr
    policy_server: PolicyServer
    environment: EnvironmentRuntime


class DynamicVLAObservationSettings(_SchemaModel):
    """Observation semantics shared by standalone runs and study points."""

    replica_batch_size: Literal[1] = 1
    antialiasing_mode: Literal["Off", "TAA", "FXAA", "DLSS", "DLAA"] | None = None
    enable_dl_denoiser: bool | None = Field(default=None, exclude_if=lambda value: value is None)
    dlss_mode: Literal["Performance", "Balanced", "Quality", "Auto"] | None = Field(default=None, exclude_if=lambda value: value is None)
    enable_dlssg: bool | None = Field(default=None, exclude_if=lambda value: value is None)
    render_mode: Literal["RaytracedLighting"] | None = Field(default=None, exclude_if=lambda value: value is None)
    disable_fabric: bool = Field(default=False, exclude_if=lambda value: value is False)


class DynamicVLARuntime(DynamicVLAObservationSettings):
    schema_version: Literal[RUNTIME_SCHEMA_VERSION]
    kind: Literal["dynamicvla"]
    name: NonEmptyStr
    dynamicvla_root: NonEmptyStr
    assets_root: NonEmptyStr
    simulator_python: NonEmptyStr
    policy_python: NonEmptyStr
    host: NonEmptyStr = "127.0.0.1"
    image_port: Annotated[int, Field(ge=1, le=65535)] = 3186
    action_port: Annotated[int, Field(ge=1, le=65535)] = 3188
    startup_timeout_s: PositiveFloat = 900.0
    reply_timeout_s: PositiveFloat = 300.0
    suite_timeout_s: PositiveFloat = 86400.0
    trace_writer_workers: Annotated[int, Field(gt=0, le=8)] = 1
    # Traces keep states, actions, and protocol; camera frames are re-rendered
    # from states unless this also stores the exact policy RGB (~0.25 MB per
    # camera per control step).
    trace_images: bool = Field(default=False, exclude_if=lambda value: value is False)
    save_videos: bool = False
    video_episode_limit: PositiveInt | None = Field(default=None, exclude_if=lambda value: value is None)
    simulator_library_paths: list[NonEmptyStr] = Field(default_factory=list)
    # Kit's carb.tasking pool defaults to one spinning worker per host CPU and
    # ignores taskset. Explicit None restores Kit's uncapped default.
    kit_tasking_threads: PositiveInt | None = 4
    simulator_gpu: NonNegativeInt | None = Field(default=None, exclude_if=lambda value: value is None)
    policy_gpu: NonNegativeInt | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_ports(self) -> Self:
        if self.image_port == self.action_port:
            raise ValueError("image_port and action_port must differ")
        return self


SweepValue = str | int | float


class StudyPolicy(_SchemaModel):
    """How the policy server shapes and computes requests.

    Everything here changes a trajectory, so it is study semantics and is
    hashed with the study, unlike host, port, and worker count.
    """

    implementation: Literal["single", "vanilla", "dynamic_batching"] = "single"
    max_batch_size: PositiveInt = 1
    batch_padding: Literal["none", "fixed"] = "none"
    max_wait_ms: NonNegativeFloat = 0.0
    deterministic_ops: bool = False
    pytorch_compile_mode: NonEmptyStr = "max-autotune-no-cudagraphs"


OrderRule = Annotated[list[SweepValue], Field(min_length=1)] | Literal["ascending", "descending"]


class DynamicVLAStudyPolicy(DynamicVLAObservationSettings):
    """Observation semantics, fixed across every point and resume."""

    kind: Literal["dynamicvla"] = "dynamicvla"


class StudySpec(_SchemaModel):
    """A sweep over one base experiment plus the policy semantics it runs under."""

    schema_version: Literal[1, STUDY_SCHEMA_VERSION]
    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    base_experiment: NonEmptyStr
    sweep: Annotated[
        dict[NonEmptyStr, Annotated[list[SweepValue], Field(min_length=1)]],
        Field(min_length=1),
    ]
    override: dict[NonEmptyStr, Any] = Field(default_factory=dict)
    policy: StudyPolicy | DynamicVLAStudyPolicy = Field(default_factory=StudyPolicy)
    order: list[dict[NonEmptyStr, OrderRule]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.schema_version == 2 and isinstance(self.policy, StudyPolicy) and (
            self.policy.implementation == "vanilla" or self.policy.implementation == "single" and (
                self.policy.max_batch_size != 1 or self.policy.batch_padding != "none" or self.policy.max_wait_ms != 0
            )
        ):
            raise ValueError("schema 2 studies use single-request inference or dynamic_batching; "
                             "batch settings require implementation: dynamic_batching")
        for key in self.sweep:
            if any(key == other or key.startswith(other + ".") or other.startswith(key + ".") for other in self.override):
                raise ValueError(f"override overlaps sweep key {key!r}; put each condition in one place")
        for key, values in self.sweep.items():
            if len(values) != len(set(values)):
                raise ValueError(f"sweep {key} lists a value twice")
        for entry in self.order:
            if len(entry) != 1:
                raise ValueError("each order entry names exactly one sweep key")
            (key, rule), = entry.items()
            if key not in self.sweep:
                raise ValueError(f"order key {key!r} is not a sweep key")
            if isinstance(rule, list) and sorted(map(str, rule)) != sorted(map(str, self.sweep[key])):
                raise ValueError(f"order for {key!r} must rank every sweep value once")
            if rule == "descending" and any(isinstance(value, str) for value in self.sweep[key]):
                raise ValueError(f"descending order needs numeric sweep values for {key!r}")
        return self


class Shard(_SchemaModel):
    policy_gpu: NonNegativeInt
    port: Annotated[int, Field(ge=1, le=65535)]
    cpus: Annotated[str, Field(pattern=r"^\d+(-\d+)?(,\d+(-\d+)?)*$")]

    @model_validator(mode="after")
    def validate_cpu_ranges(self) -> Self:
        for span in self.cpus.split(","):
            first, _, last = span.partition("-")
            if int(first) > int(last or first):
                raise ValueError("CPU ranges must be increasing")
        return self

    @property
    def cpu_set(self) -> frozenset[int]:
        cores: set[int] = set()
        for span in self.cpus.split(","):
            first, _, last = span.partition("-")
            cores.update(range(int(first), int(last or first) + 1))
        return frozenset(cores)


class Videos(_SchemaModel):
    clips_per_outcome: PositiveInt
    cameras: Annotated[list[Annotated[str, Field(pattern=r"^[A-Za-z0-9_]+$")]], Field(min_length=1)]
    resolution: PositiveInt
    dynamicvla_path_tracing_spp: Annotated[int, Field(ge=1, le=128)] | None = None
    # Absent keeps the per-point selection (and its settings hash) unchanged.
    per_task: Literal[True] | None = None

    @model_validator(mode="after")
    def validate_cameras(self) -> Self:
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError("cameras must be distinct")
        return self


class Placement(_SchemaModel):
    """Where a study runs on one host. Nothing here may change a trajectory."""

    schema_version: Literal[PLACEMENT_SCHEMA_VERSION]
    host: NonEmptyStr
    gpu_budget: PositiveInt
    shards: Annotated[list[Shard], Field(min_length=1)]
    render_gpu: NonNegativeInt
    workers_per_shard: PositiveInt
    policy_memory_fraction: Annotated[float, Field(gt=0, le=1)] | None = None
    render_policy_observations_only: bool = Field(default=False, exclude_if=lambda value: not value)
    dynamic_episode_queue: bool = Field(default=False, exclude_if=lambda value: not value)
    policy_python: NonEmptyStr
    output_root: NonEmptyStr


class DynamicVLAShard(_SchemaModel):
    policy_gpu: NonNegativeInt
    simulator_gpu: NonNegativeInt
    image_port: Annotated[int, Field(ge=1, le=65535)]
    action_port: Annotated[int, Field(ge=1, le=65535)]
    cpus: Annotated[str, Field(pattern=r"^\d+(-\d+)?(,\d+(-\d+)?)*$")]
    shared_endpoint: Annotated[
        str | None, Field(pattern=r"^tcp://127\.0\.0\.1:\d{1,5}$", exclude_if=lambda value: value is None)
    ] = None

    @model_validator(mode="after")
    def validate_cpus(self) -> Self:
        Shard(policy_gpu=self.policy_gpu, port=self.image_port, cpus=self.cpus)
        return self

    @property
    def cpu_set(self) -> frozenset[int]:
        return Shard(policy_gpu=self.policy_gpu, port=self.image_port, cpus=self.cpus).cpu_set


class DynamicVLAInstallation(_SchemaModel):
    dynamicvla_root: NonEmptyStr
    assets_root: NonEmptyStr
    simulator_python: NonEmptyStr
    policy_python: NonEmptyStr
    simulator_library_paths: list[NonEmptyStr] = Field(default_factory=list)
    startup_timeout_s: PositiveFloat = 900.0
    reply_timeout_s: PositiveFloat = 300.0
    suite_timeout_s: PositiveFloat = 86400.0
    trace_writer_workers: Annotated[int, Field(gt=0, le=8)] = 1
    trace_images: bool = Field(default=False, exclude_if=lambda value: value is False)
    save_videos: bool = Field(default=False, exclude_if=lambda value: value is False)
    video_episode_limit: PositiveInt | None = Field(default=None, exclude_if=lambda value: value is None)
    kit_tasking_threads: PositiveInt | None = 4


class DynamicVLAPlacement(_SchemaModel):
    schema_version: Literal[PLACEMENT_SCHEMA_VERSION]
    kind: Literal["dynamicvla"]
    host: NonEmptyStr
    gpu_budget: PositiveInt
    shards: Annotated[list[DynamicVLAShard], Field(min_length=1)]
    allow_gpu_sharing: bool = Field(default=False, exclude_if=lambda value: value is False)
    installation: DynamicVLAInstallation
    output_root: NonEmptyStr


class SourceConfig(_SchemaModel):
    source: NonEmptyStr
    sha256: Sha256
    resolved: NonEmptyStr


class CodeVersion(_SchemaModel):
    commit: GitCommit
    dirty: bool


class GPUAllocation(_SchemaModel):
    type: NonEmptyStr
    count: PositiveInt


class Hardware(_SchemaModel):
    hostname: NonEmptyStr
    cpu_count: PositiveInt
    gpus: Annotated[list[GPUAllocation], Field(min_length=1)]


class RunLatency(_SchemaModel):
    control_hz: PositiveInt
    requested_ms: NonNegativeInt
    effective_steps: NonNegativeInt

    @model_validator(mode="after")
    def validate_effective_steps(self) -> Self:
        expected = effective_steps(self.requested_ms, self.control_hz)
        if self.effective_steps != expected:
            raise ValueError(f"effective_steps must be {expected}")
        return self


class EpisodeResult(_SchemaModel):
    task_id: NonNegativeInt
    episode_id: NonNegativeInt
    seed: NonNegativeInt
    success: bool
    status: NonNegativeInt | None = None
    steps: NonNegativeInt
    control_steps: NonNegativeInt
    wait_steps: NonNegativeInt
    policy_requests: NonNegativeInt
    trace: NonEmptyStr
    trace_sha256: Sha256 | None = Field(default=None, exclude_if=lambda value: value is None)
    error: str | None


class EpisodeArtifact(_SchemaModel):
    format: Literal["jsonl"]
    path: NonEmptyStr
    sha256: Sha256
    records: NonNegativeInt


class TaskSummary(_SchemaModel):
    task_id: NonNegativeInt
    episodes: PositiveInt
    successes: NonNegativeInt
    success_rate: UnitFloat
    mean_active_steps: NonNegativeFloat
    mean_control_steps: NonNegativeFloat
    mean_wait_steps: NonNegativeFloat

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.successes > self.episodes:
            raise ValueError("successes must not exceed episodes")
        expected_rate = self.successes / self.episodes if self.episodes else 0.0
        if not math.isclose(self.success_rate, expected_rate, abs_tol=1e-12):
            raise ValueError("success_rate must match task counts")
        return self


class RunSummary(_SchemaModel):
    episodes: NonNegativeInt
    successes: NonNegativeInt
    success_rate: UnitFloat
    episode_errors: NonNegativeInt
    policy_requests: NonNegativeInt
    mean_active_steps: NonNegativeFloat
    mean_control_steps: NonNegativeFloat
    mean_wait_steps: NonNegativeFloat
    tasks: list[TaskSummary]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.successes > self.episodes:
            raise ValueError("successes must not exceed episodes")
        if self.episode_errors > self.episodes:
            raise ValueError("episode_errors must not exceed episodes")
        expected_rate = self.successes / self.episodes if self.episodes else 0.0
        if not math.isclose(self.success_rate, expected_rate, abs_tol=1e-12):
            raise ValueError("success_rate must match run counts")
        if sum(task.episodes for task in self.tasks) != self.episodes:
            raise ValueError("task episode counts must match run episodes")
        if sum(task.successes for task in self.tasks) != self.successes:
            raise ValueError("task success counts must match run successes")
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task summaries must have unique task ids")
        return self


class RunResult(_SchemaModel):
    schema_version: Literal[RUN_RESULT_SCHEMA_VERSION]
    run_id: NonEmptyStr
    status: Literal["completed", "failed"]
    error: str | None
    experiment: SourceConfig
    runtime: SourceConfig
    code: CodeVersion
    hardware: Hardware
    started_at: dt.datetime
    finished_at: dt.datetime
    latency: RunLatency
    episode_data: EpisodeArtifact
    summary: RunSummary
    study_name: NonEmptyStr | None = Field(default=None, exclude_if=lambda value: value is None)
    study_hash: Sha256 | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("error must be present exactly when status is 'failed'")
        if not _is_utc(self.started_at) or not _is_utc(self.finished_at):
            raise ValueError("started_at and finished_at must be UTC timestamps")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.episode_data.records != self.summary.episodes:
            raise ValueError("episode_data.records must match summary.episodes")
        return self


class ComparisonMetrics(_SchemaModel):
    success_rate: UnitFloat
    mean_episode_steps: NonNegativeFloat


class ComparisonRow(_SchemaModel):
    label: NonEmptyStr
    run_ids: Annotated[list[NonEmptyStr], Field(min_length=1)]
    metrics: ComparisonMetrics


class ComparisonResult(_SchemaModel):
    schema_version: Literal[COMPARISON_RESULT_SCHEMA_VERSION]
    name: NonEmptyStr
    created_at: dt.datetime
    run_ids: Annotated[list[NonEmptyStr], Field(min_length=1)]
    rows: Annotated[list[ComparisonRow], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_runs(self) -> Self:
        if not _is_utc(self.created_at):
            raise ValueError("created_at must be a UTC timestamp")
        if len(self.run_ids) != len(set(self.run_ids)):
            raise ValueError("run_ids must not contain duplicates")
        known = set(self.run_ids)
        for row in self.rows:
            if not set(row.run_ids) <= known:
                raise ValueError(f"row {row.label!r} references an unknown run_id")
        return self


ExperimentDocument = Experiment | DynamicVLAExperiment
RuntimeDocument = Runtime | DynamicVLARuntime
Document = (
    ExperimentDocument | RuntimeDocument | RunResult | ComparisonResult | StudySpec | Placement | DynamicVLAPlacement
)
DocumentType = TypeVar("DocumentType", bound=_SchemaModel)


def load_yaml(path: str | Path, model: type[DocumentType]) -> DocumentType:
    """Load a YAML file as one strict, typed schema model."""

    data: Any = yaml.safe_load(Path(path).read_text())
    return model.model_validate(data)


def load_experiment(path: str | Path) -> ExperimentDocument:
    path = Path(path)
    data: Any = yaml.safe_load(path.read_text())
    benchmark = data.get("benchmark") if isinstance(data, dict) else None
    model = (
        DynamicVLAExperiment
        if isinstance(benchmark, dict) and benchmark.get("kind") == "dynamicvla"
        else Experiment
    )
    return model.model_validate(data)


def load_runtime(path: str | Path) -> RuntimeDocument:
    path = Path(path)
    data: Any = yaml.safe_load(path.read_text())
    model = (
        DynamicVLARuntime
        if isinstance(data, dict) and data.get("kind") == "dynamicvla"
        else Runtime
    )
    return model.model_validate(data)


def load_run_result(path: str | Path) -> RunResult:
    return load_yaml(path, RunResult)


def load_comparison_result(path: str | Path) -> ComparisonResult:
    return load_yaml(path, ComparisonResult)


def write_yaml(document: Document, path: str | Path) -> Path:
    """Write a validated model; callers cannot emit an unvalidated result."""

    path = Path(path)
    return atomic_text(path, yaml.safe_dump(document.model_dump(mode="python"), sort_keys=False))


def _is_utc(value: dt.datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() == dt.timedelta(0)
