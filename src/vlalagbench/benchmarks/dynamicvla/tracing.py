"""Serializable, replayable traces for DynamicVLA on Isaac Lab.

A trace stores, for one episode, T+1 Isaac Lab state snapshots, the T control
actions, per-step termination outcomes and policy protocol snapshots, the final
policy state, and the policy's non-image observations (the end-effector state
it was sent). Camera frames are not stored by default, as in LIBERO traces:
state playback re-renders them (``vlalagbench dynamicvla-trace TRACE RUNTIME
--video``). A runtime with ``trace_images: true`` also stores every policy
camera frame, which costs roughly 0.25 MB per camera per control step.

State replay is exact, but Isaac's RTX render (DLSS, denoiser, temporal
antialiasing) keeps history outside the saved state, so re-rendered frames are
not guaranteed to be bitwise identical to the RGB the policy saw. Only traces
recorded with ``trace_images`` contain that exact RGB.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import platform
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

try:
    from typing import Self
except ImportError:  # Python 3.10 in the pinned Isaac Lab environment
    from typing_extensions import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vlalagbench.benchmarks.dynamicvla.motion import EpisodeMotion
from vlalagbench.simulators.isaaclab.tracing import FlatState, StatePath, capture_env_state


TRACE_SCHEMA_VERSION = 10


class ContinuationMeta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_trace: str
    source_trace_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    resume_step: Annotated[int, Field(ge=1)]
    original_num_steps: Annotated[int, Field(ge=1)]
    max_additional_steps: Annotated[int, Field(gt=0)]
    protocol_receipt_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$")
    ]
    # Missing in historical continuations, which always sent recorded frames.
    # "rerendered" history comes from restored states of a trace without
    # frames and need not be bitwise identical to what the policy first saw.
    history_images: Literal["recorded", "rerendered"] = "recorded"


class DynamicVLATraceMeta(BaseModel):
    """Identity and reproduction inputs for one DOM episode."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[5, 6, 7, 8, 9, TRACE_SCHEMA_VERSION]
    task_id: Annotated[int, Field(ge=0)]
    task_name: Annotated[str, Field(min_length=1)]
    episode: Annotated[int, Field(ge=0)]
    environment_index: Annotated[int, Field(ge=0)]
    replica_id: Annotated[int, Field(ge=0)]
    environment_path: str
    environment_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    environment: dict[str, Any]
    asset_sha256: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    instruction: str
    seed: int
    upstream_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    checkpoint: str
    checkpoint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    checkpoint_config_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    policy_noise_scheme: Literal[
        "sha256-seeded-cpu-torch-randn-v1",
        "sha256-policy-task-environment-replica-request-cpu-torch-v2",
    ]
    policy_torch_version: str
    policy_seed: Annotated[int, Field(ge=0)]
    policy_episode_seed: Annotated[int, Field(ge=0)]
    episode_length_s: Annotated[float, Field(gt=0)]
    deadline_basis: Literal["control_steps", "active_action_steps"] = "control_steps"
    control_hz: Annotated[float, Field(gt=0)]
    physics_time_step: Annotated[float, Field(gt=0)]
    termination_tolerance: Annotated[float, Field(gt=0)]
    disable_fabric: bool
    path_tracing: bool
    simulator_versions: dict[str, str]
    inference_mode: Literal["synchronous", "lazy", "continuous"]
    action_index_origin: Literal["observation", "release"]
    latency_steps: Annotated[int, Field(ge=0)]
    action_chunk_steps: Annotated[int, Field(gt=0)]
    execution_horizon_steps: Annotated[int, Field(gt=0)]
    # Missing in historical traces. New output explicitly identifies the
    # validated horizon after the full latency prefix.
    execution_horizon_basis: Literal["action_count", "post_latency_action_count"] = "action_count"
    # Missing in historical traces, whose waits repeated the last pose target.
    idle_control: Literal["repeat_last_action", "hold_pose"] = "repeat_last_action"
    replica_batch_size: Annotated[int, Field(gt=0)] = 1
    # Missing in historical traces: official state, no authored motion.
    initial_state_index: Annotated[int, Field(ge=0)] = 0
    motion: EpisodeMotion | None = None
    success_term: Annotated[str, Field(min_length=1)]
    timeout_terms: list[str]
    num_steps: Annotated[int, Field(ge=0)]
    status: int
    success: bool
    recorded_at: str
    python_version: str
    numpy_version: str
    continuation: ContinuationMeta | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_action_index_origin(cls, value: Any) -> Any:
        if not isinstance(value, Mapping) or "action_index_origin" in value:
            return value
        try:
            version = int(value.get("schema_version", -1))
        except (OverflowError, TypeError, ValueError):
            return value
        if version not in (5, 6):
            return value
        return {
            **value,
            "action_index_origin": (
                "observation"
                if value.get("inference_mode") == "continuous"
                else "release"
            ),
        }

    @model_validator(mode="after")
    def validate_inference_schedule(self) -> Self:
        if self.execution_horizon_steps > self.action_chunk_steps:
            raise ValueError("execution horizon exceeds the action chunk")
        if self.execution_horizon_basis == "post_latency_action_count":
            expected_origin = (
                "release" if self.inference_mode == "synchronous" else "observation"
            )
            if self.action_index_origin != expected_origin:
                raise ValueError("trace has an invalid action index origin")
            if (
                self.inference_mode != "synchronous"
                and self.latency_steps + self.execution_horizon_steps > self.action_chunk_steps
            ):
                raise ValueError("async latency plus execution horizon must fit in the action chunk")
            return self
        if self.inference_mode == "continuous":
            if self.action_index_origin == "observation":
                valid = (
                    self.latency_steps + self.execution_horizon_steps
                    == self.action_chunk_steps
                )
            else:
                valid = self.execution_horizon_steps == self.action_chunk_steps
            if not valid:
                raise ValueError("continuous trace has an invalid execution window")
        elif self.inference_mode == "lazy" and self.action_index_origin == "observation":
            if self.latency_steps + self.execution_horizon_steps > self.action_chunk_steps:
                raise ValueError("async latency plus execution horizon must fit in the action chunk")
        elif self.action_index_origin != "release":
            raise ValueError("horizon-mode trace has an invalid action index origin")
        return self


@dataclasses.dataclass
class DynamicVLATrace:
    """One episode's physical, control, outcome, and protocol state."""

    meta: DynamicVLATraceMeta
    states: dict[StatePath, np.ndarray]
    observations: dict[StatePath, np.ndarray]
    actions: np.ndarray
    success: np.ndarray
    done: np.ndarray
    terminations: list[dict[str, bool]]
    protocol: list[dict[str, Any]]
    final_policy: dict[str, Any]

    @property
    def num_steps(self) -> int:
        return len(self.actions)

    @property
    def has_images(self) -> bool:
        """Whether the policy camera frames were stored with the trace."""

        return any(is_image_observation(path) for path in self.observations)

    def state(self, index: int) -> FlatState:
        return {path: values[index].copy() for path, values in self.states.items()}

    def observation(self, index: int) -> FlatState:
        return {
            path: values[index].copy() for path, values in self.observations.items()
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_manifest = []
        observation_manifest = []
        arrays: dict[str, Any] = {
            "meta": self.meta.model_dump_json(),
            "actions": self.actions,
            "success": self.success,
            "done": self.done,
            "terminations": json.dumps(
                self.terminations, sort_keys=True, separators=(",", ":")
            ),
            "protocol": json.dumps(self.protocol, sort_keys=True, separators=(",", ":")),
            "final_policy": json.dumps(
                self.final_policy, sort_keys=True, separators=(",", ":")
            ),
        }
        for index, (state_path, values) in enumerate(sorted(self.states.items())):
            key = f"state_{index:04d}"
            state_manifest.append({"key": key, "path": list(state_path)})
            arrays[key] = values
        for index, (observation_path, values) in enumerate(
            sorted(self.observations.items())
        ):
            key = f"observation_{index:04d}"
            observation_manifest.append(
                {"key": key, "path": list(observation_path)}
            )
            arrays[key] = values
        arrays["state_manifest"] = json.dumps(
            state_manifest, separators=(",", ":")
        )
        arrays["observation_manifest"] = json.dumps(
            observation_manifest, separators=(",", ":")
        )
        _savez_compressed(path, arrays)
        return path

    @staticmethod
    def load(path: str | Path) -> "DynamicVLATrace":
        with np.load(path, allow_pickle=False) as data:
            manifest = json.loads(str(data["state_manifest"][()]))
            observation_manifest = json.loads(str(data["observation_manifest"][()]))
            states = {
                tuple(item["path"]): data[item["key"]].copy() for item in manifest
            }
            observations = {
                tuple(item["path"]): data[item["key"]].copy()
                for item in observation_manifest
            }
            state_keys = [item["key"] for item in manifest]
            observation_keys = [item["key"] for item in observation_manifest]
            if (
                len(states) != len(manifest)
                or len(observations) != len(observation_manifest)
                or len(state_keys) != len(set(state_keys))
                or len(observation_keys) != len(set(observation_keys))
            ):
                raise ValueError("trace manifests contain duplicate keys or tensor paths")
            trace = DynamicVLATrace(
                meta=DynamicVLATraceMeta.model_validate(
                    _migrate_meta(json.loads(str(data["meta"][()])))
                ),
                states=states,
                observations=observations,
                actions=data["actions"].copy(),
                success=data["success"].copy(),
                done=data["done"].copy(),
                terminations=json.loads(str(data["terminations"][()])),
                protocol=json.loads(str(data["protocol"][()])),
                final_policy=json.loads(str(data["final_policy"][()])),
            )
        trace.validate()
        return trace

    def validate(self) -> None:
        if self.meta.num_steps != self.num_steps:
            raise ValueError("trace metadata step count does not match actions")
        if self.actions.shape != (self.num_steps, 1, 8):
            raise ValueError("trace actions must have shape (T, 1, 8)")
        if not np.isfinite(self.actions).all():
            raise ValueError("trace actions must be finite")
        if len(self.protocol) != self.num_steps:
            raise ValueError("trace must contain one protocol snapshot per action")
        if self.success.shape != (self.num_steps,) or self.done.shape != (
            self.num_steps,
        ):
            raise ValueError("trace outcomes must contain one scalar per action")
        if self.success.dtype != np.bool_ or self.done.dtype != np.bool_:
            raise ValueError("trace outcomes must be boolean arrays")
        if len(self.terminations) != self.num_steps:
            raise ValueError("trace must contain termination outcomes for every action")
        if self.terminations:
            term_names = set(self.terminations[0])
            required_terms = {self.meta.success_term, *self.meta.timeout_terms}
            if (
                not required_terms.issubset(term_names)
                or len(self.meta.timeout_terms) != len(set(self.meta.timeout_terms))
                or any(set(terms) != term_names for terms in self.terminations)
                or any(
                    not name or type(value) is not bool
                    for terms in self.terminations
                    for name, value in terms.items()
                )
            ):
                raise ValueError("trace termination structure is inconsistent")
        if any(
            bool(any(terms.values())) != bool(done)
            or bool(terms.get(self.meta.success_term, False)) != bool(success)
            for terms, done, success in zip(
                self.terminations, self.done, self.success, strict=True
            )
        ):
            raise ValueError("trace termination outcomes disagree with done or success")
        continuation = self.meta.continuation
        if continuation is not None:
            from vlalagbench.benchmarks.dynamicvla.protocol import (
                audited_trace_receipt,
                receipt_sha256,
            )

            receipt = audited_trace_receipt(self)
            if (
                not receipt["exact"]
                or receipt_sha256(receipt) != continuation.protocol_receipt_sha256
            ):
                raise ValueError("continued trace protocol receipt is invalid")
        expected_states = self.num_steps + 1
        if not self.states or any(
            len(values) != expected_states for values in self.states.values()
        ):
            raise ValueError("trace state tensors must contain T+1 snapshots")
        if not self.observations or any(
            len(values) != expected_states for values in self.observations.values()
        ):
            raise ValueError("trace policy observations must contain T+1 snapshots")
        _validate_tensor_group(self.states, name="state", allow_rng_sentinel=True)
        _validate_tensor_group(self.observations, name="observation")
        if not any(
            path[:2] == ("state", "end_effector") for path in self.observations
        ):
            raise ValueError("trace observations need end-effector state")
        if bool(self.success.any()) != self.meta.success:
            raise ValueError("trace success states disagree with metadata")
        if (self.meta.status == 0) != self.meta.success:
            raise ValueError("trace status disagrees with success metadata")
        if self.num_steps > 1 and self.done[:-1].any():
            raise ValueError("trace continues after a terminal state")
        if self.meta.environment.get("seed") != self.meta.seed:
            raise ValueError("trace environment seed disagrees with metadata")
        if not all(isinstance(step, dict) for step in self.protocol) or not isinstance(
            self.final_policy, dict
        ):
            raise ValueError("trace policy snapshots must be JSON objects")


class DynamicVLATraceRecorder:
    """Collect an episode at the simulator's control-step boundary.

    Policy camera frames are dropped unless ``images`` is set.
    """

    def __init__(self, env: Any, *, images: bool = False):
        self._env = env
        self._images = images
        first = capture_env_state(env)
        self._states: dict[StatePath, list[np.ndarray]] = {
            path: [value] for path, value in first.items()
        }
        self._observations: dict[StatePath, list[np.ndarray]] | None = None
        self._actions: list[np.ndarray] = []
        self._success: list[bool] = []
        self._done: list[bool] = []
        self._terminations: list[dict[str, bool]] = []
        self._protocol: list[dict[str, Any]] = []
        self._post_step_state: FlatState | None = None

    def capture_post_step(self) -> None:
        """Capture terminal physics before ManagerBasedRLEnv auto-resets it."""

        self._post_step_state = capture_env_state(self._env)

    def record_observation(self, observation: Mapping[StatePath, Any]) -> None:
        values = {
            path: _as_array(value)
            for path, value in observation.items()
            if self._images or not is_image_observation(path)
        }
        if self._observations is None:
            self._observations = {path: [value] for path, value in values.items()}
            return
        if values.keys() != self._observations.keys():
            raise RuntimeError("DynamicVLA observation structure changed during the episode")
        for path, value in values.items():
            self._observations[path].append(value)

    def record_step(
        self,
        action: Any,
        *,
        success: bool,
        done: bool,
        terminations: Mapping[str, bool],
        protocol: Mapping[str, Any] | None,
    ) -> None:
        state = self._post_step_state or capture_env_state(self._env)
        self._post_step_state = None
        if state.keys() != self._states.keys():
            raise RuntimeError("Isaac Lab state structure changed during the episode")
        for path, value in state.items():
            self._states[path].append(value)
        self._actions.append(_as_array(action))
        self._success.append(success)
        self._done.append(done)
        self._terminations.append(dict(terminations))
        self._protocol.append(_jsonable(dict(protocol or {})))

    def finish(
        self, metadata: Mapping[str, Any], *, final_policy: Mapping[str, Any]
    ) -> DynamicVLATrace:
        if self._observations is None:
            raise RuntimeError("trace has no policy observations")
        meta = DynamicVLATraceMeta.model_validate(
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                **metadata,
                "num_steps": len(self._actions),
                "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
            }
        )
        trace = DynamicVLATrace(
            meta=meta,
            states={path: np.stack(values) for path, values in self._states.items()},
            observations={
                path: np.stack(values) for path, values in self._observations.items()
            },
            actions=(
                np.stack(self._actions)
                if self._actions
                else np.empty((0, 1, 8), dtype=np.float32)
            ),
            success=np.asarray(self._success, dtype=bool),
            done=np.asarray(self._done, dtype=bool),
            terminations=self._terminations,
            protocol=self._protocol,
            final_policy=_jsonable(dict(final_policy)),
        )
        trace.validate()
        return trace


def is_image_observation(path: StatePath) -> bool:
    return path[:1] == ("images",)


def _savez_compressed(path: Path, arrays: Mapping[str, Any]) -> None:
    """Write a standard lossless NPZ without NumPy's costly level-6 default."""

    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for key, value in arrays.items():
            with archive.open(f"{key}.npy", mode="w", force_zip64=True) as stream:
                np.lib.format.write_array(
                    stream,
                    np.asanyarray(value),
                    allow_pickle=False,
                )


def _as_array(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach().cpu().numpy()
    return np.asarray(value).copy()


def _migrate_meta(metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        version = int(metadata.get("schema_version", -1))
    except (OverflowError, TypeError, ValueError):
        raise ValueError("trace schema version must be an integer") from None
    if version == 1:
        metadata = {
            **metadata,
            "task_name": Path(metadata["environment_path"]).stem,
            "environment_index": int(metadata["task_id"]),
            "replica_id": int(metadata["episode"]),
        }
    if version in (1, 2):
        legacy_payload = (
            f"{metadata['seed']}\0{metadata['instruction']}".encode()
        )
        metadata = {
            **metadata,
            "policy_episode_seed": int.from_bytes(
                hashlib.sha256(legacy_payload).digest()[:8], "little"
            )
            & ((1 << 63) - 1),
            "policy_seed": 0,
        }
    if version in (1, 2, 3):
        metadata = {
            **metadata,
            "inference_mode": "continuous",
            "execution_horizon_steps": (
                int(metadata["action_chunk_steps"])
                - int(metadata["latency_steps"])
            ),
        }
    if version in (1, 2, 3, 4, 5, 6):
        metadata = {
            **metadata,
            "action_index_origin": (
                "observation"
                if metadata["inference_mode"] == "continuous"
                else "release"
            ),
        }
    if version in (1, 2, 3, 4):
        metadata = {**metadata, "schema_version": TRACE_SCHEMA_VERSION}
    return metadata


def _validate_tensor_group(
    values: Mapping[StatePath, np.ndarray],
    *,
    name: str,
    allow_rng_sentinel: bool = False,
) -> None:
    for path, array in values.items():
        if not path or any(not component for component in path):
            raise ValueError(f"trace {name} tensor paths must not be empty")
        if array.dtype.kind not in "biuf":
            raise ValueError(f"trace {name} tensors must be numeric")
        finite = np.isfinite(array)
        if allow_rng_sentinel and path == ("runtime", "rng", "python_gauss"):
            finite = finite | np.isnan(array)
        if not finite.all():
            raise ValueError(f"trace {name} tensors contain invalid values")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    detach = getattr(value, "detach", None)
    if detach is not None:
        return detach().cpu().numpy().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
