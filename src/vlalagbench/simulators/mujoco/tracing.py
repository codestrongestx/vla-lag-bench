"""Record MuJoCo episodes so any run can be reproduced later.

Works with any robosuite-based env (LIBERO and its variants, plain robosuite
tasks): the only assumptions are the robosuite ``MjSim`` interface reachable
via ``env.sim`` and the gym-style ``step`` return. A trace stores, for one
episode:

- ``states``: the flattened MuJoCo simulator state ``[time, qpos, qvel]``
  (robosuite's ``MjSimState`` layout, which is also LIBERO's init-state
  format) captured right after reset / ``set_init_state`` and again after
  every ``env.step``,
- ``actions``: every action passed to ``env.step``, in order,
- ``rewards`` / ``dones`` per step,
- ``meta``: enough metadata (task, seed, versions, …) to rebuild the same
  environment.

Replaying the recorded states frame by frame reproduces a run exactly,
independent of policy or solver nondeterminism; re-running the recorded
actions from the recorded initial state checks simulator determinism.
See :mod:`vlalagbench.simulators.mujoco.replay`.
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import math
import platform
import subprocess
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

TRACE_SCHEMA_VERSION = 3


class _TraceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EnvironmentMeta(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)

    kind: Annotated[str, Field(min_length=1)]


class TraceMeta(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)

    schema_version: Literal[TRACE_SCHEMA_VERSION]
    environment: EnvironmentMeta
    task_description: str | None = None
    control_freq: Annotated[float, Field(gt=0)]
    num_steps: Annotated[int, Field(ge=0)]
    success: bool


@dataclasses.dataclass
class EpisodeTrace:
    meta: TraceMeta
    states: np.ndarray  # float64 [T+1, 1 + nq + nv]
    actions: np.ndarray  # float64 [T, action_dim]
    rewards: np.ndarray  # float64 [T]
    dones: np.ndarray  # bool [T]
    acting: np.ndarray  # bool [T]
    chunk_ids: np.ndarray  # int64 [T], -1 for holds / settle
    action_indices: np.ndarray  # int64 [T], -1 for holds / settle
    observation_steps: np.ndarray  # int64 [T], -1 for holds / settle
    request_steps: np.ndarray  # int64 [R]
    arrival_steps: np.ndarray  # int64 [R]
    chunks: np.ndarray  # float64 [R, action_horizon, action_dim]
    # Compiled MJCF of the episode's model, so the trace stays reproducible
    # even if the scene assets it was built from later change.
    model_xml: str | None = None
    # Server-side timing per request, aligned with request_steps; NaN / -1
    # when the policy did not report it.
    queue_ms: np.ndarray = dataclasses.field(default_factory=lambda: np.empty(0))
    infer_ms: np.ndarray = dataclasses.field(default_factory=lambda: np.empty(0))
    batch_size: np.ndarray = dataclasses.field(default_factory=lambda: np.empty(0, dtype=np.int64))
    max_batch_size: np.ndarray = dataclasses.field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def __post_init__(self) -> None:
        # A trace written before timing was recorded, or built without it,
        # still aligns one timing row with every request.
        for name, fill, dtype in _REQUEST_TIMING:
            if len(getattr(self, name)) != len(self.request_steps):
                setattr(self, name, np.full(len(self.request_steps), fill, dtype=dtype))

    @property
    def num_steps(self) -> int:
        return len(self.actions)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "meta": self.meta.model_dump_json(),
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "acting": self.acting,
            "chunk_ids": self.chunk_ids,
            "action_indices": self.action_indices,
            "observation_steps": self.observation_steps,
            "request_steps": self.request_steps,
            "arrival_steps": self.arrival_steps,
            "chunks": self.chunks,
            "queue_ms": self.queue_ms,
            "infer_ms": self.infer_ms,
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
        }
        if self.model_xml is not None:
            arrays["model_xml"] = self.model_xml
        np.savez_compressed(path, **arrays)
        return path

    @staticmethod
    def load(path: str | Path) -> "EpisodeTrace":
        with np.load(path, allow_pickle=False) as data:
            timing = {name: data[name] for name, _, _ in _REQUEST_TIMING if name in data.files}
            return EpisodeTrace(
                meta=TraceMeta.model_validate_json(str(data["meta"][()])),
                states=data["states"],
                actions=data["actions"],
                rewards=data["rewards"],
                dones=data["dones"],
                acting=data["acting"],
                chunk_ids=data["chunk_ids"],
                action_indices=data["action_indices"],
                observation_steps=data["observation_steps"],
                request_steps=data["request_steps"],
                arrival_steps=data["arrival_steps"],
                chunks=data["chunks"],
                model_xml=str(data["model_xml"][()]) if "model_xml" in data.files else None,
                **timing,
            )


_REQUEST_TIMING = (
    ("queue_ms", math.nan, np.float64),
    ("infer_ms", math.nan, np.float64),
    ("batch_size", -1, np.int64),
    ("max_batch_size", -1, np.int64),
)


def get_mj_sim(env):
    """Return the underlying robosuite ``MjSim`` from a (possibly wrapped) env."""
    e = env
    for _ in range(5):
        sim = getattr(e, "sim", None)
        if sim is not None:
            return sim
        e = getattr(e, "env", None)
        if e is None:
            break
    raise AttributeError(f"could not find a MuJoCo sim on {type(env).__name__}")


def get_sim_state(env) -> np.ndarray:
    """Flattened MuJoCo state ``[time, qpos, qvel]`` (LIBERO init-state layout)."""
    return np.asarray(get_mj_sim(env).get_state().flatten(), dtype=np.float64)


def set_sim_state(env, state: np.ndarray) -> None:
    sim = get_mj_sim(env)
    sim.set_state_from_flattened(np.asarray(state, dtype=np.float64))
    sim.forward()


class TraceRecordingEnv:
    """Env adapter that records sim states and actions around every call.

    Wraps a LIBERO env (e.g. ``OffScreenRenderEnv``). ``reset`` and
    ``set_init_state`` start a fresh episode buffer whose first state is the
    simulator state right after the call; ``step`` appends the action and the
    post-step state. Everything else is delegated to the wrapped env.
    """

    def __init__(self, env, meta: dict[str, Any] | None = None):
        self.env = env
        self._base_meta = dict(meta or {})
        self._base_meta.setdefault("environment", {"kind": "generic"})
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []
        self._acting: list[bool] = []
        self._chunk_ids: list[int] = []
        self._action_indices: list[int] = []
        self._observation_steps: list[int] = []
        self._request_steps: list[int] = []
        self._arrival_steps: list[int] = []
        self._chunks: list[np.ndarray] = []
        self._timing: list[tuple[float, float, int, int]] = []
        self._request_audits: list[dict | None] = []
        self._next_step_meta: tuple[bool, int, int, int] | None = None
        self._model_xml: str | None = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _begin_episode(self) -> None:
        self._states = [get_sim_state(self.env)]
        self._actions, self._rewards, self._dones = [], [], []
        self._acting, self._chunk_ids, self._action_indices, self._observation_steps = [], [], [], []
        self._request_steps, self._arrival_steps, self._chunks = [], [], []
        self._timing = []
        self._request_audits = []
        self._next_step_meta = None
        # Capture per episode: robosuite's hard_reset recompiles the model on
        # every reset, so the XML is not constant across episodes.
        try:
            self._model_xml = get_mj_sim(self.env).model.get_xml()
        except Exception:
            self._model_xml = None

    def reset(self, *args, **kwargs):
        out = self.env.reset(*args, **kwargs)
        self._begin_episode()
        return out

    def set_init_state(self, *args, **kwargs):
        out = self.env.set_init_state(*args, **kwargs)
        self._begin_episode()
        return out

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        step_meta = self._next_step_meta or (False, -1, -1, -1)
        self._next_step_meta = None
        self._actions.append(np.asarray(action, dtype=np.float64).copy())
        self._states.append(get_sim_state(self.env))
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._acting.append(step_meta[0])
        self._chunk_ids.append(step_meta[1])
        self._action_indices.append(step_meta[2])
        self._observation_steps.append(step_meta[3])
        return obs, reward, done, info

    def record_step(
        self,
        *,
        acting: bool,
        chunk_id: int | None = None,
        action_index: int | None = None,
        observation_step: int | None = None,
    ) -> None:
        """Bind scheduler provenance to the next environment step."""

        if self._next_step_meta is not None:
            raise RuntimeError("scheduler metadata was already set for the next step")
        values = (chunk_id, action_index, observation_step)
        if acting != all(value is not None for value in values):
            raise ValueError("active steps require chunk, action-index, and observation-step provenance")
        self._next_step_meta = (
            acting,
            -1 if chunk_id is None else chunk_id,
            -1 if action_index is None else action_index,
            -1 if observation_step is None else observation_step,
        )

    def record_request(
        self,
        request_id: int,
        sent_step: int,
        arrival_step: int,
        chunk: np.ndarray,
        timing: Mapping[str, float] = {},
    ) -> None:
        """Record every predicted chunk, including actions not ultimately executed."""

        if request_id != len(self._chunks):
            raise ValueError(f"expected request id {len(self._chunks)}, got {request_id}")
        self._request_audits.append(None)
        self._request_steps.append(sent_step)
        self._arrival_steps.append(arrival_step)
        self._chunks.append(np.asarray(chunk, dtype=np.float64).copy())
        self._timing.append(
            tuple(dtype(timing.get(name, fill)) for name, fill, dtype in _REQUEST_TIMING)
        )

    def record_request_audit(self, request_id: int, audit: dict) -> None:
        self._request_audits[request_id] = dict(audit)

    def refresh_current_state(self) -> None:
        """Replace the latest state after a step-free trajectory update."""

        if not self._states:
            raise RuntimeError("no episode recorded; call reset() first")
        self._states[-1] = get_sim_state(self.env)

    def finish(self, extra_meta: dict[str, Any] | None = None) -> EpisodeTrace:
        """Snapshot the current episode buffer as an :class:`EpisodeTrace`."""
        if not self._states:
            raise RuntimeError("no episode recorded; call reset() first")
        meta = {
            "schema_version": TRACE_SCHEMA_VERSION,
            **self._base_meta,
            **collect_env_meta(self.env),
            **collect_runtime_meta(),
            **(extra_meta or {}),
        }
        if any(audit is not None for audit in self._request_audits):
            meta["request_audits"] = self._request_audits
        meta["num_steps"] = len(self._actions)
        meta["success"] = bool(any(self._dones))
        return EpisodeTrace(
            meta=TraceMeta.model_validate(meta),
            states=np.stack(self._states) if self._states else np.empty((0, 0)),
            actions=np.stack(self._actions) if self._actions else np.empty((0, 0)),
            rewards=np.asarray(self._rewards, dtype=np.float64),
            dones=np.asarray(self._dones, dtype=bool),
            acting=np.asarray(self._acting, dtype=bool),
            chunk_ids=np.asarray(self._chunk_ids, dtype=np.int64),
            action_indices=np.asarray(self._action_indices, dtype=np.int64),
            observation_steps=np.asarray(self._observation_steps, dtype=np.int64),
            request_steps=np.asarray(self._request_steps, dtype=np.int64),
            arrival_steps=np.asarray(self._arrival_steps, dtype=np.int64),
            chunks=np.stack(self._chunks) if self._chunks else np.empty((0, 0, 0), dtype=np.float64),
            model_xml=self._model_xml,
            **{
                name: np.asarray([row[index] for row in self._timing], dtype=dtype)
                for index, (name, _, dtype) in enumerate(_REQUEST_TIMING)
            },
        )


def collect_env_meta(env) -> dict[str, Any]:
    sim = get_mj_sim(env)
    meta: dict[str, Any] = {
        "nq": int(sim.data.qpos.size),
        "nv": int(sim.data.qvel.size),
        "state_dim": 1 + int(sim.data.qpos.size) + int(sim.data.qvel.size),
    }
    control_freq = _find_attr(env, "control_freq")
    if control_freq is not None:
        meta["control_freq"] = float(control_freq)
    return meta


def _find_attr(env, name):
    """Look up an attribute anywhere along the env wrapper chain."""
    e = env
    for _ in range(5):
        if e is None:
            break
        value = getattr(e, name, None)
        if value is not None:
            return value
        e = e.__dict__.get("env") if hasattr(e, "__dict__") else None
    return None


def collect_runtime_meta() -> dict[str, Any]:
    import mujoco
    import robosuite

    repo_root = Path(__file__).resolve().parents[4]
    return {
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "mujoco_version": mujoco.__version__,
        "robosuite_version": robosuite.__version__,
        "numpy_version": np.__version__,
        "repo_commit": git_commit(repo_root),
    }


@functools.cache
def git_commit(repo: Path) -> str | None:
    """HEAD commit of ``repo`` (with ``-dirty`` suffix), or None outside git.

    Cached per process: the evaluator records it on every episode, and the
    process runs the code it loaded at start regardless of later edits.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:
        return None
