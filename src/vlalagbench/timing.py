"""Pure control-step scheduling for chunked policy inference."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal, Sequence, TypeVar


Action = TypeVar("Action")
Scheme = Literal["synchronous", "lazy", "continuous"]


def effective_steps(latency_ms: int, control_hz: int) -> int:
    if latency_ms < 0:
        raise ValueError("latency_ms must be nonnegative")
    if control_hz <= 0:
        raise ValueError("control_hz must be positive")
    return (latency_ms * control_hz + 999) // 1000


@dataclass(frozen=True)
class TimingConfig:
    scheme: Scheme
    action_chunk_steps: int
    execution_horizon_steps: int
    latency_steps: int
    # Which index of a delivered chunk execution starts from. ``None`` keeps the
    # scheme's own rule: synchronous starts at zero because the world was frozen
    # while the policy thought, and the async schemes skip the prefix the
    # latency overtook, capped by the actions actually executed while the
    # request was in flight (a cold start executes from zero). Setting it
    # explicitly separates those two effects and is applied verbatim.
    promotion_offset_steps: int | None = None
    # New DynamicVLA runs count actions after the latency prefix.
    execution_horizon_basis: Literal["action_count", "post_latency_action_count"] = "action_count"

    @property
    def promotion_offset(self) -> int:
        if self.promotion_offset_steps is not None:
            return self.promotion_offset_steps
        return 0 if self.scheme == "synchronous" else self.latency_steps

    def __post_init__(self) -> None:
        if self.scheme not in ("synchronous", "lazy", "continuous"):
            raise ValueError(f"unknown scheme: {self.scheme!r}")
        if self.action_chunk_steps <= 0:
            raise ValueError("action_chunk_steps must be positive")
        if not 0 < self.execution_horizon_steps <= self.action_chunk_steps:
            raise ValueError("execution_horizon_steps must be between 1 and action_chunk_steps")
        if self.latency_steps < 0:
            raise ValueError("latency_steps must be nonnegative")
        if self.execution_horizon_basis not in ("action_count", "post_latency_action_count"):
            raise ValueError("unknown execution horizon basis")
        if self.execution_horizon_basis == "post_latency_action_count" and self.scheme != "synchronous":
            if self.latency_steps + self.execution_horizon_steps > self.action_chunk_steps:
                raise ValueError("async latency plus execution horizon must fit in the action chunk")
            if self.promotion_offset != self.latency_steps:
                raise ValueError("async promotion offset must equal latency steps")
        if self.promotion_offset_steps is None and self.scheme != "synchronous" and self.latency_steps + self.execution_horizon_steps > self.action_chunk_steps:
            raise ValueError("async latency plus execution horizon must fit in the action chunk")
        if self.promotion_offset_steps is not None:
            if self.promotion_offset_steps < 0:
                raise ValueError("promotion_offset_steps must be nonnegative")
            if self.promotion_offset_steps + self.execution_horizon_steps > self.action_chunk_steps:
                raise ValueError("promotion offset plus execution horizon must fit in the action chunk")


@dataclass(frozen=True)
class Request:
    id: int
    sent_step: int
    available_step: int


@dataclass(frozen=True)
class ScheduledAction(Generic[Action]):
    action: Action | None
    active: bool
    reason: str
    chunk_id: int | None = None
    index_in_chunk: int | None = None
    observation_step: int | None = None


@dataclass
class _Pending(Generic[Action]):
    request: Request
    chunk: tuple[Action, ...] | None = None
    executed_steps: int = 0


class ChunkScheduler(Generic[Action]):
    """Schedule sync, lazy-async, and continuous-async chunk execution.

    The horizon counts actions after the promotion offset.
    Sync waits without an active action and then consumes from action zero.
    Lazy inference overlaps the next request with the tail of the current
    horizon; continuous inference replaces the old plan when a prediction is
    released. Where execution starts in an arriving async chunk follows
    ``TimingConfig.promotion_offset_steps``: by default the prefix overtaken by
    the actions executed while the request was in flight (at most the
    configured latency, so a cold start executes from zero); an explicit
    offset is applied verbatim. Only active model actions consume the
    episode's acting-step budget.
    """

    def __init__(self, config: TimingConfig, *, start_step: int = 0):
        self.config = config
        if start_step < 0:
            raise ValueError("start_step must be nonnegative")
        self._next_step = start_step
        self._request_checked = False
        self._next_request_id = 0
        self._pending: _Pending[Action] | None = None
        self._queue: collections.deque[ScheduledAction[Action]] = collections.deque()
        self._released_request: Request | None = None
        self._released_chunk: tuple[Action, ...] | None = None

    def request(self, step: int) -> Request | None:
        """Return a request to issue from this step's current observation."""

        self._require_open_step(step)
        if self._request_checked:
            raise RuntimeError(f"request already checked at step {step}")
        self._released_request = None
        self._released_chunk = None
        self._promote(step)
        self._request_checked = True
        if self._pending is not None or not self._request_due():
            return None
        request = Request(
            id=self._next_request_id,
            sent_step=step,
            available_step=step + self.config.latency_steps,
        )
        self._next_request_id += 1
        self._pending = _Pending(request=request)
        return request

    def deliver(self, request: Request, chunk: Sequence[Action]) -> None:
        """Attach a returned action chunk to its in-flight virtual request."""

        if self._pending is None or self._pending.request != request:
            raise RuntimeError(f"request {request.id} is not in flight")
        if self._pending.chunk is not None:
            raise RuntimeError(f"request {request.id} was already delivered")
        if len(chunk) != self.config.action_chunk_steps:
            raise ValueError(
                f"policy returned {len(chunk)} actions, expected {self.config.action_chunk_steps}"
            )
        self._pending.chunk = tuple(chunk)

    def action(self, step: int) -> ScheduledAction[Action]:
        """Choose this control step's action and advance virtual time."""

        self._require_open_step(step)
        if not self._request_checked:
            raise RuntimeError(f"request must be checked before action at step {step}")
        self._promote(step)
        if self._queue:
            scheduled = self._queue.popleft()
            if self._pending is not None:
                self._pending.executed_steps += 1
        else:
            reason = "inference_wait" if self._pending is not None else "buffer_wait"
            scheduled = ScheduledAction(action=None, active=False, reason=reason)
        self._next_step += 1
        self._request_checked = False
        return scheduled

    @property
    def pending_request(self) -> Request | None:
        return None if self._pending is None else self._pending.request

    @property
    def released_request(self) -> Request | None:
        return self._released_request

    @property
    def released_chunk(self) -> tuple[Action, ...] | None:
        return self._released_chunk

    def snapshot(self) -> dict[str, Any]:
        pending = None
        if self._pending is not None:
            pending = {
                "request": _request_dict(self._pending.request),
                "chunk": (
                    None
                    if self._pending.chunk is None
                    else list(self._pending.chunk)
                ),
                "executed_steps": self._pending.executed_steps,
            }
        return {
            "next_step": self._next_step,
            "next_request_id": self._next_request_id,
            "pending": pending,
            "queue": [
                {
                    "action": item.action,
                    "active": item.active,
                    "reason": item.reason,
                    "chunk_id": item.chunk_id,
                    "index_in_chunk": item.index_in_chunk,
                    "observation_step": item.observation_step,
                }
                for item in self._queue
            ],
        }

    @classmethod
    def from_snapshot(
        cls,
        config: TimingConfig,
        snapshot: dict[str, Any],
        decode_action: Callable[[Any], Action],
    ) -> "ChunkScheduler[Action]":
        scheduler = cls(config)
        scheduler._next_step = int(snapshot["next_step"])
        scheduler._next_request_id = int(snapshot["next_request_id"])
        pending = snapshot.get("pending")
        if pending is not None:
            request = Request(**pending["request"])
            chunk = pending.get("chunk")
            scheduler._pending = _Pending(
                request=request,
                chunk=(
                    None
                    if chunk is None
                    else tuple(decode_action(action) for action in chunk)
                ),
                executed_steps=int(pending.get("executed_steps", 0)),
            )
        scheduler._queue.extend(
            ScheduledAction(
                action=(
                    None
                    if item["action"] is None
                    else decode_action(item["action"])
                ),
                active=bool(item["active"]),
                reason=str(item["reason"]),
                chunk_id=item.get("chunk_id"),
                index_in_chunk=item.get("index_in_chunk"),
                observation_step=item.get("observation_step"),
            )
            for item in snapshot["queue"]
        )
        return scheduler

    def _request_due(self) -> bool:
        if self.config.scheme == "synchronous":
            return not self._queue
        if self.config.scheme == "continuous":
            return True
        return not self._queue or len(self._queue) <= self.config.latency_steps

    def _promote(self, step: int) -> None:
        pending = self._pending
        if pending is None or pending.chunk is None or step < pending.request.available_step:
            return
        first = self.config.promotion_offset
        if self.config.promotion_offset_steps is None:
            first = min(first, pending.executed_steps)
        last = first + self.config.execution_horizon_steps
        self._queue.clear()
        self._queue.extend(
            ScheduledAction(
                action=pending.chunk[index],
                active=True,
                reason="model",
                chunk_id=pending.request.id,
                index_in_chunk=index,
                observation_step=pending.request.sent_step,
            )
            for index in range(first, last)
        )
        self._released_request = pending.request
        self._released_chunk = pending.chunk
        self._pending = None

    def _require_open_step(self, step: int) -> None:
        if step != self._next_step:
            raise RuntimeError(f"expected step {self._next_step}, got {step}")


def _request_dict(request: Request) -> dict[str, int]:
    return {
        "id": request.id,
        "sent_step": request.sent_step,
        "available_step": request.available_step,
    }
