"""Synthetic data types for the primal LP helper layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor
from typing import Mapping


@dataclass(frozen=True)
class LPRequestSnapshot:
    """Synthetic per-request LP input.

    Memory coefficients are scalar planning units. For Phase 12.3 synthetic
    tests they represent KV blocks; later vLLM integration must still validate
    any translated action with native allocation.
    """

    request_id: str
    remaining_prefill_tokens: int = 0
    max_prefill_chunk_tokens: int = 0
    decode_eligible: bool = False
    preemptible: bool = False
    prefill_eligible: bool = True
    prefill_memory_blocks_per_token: float = 0.0
    decode_memory_blocks: float = 0.0
    preempt_recoverable_blocks: float = 0.0
    arrival_time: float = 0.0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.remaining_prefill_tokens < 0:
            raise ValueError("remaining_prefill_tokens must be non-negative")
        if self.max_prefill_chunk_tokens < 0:
            raise ValueError("max_prefill_chunk_tokens must be non-negative")
        for name in (
            "prefill_memory_blocks_per_token",
            "decode_memory_blocks",
            "preempt_recoverable_blocks",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def max_prefill_tokens_this_step(self) -> int:
        if not self.prefill_eligible:
            return 0
        return min(self.remaining_prefill_tokens, self.max_prefill_chunk_tokens)


@dataclass(frozen=True)
class LPCapacities:
    """Global synthetic capacities for one LP scheduler step."""

    token_budget: int
    sequence_budget: int
    free_memory_blocks: float
    lp_memory_reserve_blocks: float = 0.0

    def __post_init__(self) -> None:
        if self.token_budget < 0:
            raise ValueError("token_budget must be non-negative")
        if self.sequence_budget < 0:
            raise ValueError("sequence_budget must be non-negative")
        if self.free_memory_blocks < 0:
            raise ValueError("free_memory_blocks must be non-negative")
        if self.lp_memory_reserve_blocks < 0:
            raise ValueError("lp_memory_reserve_blocks must be non-negative")

    @property
    def usable_memory_blocks(self) -> float:
        return self.free_memory_blocks - self.lp_memory_reserve_blocks


@dataclass(frozen=True)
class RequestUtilityWeights:
    """Objective weights for one request."""

    alpha: float
    beta: float
    gamma: float


@dataclass(frozen=True)
class LPUtilityWeights:
    """Objective weights keyed by request id."""

    by_request_id: Mapping[str, RequestUtilityWeights]

    def for_request(self, request_id: str) -> RequestUtilityWeights:
        try:
            return self.by_request_id[request_id]
        except KeyError as exc:
            raise KeyError(f"missing LP utility weights for {request_id!r}") from exc


@dataclass(frozen=True)
class RelaxedLPSolution:
    """Continuous LP relaxation output."""

    success: bool
    status: int
    message: str
    objective_value: float | None = None
    prefill_tokens: dict[str, float] = field(default_factory=dict)
    decode: dict[str, float] = field(default_factory=dict)
    preempt: dict[str, float] = field(default_factory=dict)
    prefill_admission: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LPResidualCapacities:
    """Capacities remaining after integral LP decisions are locked."""

    token_budget: int
    sequence_budget: int
    memory_blocks: float


@dataclass(frozen=True)
class LPActionPlan:
    """Integer action plan produced by LP extraction.

    This is not a vLLM SchedulerOutput and must not be returned from a vLLM
    scheduler directly.
    """

    prefill_tokens: dict[str, int]
    decode: dict[str, int]
    preempt: dict[str, int]
    prefill_admission: dict[str, int]
    solver_success: bool
    solver_status: int
    solver_message: str
    objective_value: float | None = None
    integral_request_ids: tuple[str, ...] = ()
    fractional_request_ids: tuple[str, ...] = ()
    fractional_rule_violation: bool = False
    forced_preemption_request_id: str | None = None
    residual_capacities: LPResidualCapacities | None = None

    @classmethod
    def empty_for(
        cls,
        requests: list[LPRequestSnapshot],
        *,
        solver_success: bool,
        solver_status: int,
        solver_message: str,
        objective_value: float | None = None,
    ) -> "LPActionPlan":
        ids = [request.request_id for request in requests]
        zeros = {request_id: 0 for request_id in ids}
        return cls(
            prefill_tokens=zeros.copy(),
            decode=zeros.copy(),
            preempt=zeros.copy(),
            prefill_admission=zeros.copy(),
            solver_success=solver_success,
            solver_status=solver_status,
            solver_message=solver_message,
            objective_value=objective_value,
        )


def floor_nonnegative(value: float, *, tol: float) -> int:
    """Floor a non-negative LP value while absorbing small solver noise."""
    if value <= tol:
        return 0
    return max(0, int(floor(value + tol)))
