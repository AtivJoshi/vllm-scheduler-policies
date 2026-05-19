"""Default-like LP utility weight helpers."""

from __future__ import annotations

from collections.abc import Iterable

from vllm_scheduler_policies.primal_lp.types import (
    LPRequestSnapshot,
    LPUtilityWeights,
    RequestUtilityWeights,
)


def _age(current_time: float, arrival_time: float) -> float:
    return max(0.0, current_time - arrival_time)


def default_like_alpha(*, decode_weight: float = 1_000_000.0) -> float:
    """Large finite decode utility approximating decode dominance."""
    return decode_weight


def default_like_beta(
    *,
    current_time: float,
    arrival_time: float,
    age_coefficient: float = 1.0e-3,
) -> float:
    """FCFS-like prefill utility: 1 + c * age."""
    return 1.0 + age_coefficient * _age(current_time, arrival_time)


def default_like_gamma(
    *,
    current_time: float,
    arrival_time: float,
    preemption_age_scale: float = 1.0e9,
) -> float:
    """Large age-scaled preemption penalty."""
    return preemption_age_scale * _age(current_time, arrival_time)


def compute_default_like_weights(
    requests: Iterable[LPRequestSnapshot],
    *,
    current_time: float,
    decode_weight: float = 1_000_000.0,
    prefill_age_coefficient: float = 1.0e-3,
    preemption_age_scale: float = 1.0e9,
) -> LPUtilityWeights:
    """Build documented default-like utility weights for synthetic snapshots."""
    weights = {}
    for request in requests:
        weights[request.request_id] = RequestUtilityWeights(
            alpha=default_like_alpha(decode_weight=decode_weight),
            beta=default_like_beta(
                current_time=current_time,
                arrival_time=request.arrival_time,
                age_coefficient=prefill_age_coefficient,
            ),
            gamma=default_like_gamma(
                current_time=current_time,
                arrival_time=request.arrival_time,
                preemption_age_scale=preemption_age_scale,
            ),
        )
    return LPUtilityWeights(weights)
