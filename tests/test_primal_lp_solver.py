from vllm_scheduler_policies.primal_lp import (
    LPCapacities,
    LPRequestSnapshot,
    LPUtilityWeights,
    RequestUtilityWeights,
    compute_default_like_weights,
    solve_lp_relaxation,
)


def _weights(**values):
    return LPUtilityWeights(
        {
            request_id: RequestUtilityWeights(*weights)
            for request_id, weights in values.items()
        }
    )


def test_prefill_only_case_schedules_full_chunk():
    requests = [
        LPRequestSnapshot(
            request_id="prefill-a",
            remaining_prefill_tokens=5,
            max_prefill_chunk_tokens=5,
            prefill_memory_blocks_per_token=1,
        )
    ]

    plan = solve_lp_relaxation(
        requests,
        LPCapacities(token_budget=8, sequence_budget=1, free_memory_blocks=8),
        _weights(**{"prefill-a": (100.0, 2.0, 1000.0)}),
    )

    assert plan.solver_success
    assert plan.prefill_tokens["prefill-a"] == 5
    assert plan.prefill_admission["prefill-a"] == 1
    assert plan.decode["prefill-a"] == 0
    assert plan.preempt["prefill-a"] == 0


def test_decode_dominance_uses_sequence_budget_for_decode():
    requests = [
        LPRequestSnapshot(
            request_id="decode-a",
            decode_eligible=True,
            decode_memory_blocks=1,
        ),
        LPRequestSnapshot(
            request_id="prefill-a",
            remaining_prefill_tokens=4,
            max_prefill_chunk_tokens=4,
            prefill_memory_blocks_per_token=1,
        ),
    ]

    plan = solve_lp_relaxation(
        requests,
        LPCapacities(token_budget=4, sequence_budget=1, free_memory_blocks=4),
        _weights(
            **{
                "decode-a": (1_000_000.0, 1.0, 1000.0),
                "prefill-a": (1.0, 10.0, 1000.0),
            }
        ),
    )

    assert plan.solver_success
    assert plan.decode["decode-a"] == 1
    assert plan.prefill_tokens["prefill-a"] == 0


def test_memory_limited_prefill_chunking():
    requests = [
        LPRequestSnapshot(
            request_id="prefill-a",
            remaining_prefill_tokens=10,
            max_prefill_chunk_tokens=10,
            prefill_memory_blocks_per_token=2,
        )
    ]

    plan = solve_lp_relaxation(
        requests,
        LPCapacities(token_budget=10, sequence_budget=1, free_memory_blocks=5),
        _weights(**{"prefill-a": (1.0, 10.0, 1000.0)}),
    )

    assert plan.solver_success
    assert plan.prefill_tokens["prefill-a"] == 2
    assert plan.prefill_admission["prefill-a"] == 1


def test_memory_reserve_reduces_usable_lp_capacity():
    requests = [
        LPRequestSnapshot(
            request_id="prefill-a",
            remaining_prefill_tokens=10,
            max_prefill_chunk_tokens=10,
            prefill_memory_blocks_per_token=1,
        )
    ]

    plan = solve_lp_relaxation(
        requests,
        LPCapacities(
            token_budget=10,
            sequence_budget=1,
            free_memory_blocks=8,
            lp_memory_reserve_blocks=3,
        ),
        _weights(**{"prefill-a": (1.0, 10.0, 1000.0)}),
    )

    assert plan.solver_success
    assert plan.prefill_tokens["prefill-a"] == 5


def test_solver_failure_returns_empty_plan_for_negative_usable_memory():
    requests = [
        LPRequestSnapshot(
            request_id="prefill-a",
            remaining_prefill_tokens=2,
            max_prefill_chunk_tokens=2,
            prefill_memory_blocks_per_token=1,
        )
    ]

    plan = solve_lp_relaxation(
        requests,
        LPCapacities(
            token_budget=2,
            sequence_budget=1,
            free_memory_blocks=1,
            lp_memory_reserve_blocks=2,
        ),
        _weights(**{"prefill-a": (1.0, 10.0, 1000.0)}),
    )

    assert not plan.solver_success
    assert plan.prefill_tokens == {"prefill-a": 0}
    assert plan.decode == {"prefill-a": 0}
    assert plan.preempt == {"prefill-a": 0}


def test_default_like_weight_helpers_use_documented_formulas():
    weights = compute_default_like_weights(
        [LPRequestSnapshot(request_id="r0", arrival_time=90.0)],
        current_time=100.0,
        decode_weight=123.0,
        prefill_age_coefficient=0.5,
        preemption_age_scale=10.0,
    ).for_request("r0")

    assert weights.alpha == 123.0
    assert weights.beta == 6.0
    assert weights.gamma == 100.0
