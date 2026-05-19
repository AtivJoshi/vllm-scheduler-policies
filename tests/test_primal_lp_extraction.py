from vllm_scheduler_policies.primal_lp import (
    LPCapacities,
    LPRequestSnapshot,
    RelaxedLPSolution,
    extract_fractionals,
    partition_integral_requests,
)


def test_partition_integral_requests_uses_binary_tolerance():
    requests = [
        LPRequestSnapshot(request_id="integral"),
        LPRequestSnapshot(request_id="fractional"),
    ]
    solution = RelaxedLPSolution(
        success=True,
        status=0,
        message="ok",
        decode={"integral": 1.0 - 1e-8, "fractional": 0.4},
        preempt={"integral": 0.0, "fractional": 0.0},
        prefill_admission={"integral": 0.0, "fractional": 0.6},
    )

    integral, fractional = partition_integral_requests(
        requests, solution, tol=1e-6
    )

    assert integral == ["integral"]
    assert fractional == ["fractional"]


def test_dominant_preemption_is_extracted_first():
    requests = [
        LPRequestSnapshot(
            request_id="frac-preempt",
            preemptible=True,
            preempt_recoverable_blocks=7,
        )
    ]
    solution = RelaxedLPSolution(
        success=True,
        status=0,
        message="ok",
        decode={"frac-preempt": 0.1},
        preempt={"frac-preempt": 0.7},
        prefill_admission={"frac-preempt": 0.2},
    )

    plan = extract_fractionals(
        requests,
        LPCapacities(token_budget=1, sequence_budget=1, free_memory_blocks=0),
        solution,
    )

    assert plan.preempt["frac-preempt"] == 1
    assert plan.decode["frac-preempt"] == 0
    assert plan.prefill_tokens["frac-preempt"] == 0


def test_fractional_dominant_action_tie_prefers_preemption_then_decode():
    requests = [
        LPRequestSnapshot(
            request_id="tie-all",
            decode_eligible=True,
            preemptible=True,
            decode_memory_blocks=1,
            preempt_recoverable_blocks=2,
        ),
        LPRequestSnapshot(
            request_id="tie-admit",
            remaining_prefill_tokens=4,
            max_prefill_chunk_tokens=4,
            decode_eligible=True,
            prefill_memory_blocks_per_token=1,
            decode_memory_blocks=1,
        ),
    ]
    solution = RelaxedLPSolution(
        success=True,
        status=0,
        message="ok",
        decode={"tie-all": 0.5, "tie-admit": 0.5},
        preempt={"tie-all": 0.5, "tie-admit": 0.1},
        prefill_admission={"tie-all": 0.5, "tie-admit": 0.5},
    )

    plan = extract_fractionals(
        requests,
        LPCapacities(token_budget=2, sequence_budget=2, free_memory_blocks=2),
        solution,
    )

    assert plan.preempt["tie-all"] == 1
    assert plan.decode["tie-admit"] == 1
    assert plan.prefill_tokens["tie-admit"] == 0


def test_forced_safety_preemption_uses_largest_fractional_z():
    requests = [
        LPRequestSnapshot(
            request_id="needs-memory-a",
            preemptible=True,
            preempt_recoverable_blocks=1,
        ),
        LPRequestSnapshot(
            request_id="needs-memory-b",
            preemptible=True,
            preempt_recoverable_blocks=5,
        ),
    ]
    solution = RelaxedLPSolution(
        success=True,
        status=0,
        message="ok",
        decode={"needs-memory-a": 0.6, "needs-memory-b": 0.5},
        preempt={"needs-memory-a": 0.2, "needs-memory-b": 0.4},
        prefill_admission={"needs-memory-a": 0.2, "needs-memory-b": 0.1},
    )

    plan = extract_fractionals(
        requests,
        LPCapacities(
            token_budget=2,
            sequence_budget=2,
            free_memory_blocks=0,
            lp_memory_reserve_blocks=3,
        ),
        solution,
    )

    assert plan.forced_preemption_request_id == "needs-memory-b"
    assert plan.preempt["needs-memory-b"] == 1
    assert plan.preempt["needs-memory-a"] == 0


def test_integral_zero_token_prefill_admission_is_normalized_away():
    requests = [LPRequestSnapshot(request_id="zero-admit")]
    solution = RelaxedLPSolution(
        success=True,
        status=0,
        message="ok",
        prefill_tokens={"zero-admit": 0.0},
        decode={"zero-admit": 0.0},
        preempt={"zero-admit": 0.0},
        prefill_admission={"zero-admit": 1.0},
    )

    plan = extract_fractionals(
        requests,
        LPCapacities(token_budget=1, sequence_budget=1, free_memory_blocks=1),
        solution,
    )

    assert plan.integral_request_ids == ("zero-admit",)
    assert plan.prefill_tokens["zero-admit"] == 0
    assert plan.prefill_admission["zero-admit"] == 0


def test_fractional_prefill_chunk_is_limited_by_memory_and_tokens():
    requests = [
        LPRequestSnapshot(
            request_id="frac-prefill",
            remaining_prefill_tokens=10,
            max_prefill_chunk_tokens=10,
            prefill_memory_blocks_per_token=2,
        )
    ]
    solution = RelaxedLPSolution(
        success=True,
        status=0,
        message="ok",
        prefill_tokens={"frac-prefill": 2.5},
        decode={"frac-prefill": 0.1},
        preempt={"frac-prefill": 0.1},
        prefill_admission={"frac-prefill": 0.8},
    )

    plan = extract_fractionals(
        requests,
        LPCapacities(token_budget=3, sequence_budget=1, free_memory_blocks=5),
        solution,
    )

    assert plan.prefill_tokens["frac-prefill"] == 2
    assert plan.prefill_admission["frac-prefill"] == 1
    assert plan.decode["frac-prefill"] == 0


def test_fractional_rule_violation_flags_more_than_three_fractionals():
    requests = [LPRequestSnapshot(request_id=f"frac-{idx}") for idx in range(4)]
    solution = RelaxedLPSolution(
        success=True,
        status=0,
        message="ok",
        decode={f"frac-{idx}": 0.5 for idx in range(4)},
        preempt={f"frac-{idx}": 0.0 for idx in range(4)},
        prefill_admission={f"frac-{idx}": 0.0 for idx in range(4)},
    )

    plan = extract_fractionals(
        requests,
        LPCapacities(token_budget=4, sequence_budget=4, free_memory_blocks=4),
        solution,
    )

    assert plan.fractional_rule_violation
