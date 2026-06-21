"""SciPy LP construction and solve helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from vllm_scheduler_policies.primal_lp.extraction import extract_fractionals
from vllm_scheduler_policies.primal_lp.types import (
    LPActionPlan,
    LPCapacities,
    LPRequestSnapshot,
    LPUtilityWeights,
    RelaxedLPSolution,
)


@dataclass(frozen=True)
class _VariableIndex:
    request_id: str
    x: int
    y: int
    z: int
    admission: int


def _build_variable_index(requests: list[LPRequestSnapshot]) -> list[_VariableIndex]:
    return [
        _VariableIndex(
            request_id=request.request_id,
            x=4 * offset,
            y=4 * offset + 1,
            z=4 * offset + 2,
            admission=4 * offset + 3,
        )
        for offset, request in enumerate(requests)
    ]


def _solve_relaxed_lp_timed(
    requests: list[LPRequestSnapshot],
    capacities: LPCapacities,
    weights: LPUtilityWeights,
) -> tuple[RelaxedLPSolution, dict[str, float]]:
    build_start = time.perf_counter()
    var_index = _build_variable_index(requests)
    num_vars = 4 * len(requests)

    c = np.zeros(num_vars, dtype=float)
    bounds: list[tuple[float, float]] = []
    a_ub: list[list[float]] = []
    b_ub: list[float] = []

    for request, index in zip(requests, var_index, strict=True):
        request_weights = weights.for_request(request.request_id)
        c[index.x] = -request_weights.beta
        c[index.y] = -request_weights.alpha
        c[index.z] = request_weights.gamma
        c[index.admission] = 0.0

        bounds.extend(
            [
                (0.0, float(request.max_prefill_tokens_this_step)),
                (0.0, 1.0 if request.decode_eligible else 0.0),
                (0.0, 1.0 if request.preemptible else 0.0),
                (0.0, 1.0 if request.max_prefill_tokens_this_step > 0 else 0.0),
            ]
        )

    token_row = [0.0] * num_vars
    sequence_row = [0.0] * num_vars
    memory_row = [0.0] * num_vars
    for request, index in zip(requests, var_index, strict=True):
        token_row[index.x] = 1.0
        token_row[index.y] = 1.0
        sequence_row[index.y] = 1.0
        sequence_row[index.admission] = 1.0
        memory_row[index.x] = request.prefill_memory_blocks_per_token
        memory_row[index.y] = request.decode_memory_blocks
        memory_row[index.z] = -request.preempt_recoverable_blocks
    a_ub.extend([token_row, sequence_row, memory_row])
    b_ub.extend(
        [
            float(capacities.token_budget),
            float(capacities.sequence_budget),
            float(capacities.usable_memory_blocks),
        ]
    )

    for request, index in zip(requests, var_index, strict=True):
        mutual_exclusion = [0.0] * num_vars
        mutual_exclusion[index.admission] = 1.0
        mutual_exclusion[index.y] = 1.0
        mutual_exclusion[index.z] = 1.0
        a_ub.append(mutual_exclusion)
        b_ub.append(1.0)

        prefill_coupling = [0.0] * num_vars
        prefill_coupling[index.x] = 1.0
        prefill_coupling[index.admission] = -float(
            request.max_prefill_tokens_this_step
        )
        a_ub.append(prefill_coupling)
        b_ub.append(0.0)

    a_ub_array = np.array(a_ub, dtype=float) if a_ub else None
    b_ub_array = np.array(b_ub, dtype=float) if b_ub else None
    build_wall_time_ms = (time.perf_counter() - build_start) * 1000.0

    solve_start = time.perf_counter()
    result = linprog(
        c,
        A_ub=a_ub_array,
        b_ub=b_ub_array,
        bounds=bounds,
        method="highs",
    )
    solve_wall_time_ms = (time.perf_counter() - solve_start) * 1000.0

    parse_start = time.perf_counter()
    if not result.success:
        solution = RelaxedLPSolution(
            success=False,
            status=int(result.status),
            message=str(result.message),
        )
    else:
        prefill_tokens = {}
        decode = {}
        preempt = {}
        prefill_admission = {}
        for index in var_index:
            prefill_tokens[index.request_id] = float(result.x[index.x])
            decode[index.request_id] = float(result.x[index.y])
            preempt[index.request_id] = float(result.x[index.z])
            prefill_admission[index.request_id] = float(result.x[index.admission])

        solution = RelaxedLPSolution(
            success=True,
            status=int(result.status),
            message=str(result.message),
            objective_value=float(-result.fun),
            prefill_tokens=prefill_tokens,
            decode=decode,
            preempt=preempt,
            prefill_admission=prefill_admission,
        )
    parse_wall_time_ms = (time.perf_counter() - parse_start) * 1000.0
    return solution, {
        "lp_relaxed_lp_build_wall_time_ms": build_wall_time_ms,
        "lp_highs_solve_wall_time_ms": solve_wall_time_ms,
        "lp_relaxed_solution_parse_wall_time_ms": parse_wall_time_ms,
    }


def _solve_relaxed_lp(
    requests: list[LPRequestSnapshot],
    capacities: LPCapacities,
    weights: LPUtilityWeights,
) -> RelaxedLPSolution:
    solution, _ = _solve_relaxed_lp_timed(requests, capacities, weights)
    return solution


def solve_lp_relaxation(
    requests: list[LPRequestSnapshot],
    capacities: LPCapacities,
    weights: LPUtilityWeights,
    *,
    tol: float = 1.0e-6,
) -> LPActionPlan:
    """Solve the relaxed LP and return the extracted integer action plan."""
    relaxed_solution = _solve_relaxed_lp(requests, capacities, weights)
    return extract_fractionals(
        requests,
        capacities,
        relaxed_solution,
        tol=tol,
    )


def solve_lp_relaxation_timed(
    requests: list[LPRequestSnapshot],
    capacities: LPCapacities,
    weights: LPUtilityWeights,
    *,
    tol: float = 1.0e-6,
) -> tuple[LPActionPlan, dict[str, float]]:
    """Solve and extract an action plan with scheduler attribution timings."""
    plan_start = time.perf_counter()
    relaxed_solution, timings = _solve_relaxed_lp_timed(
        requests, capacities, weights
    )

    extract_start = time.perf_counter()
    plan = extract_fractionals(
        requests,
        capacities,
        relaxed_solution,
        tol=tol,
    )
    timings["lp_extract_wall_time_ms"] = (
        time.perf_counter() - extract_start
    ) * 1000.0
    timings["lp_plan_wall_time_ms"] = (
        time.perf_counter() - plan_start
    ) * 1000.0
    return plan, timings
