"""Integral/fractional extraction for relaxed primal LP solutions."""

from __future__ import annotations

from vllm_scheduler_policies.primal_lp.types import (
    LPActionPlan,
    LPCapacities,
    LPRequestSnapshot,
    LPResidualCapacities,
    RelaxedLPSolution,
    floor_nonnegative,
)


def _is_binary_integral(value: float, *, tol: float) -> bool:
    return abs(value) <= tol or abs(value - 1.0) <= tol


def _round_binary(value: float, *, tol: float) -> int:
    if abs(value) <= tol:
        return 0
    if abs(value - 1.0) <= tol:
        return 1
    raise ValueError(f"value {value!r} is not integral within tolerance {tol}")


def _dominant_action(
    values: dict[str, float],
    *,
    priority: tuple[str, ...],
) -> str:
    """Return max-value action with an explicit deterministic tie priority."""
    best_action = priority[0]
    best_value = values[best_action]
    for action in priority[1:]:
        value = values[action]
        if value > best_value:
            best_action = action
            best_value = value
    return best_action


def partition_integral_requests(
    requests: list[LPRequestSnapshot],
    solution: RelaxedLPSolution,
    *,
    tol: float = 1.0e-6,
) -> tuple[list[str], list[str]]:
    """Partition requests by integrality of y, z, and I^P variables."""
    integral = []
    fractional = []
    for request in requests:
        request_id = request.request_id
        values = (
            solution.decode.get(request_id, 0.0),
            solution.preempt.get(request_id, 0.0),
            solution.prefill_admission.get(request_id, 0.0),
        )
        if all(_is_binary_integral(value, tol=tol) for value in values):
            integral.append(request_id)
        else:
            fractional.append(request_id)
    return integral, fractional


def extract_fractionals(
    requests: list[LPRequestSnapshot],
    capacities: LPCapacities,
    solution: RelaxedLPSolution,
    *,
    tol: float = 1.0e-6,
) -> LPActionPlan:
    """Convert a relaxed LP solution into an integer synthetic action plan."""
    if not solution.success:
        return LPActionPlan.empty_for(
            requests,
            solver_success=False,
            solver_status=solution.status,
            solver_message=solution.message,
            objective_value=solution.objective_value,
        )

    by_request_id = {request.request_id: request for request in requests}
    integral_ids, fractional_ids = partition_integral_requests(
        requests, solution, tol=tol
    )

    plan_prefill = {request.request_id: 0 for request in requests}
    plan_decode = {request.request_id: 0 for request in requests}
    plan_preempt = {request.request_id: 0 for request in requests}
    plan_admit = {request.request_id: 0 for request in requests}

    token_rem = capacities.token_budget
    seq_rem = capacities.sequence_budget
    memory_rem = capacities.usable_memory_blocks

    for request_id in integral_ids:
        request = by_request_id[request_id]
        x_value = solution.prefill_tokens.get(request_id, 0.0)
        y_value = solution.decode.get(request_id, 0.0)
        z_value = solution.preempt.get(request_id, 0.0)
        i_value = solution.prefill_admission.get(request_id, 0.0)

        x_hat = floor_nonnegative(x_value, tol=tol)
        y_hat = _round_binary(y_value, tol=tol)
        z_hat = _round_binary(z_value, tol=tol)
        i_hat = _round_binary(i_value, tol=tol)

        # LPActionPlan represents executable synthetic actions, so a rounded
        # admission with no tokens is normalized away.
        if x_hat > 0:
            i_hat = 1
        else:
            i_hat = 0

        plan_prefill[request_id] = x_hat
        plan_decode[request_id] = y_hat
        plan_preempt[request_id] = z_hat
        plan_admit[request_id] = i_hat

        token_rem -= x_hat + y_hat
        seq_rem -= i_hat + y_hat
        memory_rem -= (
            request.prefill_memory_blocks_per_token * x_hat
            + request.decode_memory_blocks * y_hat
            - request.preempt_recoverable_blocks * z_hat
        )

    token_rem = max(0, token_rem)
    seq_rem = max(0, seq_rem)

    forced_preemption_request_id = None
    fractional_unpreempted = set(fractional_ids)

    for request_id in fractional_ids:
        y_value = solution.decode.get(request_id, 0.0)
        z_value = solution.preempt.get(request_id, 0.0)
        i_value = solution.prefill_admission.get(request_id, 0.0)
        dominant = _dominant_action(
            {"preempt": z_value, "decode": y_value, "prefill": i_value},
            priority=("preempt", "decode", "prefill"),
        )
        request = by_request_id[request_id]
        if dominant == "preempt" and request.preemptible:
            plan_preempt[request_id] = 1
            memory_rem += request.preempt_recoverable_blocks
            fractional_unpreempted.discard(request_id)

    if memory_rem < -tol:
        candidates = [
            request_id
            for request_id in fractional_ids
            if request_id in fractional_unpreempted
            and by_request_id[request_id].preemptible
        ]
        if candidates:
            forced_preemption_request_id = max(
                candidates,
                key=lambda request_id: (
                    solution.preempt.get(request_id, 0.0),
                    request_id,
                ),
            )
            request = by_request_id[forced_preemption_request_id]
            plan_preempt[forced_preemption_request_id] = 1
            memory_rem += request.preempt_recoverable_blocks
            fractional_unpreempted.discard(forced_preemption_request_id)

    for request_id in fractional_ids:
        if request_id not in fractional_unpreempted:
            continue
        request = by_request_id[request_id]
        y_value = solution.decode.get(request_id, 0.0)
        i_value = solution.prefill_admission.get(request_id, 0.0)

        dominant = _dominant_action(
            {"decode": y_value, "prefill": i_value},
            priority=("decode", "prefill"),
        )

        if dominant == "decode":
            if (
                request.decode_eligible
                and token_rem >= 1
                and seq_rem >= 1
                and memory_rem + tol >= request.decode_memory_blocks
            ):
                plan_decode[request_id] = 1
                token_rem -= 1
                seq_rem -= 1
                memory_rem -= request.decode_memory_blocks
            continue

        if not request.prefill_eligible or seq_rem < 1:
            continue
        max_tokens = min(request.max_prefill_tokens_this_step, token_rem)
        if request.prefill_memory_blocks_per_token > 0:
            max_tokens = min(
                max_tokens,
                int((memory_rem + tol) // request.prefill_memory_blocks_per_token),
            )
        if max_tokens > 0:
            plan_prefill[request_id] = max_tokens
            plan_admit[request_id] = 1
            token_rem -= max_tokens
            seq_rem -= 1
            memory_rem -= request.prefill_memory_blocks_per_token * max_tokens

    return LPActionPlan(
        prefill_tokens=plan_prefill,
        decode=plan_decode,
        preempt=plan_preempt,
        prefill_admission=plan_admit,
        solver_success=True,
        solver_status=solution.status,
        solver_message=solution.message,
        objective_value=solution.objective_value,
        integral_request_ids=tuple(integral_ids),
        fractional_request_ids=tuple(fractional_ids),
        fractional_rule_violation=len(fractional_ids) > 3,
        forced_preemption_request_id=forced_preemption_request_id,
        residual_capacities=LPResidualCapacities(
            token_budget=token_rem,
            sequence_budget=seq_rem,
            memory_blocks=memory_rem,
        ),
    )
