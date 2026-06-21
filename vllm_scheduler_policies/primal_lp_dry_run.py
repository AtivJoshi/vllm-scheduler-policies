"""Dry-run primal LP scheduler bridge.

The LP path is diagnostic only: it snapshots vLLM scheduler state, solves the
Phase 12.3 synthetic LP, logs a summary when instrumentation is enabled, and
then returns native vLLM scheduling output unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_scheduler_policies.instrumentation import InstrumentedSchedulerMixin
from vllm_scheduler_policies.primal_lp.solver import (
    solve_lp_relaxation,  # noqa: F401 - retained for module compatibility
    solve_lp_relaxation_timed,
)
from vllm_scheduler_policies.primal_lp.types import (
    LPActionPlan,
    LPCapacities,
    LPRequestSnapshot,
)
from vllm_scheduler_policies.primal_lp.weights import compute_default_like_weights


@dataclass(frozen=True)
class LPStateSnapshot:
    """Read-only synthetic LP input built from one scheduler step."""

    requests: list[LPRequestSnapshot]
    capacities: LPCapacities
    kv_block_read_error: bool = False


@dataclass(frozen=True)
class _SchedulerPlanningView:
    max_num_scheduled_tokens: int
    max_num_running_reqs: int
    block_size: int
    long_prefill_token_threshold: int
    free_memory_blocks: float
    lp_memory_reserve_blocks: float


def _unsupported_reason_for_request(request: Any) -> str | None:
    try:
        status = request.status
        if len(request.spec_token_ids) > 0:
            return "spec_decode"
        if request.num_output_placeholders > 0:
            return "async_output_placeholders"
        if request.pooling_params is not None:
            return "pooling_request"
        if request.kv_transfer_params is not None:
            return "kv_transfer_params"
        if request.has_encoder_inputs:
            return "encoder_inputs"
    except AttributeError:
        return "missing_field"

    if status == RequestStatus.WAITING_FOR_REMOTE_KVS:
        return "waiting_for_remote_kvs"
    if status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
        return "structured_output_grammar_wait"
    if status not in {
        RequestStatus.WAITING,
        RequestStatus.RUNNING,
        RequestStatus.PREEMPTED,
    }:
        return "unsupported_status"
    return None


def _max_prefill_chunk_tokens(view: _SchedulerPlanningView) -> int:
    threshold = view.long_prefill_token_threshold
    if threshold and threshold > 0:
        return min(threshold, view.max_num_scheduled_tokens)
    return view.max_num_scheduled_tokens


def _request_to_lp_snapshot(
    request: Any,
    view: _SchedulerPlanningView,
    *,
    preempt_recoverable_blocks: float,
) -> LPRequestSnapshot:
    status = request.status
    remaining_prefill_tokens = 0
    if (
        status != RequestStatus.RUNNING
        or request.num_computed_tokens < request.num_prompt_tokens
    ):
        remaining_prefill_tokens = max(
            request.num_tokens - request.num_computed_tokens,
            0,
        )

    prefill_eligible = (
        status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
        or (
            status == RequestStatus.RUNNING
            and request.num_computed_tokens < request.num_prompt_tokens
        )
    ) and remaining_prefill_tokens > 0

    decode_eligible = (
        status == RequestStatus.RUNNING
        and request.num_output_placeholders == 0
        and len(request.spec_token_ids) == 0
        and (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        > 0
        and request.num_computed_tokens >= request.num_prompt_tokens
    )

    block_cost = 1.0 / float(view.block_size)
    return LPRequestSnapshot(
        request_id=request.request_id,
        arrival_time=request.arrival_time,
        remaining_prefill_tokens=remaining_prefill_tokens,
        max_prefill_chunk_tokens=_max_prefill_chunk_tokens(view),
        prefill_eligible=prefill_eligible,
        decode_eligible=decode_eligible,
        preemptible=status == RequestStatus.RUNNING,
        prefill_memory_blocks_per_token=block_cost,
        decode_memory_blocks=block_cost,
        preempt_recoverable_blocks=preempt_recoverable_blocks,
    )


def _summarize_lp_plan(plan: LPActionPlan) -> dict[str, Any]:
    residual = plan.residual_capacities
    return {
        "lp_solver_success": plan.solver_success,
        "lp_solver_status": plan.solver_status,
        "lp_solver_message": plan.solver_message,
        "lp_objective_value": plan.objective_value,
        "lp_num_integral_requests": len(plan.integral_request_ids),
        "lp_num_fractional_requests": len(plan.fractional_request_ids),
        "lp_fractional_rule_violation": plan.fractional_rule_violation,
        "lp_num_decode_actions": sum(plan.decode.values()),
        "lp_num_prefill_actions": sum(
            1 for num_tokens in plan.prefill_tokens.values() if num_tokens > 0
        ),
        "lp_num_prefill_tokens": sum(plan.prefill_tokens.values()),
        "lp_num_preemptions": sum(plan.preempt.values()),
        "lp_num_forced_preemptions": int(plan.forced_preemption_request_id is not None),
        "lp_token_budget_remaining": (
            residual.token_budget if residual is not None else None
        ),
        "lp_sequence_budget_remaining": (
            residual.sequence_budget if residual is not None else None
        ),
        "lp_memory_blocks_remaining": (
            residual.memory_blocks if residual is not None else None
        ),
    }


class PrimalLPDryRunScheduler(InstrumentedSchedulerMixin, Scheduler):
    """Instrumented scheduler that logs a dry-run primal LP plan."""

    def _delegate_default_schedule(self):  # noqa: ANN201 - vLLM return type
        native_start = time.perf_counter()
        try:
            return Scheduler.schedule(self)
        finally:
            self._instrumentation_add_scheduler_call_fields(
                native_schedule_wall_time_ms=(
                    time.perf_counter() - native_start
                )
                * 1000.0
            )

    def _schedule_impl(self):  # noqa: ANN201 - keep exact vLLM scheduler return
        dry_run_start = time.perf_counter()
        timings: dict[str, float] = {}
        try:
            snapshot_start = time.perf_counter()
            try:
                snapshot, unsupported_reason, lp_num_requests = (
                    self._collect_lp_state_snapshot()
                )
            finally:
                timings["lp_snapshot_wall_time_ms"] = (
                    time.perf_counter() - snapshot_start
                ) * 1000.0
            if unsupported_reason is not None:
                self._write_lp_dry_run_record(
                    timings=timings,
                    dry_run_start=dry_run_start,
                    lp_num_requests=lp_num_requests,
                    lp_fallback=True,
                    lp_unsupported_reason=unsupported_reason,
                )
            else:
                plan = self._run_lp_dry_run(snapshot, timings)
                summary_start = time.perf_counter()
                try:
                    summary = _summarize_lp_plan(plan)
                finally:
                    timings["lp_summary_wall_time_ms"] = (
                        time.perf_counter() - summary_start
                    ) * 1000.0
                self._write_lp_dry_run_record(
                    timings=timings,
                    dry_run_start=dry_run_start,
                    lp_num_requests=len(snapshot.requests),
                    lp_fallback=not plan.solver_success,
                    lp_unsupported_reason=None,
                    kv_block_read_error=snapshot.kv_block_read_error,
                    **summary,
                )
        except Exception as exc:
            self._write_lp_dry_run_record(
                timings=timings,
                dry_run_start=dry_run_start,
                lp_fallback=True,
                lp_unsupported_reason=None,
                lp_dry_run_error_type=type(exc).__name__,
                lp_dry_run_error_message=str(exc),
            )

        return self._delegate_default_schedule()

    def _collect_lp_state_snapshot(
        self,
    ) -> tuple[LPStateSnapshot | None, str | None, int]:
        view = _SchedulerPlanningView(
            max_num_scheduled_tokens=self.max_num_scheduled_tokens,
            max_num_running_reqs=self.max_num_running_reqs,
            block_size=self.block_size,
            long_prefill_token_threshold=(
                self.scheduler_config.long_prefill_token_threshold
            ),
            free_memory_blocks=self._get_num_free_kv_blocks(),
            lp_memory_reserve_blocks=float(
                getattr(self, "lp_memory_reserve_blocks", 0.0)
            ),
        )

        requests = (
            list(self.waiting) + list(self.skipped_waiting) + list(self.running)
        )
        lp_num_requests = len(requests)
        snapshots: list[LPRequestSnapshot] = []
        kv_block_read_error = False
        for request in requests:
            try:
                unsupported_reason = _unsupported_reason_for_request(request)
                if unsupported_reason is not None:
                    return None, unsupported_reason, lp_num_requests

                allocated_blocks, read_error = self._get_request_kv_blocks(
                    request.request_id
                )
                kv_block_read_error = kv_block_read_error or read_error
                snapshots.append(
                    _request_to_lp_snapshot(
                        request,
                        view,
                        preempt_recoverable_blocks=allocated_blocks,
                    )
                )
            except AttributeError:
                return None, "missing_field", lp_num_requests

        capacities = LPCapacities(
            token_budget=view.max_num_scheduled_tokens,
            sequence_budget=view.max_num_running_reqs,
            free_memory_blocks=view.free_memory_blocks,
            lp_memory_reserve_blocks=view.lp_memory_reserve_blocks,
        )
        return (
            LPStateSnapshot(snapshots, capacities, kv_block_read_error),
            None,
            lp_num_requests,
        )

    def _run_lp_dry_run(
        self, snapshot: LPStateSnapshot, timings: dict[str, float]
    ) -> LPActionPlan:
        if not snapshot.requests:
            plan_start = time.perf_counter()
            plan = LPActionPlan.empty_for(
                [],
                solver_success=True,
                solver_status=0,
                solver_message="no_requests",
            )
            timings["lp_plan_wall_time_ms"] = (
                time.perf_counter() - plan_start
            ) * 1000.0
            return plan
        weight_start = time.perf_counter()
        try:
            weights = compute_default_like_weights(
                snapshot.requests,
                current_time=time.time(),
            )
        finally:
            timings["lp_weight_wall_time_ms"] = (
                time.perf_counter() - weight_start
            ) * 1000.0
        plan, plan_timings = solve_lp_relaxation_timed(
            snapshot.requests, snapshot.capacities, weights
        )
        timings.update(plan_timings)
        return plan

    def _get_num_free_kv_blocks(self) -> float:
        return float(self.kv_cache_manager.block_pool.get_num_free_blocks())

    def _get_request_kv_blocks(self, request_id: str) -> tuple[float, bool]:
        try:
            kv_blocks = self.kv_cache_manager.get_blocks(request_id)
            return float(sum(len(group) for group in kv_blocks.blocks)), False
        except Exception:
            return 0.0, True

    def _write_lp_dry_run_record(
        self,
        *,
        timings: dict[str, float],
        dry_run_start: float,
        **fields: Any,
    ) -> None:
        log_prepare_start = time.perf_counter()
        call_index = getattr(self, "_vllm_sched_instr_call_index", None)
        if call_index is not None:
            call_index -= 1

        record: dict[str, Any] = {
            "event": "lp_dry_run",
            "call_index": call_index,
            "scheduler_class": type(self).__module__ + "." + type(self).__name__,
            "lp_num_requests": fields.pop("lp_num_requests", None),
            "lp_fallback": fields.pop("lp_fallback", False),
            "lp_unsupported_reason": fields.pop("lp_unsupported_reason", None),
            "lp_dry_run_error_type": fields.pop("lp_dry_run_error_type", None),
            "lp_dry_run_error_message": fields.pop(
                "lp_dry_run_error_message", None
            ),
            "lp_solver_success": fields.pop("lp_solver_success", None),
            "lp_solver_status": fields.pop("lp_solver_status", None),
            "lp_solver_message": fields.pop("lp_solver_message", None),
            "lp_objective_value": fields.pop("lp_objective_value", None),
            "lp_num_integral_requests": fields.pop(
                "lp_num_integral_requests", None
            ),
            "lp_num_fractional_requests": fields.pop(
                "lp_num_fractional_requests", None
            ),
            "lp_fractional_rule_violation": fields.pop(
                "lp_fractional_rule_violation", None
            ),
            "lp_num_decode_actions": fields.pop("lp_num_decode_actions", None),
            "lp_num_prefill_actions": fields.pop("lp_num_prefill_actions", None),
            "lp_num_prefill_tokens": fields.pop("lp_num_prefill_tokens", None),
            "lp_num_preemptions": fields.pop("lp_num_preemptions", None),
            "lp_num_forced_preemptions": fields.pop(
                "lp_num_forced_preemptions", None
            ),
            "lp_token_budget_remaining": fields.pop(
                "lp_token_budget_remaining", None
            ),
            "lp_sequence_budget_remaining": fields.pop(
                "lp_sequence_budget_remaining", None
            ),
            "lp_memory_blocks_remaining": fields.pop(
                "lp_memory_blocks_remaining", None
            ),
            "kv_block_read_error": fields.pop("kv_block_read_error", False),
        }
        record.update(fields)
        record.update(timings)
        record["lp_timing_available"] = True
        record["lp_log_prepare_wall_time_ms"] = (
            time.perf_counter() - log_prepare_start
        ) * 1000.0
        record["lp_dry_run_total_wall_time_ms"] = (
            time.perf_counter() - dry_run_start
        ) * 1000.0
        self._instrumentation_write(record)
