import inspect
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

import vllm_scheduler_policies.primal_lp_dry_run as dry_run
from vllm_scheduler_policies.instrumentation import InstrumentedSchedulerMixin
from vllm_scheduler_policies.primal_lp.types import (
    LPActionPlan,
    LPCapacities,
    LPRequestSnapshot,
    LPResidualCapacities,
)
from vllm_scheduler_policies.primal_lp_dry_run import (
    LPStateSnapshot,
    PrimalLPDryRunScheduler,
    _SchedulerPlanningView,
    _request_to_lp_snapshot,
    _summarize_lp_plan,
    _unsupported_reason_for_request,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRY_RUN_SOURCE = REPO_ROOT / "vllm_scheduler_policies" / "primal_lp_dry_run.py"


def test_root_package_import_remains_vllm_lazy():
    code = """
import sys
import vllm_scheduler_policies
assert "vllm" not in sys.modules
print("root import ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip() == "root import ok"


def test_scheduler_overrides_schedule_impl_not_schedule():
    assert PrimalLPDryRunScheduler.__dict__["_schedule_impl"]
    assert "schedule" not in PrimalLPDryRunScheduler.__dict__
    assert PrimalLPDryRunScheduler.schedule is InstrumentedSchedulerMixin.schedule
    assert issubclass(PrimalLPDryRunScheduler, InstrumentedSchedulerMixin)
    assert issubclass(PrimalLPDryRunScheduler, Scheduler)


def test_module_wide_forbidden_strings_are_absent():
    src = DRY_RUN_SOURCE.read_text(encoding="utf-8")
    for forbidden in (
        "super().schedule()",
        "allocate_slots(",
        "_preempt_request(",
        "SchedulerOutput(",
        "pop_request(",
        "prepend_request(",
        "VLLM_SCHEDULER_ITER_LOG",
    ):
        assert forbidden not in src


def test_native_delegation_is_explicit():
    src = inspect.getsource(PrimalLPDryRunScheduler._delegate_default_schedule)
    assert "Scheduler.schedule(self)" in src


def _fake_request(**overrides):
    values = {
        "request_id": "req",
        "arrival_time": 90.0,
        "status": RequestStatus.WAITING,
        "num_computed_tokens": 0,
        "num_prompt_tokens": 8,
        "num_tokens": 8,
        "num_tokens_with_spec": 8,
        "num_output_placeholders": 0,
        "spec_token_ids": [],
        "pooling_params": None,
        "has_encoder_inputs": False,
        "kv_transfer_params": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _planning_view():
    return _SchedulerPlanningView(
        max_num_scheduled_tokens=16,
        max_num_running_reqs=4,
        block_size=8,
        long_prefill_token_threshold=6,
        free_memory_blocks=32.0,
        lp_memory_reserve_blocks=0.0,
    )


def test_request_snapshot_classifies_waiting_prefill():
    snapshot = _request_to_lp_snapshot(
        _fake_request(),
        _planning_view(),
        preempt_recoverable_blocks=0.0,
    )

    assert snapshot.request_id == "req"
    assert snapshot.remaining_prefill_tokens == 8
    assert snapshot.max_prefill_chunk_tokens == 6
    assert snapshot.prefill_eligible
    assert not snapshot.decode_eligible
    assert not snapshot.preemptible
    assert snapshot.prefill_memory_blocks_per_token == pytest.approx(0.125)


def test_request_snapshot_classifies_running_decode():
    snapshot = _request_to_lp_snapshot(
        _fake_request(
            status=RequestStatus.RUNNING,
            num_computed_tokens=8,
            num_prompt_tokens=8,
            num_tokens=9,
            num_tokens_with_spec=9,
        ),
        _planning_view(),
        preempt_recoverable_blocks=3.0,
    )

    assert snapshot.remaining_prefill_tokens == 0
    assert not snapshot.prefill_eligible
    assert snapshot.decode_eligible
    assert snapshot.preemptible
    assert snapshot.preempt_recoverable_blocks == 3.0


@pytest.mark.parametrize(
    ("req_obj", "reason"),
    [
        (_fake_request(spec_token_ids=[1]), "spec_decode"),
        (_fake_request(num_output_placeholders=1), "async_output_placeholders"),
        (_fake_request(pooling_params=object()), "pooling_request"),
        (_fake_request(kv_transfer_params={"x": "y"}), "kv_transfer_params"),
        (_fake_request(has_encoder_inputs=True), "encoder_inputs"),
        (
            _fake_request(status=RequestStatus.WAITING_FOR_REMOTE_KVS),
            "waiting_for_remote_kvs",
        ),
        (
            _fake_request(
                status=RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
            ),
            "structured_output_grammar_wait",
        ),
        (
            _fake_request(status=RequestStatus.WAITING_FOR_STREAMING_REQ),
            "unsupported_status",
        ),
        (SimpleNamespace(request_id="missing"), "missing_field"),
    ],
)
def test_unsupported_state_cases_fallback_conservatively(req_obj, reason):
    assert _unsupported_reason_for_request(req_obj) == reason


def test_collect_snapshot_uses_non_mutating_iterables_and_kv_reads():
    scheduler = object.__new__(PrimalLPDryRunScheduler)
    scheduler.max_num_scheduled_tokens = 16
    scheduler.max_num_running_reqs = 4
    scheduler.block_size = 8
    scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    scheduler.waiting = [_fake_request(request_id="waiting")]
    scheduler.skipped_waiting = [_fake_request(request_id="skipped")]
    scheduler.running = [
        _fake_request(
            request_id="running",
            status=RequestStatus.RUNNING,
            num_computed_tokens=8,
            num_prompt_tokens=8,
            num_tokens=9,
            num_tokens_with_spec=9,
        )
    ]
    scheduler.kv_cache_manager = SimpleNamespace(
        block_pool=SimpleNamespace(get_num_free_blocks=lambda: 20),
        get_blocks=lambda request_id: SimpleNamespace(blocks=([object()], [])),
    )

    snapshot, reason, lp_num_requests = scheduler._collect_lp_state_snapshot()

    assert reason is None
    assert lp_num_requests == 3
    assert snapshot is not None
    assert [request.request_id for request in snapshot.requests] == [
        "waiting",
        "skipped",
        "running",
    ]
    assert snapshot.capacities == LPCapacities(
        token_budget=16,
        sequence_budget=4,
        free_memory_blocks=20.0,
    )
    assert all(
        request.preempt_recoverable_blocks == 1.0
        for request in snapshot.requests
    )


def test_unsupported_state_logs_observed_count_and_delegates(monkeypatch):
    records = []
    scheduler = object.__new__(PrimalLPDryRunScheduler)
    scheduler._instrumentation_write = records.append
    scheduler._vllm_sched_instr_call_index = 8
    scheduler.max_num_scheduled_tokens = 16
    scheduler.max_num_running_reqs = 4
    scheduler.block_size = 8
    scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    scheduler.waiting = [
        _fake_request(request_id="supported"),
        _fake_request(request_id="unsupported", spec_token_ids=[1]),
    ]
    scheduler.skipped_waiting = [_fake_request(request_id="skipped")]
    scheduler.running = []
    scheduler.kv_cache_manager = SimpleNamespace(
        block_pool=SimpleNamespace(get_num_free_blocks=lambda: 20),
        get_blocks=lambda request_id: SimpleNamespace(blocks=([], [])),
    )
    native_calls = []

    def native_schedule(self):
        native_calls.append(self)
        return "native-output"

    monkeypatch.setattr(Scheduler, "schedule", native_schedule)

    assert scheduler._schedule_impl() == "native-output"
    assert native_calls == [scheduler]
    assert len(records) == 1
    assert records[0]["event"] == "lp_dry_run"
    assert records[0]["lp_fallback"] is True
    assert records[0]["lp_unsupported_reason"] == "spec_decode"
    assert records[0]["lp_num_requests"] == 3


def test_diagnostic_summarization_from_action_plan():
    plan = LPActionPlan(
        prefill_tokens={"a": 3, "b": 0},
        decode={"a": 0, "b": 1},
        preempt={"a": 1, "b": 0},
        prefill_admission={"a": 1, "b": 0},
        solver_success=True,
        solver_status=0,
        solver_message="ok",
        objective_value=12.5,
        integral_request_ids=("a",),
        fractional_request_ids=("b",),
        fractional_rule_violation=False,
        forced_preemption_request_id="a",
        residual_capacities=LPResidualCapacities(
            token_budget=4,
            sequence_budget=2,
            memory_blocks=8.5,
        ),
    )

    summary = _summarize_lp_plan(plan)

    assert summary["lp_solver_success"] is True
    assert summary["lp_objective_value"] == 12.5
    assert summary["lp_num_integral_requests"] == 1
    assert summary["lp_num_fractional_requests"] == 1
    assert summary["lp_num_decode_actions"] == 1
    assert summary["lp_num_prefill_actions"] == 1
    assert summary["lp_num_prefill_tokens"] == 3
    assert summary["lp_num_preemptions"] == 1
    assert summary["lp_num_forced_preemptions"] == 1
    assert summary["lp_token_budget_remaining"] == 4
    assert summary["lp_sequence_budget_remaining"] == 2
    assert summary["lp_memory_blocks_remaining"] == 8.5


def test_lp_dry_run_exception_is_logged_and_default_schedule_runs(monkeypatch):
    records = []
    scheduler = object.__new__(PrimalLPDryRunScheduler)
    scheduler._instrumentation_write = records.append
    scheduler._vllm_sched_instr_call_index = 3

    monkeypatch.setattr(
        PrimalLPDryRunScheduler,
        "_collect_lp_state_snapshot",
        lambda self: (_ for _ in ()).throw(RuntimeError("dry-run failed")),
    )
    monkeypatch.setattr(Scheduler, "schedule", lambda self: "native-output")

    assert scheduler._schedule_impl() == "native-output"
    assert records[-1]["event"] == "lp_dry_run"
    assert records[-1]["call_index"] == 2
    assert records[-1]["lp_fallback"] is True
    assert records[-1]["lp_dry_run_error_type"] == "RuntimeError"
    assert records[-1]["lp_dry_run_error_message"] == "dry-run failed"


def test_native_scheduler_exceptions_are_not_swallowed(monkeypatch):
    scheduler = object.__new__(PrimalLPDryRunScheduler)
    scheduler._instrumentation_write = lambda record: None
    monkeypatch.setattr(
        PrimalLPDryRunScheduler,
        "_collect_lp_state_snapshot",
        lambda self: (
            LPStateSnapshot(
                [],
                LPCapacities(
                    token_budget=1,
                    sequence_budget=1,
                    free_memory_blocks=1,
                ),
            ),
            None,
            0,
        ),
    )
    monkeypatch.setattr(
        dry_run,
        "solve_lp_relaxation",
        lambda requests, capacities, weights: LPActionPlan.empty_for(
            requests,
            solver_success=True,
            solver_status=0,
            solver_message="ok",
        ),
    )

    def raise_native(self):
        raise ValueError("native failed")

    monkeypatch.setattr(Scheduler, "schedule", raise_native)

    with pytest.raises(ValueError, match="native failed"):
        scheduler._schedule_impl()
