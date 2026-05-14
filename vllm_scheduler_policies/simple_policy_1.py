"""Phase 9 instrumented passthrough scheduler.

This scheduler intentionally preserves vLLM's default scheduling behavior.
It only adds lightweight per-scheduler-call JSONL instrumentation via
InstrumentedSchedulerMixin.

Enable logging with:

    SCHEDULER_POLICIES_ITER_LOG=/path/to/scheduler_iter.jsonl
"""

from vllm.v1.core.sched.scheduler import Scheduler

from vllm_scheduler_policies.instrumentation import InstrumentedSchedulerMixin


class SimplePolicy1Scheduler(InstrumentedSchedulerMixin, Scheduler):
    """Behavior-preserving scheduler with Phase 9 JSONL instrumentation."""
