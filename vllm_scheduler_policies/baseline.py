"""Baseline passthrough scheduler.

This module defines an external scheduler class that is intentionally
behavior-identical to vLLM's default v1 scheduler. It exists only to validate
that vLLM can load scheduler classes from an external package via
`--scheduler-cls`.
"""

from vllm.v1.core.sched.scheduler import Scheduler


class BaselinePassthroughScheduler(Scheduler):
    """Behavior-preserving subclass of vLLM's default v1 scheduler.

    No methods are overridden. Construction and scheduling are inherited
    directly from `vllm.v1.core.sched.scheduler.Scheduler`.
    """
