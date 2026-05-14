"""Lightweight scheduler instrumentation helpers.

Phase 9 goal:
- preserve vLLM default scheduler behavior
- measure Python scheduler wall time with time.perf_counter()
- emit one JSONL record per scheduler call when enabled by environment variable

Enable logging by setting:

    SCHEDULER_POLICIES_ITER_LOG=/path/to/scheduler_iter.jsonl
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import time
from typing import Any


_LOG_ENV_VAR = "SCHEDULER_POLICIES_ITER_LOG"


def _safe_len(obj: Any) -> int | None:
    """Return len(obj), or None if the object has no cheap length."""
    if obj is None:
        return None
    try:
        return len(obj)
    except Exception:
        return None


class InstrumentedSchedulerMixin:
    """Mixin that wraps Scheduler.schedule() with lightweight JSONL logging.

    This mixin must appear before vLLM's Scheduler in the subclass MRO:

        class MyScheduler(InstrumentedSchedulerMixin, Scheduler):
            pass

    It intentionally delegates scheduling to super().schedule() without changing
    queue order, token budgets, KV-cache allocation, preemption, or outputs.
    """

    def _instrumentation_init_once(self) -> None:
        if hasattr(self, "_vllm_sched_instr_initialized"):
            return

        self._vllm_sched_instr_initialized = True
        self._vllm_sched_instr_call_index = 0
        self._vllm_sched_instr_path = os.environ.get(_LOG_ENV_VAR, "").strip()
        self._vllm_sched_instr_file = None
        self._vllm_sched_instr_hostname = socket.gethostname()
        self._vllm_sched_instr_pid = os.getpid()

        if self._vllm_sched_instr_path:
            log_dir = os.path.dirname(os.path.abspath(self._vllm_sched_instr_path))
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self._vllm_sched_instr_file = open(
                self._vllm_sched_instr_path,
                "a",
                buffering=1,
                encoding="utf-8",
            )
            atexit.register(self._instrumentation_close)

    def _instrumentation_close(self) -> None:
        f = getattr(self, "_vllm_sched_instr_file", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
            self._vllm_sched_instr_file = None

    def _instrumentation_write(self, record: dict[str, Any]) -> None:
        f = getattr(self, "_vllm_sched_instr_file", None)
        if f is None:
            return
        try:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception:
            # Instrumentation must never break serving.
            pass

    def schedule(self):  # noqa: ANN201 - keep exact vLLM scheduler signature
        self._instrumentation_init_once()

        call_index = self._vllm_sched_instr_call_index
        self._vllm_sched_instr_call_index += 1

        waiting_before = _safe_len(getattr(self, "waiting", None))
        skipped_waiting_before = _safe_len(getattr(self, "skipped_waiting", None))
        running_before = _safe_len(getattr(self, "running", None))

        start_perf = time.perf_counter()
        start_wall = time.time()
        output = None
        error_type = None
        error_message = None

        try:
            output = super().schedule()
            return output
        except BaseException as exc:
            error_type = type(exc).__name__
            error_message = str(exc)
            raise
        finally:
            end_perf = time.perf_counter()
            wall_time_ms = (end_perf - start_perf) * 1000.0

            record: dict[str, Any] = {
                "event": "scheduler_call",
                "call_index": call_index,
                "hostname": self._vllm_sched_instr_hostname,
                "pid": self._vllm_sched_instr_pid,
                "scheduler_class": type(self).__module__ + "." + type(self).__name__,
                "start_time_unix_s": start_wall,
                "scheduler_wall_time_ms": wall_time_ms,
                "waiting_before": waiting_before,
                "skipped_waiting_before": skipped_waiting_before,
                "running_before": running_before,
                "waiting_after": _safe_len(getattr(self, "waiting", None)),
                "skipped_waiting_after": _safe_len(
                    getattr(self, "skipped_waiting", None)
                ),
                "running_after": _safe_len(getattr(self, "running", None)),
                "ok": error_type is None,
            }

            if output is not None:
                num_scheduled_tokens = getattr(output, "num_scheduled_tokens", None)
                preempted_req_ids = getattr(output, "preempted_req_ids", None)

                record.update(
                    {
                        "num_scheduled_requests": _safe_len(num_scheduled_tokens),
                        "num_scheduled_tokens": getattr(
                            output, "total_num_scheduled_tokens", None
                        ),
                        "num_preemptions": _safe_len(preempted_req_ids),
                        "num_scheduled_new_reqs": _safe_len(
                            getattr(output, "scheduled_new_reqs", None)
                        ),
                    }
                )

            if error_type is not None:
                record["error_type"] = error_type
                record["error_message"] = error_message

            self._instrumentation_write(record)
