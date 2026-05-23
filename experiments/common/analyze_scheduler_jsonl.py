#!/usr/bin/env python3
"""Analyze scheduler JSONL artifacts for Phase 13 experiments.

This script is intentionally dependency-free. It reads the JSONL file produced
by SCHEDULER_POLICIES_ITER_LOG and writes compact text/JSON summaries.

It is strict about the LP timing caveat: unless a dedicated LP-specific timing
field is present in the records, it reports scheduler_wall_time_ms as the timing
field used for dry-run overhead comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


LP_TIMING_CANDIDATE_FIELDS = (
    "lp_wall_time_ms",
    "lp_solver_wall_time_ms",
    "lp_planning_wall_time_ms",
    "lp_dry_run_wall_time_ms",
)


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: JSON decode error: {exc}")
                continue
            if not isinstance(value, dict):
                errors.append(f"line {line_number}: record is not a JSON object")
                continue
            records.append(value)

    return records, errors


def _numeric_values(records: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _sum_numeric(records: list[dict[str, Any]], field: str) -> float | int | None:
    values = _numeric_values(records, field)
    if not values:
        return None
    total = sum(values)
    if all(float(v).is_integer() for v in values):
        return int(total)
    return total


def _percentile_nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)
    sorted_values = sorted(values)
    rank = math.ceil((percentile / 100.0) * len(sorted_values))
    index = max(0, min(rank - 1, len(sorted_values) - 1))
    return sorted_values[index]


def _summary_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile_nearest_rank(values, 95.0),
        "min": min(values),
        "max": max(values),
    }


def _counter(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        value = record.get(field)
        if value is None:
            counts["<none>"] += 1
        else:
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def analyze(path: Path) -> dict[str, Any]:
    records, parse_errors = _read_jsonl(path)

    event_counts = Counter(str(record.get("event", "<missing>")) for record in records)
    scheduler_records = [r for r in records if r.get("event") == "scheduler_call"]
    lp_records = [r for r in records if r.get("event") == "lp_dry_run"]

    wall_times = _numeric_values(scheduler_records, "scheduler_wall_time_ms")

    lp_specific_timing_fields_present = [
        field
        for field in LP_TIMING_CANDIDATE_FIELDS
        if any(field in record for record in lp_records)
    ]

    lp_specific_timing_available = bool(lp_specific_timing_fields_present)
    if lp_specific_timing_available:
        lp_timing_field = lp_specific_timing_fields_present[0]
        lp_timing_stats = _summary_stats(_numeric_values(lp_records, lp_timing_field))
        overhead_metric_used = lp_timing_field
    else:
        lp_timing_field = None
        lp_timing_stats = None
        overhead_metric_used = "scheduler_wall_time_ms"

    scheduler_classes = sorted(
        {
            str(record.get("scheduler_class"))
            for record in scheduler_records
            if record.get("scheduler_class") is not None
        }
    )

    scheduler_ok_values = [record.get("ok") for record in scheduler_records]
    all_scheduler_calls_ok = bool(scheduler_records) and all(
        value is True for value in scheduler_ok_values
    )

    summary: dict[str, Any] = {
        "input_path": str(path),
        "num_records_total": len(records),
        "num_parse_errors": len(parse_errors),
        "parse_errors": parse_errors[:20],
        "event_counts": dict(sorted(event_counts.items())),
        "num_scheduler_call_records": len(scheduler_records),
        "num_lp_dry_run_records": len(lp_records),
        "scheduler_classes": scheduler_classes,
        "all_scheduler_calls_ok": all_scheduler_calls_ok,
        "num_scheduler_error_records": sum(
            1 for record in scheduler_records if record.get("ok") is not True
        ),
        "scheduler_wall_time_ms": _summary_stats(wall_times),
        "total_scheduled_tokens": _sum_numeric(
            scheduler_records, "num_scheduled_tokens"
        ),
        "total_preemptions": _sum_numeric(scheduler_records, "num_preemptions"),
        "total_scheduled_requests": _sum_numeric(
            scheduler_records, "num_scheduled_requests"
        ),
        "lp_specific_timing_available": lp_specific_timing_available,
        "lp_specific_timing_fields_present": lp_specific_timing_fields_present,
        "lp_timing_field": lp_timing_field,
        "lp_timing_stats": lp_timing_stats,
        "overhead_metric_used": overhead_metric_used,
        "measurement_caveat": (
            "LP-specific timing field found; use lp_timing_field for isolated "
            "LP timing if the field semantics are verified."
            if lp_specific_timing_available
            else "No LP-specific timing field found; overhead comparisons use "
            "scheduler_wall_time_ms and therefore measure total dry-run scheduler "
            "overhead, not isolated LP solver time."
        ),
    }

    if lp_records:
        summary.update(
            {
                "lp_fallback_counts": _counter(lp_records, "lp_fallback"),
                "lp_unsupported_reason_counts": _counter(
                    lp_records, "lp_unsupported_reason"
                ),
                "lp_solver_success_counts": _counter(lp_records, "lp_solver_success"),
                "lp_solver_status_counts": _counter(lp_records, "lp_solver_status"),
                "lp_solver_message_counts": _counter(lp_records, "lp_solver_message"),
                "lp_error_type_counts": _counter(
                    lp_records, "lp_dry_run_error_type"
                ),
                "total_lp_num_requests": _sum_numeric(lp_records, "lp_num_requests"),
                "total_lp_num_prefill_tokens": _sum_numeric(
                    lp_records, "lp_num_prefill_tokens"
                ),
                "total_lp_num_preemptions": _sum_numeric(
                    lp_records, "lp_num_preemptions"
                ),
                "total_lp_num_forced_preemptions": _sum_numeric(
                    lp_records, "lp_num_forced_preemptions"
                ),
            }
        )

    return summary


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "null"
    return str(value)


def format_text(summary: dict[str, Any]) -> str:
    wall = summary["scheduler_wall_time_ms"]

    lines = [
        "=== scheduler JSONL analysis ===",
        f"input_path={summary['input_path']}",
        f"num_records_total={summary['num_records_total']}",
        f"num_parse_errors={summary['num_parse_errors']}",
        f"event_counts={json.dumps(summary['event_counts'], sort_keys=True)}",
        f"num_scheduler_call_records={summary['num_scheduler_call_records']}",
        f"num_lp_dry_run_records={summary['num_lp_dry_run_records']}",
        f"scheduler_classes={json.dumps(summary['scheduler_classes'])}",
        f"all_scheduler_calls_ok={summary['all_scheduler_calls_ok']}",
        f"num_scheduler_error_records={summary['num_scheduler_error_records']}",
        "scheduler_wall_time_ms.count="
        f"{_format_value(wall['count'])}",
        "scheduler_wall_time_ms.mean="
        f"{_format_value(wall['mean'])}",
        "scheduler_wall_time_ms.median="
        f"{_format_value(wall['median'])}",
        "scheduler_wall_time_ms.p95="
        f"{_format_value(wall['p95'])}",
        "scheduler_wall_time_ms.max="
        f"{_format_value(wall['max'])}",
        f"total_scheduled_tokens={_format_value(summary['total_scheduled_tokens'])}",
        f"total_preemptions={_format_value(summary['total_preemptions'])}",
        "lp_specific_timing_available="
        f"{summary['lp_specific_timing_available']}",
        f"lp_timing_field={summary['lp_timing_field']}",
        f"overhead_metric_used={summary['overhead_metric_used']}",
        f"measurement_caveat={summary['measurement_caveat']}",
    ]

    if summary.get("num_lp_dry_run_records", 0) > 0:
        lines.extend(
            [
                "lp_fallback_counts="
                f"{json.dumps(summary.get('lp_fallback_counts', {}), sort_keys=True)}",
                "lp_unsupported_reason_counts="
                f"{json.dumps(summary.get('lp_unsupported_reason_counts', {}), sort_keys=True)}",
                "lp_solver_success_counts="
                f"{json.dumps(summary.get('lp_solver_success_counts', {}), sort_keys=True)}",
                "lp_solver_message_counts="
                f"{json.dumps(summary.get('lp_solver_message_counts', {}), sort_keys=True)}",
                "lp_error_type_counts="
                f"{json.dumps(summary.get('lp_error_type_counts', {}), sort_keys=True)}",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, help="Path to scheduler JSONL file")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for machine-readable JSON summary",
    )
    parser.add_argument(
        "--text-out",
        type=Path,
        default=None,
        help="Optional path for compact text summary",
    )
    args = parser.parse_args()

    if not args.jsonl.exists():
        parser.error(f"JSONL file does not exist: {args.jsonl}")
    if not args.jsonl.is_file():
        parser.error(f"JSONL path is not a file: {args.jsonl}")

    summary = analyze(args.jsonl)
    text = format_text(summary)

    print(text, end="")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.text_out is not None:
        args.text_out.parent.mkdir(parents=True, exist_ok=True)
        args.text_out.write_text(text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())