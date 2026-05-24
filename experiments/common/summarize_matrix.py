#!/usr/bin/env python3
"""Summarize a Phase 13 LP dry-run overhead matrix directory."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_cell_name(name: str) -> tuple[str, str]:
    if "_seed" not in name:
        return name, ""
    scheduler, seed = name.rsplit("_seed", 1)
    return scheduler, seed


def _bench_metrics(cell_dir: Path) -> dict[str, str]:
    logs = sorted(cell_dir.glob("bench_*.log"))
    if not logs:
        return {}

    metrics: dict[str, str] = {}
    for line in logs[0].read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {
            "Successful requests",
            "Failed requests",
            "Request throughput (req/s)",
            "Mean TTFT (ms)",
            "Mean TPOT (ms)",
        }:
            metrics[key] = value
    return metrics


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def _format_float(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:.6g}"


def summarize(matrix_root: Path) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    for cell_dir in sorted(p for p in matrix_root.iterdir() if p.is_dir()):
        analysis_path = cell_dir / "analysis.json"
        if not analysis_path.exists():
            continue

        analysis = _load_json(analysis_path)
        scheduler, seed = _parse_cell_name(cell_dir.name)
        wall = analysis.get("scheduler_wall_time_ms", {})
        bench = _bench_metrics(cell_dir)

        cell = {
            "cell": cell_dir.name,
            "scheduler": scheduler,
            "seed": seed,
            "successful_requests": _float_or_none(
                bench.get("Successful requests")
            ),
            "failed_requests": _float_or_none(bench.get("Failed requests")),
            "request_throughput_req_s": _float_or_none(
                bench.get("Request throughput (req/s)")
            ),
            "mean_ttft_ms": _float_or_none(bench.get("Mean TTFT (ms)")),
            "mean_tpot_ms": _float_or_none(bench.get("Mean TPOT (ms)")),
            "num_records_total": analysis.get("num_records_total"),
            "num_scheduler_call_records": analysis.get(
                "num_scheduler_call_records"
            ),
            "num_lp_dry_run_records": analysis.get("num_lp_dry_run_records"),
            "all_scheduler_calls_ok": analysis.get("all_scheduler_calls_ok"),
            "scheduler_wall_time_ms_mean": wall.get("mean"),
            "scheduler_wall_time_ms_median": wall.get("median"),
            "scheduler_wall_time_ms_p95": wall.get("p95"),
            "scheduler_wall_time_ms_max": wall.get("max"),
            "total_scheduled_tokens": analysis.get("total_scheduled_tokens"),
            "total_preemptions": analysis.get("total_preemptions"),
            "lp_specific_timing_available": analysis.get(
                "lp_specific_timing_available"
            ),
            "overhead_metric_used": analysis.get("overhead_metric_used"),
            "lp_fallback_counts": analysis.get("lp_fallback_counts"),
            "lp_error_type_counts": analysis.get("lp_error_type_counts"),
        }
        cells.append(cell)

    by_scheduler: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        by_scheduler.setdefault(str(cell["scheduler"]), []).append(cell)

    grouped: dict[str, Any] = {}
    for scheduler, scheduler_cells in sorted(by_scheduler.items()):
        wall_means = [
            float(c["scheduler_wall_time_ms_mean"])
            for c in scheduler_cells
            if c.get("scheduler_wall_time_ms_mean") is not None
        ]
        wall_p95s = [
            float(c["scheduler_wall_time_ms_p95"])
            for c in scheduler_cells
            if c.get("scheduler_wall_time_ms_p95") is not None
        ]
        throughputs = [
            float(c["request_throughput_req_s"])
            for c in scheduler_cells
            if c.get("request_throughput_req_s") is not None
        ]
        failed_requests = [
            float(c["failed_requests"])
            for c in scheduler_cells
            if c.get("failed_requests") is not None
        ]

        grouped[scheduler] = {
            "num_cells": len(scheduler_cells),
            "seeds": [c["seed"] for c in scheduler_cells],
            "all_scheduler_calls_ok": all(
                c.get("all_scheduler_calls_ok") is True for c in scheduler_cells
            ),
            "total_failed_requests": sum(failed_requests)
            if failed_requests
            else None,
            "mean_of_scheduler_wall_time_ms_mean": _mean(wall_means),
            "mean_of_scheduler_wall_time_ms_p95": _mean(wall_p95s),
            "mean_request_throughput_req_s": _mean(throughputs),
            "lp_specific_timing_available_any": any(
                c.get("lp_specific_timing_available") is True
                for c in scheduler_cells
            ),
            "overhead_metric_used": sorted(
                {str(c.get("overhead_metric_used")) for c in scheduler_cells}
            ),
        }

    comparison: dict[str, Any] = {}
    baseline = grouped.get("simple_policy_1", {})
    lp = grouped.get("primal_lp_dry_run", {})
    baseline_mean = baseline.get("mean_of_scheduler_wall_time_ms_mean")
    lp_mean = lp.get("mean_of_scheduler_wall_time_ms_mean")
    if isinstance(baseline_mean, (int, float)) and isinstance(lp_mean, (int, float)):
        comparison["mean_wall_time_delta_ms"] = lp_mean - baseline_mean
        comparison["mean_wall_time_ratio"] = (
            lp_mean / baseline_mean if baseline_mean != 0 else None
        )

    return {
        "matrix_root": str(matrix_root),
        "num_cells": len(cells),
        "cells": cells,
        "by_scheduler": grouped,
        "comparison": comparison,
        "measurement_caveat": (
            "Unless lp_specific_timing_available is true, comparisons use "
            "scheduler_wall_time_ms and measure total scheduler overhead, not "
            "isolated LP solver time."
        ),
    }


def format_text(summary: dict[str, Any]) -> str:
    lines = [
        "=== matrix summary ===",
        f"matrix_root={summary['matrix_root']}",
        f"num_cells={summary['num_cells']}",
        f"measurement_caveat={summary['measurement_caveat']}",
        "",
        "cell,scheduler,seed,success,failed,throughput,wall_mean_ms,wall_p95_ms,wall_max_ms,scheduler_calls,lp_records",
    ]

    for cell in summary["cells"]:
        lines.append(
            ",".join(
                [
                    str(cell["cell"]),
                    str(cell["scheduler"]),
                    str(cell["seed"]),
                    _format_float(cell["successful_requests"]),
                    _format_float(cell["failed_requests"]),
                    _format_float(cell["request_throughput_req_s"]),
                    _format_float(cell["scheduler_wall_time_ms_mean"]),
                    _format_float(cell["scheduler_wall_time_ms_p95"]),
                    _format_float(cell["scheduler_wall_time_ms_max"]),
                    str(cell["num_scheduler_call_records"]),
                    str(cell["num_lp_dry_run_records"]),
                ]
            )
        )

    lines.extend(["", "=== grouped summary ==="])
    for scheduler, group in summary["by_scheduler"].items():
        lines.extend(
            [
                f"scheduler={scheduler}",
                f"  num_cells={group['num_cells']}",
                f"  seeds={group['seeds']}",
                f"  all_scheduler_calls_ok={group['all_scheduler_calls_ok']}",
                f"  total_failed_requests={_format_float(group['total_failed_requests'])}",
                "  mean_of_scheduler_wall_time_ms_mean="
                f"{_format_float(group['mean_of_scheduler_wall_time_ms_mean'])}",
                "  mean_of_scheduler_wall_time_ms_p95="
                f"{_format_float(group['mean_of_scheduler_wall_time_ms_p95'])}",
                "  mean_request_throughput_req_s="
                f"{_format_float(group['mean_request_throughput_req_s'])}",
                "  lp_specific_timing_available_any="
                f"{group['lp_specific_timing_available_any']}",
                f"  overhead_metric_used={group['overhead_metric_used']}",
            ]
        )

    lines.extend(["", "=== comparison ==="])
    for key, value in summary["comparison"].items():
        lines.append(f"{key}={_format_float(value)}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--text-out", type=Path, default=None)
    args = parser.parse_args()

    if not args.matrix_root.exists() or not args.matrix_root.is_dir():
        parser.error(f"matrix root does not exist: {args.matrix_root}")

    summary = summarize(args.matrix_root)
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