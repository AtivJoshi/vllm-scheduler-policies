#!/usr/bin/env python3
"""Summarize an LP dry-run overhead matrix directory.

The summary combines:
- benchmark-level metrics parsed from bench_*.log;
- scheduler JSONL analysis metrics from analysis.json;
- grouped comparisons between simple_policy_1 and primal_lp_dry_run.

This script is intentionally dependency-free.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


BENCH_NUMERIC_KEYS = (
    "Successful requests",
    "Failed requests",
    "Request throughput (req/s)",
    "Output token throughput (tok/s)",
    "Total Token throughput (tok/s)",
    "Mean TTFT (ms)",
    "Median TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean TPOT (ms)",
    "Median TPOT (ms)",
    "P99 TPOT (ms)",
    "Mean ITL (ms)",
    "Median ITL (ms)",
    "P99 ITL (ms)",
    "Mean E2EL (ms)",
    "Median E2EL (ms)",
    "P99 E2EL (ms)",
)

COMPARISON_BASELINE = "simple_policy_1"
COMPARISON_TREATMENT = "primal_lp_dry_run"
TIMING_FIELDS = (
    "scheduler_wall_time_ms",
    "native_schedule_wall_time_ms",
    "lp_dry_run_total_wall_time_ms",
    "lp_snapshot_wall_time_ms",
    "lp_weight_wall_time_ms",
    "lp_relaxed_lp_build_wall_time_ms",
    "lp_highs_solve_wall_time_ms",
    "lp_relaxed_solution_parse_wall_time_ms",
    "lp_extract_wall_time_ms",
    "lp_plan_wall_time_ms",
    "lp_summary_wall_time_ms",
    "lp_log_prepare_wall_time_ms",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_cell_name(name: str) -> tuple[str, str]:
    if "_seed" not in name:
        return name, ""
    scheduler, seed = name.rsplit("_seed", 1)
    return scheduler, seed


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        value_f = float(value)
        return value_f if math.isfinite(value_f) else None

    text = str(value).strip()
    if not text:
        return None

    # vLLM benchmark values are usually plain numbers, but tolerate commas.
    text = text.replace(",", "")
    try:
        value_f = float(text)
    except ValueError:
        return None
    return value_f if math.isfinite(value_f) else None


def _bench_metrics(cell_dir: Path) -> dict[str, float]:
    """Parse known numeric benchmark metrics from the first bench_*.log file."""
    logs = sorted(cell_dir.glob("bench_*.log"))
    if not logs:
        return {}

    metrics: dict[str, float] = {}
    for line in logs[0].read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value_f = _float_or_none(value)

        if key in BENCH_NUMERIC_KEYS and value_f is not None:
            metrics[key] = value_f

    return metrics


def _numeric_list(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        value_f = _float_or_none(value)
        if value_f is not None:
            out.append(value_f)
    return out


def _mean(values: list[Any]) -> float | None:
    xs = _numeric_list(values)
    if not xs:
        return None
    return statistics.fmean(xs)


def _median(values: list[Any]) -> float | None:
    xs = _numeric_list(values)
    if not xs:
        return None
    return statistics.median(xs)


def _sample_stdev(values: list[Any]) -> float | None:
    xs = _numeric_list(values)
    if len(xs) < 2:
        return None
    return statistics.stdev(xs)


def _min(values: list[Any]) -> float | None:
    xs = _numeric_list(values)
    if not xs:
        return None
    return min(xs)


def _max(values: list[Any]) -> float | None:
    xs = _numeric_list(values)
    if not xs:
        return None
    return max(xs)


def _sum(values: list[Any]) -> float | None:
    xs = _numeric_list(values)
    if not xs:
        return None
    return sum(xs)


def _format_float(value: Any) -> str:
    value_f = _float_or_none(value)
    if value_f is None:
        return "null"
    return f"{value_f:.6g}"


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    numerator_f = _float_or_none(numerator)
    denominator_f = _float_or_none(denominator)
    if numerator_f is None or denominator_f is None or denominator_f == 0:
        return None
    return numerator_f / denominator_f


def _metric_stats(cells: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [cell.get(key) for cell in cells]
    return {
        "mean": _mean(values),
        "median": _median(values),
        "stdev": _sample_stdev(values),
        "min": _min(values),
        "max": _max(values),
    }


def _timing_cell_fields(analysis: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for timing_field in TIMING_FIELDS:
        stats = analysis.get(timing_field, {})
        if not isinstance(stats, dict):
            stats = {}
        mean = _float_or_none(stats.get("mean"))
        count = _float_or_none(stats.get("count"))
        fields[f"{timing_field}_mean"] = mean
        fields[f"{timing_field}_p95"] = _float_or_none(stats.get("p95"))
        fields[f"{timing_field}_sum"] = (
            mean * count if mean is not None and count is not None else None
        )
    return fields


def _metric_summary_for_scheduler(cells: list[dict[str, Any]]) -> dict[str, Any]:
    failed_requests = [cell.get("failed_requests") for cell in cells]

    summary = {
        "num_cells": len(cells),
        "seeds": [cell["seed"] for cell in cells],
        "all_scheduler_calls_ok": all(
            cell.get("all_scheduler_calls_ok") is True for cell in cells
        ),
        "total_failed_requests": _sum(failed_requests),
        "successful_requests": _metric_stats(cells, "successful_requests"),
        "request_throughput_req_s": _metric_stats(
            cells, "request_throughput_req_s"
        ),
        "output_token_throughput_tok_s": _metric_stats(
            cells, "output_token_throughput_tok_s"
        ),
        "total_token_throughput_tok_s": _metric_stats(
            cells, "total_token_throughput_tok_s"
        ),
        "mean_ttft_ms": _metric_stats(cells, "mean_ttft_ms"),
        "median_ttft_ms": _metric_stats(cells, "median_ttft_ms"),
        "p99_ttft_ms": _metric_stats(cells, "p99_ttft_ms"),
        "mean_tpot_ms": _metric_stats(cells, "mean_tpot_ms"),
        "median_tpot_ms": _metric_stats(cells, "median_tpot_ms"),
        "p99_tpot_ms": _metric_stats(cells, "p99_tpot_ms"),
        "mean_itl_ms": _metric_stats(cells, "mean_itl_ms"),
        "median_itl_ms": _metric_stats(cells, "median_itl_ms"),
        "p99_itl_ms": _metric_stats(cells, "p99_itl_ms"),
        "mean_e2el_ms": _metric_stats(cells, "mean_e2el_ms"),
        "median_e2el_ms": _metric_stats(cells, "median_e2el_ms"),
        "p99_e2el_ms": _metric_stats(cells, "p99_e2el_ms"),
        "scheduler_wall_time_ms_median": _metric_stats(
            cells, "scheduler_wall_time_ms_median"
        ),
        "scheduler_wall_time_ms_max": _metric_stats(
            cells, "scheduler_wall_time_ms_max"
        ),
        "num_scheduler_call_records": _metric_stats(
            cells, "num_scheduler_call_records"
        ),
        "num_lp_dry_run_records": _metric_stats(cells, "num_lp_dry_run_records"),
        "total_scheduled_tokens": _metric_stats(cells, "total_scheduled_tokens"),
        "total_preemptions": _metric_stats(cells, "total_preemptions"),
        "lp_specific_timing_available_any": any(
            cell.get("lp_specific_timing_available") is True for cell in cells
        ),
        "overhead_metric_used": sorted(
            {str(cell.get("overhead_metric_used")) for cell in cells}
        ),
    }
    for timing_field in TIMING_FIELDS:
        for stat in ("mean", "p95", "sum"):
            key = f"{timing_field}_{stat}"
            summary[key] = _metric_stats(cells, key)
    return summary


def _comparison_from_groups(grouped: dict[str, Any]) -> dict[str, Any]:
    baseline = grouped.get(COMPARISON_BASELINE)
    treatment = grouped.get(COMPARISON_TREATMENT)
    if baseline is None or treatment is None:
        return {}

    comparison: dict[str, Any] = {}

    metric_paths = {
        "mean_tpot_ms": (
            baseline["mean_tpot_ms"]["mean"],
            treatment["mean_tpot_ms"]["mean"],
        ),
        "median_tpot_ms": (
            baseline["median_tpot_ms"]["mean"],
            treatment["median_tpot_ms"]["mean"],
        ),
        "p99_tpot_ms": (
            baseline["p99_tpot_ms"]["mean"],
            treatment["p99_tpot_ms"]["mean"],
        ),
        "mean_ttft_ms": (
            baseline["mean_ttft_ms"]["mean"],
            treatment["mean_ttft_ms"]["mean"],
        ),
        "request_throughput_req_s": (
            baseline["request_throughput_req_s"]["mean"],
            treatment["request_throughput_req_s"]["mean"],
        ),
        "output_token_throughput_tok_s": (
            baseline["output_token_throughput_tok_s"]["mean"],
            treatment["output_token_throughput_tok_s"]["mean"],
        ),
        "total_token_throughput_tok_s": (
            baseline["total_token_throughput_tok_s"]["mean"],
            treatment["total_token_throughput_tok_s"]["mean"],
        ),
    }
    for timing_field in TIMING_FIELDS:
        for stat in ("mean", "p95", "sum"):
            key = f"{timing_field}_{stat}"
            metric_paths[key] = (
                baseline[key]["mean"],
                treatment[key]["mean"],
            )

    for metric, (baseline_value, treatment_value) in metric_paths.items():
        baseline_f = _float_or_none(baseline_value)
        treatment_f = _float_or_none(treatment_value)
        if baseline_f is None or treatment_f is None:
            continue

        comparison[f"{metric}_baseline_mean"] = baseline_f
        comparison[f"{metric}_treatment_mean"] = treatment_f
        comparison[f"{metric}_delta"] = treatment_f - baseline_f
        comparison[f"{metric}_ratio"] = _safe_ratio(treatment_f, baseline_f)

    scheduler_delta = comparison.get("scheduler_wall_time_ms_mean_delta")
    tpot_delta = comparison.get("mean_tpot_ms_delta")
    if scheduler_delta is not None and tpot_delta is not None:
        comparison["mean_tpot_delta_per_scheduler_mean_delta"] = _safe_ratio(
            tpot_delta, scheduler_delta
        )

    return comparison


def summarize(matrix_root: Path) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    for cell_dir in sorted(p for p in matrix_root.iterdir() if p.is_dir()):
        analysis_path = cell_dir / "analysis.json"
        if not analysis_path.exists():
            continue

        analysis = _load_json(analysis_path)
        scheduler, seed = _parse_cell_name(cell_dir.name)
        wall = analysis.get("scheduler_wall_time_ms", {})
        if not isinstance(wall, dict):
            wall = {}
        bench = _bench_metrics(cell_dir)

        cell = {
            "cell": cell_dir.name,
            "scheduler": scheduler,
            "seed": seed,
            "successful_requests": bench.get("Successful requests"),
            "failed_requests": bench.get("Failed requests"),
            "request_throughput_req_s": bench.get("Request throughput (req/s)"),
            "output_token_throughput_tok_s": bench.get(
                "Output token throughput (tok/s)"
            ),
            "total_token_throughput_tok_s": bench.get(
                "Total Token throughput (tok/s)"
            ),
            "mean_ttft_ms": bench.get("Mean TTFT (ms)"),
            "median_ttft_ms": bench.get("Median TTFT (ms)"),
            "p99_ttft_ms": bench.get("P99 TTFT (ms)"),
            "mean_tpot_ms": bench.get("Mean TPOT (ms)"),
            "median_tpot_ms": bench.get("Median TPOT (ms)"),
            "p99_tpot_ms": bench.get("P99 TPOT (ms)"),
            "mean_itl_ms": bench.get("Mean ITL (ms)"),
            "median_itl_ms": bench.get("Median ITL (ms)"),
            "p99_itl_ms": bench.get("P99 ITL (ms)"),
            "mean_e2el_ms": bench.get("Mean E2EL (ms)"),
            "median_e2el_ms": bench.get("Median E2EL (ms)"),
            "p99_e2el_ms": bench.get("P99 E2EL (ms)"),
            "num_records_total": analysis.get("num_records_total"),
            "num_scheduler_call_records": analysis.get(
                "num_scheduler_call_records"
            ),
            "num_lp_dry_run_records": analysis.get("num_lp_dry_run_records"),
            "all_scheduler_calls_ok": analysis.get("all_scheduler_calls_ok"),
            "scheduler_wall_time_ms_median": wall.get("median"),
            "scheduler_wall_time_ms_max": wall.get("max"),
            "total_scheduled_tokens": analysis.get("total_scheduled_tokens"),
            "total_preemptions": analysis.get("total_preemptions"),
            "lp_specific_timing_available": analysis.get(
                "lp_specific_timing_available"
            ),
            "phase14_timing_fields_present": analysis.get(
                "phase14_timing_fields_present", []
            ),
            "overhead_metric_used": analysis.get("overhead_metric_used"),
            "lp_fallback_counts": analysis.get("lp_fallback_counts"),
            "lp_unsupported_reason_counts": analysis.get(
                "lp_unsupported_reason_counts"
            ),
            "lp_solver_success_counts": analysis.get("lp_solver_success_counts"),
            "lp_error_type_counts": analysis.get("lp_error_type_counts"),
        }
        cell.update(_timing_cell_fields(analysis))
        cells.append(cell)

    by_scheduler_cells: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        by_scheduler_cells.setdefault(str(cell["scheduler"]), []).append(cell)

    grouped = {
        scheduler: _metric_summary_for_scheduler(scheduler_cells)
        for scheduler, scheduler_cells in sorted(by_scheduler_cells.items())
    }

    comparison = _comparison_from_groups(grouped)

    return {
        "matrix_root": str(matrix_root),
        "num_cells": len(cells),
        "cells": cells,
        "by_scheduler": grouped,
        "comparison": comparison,
        "measurement_caveat": (
            "Phase 14 timing fields are reported when present. Missing timing "
            "values remain null and are excluded from grouped statistics and "
            "treatment-versus-baseline comparisons."
        ),
    }


def _stats_inline(stats: dict[str, Any]) -> str:
    return (
        f"mean={_format_float(stats.get('mean'))}, "
        f"median={_format_float(stats.get('median'))}, "
        f"stdev={_format_float(stats.get('stdev'))}, "
        f"min={_format_float(stats.get('min'))}, "
        f"max={_format_float(stats.get('max'))}"
    )


def format_text(summary: dict[str, Any]) -> str:
    lines = [
        "=== matrix summary ===",
        f"matrix_root={summary['matrix_root']}",
        f"num_cells={summary['num_cells']}",
        f"measurement_caveat={summary['measurement_caveat']}",
        "",
        (
            "cell,scheduler,seed,success,failed,req_throughput,"
            "mean_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,"
            "wall_mean_ms,wall_p95_ms,wall_sum_ms,wall_max_ms,"
            "native_mean_ms,lp_total_mean_ms,lp_highs_mean_ms,"
            "scheduler_calls,lp_records"
        ),
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
                    _format_float(cell["mean_ttft_ms"]),
                    _format_float(cell["mean_tpot_ms"]),
                    _format_float(cell["median_tpot_ms"]),
                    _format_float(cell["p99_tpot_ms"]),
                    _format_float(cell["scheduler_wall_time_ms_mean"]),
                    _format_float(cell["scheduler_wall_time_ms_p95"]),
                    _format_float(cell["scheduler_wall_time_ms_sum"]),
                    _format_float(cell["scheduler_wall_time_ms_max"]),
                    _format_float(cell["native_schedule_wall_time_ms_mean"]),
                    _format_float(cell["lp_dry_run_total_wall_time_ms_mean"]),
                    _format_float(cell["lp_highs_solve_wall_time_ms_mean"]),
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
                f"  successful_requests: {_stats_inline(group['successful_requests'])}",
                "  request_throughput_req_s: "
                f"{_stats_inline(group['request_throughput_req_s'])}",
                "  output_token_throughput_tok_s: "
                f"{_stats_inline(group['output_token_throughput_tok_s'])}",
                "  total_token_throughput_tok_s: "
                f"{_stats_inline(group['total_token_throughput_tok_s'])}",
                f"  mean_ttft_ms: {_stats_inline(group['mean_ttft_ms'])}",
                f"  median_ttft_ms: {_stats_inline(group['median_ttft_ms'])}",
                f"  p99_ttft_ms: {_stats_inline(group['p99_ttft_ms'])}",
                f"  mean_tpot_ms: {_stats_inline(group['mean_tpot_ms'])}",
                f"  median_tpot_ms: {_stats_inline(group['median_tpot_ms'])}",
                f"  p99_tpot_ms: {_stats_inline(group['p99_tpot_ms'])}",
                f"  mean_itl_ms: {_stats_inline(group['mean_itl_ms'])}",
                f"  median_itl_ms: {_stats_inline(group['median_itl_ms'])}",
                f"  p99_itl_ms: {_stats_inline(group['p99_itl_ms'])}",
                "  scheduler_wall_time_ms_mean: "
                f"{_stats_inline(group['scheduler_wall_time_ms_mean'])}",
                "  scheduler_wall_time_ms_p95: "
                f"{_stats_inline(group['scheduler_wall_time_ms_p95'])}",
                "  scheduler_wall_time_ms_sum: "
                f"{_stats_inline(group['scheduler_wall_time_ms_sum'])}",
                "  num_scheduler_call_records: "
                f"{_stats_inline(group['num_scheduler_call_records'])}",
                "  num_lp_dry_run_records: "
                f"{_stats_inline(group['num_lp_dry_run_records'])}",
                "  lp_specific_timing_available_any="
                f"{group['lp_specific_timing_available_any']}",
                f"  overhead_metric_used={group['overhead_metric_used']}",
            ]
        )
        for timing_field in TIMING_FIELDS[1:]:
            lines.append(
                f"  {timing_field}: "
                "cell_mean_avg="
                f"{_format_float(group[f'{timing_field}_mean']['mean'])}, "
                "cell_p95_avg="
                f"{_format_float(group[f'{timing_field}_p95']['mean'])}, "
                "cell_sum_avg="
                f"{_format_float(group[f'{timing_field}_sum']['mean'])}"
            )

    lines.extend(["", "=== comparison ==="])
    for key, value in sorted(summary["comparison"].items()):
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
