#!/usr/bin/env python3
"""Generate a Markdown experiment record for one run directory.

The record is intentionally compact. It points to raw artifacts but does not
inline raw logs or JSONL contents.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_text_if_exists(path: Path, max_chars: int = 20_000) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated by make_record.py]...\n"
    return text


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(value, dict):
        return value
    return None


def _artifact_status(run_dir: Path, names: list[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for name in names:
        path = run_dir / name
        if path.exists():
            if path.is_file():
                rows.append((name, f"present, {path.stat().st_size} bytes"))
            elif path.is_dir():
                rows.append((name, "present, directory"))
            else:
                rows.append((name, "present"))
        else:
            rows.append((name, "missing"))
    return rows


def _format_markdown_table(rows: list[tuple[str, str]]) -> str:
    lines = [
        "| Artifact | Status |",
        "|---|---|",
    ]
    for artifact, status in rows:
        lines.append(f"| `{artifact}` | {status} |")
    return "\n".join(lines)


def _format_analysis_summary(analysis: dict[str, Any] | None) -> str:
    if analysis is None:
        return "No `analysis.json` file was found or it could not be parsed.\n"

    wall = analysis.get("scheduler_wall_time_ms", {})
    lines = [
        f"- total records: `{analysis.get('num_records_total')}`",
        f"- scheduler-call records: `{analysis.get('num_scheduler_call_records')}`",
        f"- LP dry-run records: `{analysis.get('num_lp_dry_run_records')}`",
        f"- scheduler classes: `{analysis.get('scheduler_classes')}`",
        f"- all scheduler calls ok: `{analysis.get('all_scheduler_calls_ok')}`",
        f"- scheduler wall-time mean ms: `{wall.get('mean')}`",
        f"- scheduler wall-time median ms: `{wall.get('median')}`",
        f"- scheduler wall-time p95 ms: `{wall.get('p95')}`",
        f"- scheduler wall-time max ms: `{wall.get('max')}`",
        f"- total scheduled tokens: `{analysis.get('total_scheduled_tokens')}`",
        f"- total preemptions: `{analysis.get('total_preemptions')}`",
        f"- LP-specific timing available: `{analysis.get('lp_specific_timing_available')}`",
        f"- timing metric used: `{analysis.get('overhead_metric_used')}`",
        f"- measurement caveat: {analysis.get('measurement_caveat')}",
    ]

    if analysis.get("num_lp_dry_run_records", 0):
        lines.extend(
            [
                f"- LP fallback counts: `{analysis.get('lp_fallback_counts')}`",
                "- LP unsupported-reason counts: "
                f"`{analysis.get('lp_unsupported_reason_counts')}`",
                f"- LP solver-success counts: `{analysis.get('lp_solver_success_counts')}`",
                f"- LP solver-message counts: `{analysis.get('lp_solver_message_counts')}`",
                f"- LP error-type counts: `{analysis.get('lp_error_type_counts')}`",
            ]
        )

    return "\n".join(lines) + "\n"


def build_record(run_dir: Path, title: str) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    env_text = _read_text_if_exists(run_dir / "env.txt")
    git_state_text = _read_text_if_exists(run_dir / "git_state.txt")
    commands_text = _read_text_if_exists(run_dir / "commands.sh")
    analysis_text = _read_text_if_exists(run_dir / "analysis.txt")
    analysis_json = _read_json_if_exists(run_dir / "analysis.json")

    artifacts = _artifact_status(
        run_dir,
        [
            "env.txt",
            "git_state.txt",
            "commands.sh",
            "server.log",
            "server.pid",
            "scheduler_iter.jsonl",
            "analysis.json",
            "analysis.txt",
            "record.md",
        ],
    )

    sections = [
        f"# {title}",
        "",
        "## Purpose",
        "",
        (
            "Record one run of the LP dry-run overhead experiment. This file is a "
            "compact human-readable summary and intentionally does not inline raw "
            "server logs, benchmark logs, or scheduler JSONL records."
        ),
        "",
        "## Run identity",
        "",
        f"- run directory: `{run_dir}`",
        f"- generated at UTC: `{generated_at}`",
        "",
        "## Artifact inventory",
        "",
        _format_markdown_table(artifacts),
        "",
        "## Git state",
        "",
    ]

    if git_state_text:
        sections.extend(["```text", git_state_text.rstrip(), "```", ""])
    else:
        sections.extend(["No `git_state.txt` file was found.", ""])

    sections.extend(["## Environment", ""])
    if env_text:
        sections.extend(["```text", env_text.rstrip(), "```", ""])
    else:
        sections.extend(["No `env.txt` file was found.", ""])

    sections.extend(["## Commands", ""])
    if commands_text:
        sections.extend(["```bash", commands_text.rstrip(), "```", ""])
    else:
        sections.extend(["No `commands.sh` file was found.", ""])

    sections.extend(["## Results summary", "", _format_analysis_summary(analysis_json), ""])

    sections.extend(["## Analysis text", ""])
    if analysis_text:
        sections.extend(["```text", analysis_text.rstrip(), "```", ""])
    else:
        sections.extend(["No `analysis.txt` file was found.", ""])

    sections.extend(
        [
            "## Interpretation",
            "",
            (
                "Interpret this run using the timing field reported above. If "
                "`lp_specific_timing_available` is false, the result measures total "
                "dry-run scheduler overhead via `scheduler_wall_time_ms`, not isolated "
                "LP solver time."
            ),
            "",
            "## Limitations",
            "",
            (
                "- Raw logs are not included in this record.\n"
                "- This record is only as complete as the files present in the run directory.\n"
                "- A tiny harness-validation run is not a broad benchmark campaign."
            ),
            "",
            "## Reproducibility notes",
            "",
            (
                "Use the tracked experiment scripts and config files in "
                "`experiments/lp_dry_run_overhead/`. The exact local run artifacts are "
                "stored in the run directory listed above."
            ),
            "",
        ]
    )

    return "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--title",
        default="LP Dry-Run Overhead Run Record",
    )
    args = parser.parse_args()

    if not args.run_dir.exists() or not args.run_dir.is_dir():
        parser.error(f"run directory does not exist: {args.run_dir}")

    out = args.out if args.out is not None else args.run_dir / "record.md"
    record = build_record(args.run_dir, args.title)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(record, encoding="utf-8")

    print(f"record_md={out}")
    print("make_record_completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())