"""calibration_log.py — consumer of per-pass instrumentation JSONL.

Phase 2 of the 2026-05-29 follow-ups plan (post-d90e4cf). Reads the per-run
instrumentation written by extended_research_runner.py:_emit_pass_instrumentation
and produces summary statistics that drive future tuning of the INTERIM
constants (ARTIFACT_TEXT_TOKEN_CAP=5000, reasoning-trail-threshold=2500,
mount-threshold=200).

Three public functions:
  - aggregate_run(workdir) — per-run summary dict
  - append_to_global_summary(run_summary) — append to global JSONL with 10 MB
    tail-truncate at write time
  - summarize_recent(n=20) — read last N global summaries, return readable text

CLI entry: `python calibration_log.py` prints summarize_recent(20).

Fail-open: every public function wraps its I/O in try/except returning safe
empty defaults. Never raises. Never blocks the runner.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

GLOBAL_SUMMARY_PATH = (
    Path.home() / ".claude" / "council-automation" / "calibration-summary.jsonl"
)
GLOBAL_SUMMARY_CAP_BYTES = 10 * 1024 * 1024  # 10 MB


def _append_with_tail_truncate(
    path: Path, line: str, cap_bytes: int = GLOBAL_SUMMARY_CAP_BYTES
) -> None:
    """Append a line, tail-truncating to 8 MB if file exceeds cap_bytes."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > cap_bytes:
            data = path.read_bytes()[-(8 * 1024 * 1024):]
            first_nl = data.find(b"\n")
            if first_nl > 0:
                data = data[first_nl + 1:]
            path.write_bytes(data)
        with path.open("a", encoding="utf-8") as f:
            f.write(line if line.endswith("\n") else line + "\n")
    except Exception:
        pass


def aggregate_run(workdir: Path) -> dict[str, Any]:
    """Read {workdir}/instrumentation.jsonl, return per-run summary stats.

    Always returns a dict; on any error returns one with entry_count=0 so
    the caller's WARN-on-zero-rows-from-non-empty-source check works.
    """
    summary: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "workdir": str(workdir),
        "entry_count": 0,
        "passes_completed": 0,
        "passes_parse_failed": 0,
        "passes_skipped_network": 0,
        "retry_fired_count": 0,
        "partial_prefix_attempted_count": 0,
        "partial_prefix_succeeded_count": 0,
        "total_prompt_chars": 0,
        "total_raw_response_chars": 0,
        "mean_prompt_chars": 0.0,
        "mean_raw_response_chars": 0.0,
        "mean_backoff_s_when_retry": 0.0,
        "pass_types": {},  # dict[str, int]
    }
    try:
        jsonl_path = workdir / "instrumentation.jsonl"
        if not jsonl_path.exists():
            return summary
        backoff_sum = 0.0
        backoff_count = 0
        for raw_line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            summary["entry_count"] += 1
            status = rec.get("status", "")
            if status == "COMPLETED":
                summary["passes_completed"] += 1
            elif status == "PARSE-FAILED":
                summary["passes_parse_failed"] += 1
            elif status == "SKIPPED-NETWORK":
                summary["passes_skipped_network"] += 1
            if rec.get("retry_fired"):
                summary["retry_fired_count"] += 1
                bo = rec.get("backoff_s", 0.0) or 0.0
                if isinstance(bo, (int, float)) and bo > 0:
                    backoff_sum += float(bo)
                    backoff_count += 1
            if rec.get("partial_prefix_attempted"):
                summary["partial_prefix_attempted_count"] += 1
            if rec.get("partial_prefix_succeeded"):
                summary["partial_prefix_succeeded_count"] += 1
            summary["total_prompt_chars"] += int(rec.get("prompt_len_chars", 0) or 0)
            summary["total_raw_response_chars"] += int(rec.get("raw_response_len_chars", 0) or 0)
            pt = rec.get("pass_type", "")
            if pt:
                summary["pass_types"][pt] = summary["pass_types"].get(pt, 0) + 1
        n = max(summary["entry_count"], 1)
        summary["mean_prompt_chars"] = round(summary["total_prompt_chars"] / n, 1)
        summary["mean_raw_response_chars"] = round(summary["total_raw_response_chars"] / n, 1)
        if backoff_count:
            summary["mean_backoff_s_when_retry"] = round(backoff_sum / backoff_count, 1)
    except Exception:
        pass
    return summary


def append_to_global_summary(
    run_summary: dict[str, Any], summary_path: Path | None = None
) -> None:
    """Append one JSONL line to ~/.claude/council-automation/calibration-summary.jsonl."""
    if summary_path is None:
        summary_path = GLOBAL_SUMMARY_PATH
    try:
        _append_with_tail_truncate(summary_path, json.dumps(run_summary, default=str))
    except Exception:
        pass


def summarize_recent(n: int = 20, summary_path: Path | None = None) -> str:
    """Return readable text summary of last N global-summary entries."""
    if summary_path is None:
        summary_path = GLOBAL_SUMMARY_PATH
    if not summary_path.exists():
        return f"(no calibration data yet at {summary_path})"
    try:
        lines = [
            ln for ln in summary_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
    except Exception as e:
        return f"(failed to read {summary_path}: {type(e).__name__})"
    if not lines:
        return f"(empty calibration log at {summary_path})"
    recent = lines[-n:]
    runs: list[dict[str, Any]] = []
    for ln in recent:
        try:
            runs.append(json.loads(ln))
        except Exception:
            continue
    if not runs:
        return f"({len(recent)} entries unparseable in {summary_path})"
    total_passes = sum(r.get("entry_count", 0) for r in runs)
    if total_passes == 0:
        return f"({len(runs)} runs in last {n}, but all reported 0 passes)"
    total_retries = sum(r.get("retry_fired_count", 0) for r in runs)
    total_completed = sum(r.get("passes_completed", 0) for r in runs)
    total_parse_failed = sum(r.get("passes_parse_failed", 0) for r in runs)
    total_partial_prefix_succ = sum(
        r.get("partial_prefix_succeeded_count", 0) for r in runs
    )
    sum_prompt = sum(r.get("total_prompt_chars", 0) for r in runs)
    sum_response = sum(r.get("total_raw_response_chars", 0) for r in runs)
    pass_types: dict[str, int] = {}
    for r in runs:
        for pt, cnt in (r.get("pass_types") or {}).items():
            pass_types[pt] = pass_types.get(pt, 0) + cnt
    lines_out = [
        f"# Calibration summary — last {len(runs)} runs ({total_passes} total passes)",
        "",
        f"- pct_retry_fired:           {100.0 * total_retries / total_passes:.1f}%  ({total_retries}/{total_passes})",
        f"- pct_partial_prefix_salvage: {100.0 * total_partial_prefix_succ / total_passes:.1f}%  ({total_partial_prefix_succ}/{total_passes})",
        f"- pct_completed:             {100.0 * total_completed / total_passes:.1f}%  ({total_completed}/{total_passes})",
        f"- pct_parse_failed:          {100.0 * total_parse_failed / total_passes:.1f}%  ({total_parse_failed}/{total_passes})",
        f"- mean_prompt_chars:         {sum_prompt / total_passes:.0f}",
        f"- mean_raw_response_chars:   {sum_response / total_passes:.0f}",
        "",
        "Pass-type distribution:",
    ]
    for pt, cnt in sorted(pass_types.items(), key=lambda kv: -kv[1]):
        lines_out.append(f"  {pt:20s} {cnt:4d}  ({100.0 * cnt / total_passes:.1f}%)")
    return "\n".join(lines_out)


if __name__ == "__main__":
    n = 20
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    print(summarize_recent(n))
