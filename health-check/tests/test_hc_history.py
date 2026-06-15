#!/usr/bin/env python3
"""Tests for hc_history.py — rolling row-count history with first-run baseline guard.

Written test-first (TDD). All filesystem state is isolated via ``tmp_path`` so
tests never touch the real ``~/.claude/health-check/row-counts.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import hc_history


def _counts(entries: list[dict]) -> list[int]:
    """Extract just the count values from a history list."""
    return [e["count"] for e in entries]


# ---------------------------------------------------------------------------
# record / history round-trip + key isolation
# ---------------------------------------------------------------------------

def test_record_then_history_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "row-counts.json"
    hc_history.record("app", "tbl", 247, "2026-06-15T08:00:00", path)

    hist = hc_history.history("app", "tbl", path)
    assert hist == [{"ts": "2026-06-15T08:00:00", "count": 247}]


def test_history_missing_file_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "does-not-exist.json"
    assert hc_history.history("app", "tbl", path) == []


def test_multiple_apps_tables_isolated_by_key(tmp_path: Path) -> None:
    path = tmp_path / "row-counts.json"
    hc_history.record("app1", "users", 10, "2026-06-15T08:00:00", path)
    hc_history.record("app1", "orders", 20, "2026-06-15T08:00:00", path)
    hc_history.record("app2", "users", 30, "2026-06-15T08:00:00", path)

    assert _counts(hc_history.history("app1", "users", path)) == [10]
    assert _counts(hc_history.history("app1", "orders", path)) == [20]
    assert _counts(hc_history.history("app2", "users", path)) == [30]


# ---------------------------------------------------------------------------
# pruning
# ---------------------------------------------------------------------------

def test_pruning_keeps_last_30_newest(tmp_path: Path) -> None:
    path = tmp_path / "row-counts.json"
    for i in range(35):
        hc_history.record("app", "tbl", i, f"2026-06-15T08:{i:02d}:00", path)

    hist = hc_history.history("app", "tbl", path)
    assert len(hist) == 30
    # The newest 30 are counts 5..34 (0..4 dropped as oldest).
    assert _counts(hist) == list(range(5, 35))


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------

def test_baseline_none_on_empty_history(tmp_path: Path) -> None:
    path = tmp_path / "row-counts.json"
    assert hc_history.baseline("app", "tbl", path) is None


def test_baseline_mean_after_entries(tmp_path: Path) -> None:
    path = tmp_path / "row-counts.json"
    hc_history.record("app", "tbl", 100, "2026-06-15T08:00:00", path)
    hc_history.record("app", "tbl", 200, "2026-06-15T08:01:00", path)
    hc_history.record("app", "tbl", 300, "2026-06-15T08:02:00", path)

    assert hc_history.baseline("app", "tbl", path) == 200.0


# ---------------------------------------------------------------------------
# is_anomaly
# ---------------------------------------------------------------------------

def test_is_anomaly_none_base_is_false() -> None:
    assert hc_history.is_anomaly(999, None) is False


def test_is_anomaly_zero_base_is_false() -> None:
    assert hc_history.is_anomaly(999, 0) is False
    assert hc_history.is_anomaly(999, 0.0) is False


def test_is_anomaly_within_pct_is_false() -> None:
    # base=100, count=120 → 20% change, under default 0.5 → not anomaly
    assert hc_history.is_anomaly(120, 100.0) is False


def test_is_anomaly_beyond_pct_is_true() -> None:
    # base=100, count=200 → 100% change, over default 0.5 → anomaly
    assert hc_history.is_anomaly(200, 100.0) is True


def test_is_anomaly_custom_pct() -> None:
    # base=100, count=120 → 20% change
    assert hc_history.is_anomaly(120, 100.0, pct=0.1) is True
    assert hc_history.is_anomaly(120, 100.0, pct=0.5) is False


def test_is_anomaly_drop_beyond_pct_is_true() -> None:
    # base=100, count=40 → 60% drop, over default 0.5 → anomaly
    assert hc_history.is_anomaly(40, 100.0) is True


# ---------------------------------------------------------------------------
# first-run guard (end-to-end)
# ---------------------------------------------------------------------------

def test_first_run_guard_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / "row-counts.json"
    # Typical call order: baseline BEFORE record for the current run.
    base = hc_history.baseline("app", "tbl", path)
    assert base is None
    # First observation must never be flagged as an anomaly.
    assert hc_history.is_anomaly(99999, base) is False
    # Now record the first reading for future runs.
    hc_history.record("app", "tbl", 99999, "2026-06-15T08:00:00", path)
    # Second run now has a baseline.
    assert hc_history.baseline("app", "tbl", path) == 99999.0


# ---------------------------------------------------------------------------
# corrupt file resilience
# ---------------------------------------------------------------------------

def test_corrupt_file_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "row-counts.json"
    path.write_text("not json{", encoding="utf-8")

    # record must not crash on corrupt input.
    hc_history.record("app", "tbl", 42, "2026-06-15T08:00:00", path)

    # File is now valid JSON with the recorded entry.
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"app:tbl": [{"ts": "2026-06-15T08:00:00", "count": 42}]}


def test_empty_file_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "row-counts.json"
    path.write_text("", encoding="utf-8")

    hc_history.record("app", "tbl", 7, "2026-06-15T08:00:00", path)
    assert _counts(hc_history.history("app", "tbl", path)) == [7]


# ---------------------------------------------------------------------------
# atomic write
# ---------------------------------------------------------------------------

def test_atomic_write_valid_json_no_temp_leftover(tmp_path: Path) -> None:
    path = tmp_path / "row-counts.json"
    hc_history.record("app", "tbl", 1, "2026-06-15T08:00:00", path)
    hc_history.record("app", "tbl", 2, "2026-06-15T08:01:00", path)

    # File parses as valid JSON.
    json.loads(path.read_text(encoding="utf-8"))

    # Only the target file remains in the directory — no temp files leaked.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "row-counts.json"]
    assert leftovers == []


def test_record_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "row-counts.json"
    hc_history.record("app", "tbl", 5, "2026-06-15T08:00:00", path)
    assert path.exists()
    assert _counts(hc_history.history("app", "tbl", path)) == [5]


def test_path_accepts_str(tmp_path: Path) -> None:
    path = str(tmp_path / "row-counts.json")
    hc_history.record("app", "tbl", 5, "2026-06-15T08:00:00", path)
    assert _counts(hc_history.history("app", "tbl", path)) == [5]
