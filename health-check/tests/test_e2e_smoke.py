# tests/test_e2e_smoke.py
"""End-to-end smoke test: the command-layer contract against a synthetic app.

This exercises the full contract between the /health-check command layer and the
scripts (hc_db.py CLI, hc_history.py, hc_provision.py) against a throwaway
SQLite database built by ``fixtures/make_demo.py``. Every op is asserted against
deterministic expected values returned by ``build_demo_db`` — not just shape.

Contracts pinned here:
  * Integrity + audit ops produce the JSON shapes/values the command layer reads
    (rowcount, fk_orphans, duplicates, null_drift, stale, audit_user_summary,
    audit_velocity, audit_fingerprint, audit_recency).
  * History: callers MUST compute baseline() BEFORE record() (first-run guard).
  * Provision: write-to-file only — the DB is never touched; output is reviewable
    and contains no write grants.
  * Read-only safety: a write via --raw is refused even with HC_ALLOW_RAW=1, and
    the DB row counts are unchanged afterward.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

# Resolve sibling scripts + the fixtures package relative to this file.
_TESTS_DIR = Path(__file__).resolve().parent
_HC_DIR = _TESTS_DIR.parent
HC = str(_HC_DIR / "hc_db.py")

if str(_HC_DIR) not in sys.path:
    sys.path.insert(0, str(_HC_DIR))
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


def _load_module(name: str, filename: str):
    """Load a sibling script as a module by absolute path."""
    spec = importlib.util.spec_from_file_location(name, str(_HC_DIR / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hc_history = _load_module("hc_history", "hc_history.py")
hc_provision = _load_module("hc_provision", "hc_provision.py")

from fixtures.make_demo import build_demo_db  # noqa: E402


def _run(args, **kw):
    """Invoke the hc_db.py CLI as a subprocess, like the other suites."""
    return subprocess.run([sys.executable, HC, *args],
                          capture_output=True, text=True, **kw)


def _ok_json(args, **kw):
    """Run the CLI, assert success, return parsed JSON stdout."""
    r = _run(args, **kw)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def _build(tmp_path):
    """Build the demo DB in tmp_path and return (db_path, expected)."""
    db = tmp_path / "demo.db"
    expected = build_demo_db(db)
    return db, expected


# --- integrity ops ------------------------------------------------------------

def test_e2e_rowcount_orders(tmp_path):
    db, exp = _build(tmp_path)
    out = _ok_json(["--engine", "sqlite", "--db", str(db),
                    "--op", "rowcount", "--table", "orders"])
    assert out["table"] == "orders"
    assert out["count"] == exp["orders_rowcount"]


def test_e2e_fk_orphans(tmp_path):
    db, exp = _build(tmp_path)
    out = _ok_json(["--engine", "sqlite", "--db", str(db),
                    "--op", "fk_orphans", "--child", "orders",
                    "--column", "customer_id", "--parent", "customers",
                    "--parent-column", "id"])
    assert out["op"] == "fk_orphans"
    assert out["orphans"] == exp["fk_orphans"]


def test_e2e_duplicates(tmp_path):
    db, exp = _build(tmp_path)
    out = _ok_json(["--engine", "sqlite", "--db", str(db),
                    "--op", "duplicates", "--table", "users",
                    "--column", "email"])
    assert out["op"] == "duplicates"
    assert out["duplicate_groups"] == exp["duplicate_groups"]


def test_e2e_null_drift(tmp_path):
    db, exp = _build(tmp_path)
    out = _ok_json(["--engine", "sqlite", "--db", str(db),
                    "--op", "null_drift", "--table", "orders",
                    "--column", "status"])
    assert out["op"] == "null_drift"
    assert out["nulls"] == exp["null_count"]


def test_e2e_stale_one_hour_threshold(tmp_path):
    db, exp = _build(tmp_path)
    out = _ok_json(["--engine", "sqlite", "--db", str(db),
                    "--op", "stale", "--table", "orders",
                    "--column", "status", "--value", "processing",
                    "--age-column", "updated_at", "--older-than-hours", "1"])
    assert out["op"] == "stale"
    assert out["stale"] == exp["stale_count"]


# --- audit ops ----------------------------------------------------------------

def test_e2e_audit_user_summary_alice(tmp_path):
    db, exp = _build(tmp_path)
    out = _ok_json(["--engine", "sqlite", "--db", str(db),
                    "--op", "audit_user_summary", "--table", "audit_log",
                    "--user-column", "user_id", "--action-column", "action"])
    assert out["op"] == "audit_user_summary"
    users = {u["user_id"]: u for u in out["users"]}
    assert users["alice"]["actions"] == exp["alice_actions"]
    assert users["alice"]["total"] == exp["alice_total"]
    # users sorted by total desc.
    totals = [u["total"] for u in out["users"]]
    assert totals == sorted(totals, reverse=True)


def test_e2e_audit_velocity_flags_mallory(tmp_path):
    db, exp = _build(tmp_path)
    out = _ok_json(["--engine", "sqlite", "--db", str(db),
                    "--op", "audit_velocity", "--table", "audit_log",
                    "--user-column", "user_id", "--ts-column", "ts",
                    "--window-seconds", str(exp["velocity_window_seconds"]),
                    "--threshold", str(exp["velocity_threshold"])])
    assert out["op"] == "audit_velocity"
    assert out["window_seconds"] == exp["velocity_window_seconds"]
    assert out["threshold"] == exp["velocity_threshold"]
    bursts = {b["user_id"]: b for b in out["bursts"]}
    assert "mallory" in bursts
    assert bursts["mallory"]["count"] == exp["mallory_burst_count"]
    assert "bucket_start_epoch" in bursts["mallory"]
    # alice (max 5/window) and bob (1) are below the threshold.
    assert "alice" not in bursts
    assert "bob" not in bursts


def test_e2e_audit_fingerprint_alice_shape(tmp_path):
    db, exp = _build(tmp_path)
    out = _ok_json(["--engine", "sqlite", "--db", str(db),
                    "--op", "audit_fingerprint", "--table", "audit_log",
                    "--user-column", "user_id", "--action-column", "action"])
    assert out["op"] == "audit_fingerprint"
    users = {u["user_id"]: u for u in out["users"]}
    assert users["alice"]["actions"] == exp["alice_actions"]


def test_e2e_audit_recency(tmp_path):
    db, exp = _build(tmp_path)
    out = _ok_json(["--engine", "sqlite", "--db", str(db),
                    "--op", "audit_recency", "--table", "audit_log",
                    "--ts-column", "ts"])
    assert out["op"] == "audit_recency"
    assert out["events_24h"] == exp["events_24h"]
    assert out["events_7d"] == exp["events_7d"]
    assert out["last_write_ts"] is not None


# --- history integration: baseline-before-record ordering contract ------------

def test_e2e_history_baseline_before_record_contract(tmp_path):
    """Pin the baseline-before-record call order (first-run guard)."""
    db, _ = _build(tmp_path)
    hist_path = tmp_path / "row-counts.json"
    app, table = "demo", "orders"

    # The current run's count comes from the CLI rowcount op.
    out = _ok_json(["--engine", "sqlite", "--db", str(db),
                    "--op", "rowcount", "--table", table])
    count = out["count"]

    # Run 1 — FIRST run: baseline read BEFORE record -> None -> no anomaly.
    base1 = hc_history.baseline(app, table, hist_path)
    assert base1 is None
    assert hc_history.is_anomaly(count, base1) is False
    hc_history.record(app, table, count, "2026-06-15T08:00:00", hist_path)

    # Run 2 — baseline now returns the prior count (not None).
    base2 = hc_history.baseline(app, table, hist_path)
    assert base2 is not None
    assert base2 == float(count)
    assert hc_history.is_anomaly(count, base2) is False
    hc_history.record(app, table, count, "2026-06-15T09:00:00", hist_path)

    # An anomalous current count (10x) is flagged against the established base.
    big = count * 10
    assert hc_history.is_anomaly(big, base2) is True


# --- provision integration: write-to-file only --------------------------------

def test_e2e_provision_writes_file_only(tmp_path):
    """Provision generates a reviewable SQL file and never touches the DB."""
    db, _ = _build(tmp_path)

    # Capture DB state before provisioning.
    db_mtime_before = db.stat().st_mtime_ns
    db_bytes_before = db.read_bytes()

    role_sql = hc_provision.gen_readonly_role()
    table_sql = hc_provision.gen_audit_log_table()
    out_path = tmp_path / "out.sql"
    written = hc_provision.write_provision_sql(
        [role_sql, table_sql], out_path, app="demo"
    )

    assert written.exists()
    content = written.read_text(encoding="utf-8")
    assert "REVIEW BEFORE RUNNING" in content
    # Role + table SQL are present.
    assert "CREATE ROLE" in content
    assert 'GRANT SELECT ON ALL TABLES' in content
    assert "CREATE TABLE IF NOT EXISTS" in content
    assert "audit_log" in content
    # Read-only role SQL contains NO write grants.
    lowered = content.lower()
    for forbidden in ("grant insert", "grant update", "grant delete",
                      "all privileges"):
        assert forbidden not in lowered, f"write grant leaked: {forbidden}"

    # No DB connection happened: the DB file is byte-for-byte unchanged.
    assert db.stat().st_mtime_ns == db_mtime_before
    assert db.read_bytes() == db_bytes_before


# --- read-only safety end-to-end ----------------------------------------------

def test_e2e_raw_write_refused_db_unchanged(tmp_path):
    """A write via --raw is refused even with HC_ALLOW_RAW=1; DB unchanged."""
    db, exp = _build(tmp_path)

    def _orders_count():
        out = _ok_json(["--engine", "sqlite", "--db", str(db),
                        "--op", "rowcount", "--table", "orders"])
        return out["count"]

    before = _orders_count()
    assert before == exp["orders_rowcount"]

    # Attempt a write via --raw with the gate opened — is_safe_sql must refuse.
    r = _run(["--engine", "sqlite", "--db", str(db),
              "--raw", "DELETE FROM orders"],
             env={**os.environ, "HC_ALLOW_RAW": "1"})
    assert r.returncode != 0
    assert "read-only" in (r.stdout + r.stderr).lower()

    # Prove nothing was written: row count is unchanged.
    after = _orders_count()
    assert after == before == exp["orders_rowcount"]
