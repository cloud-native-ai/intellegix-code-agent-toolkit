# tests/test_hc_db_checks.py
"""DB-integrity ops over sqlite fixtures (no external DB).

Covers fk_orphans / duplicates / null_drift / stale plus negative tests for
identifier injection and a negative --older-than-hours value. Every assertion
goes through the CLI via subprocess, matching the style of the safety suite.
"""
import json
import os
import sqlite3
import subprocess
import sys

HC = os.path.join(os.path.dirname(__file__), "..", "hc_db.py")


def _run(args, **kw):
    return subprocess.run([sys.executable, HC, *args],
                          capture_output=True, text=True, **kw)


def _make_db(tmp_path):
    """Build schema + seed deterministic rows for every integrity op."""
    db = tmp_path / "checks.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE customers(id INTEGER PRIMARY KEY)")
    con.execute(
        "CREATE TABLE orders("
        "id INTEGER PRIMARY KEY, customer_id INTEGER, status TEXT, updated_at TEXT)"
    )
    con.execute("CREATE TABLE users(id INTEGER PRIMARY KEY, email TEXT)")

    # customers: ids 1, 2 exist (no id 99 / 100).
    con.executemany("INSERT INTO customers(id) VALUES (?)", [(1,), (2,)])

    # orders:
    #   - 2 rows reference a non-existent customer (99, 100) -> fk orphans = 2
    #   - 3 rows with valid customer_id
    #   - status NULL on exactly 2 rows -> null_drift = 2
    #   - 'processing' rows: 2 old (3h ago) + 1 fresh (5m ago) -> stale(>1h) = 2
    con.executemany(
        "INSERT INTO orders(id, customer_id, status, updated_at) "
        "VALUES (?, ?, ?, datetime('now', ?))",
        [
            (1, 99, "shipped", "-2 hours"),    # orphan
            (2, 100, "shipped", "-2 hours"),   # orphan
            (3, 1, "processing", "-3 hours"),  # stale old
            (4, 2, "processing", "-3 hours"),  # stale old
            (5, 1, "processing", "-5 minutes"),  # fresh, not stale
            (6, 2, None, "-1 hours"),          # null status
            (7, 1, None, "-1 hours"),          # null status
        ],
    )

    # users: one duplicated email pair + two unique others -> duplicate_groups = 1
    con.executemany(
        "INSERT INTO users(id, email) VALUES (?, ?)",
        [
            (1, "dup@example.com"),
            (2, "dup@example.com"),
            (3, "a@example.com"),
            (4, "b@example.com"),
        ],
    )
    con.commit()
    con.close()
    return db


def test_fk_orphans_counts_unmatched_children(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--op", "fk_orphans",
              "--child", "orders", "--column", "customer_id",
              "--parent", "customers", "--parent-column", "id"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["op"] == "fk_orphans"
    assert out["child"] == "orders"
    assert out["column"] == "customer_id"
    assert out["parent"] == "customers"
    assert out["orphans"] == 2


def test_duplicates_counts_value_groups(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--op", "duplicates",
              "--table", "users", "--column", "email"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["op"] == "duplicates"
    assert out["table"] == "users"
    assert out["column"] == "email"
    assert out["duplicate_groups"] == 1


def test_null_drift_counts_nulls(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--op", "null_drift",
              "--table", "orders", "--column", "status"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["op"] == "null_drift"
    assert out["table"] == "orders"
    assert out["column"] == "status"
    assert out["nulls"] == 2


def test_stale_counts_only_old_rows(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--op", "stale",
              "--table", "orders", "--column", "status", "--value", "processing",
              "--age-column", "updated_at", "--older-than-hours", "1"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["op"] == "stale"
    assert out["table"] == "orders"
    assert out["column"] == "status"
    assert out["value"] == "processing"
    assert out["stale"] == 2


def test_fk_orphans_rejects_bad_child_identifier(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--op", "fk_orphans",
              "--child", "orders; DROP TABLE x", "--column", "customer_id",
              "--parent", "customers", "--parent-column", "id"])
    assert r.returncode != 0
    combined = (r.stdout + r.stderr).lower()
    assert "identifier" in combined or "invalid" in combined


def test_stale_rejects_negative_hours(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--op", "stale",
              "--table", "orders", "--column", "status", "--value", "processing",
              "--age-column", "updated_at", "--older-than-hours", "-5"])
    assert r.returncode != 0
