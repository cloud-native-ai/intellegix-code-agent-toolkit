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


def _make_schema_db(tmp_path):
    """Build a DB whose schema exercises pk / notnull / FK introspection."""
    db = tmp_path / "schema.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE customers(id INTEGER PRIMARY KEY)")
    con.execute(
        "CREATE TABLE orders("
        "id INTEGER PRIMARY KEY, "
        "customer_id INTEGER REFERENCES customers(id), "
        "status TEXT NOT NULL)"
    )
    con.execute("CREATE TABLE users(id INTEGER PRIMARY KEY, email TEXT)")
    con.commit()
    con.close()
    return db


def _assert_no_raw_gate(env):
    """Fail the test if HC_ALLOW_RAW leaked into the call environment."""
    assert env.get("HC_ALLOW_RAW") in (None, ""), (
        "schema must not require HC_ALLOW_RAW")


def test_schema_lists_all_tables_with_columns_and_fks(tmp_path):
    db = _make_schema_db(tmp_path)
    # Build an env that explicitly has NO HC_ALLOW_RAW set.
    env = {k: v for k, v in os.environ.items() if k != "HC_ALLOW_RAW"}
    _assert_no_raw_gate(env)
    r = _run(["--engine", "sqlite", "--db", str(db), "--op", "schema"], env=env)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["op"] == "schema"
    by_name = {t["table"]: t for t in out["tables"]}
    # All three tables discovered.
    assert set(by_name) == {"customers", "orders", "users"}

    orders = by_name["orders"]
    cols = {c["name"]: c for c in orders["columns"]}
    assert "customer_id" in cols
    # id columns show pk:true.
    assert cols["id"]["pk"] is True
    assert by_name["customers"]["columns"][0]["name"] == "id"
    assert {c["name"]: c for c in by_name["customers"]["columns"]}["id"]["pk"] \
        is True
    # NOT NULL column shows notnull:true.
    assert cols["status"]["notnull"] is True
    # FK relationship surfaced.
    assert {"column": "customer_id", "ref_table": "customers",
            "ref_column": "id"} in orders["foreign_keys"]


def test_schema_scoped_to_one_table(tmp_path):
    db = _make_schema_db(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "HC_ALLOW_RAW"}
    _assert_no_raw_gate(env)
    r = _run(["--engine", "sqlite", "--db", str(db), "--op", "schema",
              "--table", "orders"], env=env)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert [t["table"] for t in out["tables"]] == ["orders"]


def test_schema_rejects_bad_table_identifier(tmp_path):
    db = _make_schema_db(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "HC_ALLOW_RAW"}
    r = _run(["--engine", "sqlite", "--db", str(db), "--op", "schema",
              "--table", "x; DROP TABLE y"], env=env)
    assert r.returncode != 0
    combined = (r.stdout + r.stderr).lower()
    assert "identifier" in combined or "invalid" in combined


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


def test_stale_sqlite_t_separated_timestamp_is_known_limitation(tmp_path):
    # FIX B (I1) — DOCUMENTING A KNOWN LIMITATION, not asserting desired behavior.
    #
    # SQLite stale compares age_column lexically against datetime('now', ...),
    # which emits canonical 'YYYY-MM-DD HH:MM:SS' (a SPACE between date and time).
    # A timestamp stored with a 'T' separator (ISO-8601, e.g. '2026-06-15T13:00:00')
    # sorts lexically AFTER the space form when the DATE prefix is identical,
    # because 'T' (0x54) > ' ' (0x20) at the separator position. So a T-separated
    # row that is chronologically OLDER than the threshold but shares the SAME
    # calendar date as `datetime('now','-1 hours')` fails the
    # `age_column < threshold` comparison and is NOT counted as stale. This test
    # pins that current (wrong-for-T-timestamps) behavior so the limitation is
    # visible, not latent. NOTE: the bug only surfaces on same-date comparisons;
    # a T-separated row on an earlier calendar date still compares correctly
    # because the date difference dominates before the separator char is reached.
    db = tmp_path / "tsep.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE orders(id INTEGER PRIMARY KEY, status TEXT, "
                "updated_at TEXT)")
    # A timestamp a few hours in the past TODAY (same date as the threshold),
    # stored with a 'T' separator. Chronologically older than a 1-hour threshold,
    # but lexically NOT less-than it due to the 'T' separator.
    con.execute(
        "INSERT INTO orders(id, status, updated_at) VALUES "
        "(1, 'processing', strftime('%Y-%m-%dT%H:%M:%S', datetime('now', "
        "'-3 hours')))"
    )
    # Control row: same instant but canonical (space) format -> IS counted.
    con.execute(
        "INSERT INTO orders(id, status, updated_at) VALUES "
        "(2, 'processing', datetime('now', '-3 hours'))"
    )
    con.commit()
    con.close()

    r = _run(["--engine", "sqlite", "--db", str(db), "--op", "stale",
              "--table", "orders", "--column", "status", "--value", "processing",
              "--age-column", "updated_at", "--older-than-hours", "1"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    # Only the canonical-format row (id 2) is counted; the T-separated row (id 1),
    # despite being equally old, is silently missed by the lexical comparison.
    assert out["stale"] == 1


def test_stale_older_than_hours_zero_counts_all_past_rows(tmp_path):
    # FIX C — pin the behavior of --older-than-hours 0. With H=0 the comparison
    # is `age_column < datetime('now')` (now-0-hours == now), so EVERY matching
    # row whose timestamp is strictly before "now" is counted as stale. In the
    # _make_db fixture all 3 'processing' rows (ids 3,4 at -3h and id 5 at -5m)
    # are in the past, so all 3 count. Pinned so the behavior can't silently drift.
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--op", "stale",
              "--table", "orders", "--column", "status", "--value", "processing",
              "--age-column", "updated_at", "--older-than-hours", "0"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["op"] == "stale"
    assert out["stale"] == 3


def test_connections_sqlite_returns_na_note(tmp_path):
    # Connection-saturation preflight is a Postgres concept; sqlite has no
    # server connection pool to saturate, so the op returns an n/a note rather
    # than erroring.
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--op", "connections"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["op"] == "connections"
    assert out["note"] == "n/a for sqlite"
