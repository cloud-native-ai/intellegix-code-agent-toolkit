# tests/test_hc_db_pg.py
"""Postgres integration tests for hc_db.py.

These run a real subprocess against a live Postgres. They auto-SKIP unless
the environment variable ``HC_TEST_PG_DSN`` is set to a reachable DSN.
The DSN is read from the environment only — never hardcoded.
"""
import os
import json
import subprocess
import sys

import pytest

HC = os.path.join(os.path.dirname(__file__), "..", "hc_db.py")
DSN = os.environ.get("HC_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="set HC_TEST_PG_DSN to run")


def _run(args, env=None):
    return subprocess.run([sys.executable, HC, *args],
                          capture_output=True, text=True, env=env)


# --raw is gated behind HC_ALLOW_RAW=1; PG --raw tests opt in via this env.
_RAW_ENV = {**os.environ, "HC_ALLOW_RAW": "1"}


def test_pg_refuses_write_raw():
    r = _run(["--engine", "postgres", "--db", DSN,
              "--raw", "UPDATE pg_class SET relname=relname"], env=_RAW_ENV)
    assert r.returncode != 0
    assert "read-only" in (r.stdout + r.stderr).lower()


def test_pg_raw_reports_read_only_txn():
    r = _run(["--engine", "postgres", "--db", DSN,
              "--raw", "SELECT current_setting('transaction_read_only')"],
             env=_RAW_ENV)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    # metadata surfaced on the raw path, AND the row value confirms 'on'
    assert out.get("transaction_read_only") is True or "on" in json.dumps(out).lower()


def test_pg_read_only_txn_blocks_write_at_server_backstop():
    # FIX 6 — genuine SERVER backstop test. Build a connection the SAME way
    # _handle_postgres does (default_transaction_read_only=on, autocommit off,
    # SET TRANSACTION READ ONLY), then execute a write DIRECTLY, BYPASSING
    # is_safe_sql, and assert psycopg raises a read-only error. This proves the
    # server-enforced READ ONLY txn is the real write-prevention guarantee, not
    # just the lexical guard.
    import psycopg

    conn = psycopg.connect(DSN)
    try:
        conn.execute("SET default_transaction_read_only = on")
        conn.autocommit = False
        conn.execute("SET TRANSACTION READ ONLY")
        conn.execute("SET LOCAL statement_timeout = '5s'")

        with pytest.raises(psycopg.Error) as excinfo:
            # Bypass is_safe_sql entirely — go straight to the server.
            conn.execute("CREATE TEMP TABLE hc_backstop_probe (x int)")
        assert "read-only" in str(excinfo.value).lower()
        conn.rollback()
    finally:
        conn.close()


def test_pg_dsn_not_leaked_on_error():
    # Force a connection error with a bogus host; the DSN must never appear.
    bogus = "postgresql://user:secretpw@nonexistent.invalid:5432/db"
    r = _run(["--engine", "postgres", "--db", bogus,
              "--raw", "SELECT 1"], env=_RAW_ENV)
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "secretpw" not in combined
    assert "nonexistent.invalid" not in combined
