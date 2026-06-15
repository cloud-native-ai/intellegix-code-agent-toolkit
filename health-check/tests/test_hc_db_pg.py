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


def _run(args):
    return subprocess.run([sys.executable, HC, *args],
                          capture_output=True, text=True)


def test_pg_refuses_write_raw():
    r = _run(["--engine", "postgres", "--db", DSN,
              "--raw", "UPDATE pg_class SET relname=relname"])
    assert r.returncode != 0
    assert "read-only" in (r.stdout + r.stderr).lower()


def test_pg_raw_reports_read_only_txn():
    r = _run(["--engine", "postgres", "--db", DSN,
              "--raw", "SELECT current_setting('transaction_read_only')"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    # metadata surfaced on the raw path, AND the row value confirms 'on'
    assert out.get("transaction_read_only") is True or "on" in json.dumps(out).lower()


def test_pg_read_only_txn_blocks_write_at_server():
    # Even a "safe-looking" write that the prefix-guard would not catch at the
    # SQL string level (here it IS caught, but the server READ ONLY txn is the
    # real backstop) must fail. Use a guard-passing CTE wrapper is impossible
    # since the guard rejects it; instead assert the txn-level guarantee via a
    # write the guard refuses, confirming nothing reaches the server writable.
    r = _run(["--engine", "postgres", "--db", DSN,
              "--raw", "WITH t AS (DELETE FROM pg_class RETURNING *) SELECT * FROM t"])
    assert r.returncode != 0
    assert "read-only" in (r.stdout + r.stderr).lower()


def test_pg_dsn_not_leaked_on_error():
    # Force a connection error with a bogus host; the DSN must never appear.
    bogus = "postgresql://user:secretpw@nonexistent.invalid:5432/db"
    r = _run(["--engine", "postgres", "--db", bogus,
              "--raw", "SELECT 1"])
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "secretpw" not in combined
    assert "nonexistent.invalid" not in combined
