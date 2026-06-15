# tests/test_hc_db_safety.py
import json, sqlite3, subprocess, sys, os
import argparse
import importlib.util
import pytest

HC = os.path.join(os.path.dirname(__file__), "..", "hc_db.py")

# Import is_safe_sql directly from the module (no DB, pure guard checks).
_HC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _HC_DIR not in sys.path:
    sys.path.insert(0, _HC_DIR)
_spec = importlib.util.spec_from_file_location("hc_db", os.path.join(_HC_DIR, "hc_db.py"))
_hc_db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hc_db)
is_safe_sql = _hc_db.is_safe_sql


def _run(args, **kw):
    return subprocess.run([sys.executable, HC, *args],
                          capture_output=True, text=True, **kw)


def _make_db(tmp_path):
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE a(id INTEGER PRIMARY KEY)")
    con.executemany("INSERT INTO a(id) VALUES (?)", [(1,), (2,), (3,)])
    con.commit()
    con.close()
    return db


# --- contract tests (given) ---

def test_sqlite_select_only_returns_json(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--op", "rowcount", "--table", "a"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["table"] == "a" and out["count"] == 3


def test_sqlite_refuses_write_sql(tmp_path):
    db = tmp_path / "t.db"
    sqlite3.connect(db).close()
    r = _run(["--engine", "sqlite", "--db", str(db), "--raw", "DELETE FROM sqlite_master"],
             env={**os.environ, "HC_ALLOW_RAW": "1"})
    assert r.returncode != 0
    assert "read-only" in (r.stdout + r.stderr).lower()


# --- additional safety tests ---

def test_sqlite_raw_select_with_whitespace_and_case_allowed(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--raw", "  SeLeCt 1 "],
             env={**os.environ, "HC_ALLOW_RAW": "1"})
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert "rows" in out or "result" in out


def test_sqlite_raw_cte_allowed(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--raw",
              "WITH x AS (SELECT 1) SELECT * FROM x"],
             env={**os.environ, "HC_ALLOW_RAW": "1"})
    assert r.returncode == 0, r.stderr
    json.loads(r.stdout)


def test_sqlite_raw_explain_allowed(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--raw", "EXPLAIN SELECT 1"],
             env={**os.environ, "HC_ALLOW_RAW": "1"})
    assert r.returncode == 0, r.stderr
    json.loads(r.stdout)


@pytest.mark.parametrize("sql", [
    "INSERT INTO a VALUES (9)",
    "UPDATE a SET id=9",
    "DROP TABLE a",
    "PRAGMA writable_schema=1",
    "SELECT 1; DROP TABLE a",
])
def test_sqlite_raw_write_variants_refused(tmp_path, sql):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--raw", sql],
             env={**os.environ, "HC_ALLOW_RAW": "1"})
    assert r.returncode != 0, f"expected refusal for: {sql}"
    assert "read-only" in (r.stdout + r.stderr).lower()


def test_sqlite_raw_gated_without_env_var(tmp_path):
    # FIX 4 — --raw must be refused as gated when HC_ALLOW_RAW is not set, and
    # the guard (is_safe_sql) is never even reached for an otherwise-valid SELECT.
    db = _make_db(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "HC_ALLOW_RAW"}
    r = _run(["--engine", "sqlite", "--db", str(db), "--raw", "SELECT 1"], env=env)
    assert r.returncode != 0
    assert "gated" in (r.stdout + r.stderr).lower()


def test_sqlite_rowcount_rejects_bad_identifier(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--op", "rowcount",
              "--table", "a; DROP TABLE x"])
    assert r.returncode != 0
    # error JSON emitted to stdout
    combined = (r.stdout + r.stderr).lower()
    assert "identifier" in combined or "invalid" in combined


# --- guard-hardening unit tests (no DB; pure is_safe_sql) ---

@pytest.mark.parametrize("sql", [
    "WITH t AS (DELETE FROM x RETURNING *) SELECT * FROM t",
    "WITH t AS (SELECT 1) DELETE FROM x",
    "EXPLAIN ANALYZE DELETE FROM x",
    "EXPLAIN ANALYZE SELECT 1",
])
def test_is_safe_sql_rejects_pg_unsafe(sql):
    assert is_safe_sql(sql) is False, f"should reject: {sql}"


@pytest.mark.parametrize("sql", [
    "WITH t AS (SELECT 1) SELECT * FROM t",
    "SELECT 1",
    "EXPLAIN SELECT 1",
    "select count(*) from a",
])
def test_is_safe_sql_allows_reads(sql):
    assert is_safe_sql(sql) is True, f"should allow: {sql}"


# --- FIX 1 lock-in: postgres handler fails closed on non-READ-ONLY txn ---
#
# No live DB. We stub psycopg.connect to hand back a fake connection that
# accepts the session/transaction SETs, then monkeypatch the module's
# _pg_read_only_flag to report False (i.e. the READ ONLY txn did NOT take).
# The handler MUST abort (SystemExit, non-zero) BEFORE running any user query,
# and the emitted error must mention "read-only". This proves the fail-closed
# precondition branch rather than merely reporting the flag.

class _FakeCursor:
    description = None

    def fetchone(self):
        return (0,)

    def fetchall(self):
        return []


class _FakeConn:
    """Minimal stand-in for a psycopg3 connection used by _handle_postgres."""

    def __init__(self):
        self.autocommit = True
        self.rolled_back = False
        self.closed = False
        self.executed: list[str] = []

    def execute(self, sql, params=None):
        # If the handler ever reaches the user query after the read-only flag
        # is False, this records it — the test asserts that never happens.
        self.executed.append(sql)
        return _FakeCursor()

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def test_postgres_handler_fails_closed_when_not_read_only(monkeypatch, capsys):
    import types

    fake_conn = _FakeConn()
    fake_psycopg = types.SimpleNamespace(
        connect=lambda dsn: fake_conn,
        Error=Exception,
    )
    # _handle_postgres does `import psycopg` lazily inside the function, so
    # injecting it into sys.modules makes the lazy import resolve to our stub.
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    # Force the read-only precondition to fail.
    monkeypatch.setattr(_hc_db, "_pg_read_only_flag", lambda conn: False)

    args = argparse.Namespace(
        engine="postgres", db="postgresql://x@localhost/db",
        op="rowcount", table="a", raw=None, exact=False,
        child=None, column=None, parent=None, parent_column=None,
        value=None, age_column=None, older_than_hours=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        _hc_db._handle_postgres(args)

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    combined = (captured.out + captured.err).lower()
    assert "read-only" in combined or "read only" in combined
    # The user query must never have run after the precondition failed.
    assert not any('from "a"' in s.lower() or "n_live_tup" in s.lower()
                   for s in fake_conn.executed), fake_conn.executed
    assert fake_conn.rolled_back is True
