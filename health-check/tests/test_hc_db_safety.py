# tests/test_hc_db_safety.py
import json, sqlite3, subprocess, sys, os
import pytest

HC = os.path.join(os.path.dirname(__file__), "..", "hc_db.py")


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
    r = _run(["--engine", "sqlite", "--db", str(db), "--raw", "DELETE FROM sqlite_master"])
    assert r.returncode != 0
    assert "read-only" in (r.stdout + r.stderr).lower()


# --- additional safety tests ---

def test_sqlite_raw_select_with_whitespace_and_case_allowed(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--raw", "  SeLeCt 1 "])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert "rows" in out or "result" in out


def test_sqlite_raw_cte_allowed(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--raw",
              "WITH x AS (SELECT 1) SELECT * FROM x"])
    assert r.returncode == 0, r.stderr
    json.loads(r.stdout)


def test_sqlite_raw_explain_allowed(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--raw", "EXPLAIN SELECT 1"])
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
    r = _run(["--engine", "sqlite", "--db", str(db), "--raw", sql])
    assert r.returncode != 0, f"expected refusal for: {sql}"
    assert "read-only" in (r.stdout + r.stderr).lower()


def test_sqlite_rowcount_rejects_bad_identifier(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db), "--op", "rowcount",
              "--table", "a; DROP TABLE x"])
    assert r.returncode != 0
    # error JSON emitted to stdout
    combined = (r.stdout + r.stderr).lower()
    assert "identifier" in combined or "invalid" in combined


def test_postgres_not_implemented(tmp_path):
    r = _run(["--engine", "postgres", "--db", "postgresql://x", "--op", "rowcount",
              "--table", "a"])
    assert r.returncode != 0
    out = json.loads(r.stdout)
    assert "error" in out
