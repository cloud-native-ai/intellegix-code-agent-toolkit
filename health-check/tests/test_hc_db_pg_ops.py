# tests/test_hc_db_pg_ops.py
"""Stubbed-connection unit tests for the four Postgres integrity ops.

NO live DB. These guard the net-new Postgres code paths (_pg_fk_orphans,
_pg_duplicates, _pg_null_drift, _pg_stale) by injecting a fake ``psycopg`` into
``sys.modules`` (the same pattern used by the fail-closed test in
``test_hc_db_safety.py``). The fake connection RECORDS every executed
``(sql, params)`` and returns a canned scalar ``(0,)`` for COUNT queries.

We drive each op through ``_handle_postgres`` with a parsed args namespace so the
full read-only transaction setup (SET default_transaction_read_only / SET
TRANSACTION READ ONLY / SET LOCAL caps) runs exactly as in production. The
read-only precondition flag is monkeypatched to True so the ops actually run
(they must NOT trip the fail-closed branch).

DSN-gated live Postgres tests, if any, live elsewhere and SKIP without a DSN;
this file never connects to anything.
"""
import argparse
import importlib.util
import os
import sys
import types

import pytest

# Import the module by path (mirrors test_hc_db_safety.py).
_HC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _HC_DIR not in sys.path:
    sys.path.insert(0, _HC_DIR)
_spec = importlib.util.spec_from_file_location(
    "hc_db", os.path.join(_HC_DIR, "hc_db.py"))
_hc_db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hc_db)


class _RecordingCursor:
    """Cursor stand-in: every execute records (sql, params); COUNT -> (0,)."""

    def __init__(self, log: list[tuple[str, object]]):
        self._log = log
        self.description = [("count",)]  # non-None so _pg_raw would fetchall

    def fetchone(self):
        return (0,)

    def fetchall(self):
        return []


class _RecordingConn:
    """Fake psycopg3 connection that records every executed (sql, params).

    ``execute`` returns a cursor (psycopg3 style) AND the connection also exposes
    ``executed`` as the ordered list of (sql, params) tuples for assertions.
    """

    def __init__(self):
        self.autocommit = True
        self.rolled_back = False
        self.closed = False
        self.executed: list[tuple[str, object]] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return _RecordingCursor(self.executed)

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def _base_args(**overrides) -> argparse.Namespace:
    """A fully-populated args namespace; override only what the op needs."""
    defaults = dict(
        engine="postgres", db="postgresql://u@localhost/db",
        op=None, table=None, child=None, column=None, parent=None,
        parent_column=None, value=None, age_column=None, older_than_hours=None,
        raw=None, exact=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _run_op(monkeypatch, args) -> tuple[_RecordingConn, dict]:
    """Drive ``_handle_postgres`` against a recording fake conn.

    Returns (fake_conn, result_dict). Read-only flag forced True so the op runs.
    """
    fake_conn = _RecordingConn()
    fake_psycopg = types.SimpleNamespace(
        connect=lambda dsn: fake_conn,
        Error=Exception,
    )
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    # Force the read-only precondition to pass so the op actually executes.
    monkeypatch.setattr(_hc_db, "_pg_read_only_flag", lambda conn: True)
    # writable_role probe also runs conn.execute -> (0,); keep it deterministic.
    monkeypatch.setattr(_hc_db, "_pg_writable_role", lambda conn: False)

    result = _hc_db._handle_postgres(args)
    return fake_conn, result


def _executed_sql(fake_conn: _RecordingConn) -> list[str]:
    return [sql for (sql, _params) in fake_conn.executed]


def _read_only_setup_ran_before(fake_conn: _RecordingConn, op_sql: str) -> bool:
    """True if a READ ONLY / read-only setup statement executed before op_sql.

    Matches either ``SET TRANSACTION READ ONLY`` or
    ``SET default_transaction_read_only`` (case-insensitive), and confirms its
    index precedes the first index of the op's COUNT query.
    """
    sqls = _executed_sql(fake_conn)
    op_idx = next(i for i, s in enumerate(sqls) if s == op_sql)
    for i, s in enumerate(sqls[:op_idx]):
        low = s.lower()
        if "transaction read only" in low or "default_transaction_read_only" in low:
            return True
    return False


def _op_count_stmt(fake_conn: _RecordingConn) -> tuple[str, object]:
    """Return the (sql, params) of the op's COUNT query.

    The op's COUNT is the LAST executed statement before the final rollback;
    setup statements are SETs and the read-only/writable probes are SELECTs that
    don't contain ``count(``. We pick the last executed statement whose SQL
    contains ``count(`` case-insensitively and references a quoted identifier.
    """
    for sql, params in reversed(fake_conn.executed):
        if "count(" in sql.lower():
            return sql, params
    raise AssertionError(f"no COUNT statement found in {fake_conn.executed!r}")


# --- schema (Task: read-only introspection, no --raw gate) --------------------

def test_pg_schema_queries_information_schema_after_ro_setup(monkeypatch):
    args = _base_args(op="schema")
    fake_conn, result = _run_op(monkeypatch, args)
    sqls = _executed_sql(fake_conn)
    info_sqls = [s for s in sqls if "information_schema" in s.lower()]
    # Tables/columns/PK/FK introspection all hit information_schema.
    assert info_sqls, f"no information_schema query in {sqls!r}"
    assert any("information_schema.tables" in s.lower() for s in info_sqls)
    assert any("information_schema.columns" in s.lower() for s in info_sqls)
    assert any("foreign key" in s.lower() for s in info_sqls)
    # Read-only txn setup ran before the first information_schema query.
    assert _read_only_setup_ran_before(fake_conn, info_sqls[0])
    # No --table filter -> no param binding on the table query.
    tbl_stmt = next((sql, params) for sql, params in fake_conn.executed
                    if "information_schema.tables" in sql.lower())
    assert tbl_stmt[1] in (None, (), [])
    assert result["op"] == "schema"
    assert result["tables"] == []  # empty recording result folds to no tables


def test_pg_schema_binds_table_filter_as_param(monkeypatch):
    args = _base_args(op="schema", table="orders")
    fake_conn, result = _run_op(monkeypatch, args)
    # The table filter is a VALUE in information_schema -> bound as a param,
    # never interpolated as an identifier.
    tbl_stmt = next((sql, params) for sql, params in fake_conn.executed
                    if "information_schema.tables" in sql.lower())
    assert "table_name = %s" in tbl_stmt[0].lower()
    assert tbl_stmt[1] == ("orders",)
    assert result["op"] == "schema"


# --- fk_orphans ---------------------------------------------------------------

def test_pg_fk_orphans_sql_shape_and_quoting(monkeypatch):
    args = _base_args(op="fk_orphans", child="orders", column="customer_id",
                      parent="customers", parent_column="id")
    fake_conn, result = _run_op(monkeypatch, args)
    sql, params = _op_count_stmt(fake_conn)

    # Correctly double-quoted identifiers.
    assert '"orders"' in sql
    assert '"customer_id"' in sql
    assert '"customers"' in sql
    assert '"id"' in sql
    # COUNT query selecting no row contents.
    assert "count(" in sql.lower()
    assert "select 1 from" in sql.lower()  # the EXISTS subquery, not row data
    # fk_orphans binds no params (pure identifier interpolation).
    assert params is None or params == () or params == []
    # Read-only setup ran before the op.
    assert _read_only_setup_ran_before(fake_conn, sql)
    # Result keys intact.
    assert result["op"] == "fk_orphans"
    assert result["orphans"] == 0


# --- duplicates ---------------------------------------------------------------

def test_pg_duplicates_sql_shape_and_quoting(monkeypatch):
    args = _base_args(op="duplicates", table="users", column="email")
    fake_conn, result = _run_op(monkeypatch, args)
    sql, params = _op_count_stmt(fake_conn)

    assert '"users"' in sql
    assert '"email"' in sql
    assert "count(" in sql.lower()
    assert "group by" in sql.lower()
    assert "having" in sql.lower()
    assert params is None or params == () or params == []
    assert _read_only_setup_ran_before(fake_conn, sql)
    assert result["op"] == "duplicates"
    assert result["duplicate_groups"] == 0


# --- null_drift ---------------------------------------------------------------

def test_pg_null_drift_sql_shape_and_quoting(monkeypatch):
    args = _base_args(op="null_drift", table="orders", column="status")
    fake_conn, result = _run_op(monkeypatch, args)
    sql, params = _op_count_stmt(fake_conn)

    assert '"orders"' in sql
    assert '"status"' in sql
    assert "count(" in sql.lower()
    assert "is null" in sql.lower()
    assert params is None or params == () or params == []
    assert _read_only_setup_ran_before(fake_conn, sql)
    assert result["op"] == "null_drift"
    assert result["nulls"] == 0


# --- stale --------------------------------------------------------------------

def test_pg_stale_sql_shape_param_order_and_quoting(monkeypatch):
    args = _base_args(op="stale", table="orders", column="status",
                      value="processing", age_column="updated_at",
                      older_than_hours=3)
    fake_conn, result = _run_op(monkeypatch, args)
    sql, params = _op_count_stmt(fake_conn)

    # Quoted identifiers for table + both columns.
    assert '"orders"' in sql
    assert '"status"' in sql
    assert '"updated_at"' in sql
    # COUNT query, no row contents.
    assert "count(" in sql.lower()
    # Typed interval arithmetic via make_interval(hours => %s).
    assert "make_interval(hours => %s)" in sql.lower()
    # PIN param order: (value, hours) in THAT order so a future edit can't
    # silently swap them.
    assert params == ("processing", 3)
    # Read-only setup ran before the op.
    assert _read_only_setup_ran_before(fake_conn, sql)
    assert result["op"] == "stale"
    assert result["value"] == "processing"
    assert result["stale"] == 0


def test_pg_ops_run_inside_read_only_txn_setup(monkeypatch):
    """Sanity: the canonical read-only transaction statements all executed."""
    args = _base_args(op="null_drift", table="orders", column="status")
    fake_conn, _ = _run_op(monkeypatch, args)
    joined = " | ".join(_executed_sql(fake_conn)).lower()
    assert "set default_transaction_read_only = on" in joined
    assert "set transaction read only" in joined
    # Never committed.
    assert fake_conn.rolled_back is True


# --- audit ops (Task 4) -------------------------------------------------------
#
# The recording cursor returns fetchall()->[] and fetchone()->(0,). The audit
# ops fold an empty result set without error, so these assert SQL SHAPE +
# quoted-identifier interpolation + read-only setup, not row math (that is
# covered by the sqlite-fixture suite in test_hc_db_audit.py).

def _first_sql_containing(fake_conn: _RecordingConn, needle: str) -> str:
    needle = needle.lower()
    for sql, _params in fake_conn.executed:
        if needle in sql.lower():
            return sql
    raise AssertionError(f"no statement containing {needle!r} in "
                         f"{fake_conn.executed!r}")


def test_pg_audit_user_summary_sql_shape_and_quoting(monkeypatch):
    args = _base_args(op="audit_user_summary", table="audit_log",
                      user_column="user_id", action_column="action")
    fake_conn, result = _run_op(monkeypatch, args)
    sql = _first_sql_containing(fake_conn, "group by")
    assert '"audit_log"' in sql
    assert '"user_id"' in sql
    assert '"action"' in sql
    assert "count(" in sql.lower()
    assert "group by" in sql.lower()
    assert _read_only_setup_ran_before(fake_conn, sql)
    assert result["op"] == "audit_user_summary"
    assert result["table"] == "audit_log"
    assert result["users"] == []  # empty recording result folds to no users


def test_pg_audit_velocity_sql_shape_param_order_and_quoting(monkeypatch):
    args = _base_args(op="audit_velocity", table="audit_log",
                      user_column="user_id", ts_column="ts",
                      window_seconds=60, threshold=20)
    fake_conn, result = _run_op(monkeypatch, args)
    sql, params = _op_count_stmt(fake_conn)
    assert '"audit_log"' in sql
    assert '"user_id"' in sql
    assert '"ts"' in sql
    # Postgres epoch bucketing.
    assert "extract(epoch from" in sql.lower()
    assert "::bigint" in sql.lower()
    assert "having" in sql.lower()
    # PIN param order: (window, threshold) so a future edit can't swap them.
    assert params == (60, 20)
    assert _read_only_setup_ran_before(fake_conn, sql)
    assert result["op"] == "audit_velocity"
    assert result["window_seconds"] == 60
    assert result["threshold"] == 20
    assert result["bursts"] == []


def test_pg_audit_fingerprint_sql_shape_and_quoting(monkeypatch):
    args = _base_args(op="audit_fingerprint", table="audit_log",
                      user_column="user_id", action_column="action")
    fake_conn, result = _run_op(monkeypatch, args)
    sql = _first_sql_containing(fake_conn, "group by")
    assert '"audit_log"' in sql
    assert '"user_id"' in sql
    assert '"action"' in sql
    assert "count(" in sql.lower()
    assert _read_only_setup_ran_before(fake_conn, sql)
    assert result["op"] == "audit_fingerprint"
    assert result["users"] == []


def test_pg_audit_recency_sql_shape_and_quoting(monkeypatch):
    args = _base_args(op="audit_recency", table="audit_log", ts_column="ts")
    fake_conn, result = _run_op(monkeypatch, args)
    max_sql = _first_sql_containing(fake_conn, "max(")
    interval_sql = _first_sql_containing(fake_conn, "make_interval")
    assert '"audit_log"' in max_sql
    assert '"ts"' in max_sql
    assert '"ts"' in interval_sql
    assert "make_interval(days => %s)" in interval_sql.lower()
    # The 24h/7d count queries bind the day count.
    day_params = [params for sql, params in fake_conn.executed
                  if "make_interval" in sql.lower()]
    assert (1,) in day_params
    assert (7,) in day_params
    assert _read_only_setup_ran_before(fake_conn, interval_sql)
    assert result["op"] == "audit_recency"
    assert result["table"] == "audit_log"


def test_pg_audit_velocity_rejects_zero_window(monkeypatch):
    args = _base_args(op="audit_velocity", table="audit_log",
                      user_column="user_id", ts_column="ts",
                      window_seconds=0, threshold=20)
    with pytest.raises(SystemExit):
        _run_op(monkeypatch, args)


def test_pg_audit_velocity_rejects_zero_threshold(monkeypatch):
    args = _base_args(op="audit_velocity", table="audit_log",
                      user_column="user_id", ts_column="ts",
                      window_seconds=60, threshold=0)
    with pytest.raises(SystemExit):
        _run_op(monkeypatch, args)
