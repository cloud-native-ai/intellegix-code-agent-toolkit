#!/usr/bin/env python3
"""hc_db.py — read-only database safety boundary for the /health-check tool.

This module is the single code-enforced guard that guarantees the health-check
tool can only ever READ from a target database. Every SQL path flows through
``is_safe_sql`` (SELECT-only, no stacked statements) and every identifier
flows through ``is_valid_identifier`` (strict regex). SQLite connections are
opened with OS-level read-only mode (``mode=ro``).

Output contract: all results are JSON on stdout. Aggregate/scalar values only —
row contents are never selected, EXCEPT via the ``--raw`` test/escape hatch,
which returns the rows of a guarded SELECT-only query (used for diagnostics and
tests). Because ``--raw`` can return full row contents from a live database, it
is GATED: it is refused unless the environment variable ``HC_ALLOW_RAW=1`` is
set. The gate is enforced centrally before engine dispatch and applies to both
SQLite and Postgres. On refusal or error a JSON ``{"error": ...}`` object is
printed to stdout, a short message to stderr, and the process exits non-zero.

The ``rowcount`` op + raw path are implemented for both SQLite and Postgres.
Postgres additionally runs every query inside a READ ONLY transaction (the real
server-enforced backstop) on top of the shared ``is_safe_sql`` guard. The
argument surface and dispatch are structured so future ops (fk_orphans,
duplicates, audit_*, ...) slot in cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from typing import Any, NoReturn
from urllib.parse import urlparse

# SQL whose first keyword is one of these is considered a read.
_READ_ONLY_PREFIXES: tuple[str, ...] = ("select", "with", "explain")

# Strict SQL identifier: letter/underscore start, then word chars only.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Data-modifying keywords that must never appear as a top-level clause, even
# inside a CTE body. On SQLite these are neutralized by ``mode=ro``; on Postgres
# a data-modifying CTE (``WITH t AS (DELETE ...) ...``) genuinely writes.
_DATA_MODIFYING_KEYWORDS: tuple[str, ...] = (
    "insert", "update", "delete", "merge", "drop", "alter",
    "create", "truncate", "grant", "revoke",
)

# Matches any of the data-modifying keywords as a whole word.
_DATA_MODIFYING_RE = re.compile(
    r"\b(?:" + "|".join(_DATA_MODIFYING_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def is_valid_identifier(name: str) -> bool:
    """Return True if ``name`` is a safe, unquoted SQL identifier.

    Used for all table/column arguments to prevent identifier injection. This
    guarantees injection-safety ONLY (the name cannot break out of an
    identifier position); callers must still quote the identifier themselves
    (e.g. ``"name"``) before interpolating it into SQL.
    """
    return bool(_IDENTIFIER_RE.match(name))


def is_safe_sql(sql: str) -> bool:
    """Return True if ``sql`` is a single read-only statement.

    Rules:
      * must (after stripping/lowercasing) start with select / with / explain
      * must not contain a statement separator (``;``) that yields a second
        non-empty statement — blocks stacked statements / injection.
      * ``explain analyze`` is rejected outright — on Postgres it actually RUNS
        the analyzed statement (so ``EXPLAIN ANALYZE DELETE ...`` would write).
      * a ``with``-prefixed statement whose body contains any data-modifying
        keyword (insert/update/delete/merge/drop/alter/create/truncate/grant/
        revoke) is rejected — blocks data-modifying CTEs which are harmless on
        SQLite (mode=ro) but write on Postgres.

    A trailing ``;`` with only whitespace after it is allowed.

    NOTE — lexical guard limitation: this function does NOT parse SQL. It is a
    purely lexical check (prefix match + whole-word keyword regex). It may
    produce false REFUSALS when a data-modifying keyword appears inside a string
    literal or a quoted identifier within a ``WITH`` statement (e.g.
    ``WITH t AS (SELECT 'delete me') SELECT * FROM t``). On Postgres the
    server-side ``READ ONLY`` transaction is the AUTHORITATIVE write-prevention
    guarantee; ``is_safe_sql`` is defense-in-depth layered on top of it.
    """
    if sql is None:
        return False
    stripped = sql.strip()
    if not stripped:
        return False

    # Reject stacked statements: any ';' followed by further non-whitespace.
    parts = [p for p in stripped.split(";") if p.strip()]
    if len(parts) > 1:
        return False

    lowered = stripped.lower()

    if not any(lowered.startswith(prefix) for prefix in _READ_ONLY_PREFIXES):
        return False

    # EXPLAIN ANALYZE executes the statement on Postgres — reject outright.
    if re.match(r"^explain\s+analyze\b", lowered):
        return False

    # Data-modifying CTE: WITH ... (DELETE/UPDATE/... ) ... writes on Postgres.
    if lowered.startswith("with") and _DATA_MODIFYING_RE.search(stripped):
        return False

    return True


def _fail(message: str, *, error_obj: dict[str, Any] | None = None) -> NoReturn:
    """Print a JSON error to stdout + a short message to stderr, exit non-zero."""
    payload = error_obj if error_obj is not None else {"error": message}
    print(json.dumps(payload))
    print(message, file=sys.stderr)
    sys.exit(2)


def _emit(payload: dict[str, Any]) -> None:
    """Print a successful JSON result to stdout."""
    print(json.dumps(payload))


def _require_identifier(name: str | None, what: str) -> str:
    """Validate ``name`` as an identifier or fail closed.

    Returns the validated name so call sites read as ``t = _require_identifier(...)``.
    ``what`` is the human label used in the error (e.g. ``--child``).
    """
    if not name:
        _fail(f"{what} is required",
              error_obj={"error": f"{what} is required"})
    if not is_valid_identifier(name):
        _fail(f"invalid identifier for {what}: {name!r}",
              error_obj={"error": f"invalid identifier for {what}: {name}"})
    return name


def _require_non_negative_int(value: int | None, what: str) -> int:
    """Validate ``value`` is a present, non-negative int or fail closed."""
    if value is None:
        _fail(f"{what} is required",
              error_obj={"error": f"{what} is required"})
    if value < 0:
        _fail(f"{what} must be a non-negative integer (got {value})",
              error_obj={"error": f"{what} must be a non-negative integer"})
    return value


# --- SQLite path ---------------------------------------------------------------

def _connect_sqlite_ro(db: str) -> sqlite3.Connection:
    """Open ``db`` with OS-level read-only mode (mode=ro)."""
    uri = f"file:{db}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _sqlite_rowcount(con: sqlite3.Connection, table: str) -> dict[str, Any]:
    """Return {"table": table, "count": N} for an aggregate COUNT(*)."""
    if not is_valid_identifier(table):
        _fail(f"invalid identifier: {table!r}",
              error_obj={"error": f"invalid identifier: {table}"})
    # table is identifier-validated; quote it defensively all the same.
    cur = con.execute(f'SELECT COUNT(*) FROM "{table}"')
    count = cur.fetchone()[0]
    return {"table": table, "count": int(count)}


def _sqlite_raw(con: sqlite3.Connection, sql: str) -> dict[str, Any]:
    """Run a guarded read-only raw query, returning scalar/aggregate rows."""
    if not is_safe_sql(sql):
        _fail("refused: query is not read-only",
              error_obj={"error": "refused: query is not read-only (SELECT-only)"})
    cur = con.execute(sql)
    rows = cur.fetchall()
    return {"rows": [list(row) for row in rows]}


def _sqlite_fk_orphans(con: sqlite3.Connection, child: str, column: str,
                       parent: str, parent_column: str) -> dict[str, Any]:
    """Count child rows whose ``column`` is set but has no matching parent row.

    All four identifiers are validated + quoted; no values are interpolated.
    """
    child = _require_identifier(child, "--child")
    column = _require_identifier(column, "--column")
    parent = _require_identifier(parent, "--parent")
    parent_column = _require_identifier(parent_column, "--parent-column")
    sql = (
        f'SELECT COUNT(*) FROM "{child}" ch '
        f'WHERE ch."{column}" IS NOT NULL '
        f'AND NOT EXISTS (SELECT 1 FROM "{parent}" p '
        f'WHERE p."{parent_column}" = ch."{column}")'
    )
    count = int(con.execute(sql).fetchone()[0])
    return {"op": "fk_orphans", "child": child, "column": column,
            "parent": parent, "orphans": count}


def _sqlite_duplicates(con: sqlite3.Connection, table: str,
                       column: str) -> dict[str, Any]:
    """Count value-groups in ``column`` (non-NULL) that occur more than once."""
    table = _require_identifier(table, "--table")
    column = _require_identifier(column, "--column")
    sql = (
        f'SELECT COUNT(*) FROM (SELECT "{column}" FROM "{table}" '
        f'WHERE "{column}" IS NOT NULL GROUP BY "{column}" '
        f'HAVING COUNT(*) > 1) d'
    )
    count = int(con.execute(sql).fetchone()[0])
    return {"op": "duplicates", "table": table, "column": column,
            "duplicate_groups": count}


def _sqlite_null_drift(con: sqlite3.Connection, table: str,
                       column: str) -> dict[str, Any]:
    """Count rows where ``column`` IS NULL."""
    table = _require_identifier(table, "--table")
    column = _require_identifier(column, "--column")
    sql = f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" IS NULL'
    count = int(con.execute(sql).fetchone()[0])
    return {"op": "null_drift", "table": table, "column": column, "nulls": count}


def _sqlite_stale(con: sqlite3.Connection, table: str, column: str, value: str,
                  age_column: str, older_than_hours: int) -> dict[str, Any]:
    """Count rows matching ``column = value`` whose ``age_column`` is older than H hours.

    ``age_column`` is assumed to be a timestamp/text column comparable against
    SQLite's ``datetime('now', ...)`` (ISO-8601 / SQLite datetime format). The
    matched value is bound as a parameter; the relative interval string is built
    from the validated integer H (no value interpolation of untrusted data).
    """
    table = _require_identifier(table, "--table")
    column = _require_identifier(column, "--column")
    age_column = _require_identifier(age_column, "--age-column")
    hours = _require_non_negative_int(older_than_hours, "--older-than-hours")
    sql = (
        f'SELECT COUNT(*) FROM "{table}" '
        f'WHERE "{column}" = ? AND "{age_column}" < datetime(\'now\', ?)'
    )
    interval = f"-{hours} hours"
    count = int(con.execute(sql, (value, interval)).fetchone()[0])
    return {"op": "stale", "table": table, "column": column, "value": value,
            "stale": count}


def _handle_sqlite(args: argparse.Namespace) -> dict[str, Any]:
    con = _connect_sqlite_ro(args.db)
    try:
        if args.raw is not None:
            return _sqlite_raw(con, args.raw)
        if args.op == "rowcount":
            if not args.table:
                _fail("--table is required for --op rowcount",
                      error_obj={"error": "--table is required for rowcount"})
            return _sqlite_rowcount(con, args.table)
        if args.op == "fk_orphans":
            return _sqlite_fk_orphans(con, args.child, args.column,
                                      args.parent, args.parent_column)
        if args.op == "duplicates":
            return _sqlite_duplicates(con, args.table, args.column)
        if args.op == "null_drift":
            return _sqlite_null_drift(con, args.table, args.column)
        if args.op == "stale":
            if args.value is None:
                _fail("--value is required for --op stale",
                      error_obj={"error": "--value is required for stale"})
            return _sqlite_stale(con, args.table, args.column, args.value,
                                 args.age_column, args.older_than_hours)
        _fail(f"unknown op: {args.op!r}",
              error_obj={"error": f"unknown op: {args.op}"})
    finally:
        con.close()


# --- Postgres path -------------------------------------------------------------

def _scrub_dsn(text: str, dsn: str) -> str:
    """Remove any occurrence of the DSN (and its host) from ``text``.

    The DSN may carry credentials — it must never appear in error output.
    """
    scrubbed = text.replace(dsn, "<dsn>") if dsn else text
    try:
        host = urlparse(dsn).hostname
    except Exception:  # noqa: BLE001 — never let scrubbing raise
        host = None
    if host:
        scrubbed = scrubbed.replace(host, "<host>")
    return scrubbed


def _detect_pooler(dsn: str) -> dict[str, Any]:
    """Detect Supabase pooler / transaction-mode connections from the DSN.

    Returns a dict with ``pooler`` (bool) and, when pooled, a ``note`` warning
    that DDL requires the direct connection URL.
    """
    parsed = urlparse(dsn)
    host = (parsed.hostname or "").lower()
    port = parsed.port
    is_pooler = "pooler.supabase.com" in host or port == 6543
    meta: dict[str, Any] = {"pooler": bool(is_pooler)}
    if is_pooler:
        meta["note"] = ("pooler/transaction mode — DDL (provision) requires "
                        "the direct connection URL")
    return meta


def _pg_read_only_flag(conn: Any) -> bool:
    """Return True if the current transaction is READ ONLY."""
    row = conn.execute("SELECT current_setting('transaction_read_only')").fetchone()
    return bool(row and str(row[0]).lower() == "on")


def _pg_writable_role(conn: Any) -> bool:
    """Return True if the current role COULD write to the database/public schema.

    Advisory probe only: runs as a guarded SELECT (binds nothing) and reports
    whether the role holds CREATE on the database or the ``public`` schema. This
    is informational — the READ ONLY transaction is the hard write-prevention
    guarantee regardless of the role's privileges.
    """
    row = conn.execute(
        "SELECT has_database_privilege(current_user, current_database(), "
        "'CREATE') OR has_schema_privilege(current_user, 'public', 'CREATE')"
    ).fetchone()
    return bool(row and row[0])


def _pg_rowcount(conn: Any, table: str, exact: bool) -> dict[str, Any]:
    """Row count for a Postgres table.

    Default: fast approximate count from ``pg_stat_user_tables`` (bound param).
    With ``exact``: ``SELECT COUNT(*)`` on the identifier-validated, quoted table.
    """
    if not is_valid_identifier(table):
        _fail(f"invalid identifier: {table!r}",
              error_obj={"error": f"invalid identifier: {table}"})

    if exact:
        # table is identifier-validated; quote it defensively all the same.
        row = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
        count = int(row[0])
        approximate = False
    else:
        row = conn.execute(
            "SELECT n_live_tup FROM pg_stat_user_tables WHERE relname = %s",
            (table,),
        ).fetchone()
        if row is None or row[0] is None:
            # No stats yet (fresh table / never analyzed) — fall back to exact.
            row = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
            count = int(row[0])
            approximate = False
        else:
            count = int(row[0])
            approximate = True

    return {"table": table, "count": count, "approximate": approximate}


def _pg_raw(conn: Any, sql: str) -> dict[str, Any]:
    """Run a guarded read-only raw query inside the READ ONLY transaction."""
    if not is_safe_sql(sql):
        _fail("refused: query is not read-only",
              error_obj={"error": "refused: query is not read-only (SELECT-only)"})
    cur = conn.execute(sql)
    rows = cur.fetchall() if cur.description is not None else []
    return {"rows": [list(row) for row in rows]}


def _pg_fk_orphans(conn: Any, child: str, column: str, parent: str,
                   parent_column: str) -> dict[str, Any]:
    """Count child rows whose ``column`` is set but has no matching parent row."""
    child = _require_identifier(child, "--child")
    column = _require_identifier(column, "--column")
    parent = _require_identifier(parent, "--parent")
    parent_column = _require_identifier(parent_column, "--parent-column")
    sql = (
        f'SELECT COUNT(*) FROM "{child}" ch '
        f'WHERE ch."{column}" IS NOT NULL '
        f'AND NOT EXISTS (SELECT 1 FROM "{parent}" p '
        f'WHERE p."{parent_column}" = ch."{column}")'
    )
    count = int(conn.execute(sql).fetchone()[0])
    return {"op": "fk_orphans", "child": child, "column": column,
            "parent": parent, "orphans": count}


def _pg_duplicates(conn: Any, table: str, column: str) -> dict[str, Any]:
    """Count value-groups in ``column`` (non-NULL) that occur more than once."""
    table = _require_identifier(table, "--table")
    column = _require_identifier(column, "--column")
    sql = (
        f'SELECT COUNT(*) FROM (SELECT "{column}" FROM "{table}" '
        f'WHERE "{column}" IS NOT NULL GROUP BY "{column}" '
        f'HAVING COUNT(*) > 1) d'
    )
    count = int(conn.execute(sql).fetchone()[0])
    return {"op": "duplicates", "table": table, "column": column,
            "duplicate_groups": count}


def _pg_null_drift(conn: Any, table: str, column: str) -> dict[str, Any]:
    """Count rows where ``column`` IS NULL."""
    table = _require_identifier(table, "--table")
    column = _require_identifier(column, "--column")
    sql = f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" IS NULL'
    count = int(conn.execute(sql).fetchone()[0])
    return {"op": "null_drift", "table": table, "column": column, "nulls": count}


def _pg_stale(conn: Any, table: str, column: str, value: str, age_column: str,
              older_than_hours: int) -> dict[str, Any]:
    """Count rows matching ``column = value`` older than H hours.

    ``age_column`` is assumed to be a timestamp/timestamptz column. The matched
    value and the integer hour count are both bound as parameters; only the
    validated identifiers are interpolated into the SQL text.
    """
    table = _require_identifier(table, "--table")
    column = _require_identifier(column, "--column")
    age_column = _require_identifier(age_column, "--age-column")
    hours = _require_non_negative_int(older_than_hours, "--older-than-hours")
    sql = (
        f'SELECT COUNT(*) FROM "{table}" '
        f'WHERE "{column}" = %s '
        f'AND "{age_column}" < now() - make_interval(hours => %s)'
    )
    count = int(conn.execute(sql, (value, hours)).fetchone()[0])
    return {"op": "stale", "table": table, "column": column, "value": value,
            "stale": count}


def _handle_postgres(args: argparse.Namespace) -> dict[str, Any]:
    """Postgres read-only handler.

    Hard guarantee: every query runs inside a ``READ ONLY`` transaction, which
    the Postgres server enforces — INSERT/UPDATE/DELETE/DDL (including inside
    CTEs) raise an error. ``is_safe_sql`` is defense-in-depth on top of that.

    ``psycopg`` (psycopg3) is imported lazily here, never at module top.
    """
    import psycopg  # lazy import — psycopg3

    dsn = args.db
    pooler_meta = _detect_pooler(dsn)

    try:
        conn = psycopg.connect(dsn)
    except Exception as exc:  # noqa: BLE001 — scrub DSN before surfacing
        msg = _scrub_dsn(f"connection failed: {exc}", dsn)
        _fail(msg, error_obj={"error": msg})

    try:
        # Belt-and-suspenders: make the session default read-only so the next
        # transaction inherits it. With autocommit on (psycopg3 default) this is
        # a session-level SET. The txn-local SET TRANSACTION READ ONLY below is
        # the load-bearing guarantee on transaction-mode poolers.
        conn.execute("SET default_transaction_read_only = on")

        # Run all real work inside an explicit READ ONLY transaction. Turning
        # autocommit off makes the connection start a transaction implicitly;
        # SET TRANSACTION READ ONLY marks it. This is the server-enforced
        # backstop against writes (including data-modifying CTEs).
        conn.autocommit = False
        conn.execute("SET TRANSACTION READ ONLY")

        # FIX 3 — pooler-safe resource caps applied with SET LOCAL INSIDE the
        # open read-only transaction, so they are scoped to the txn that runs
        # the user's query and survive transaction-mode poolers (which reset
        # session-level SETs between checkouts).
        conn.execute("SET LOCAL statement_timeout = '5s'")
        conn.execute("SET LOCAL work_mem = '4MB'")

        read_only = _pg_read_only_flag(conn)

        # FIX 1 — fail closed: ENFORCE read-only as a precondition. If the txn
        # did not actually establish READ ONLY, abort BEFORE running the user's
        # query rather than merely reporting it in the JSON.
        if not read_only:
            conn.rollback()
            _fail(
                "refused: could not establish READ ONLY transaction (read-only "
                "precondition not met)",
                error_obj={"error": "refused: could not establish read-only "
                                    "transaction"},
            )

        # FIX 2 — write-permission probe (advisory). The READ ONLY txn remains
        # the hard guarantee; this only surfaces whether the role COULD write,
        # so Task 8's P0 gate can read "writable_role" by name. Do NOT refuse on
        # a writable role — the transaction is the guarantee.
        writable_role = _pg_writable_role(conn)

        if args.raw is not None:
            result = _pg_raw(conn, args.raw)
        elif args.op == "rowcount":
            if not args.table:
                _fail("--table is required for --op rowcount",
                      error_obj={"error": "--table is required for rowcount"})
            result = _pg_rowcount(conn, args.table, bool(args.exact))
        elif args.op == "fk_orphans":
            result = _pg_fk_orphans(conn, args.child, args.column,
                                    args.parent, args.parent_column)
        elif args.op == "duplicates":
            result = _pg_duplicates(conn, args.table, args.column)
        elif args.op == "null_drift":
            result = _pg_null_drift(conn, args.table, args.column)
        elif args.op == "stale":
            if args.value is None:
                _fail("--value is required for --op stale",
                      error_obj={"error": "--value is required for stale"})
            result = _pg_stale(conn, args.table, args.column, args.value,
                               args.age_column, args.older_than_hours)
        else:
            _fail(f"unknown op: {args.op!r}",
                  error_obj={"error": f"unknown op: {args.op}"})

        # Never commit — roll back the read-only txn cleanly.
        conn.rollback()

        result["transaction_read_only"] = read_only
        result["writable_role"] = writable_role
        result.update(pooler_meta)
        return result
    except SystemExit:
        raise
    except psycopg.Error as exc:
        msg = _scrub_dsn(f"postgres error: {exc}", dsn)
        _fail(msg, error_obj={"error": msg})
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001 — best-effort close
            pass


# --- CLI -----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI. Extensible for future ops."""
    parser = argparse.ArgumentParser(
        prog="hc_db.py",
        description="Read-only database safety boundary for /health-check.",
    )
    parser.add_argument("--engine", required=True, choices=["sqlite", "postgres"],
                        help="Database engine.")
    parser.add_argument("--db", required=True,
                        help="Connection string / path (CLI-only; never hardcoded).")
    parser.add_argument("--op", default=None,
                        help="Operation to run (e.g. rowcount). Future: "
                             "fk_orphans, duplicates, audit_*.")
    parser.add_argument("--table", default=None,
                        help="Table identifier for table-scoped ops.")
    parser.add_argument("--child", default=None,
                        help="Child table identifier (fk_orphans).")
    parser.add_argument("--column", default=None,
                        help="Column identifier for column-scoped ops "
                             "(duplicates, null_drift, fk_orphans child column, "
                             "stale match column).")
    parser.add_argument("--parent", default=None,
                        help="Parent table identifier (fk_orphans).")
    parser.add_argument("--parent-column", default=None,
                        help="Parent column identifier referenced by the child "
                             "column (fk_orphans).")
    parser.add_argument("--value", default=None,
                        help="Data value to match (stale). Bound as a parameter; "
                             "NOT validated as an identifier.")
    parser.add_argument("--age-column", default=None,
                        help="Timestamp column to age-compare (stale).")
    parser.add_argument("--older-than-hours", type=int, default=None,
                        help="Age threshold in hours for the stale op. Must be a "
                             "non-negative integer.")
    parser.add_argument("--raw", default=None,
                        help="Raw SQL escape hatch. Still SELECT-only guarded.")
    parser.add_argument("--exact", action="store_true",
                        help="For rowcount on Postgres: force exact COUNT(*) "
                             "instead of the fast approximate pg_stat estimate.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.raw is None and args.op is None:
        _fail("nothing to do: provide --op or --raw",
              error_obj={"error": "nothing to do: provide --op or --raw"})

    # FIX 4 — central gate for the --raw row-dump escape hatch. --raw can return
    # full row contents from a live database; refuse it unless explicitly opted
    # in via HC_ALLOW_RAW=1. Enforced once here, before engine dispatch, so it
    # covers both SQLite and Postgres. (is_safe_sql still runs after the gate.)
    if args.raw is not None and os.environ.get("HC_ALLOW_RAW") != "1":
        _fail(
            "refused: --raw is gated; set HC_ALLOW_RAW=1 to use the diagnostic "
            "escape hatch",
            error_obj={"error": "refused: --raw is gated; set HC_ALLOW_RAW=1 "
                                "to use the diagnostic escape hatch"},
        )

    if args.engine == "sqlite":
        result = _handle_sqlite(args)
    else:
        result = _handle_postgres(args)

    _emit(result)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except sqlite3.Error as exc:
        _fail(f"sqlite error: {exc}", error_obj={"error": f"sqlite error: {exc}"})
    except Exception as exc:  # noqa: BLE001 — top-level safety net
        _fail(f"unexpected error: {exc}", error_obj={"error": str(exc)})
