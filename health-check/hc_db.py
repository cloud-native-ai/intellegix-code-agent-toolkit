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
tests). On refusal or error a JSON ``{"error": ...}`` object is printed to
stdout, a short message to stderr, and the process exits non-zero.

The ``rowcount`` op + raw path are implemented for both SQLite and Postgres.
Postgres additionally runs every query inside a READ ONLY transaction (the real
server-enforced backstop) on top of the shared ``is_safe_sql`` guard. The
argument surface and dispatch are structured so future ops (fk_orphans,
duplicates, audit_*, ...) slot in cleanly.
"""

from __future__ import annotations

import argparse
import json
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
        # Resource caps applied before any READ ONLY transaction starts. With
        # autocommit on (psycopg3 default), each statement is its own txn.
        conn.execute("SET statement_timeout = '5s'")
        conn.execute("SET work_mem = '4MB'")
        # Make the session default read-only so the next transaction inherits it.
        conn.execute("SET default_transaction_read_only = on")

        # Run all real work inside an explicit READ ONLY transaction. Turning
        # autocommit off makes the connection start a transaction implicitly;
        # SET TRANSACTION READ ONLY marks it. This is the server-enforced
        # backstop against writes (including data-modifying CTEs).
        conn.autocommit = False
        conn.execute("SET TRANSACTION READ ONLY")

        read_only = _pg_read_only_flag(conn)

        if args.raw is not None:
            result = _pg_raw(conn, args.raw)
        elif args.op == "rowcount":
            if not args.table:
                _fail("--table is required for --op rowcount",
                      error_obj={"error": "--table is required for rowcount"})
            result = _pg_rowcount(conn, args.table, bool(args.exact))
        else:
            _fail(f"unknown op: {args.op!r}",
                  error_obj={"error": f"unknown op: {args.op}"})

        # Never commit — roll back the read-only txn cleanly.
        conn.rollback()

        result["transaction_read_only"] = read_only
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
