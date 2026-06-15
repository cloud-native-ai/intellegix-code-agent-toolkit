#!/usr/bin/env python3
"""hc_db.py — read-only database safety boundary for the /health-check tool.

This module is the single code-enforced guard that guarantees the health-check
tool can only ever READ from a target database. Every SQL path flows through
``is_safe_sql`` (SELECT-only, no stacked statements) and every identifier
flows through ``is_valid_identifier`` (strict regex). SQLite connections are
opened with OS-level read-only mode (``mode=ro``).

Output contract: all results are JSON on stdout. Aggregate/scalar values only —
row contents are never selected. On refusal or error a JSON ``{"error": ...}``
object is printed to stdout, a short message to stderr, and the process exits
non-zero.

Only the ``rowcount`` op and the SQLite path are implemented here. The argument
surface and dispatch are structured so future ops (fk_orphans, duplicates,
audit_*, ...) and the Postgres path slot in cleanly.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from typing import Any, NoReturn

# SQL whose first keyword is one of these is considered a read.
_READ_ONLY_PREFIXES: tuple[str, ...] = ("select", "with", "explain")

# Strict SQL identifier: letter/underscore start, then word chars only.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_valid_identifier(name: str) -> bool:
    """Return True if ``name`` is a safe, unquoted SQL identifier.

    Used for all table/column arguments to prevent identifier injection.
    """
    return bool(_IDENTIFIER_RE.match(name))


def is_safe_sql(sql: str) -> bool:
    """Return True if ``sql`` is a single read-only statement.

    Rules:
      * must (after stripping/lowercasing) start with select / with / explain
      * must not contain a statement separator (``;``) that yields a second
        non-empty statement — blocks stacked statements / injection.

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
    return any(lowered.startswith(prefix) for prefix in _READ_ONLY_PREFIXES)


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


# --- Postgres path (later task) ------------------------------------------------

def _handle_postgres(args: argparse.Namespace) -> dict[str, Any]:
    """Placeholder. Postgres connection logic lands in a later task.

    Structured as a separate handler so adding it is clean. ``psycopg`` is
    imported lazily inside the real implementation only — never at module top.
    """
    _fail("postgres path not yet implemented",
          error_obj={"error": "postgres path not yet implemented"})


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
