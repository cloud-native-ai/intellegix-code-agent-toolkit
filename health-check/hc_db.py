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


def _require_positive_int(value: int | None, what: str) -> int:
    """Validate ``value`` is a present int >= 1 or fail closed.

    Used for the velocity op's --window-seconds and --threshold, where a value
    of 0 is meaningless (a zero-width window / a zero-event burst threshold).
    """
    if value is None:
        _fail(f"{what} is required",
              error_obj={"error": f"{what} is required"})
    if value < 1:
        _fail(f"{what} must be a positive integer >= 1 (got {value})",
              error_obj={"error": f"{what} must be a positive integer >= 1"})
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


def _sqlite_schema(con: sqlite3.Connection,
                   table: str | None) -> dict[str, Any]:
    """Introspect tables/columns/foreign-keys via read-only PRAGMA functions.

    Dedicated code path that does NOT route through ``is_safe_sql`` or the
    ``--raw`` gate: it issues only fixed introspection statements. Table names
    come from ``sqlite_master`` (or the validated ``--table`` arg); every name
    interpolated into a PRAGMA is re-validated with ``is_valid_identifier`` and
    double-quoted. Metadata only — no row contents are read. PRAGMA on a
    ``mode=ro`` connection cannot write.
    """
    if table is not None:
        # Already validated by _require_identifier at the call site, but the
        # name is interpolated into PRAGMA text below, so re-assert here.
        table = _require_identifier(table, "--table")
        names = [table]
    else:
        rows = con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        names = [r[0] for r in rows]

    tables: list[dict[str, Any]] = []
    for name in names:
        # Defense-in-depth: every name interpolated into a PRAGMA is validated
        # and quoted, even those sourced from sqlite_master.
        if not is_valid_identifier(name):
            _fail(f"invalid identifier: {name!r}",
                  error_obj={"error": f"invalid identifier: {name}"})
        columns: list[dict[str, Any]] = []
        for row in con.execute(f'PRAGMA table_info("{name}")').fetchall():
            # PRAGMA table_info columns: (cid, name, type, notnull, dflt, pk)
            columns.append({
                "name": row[1],
                "type": row[2],
                "notnull": bool(row[3]),
                "pk": bool(row[5]),
            })
        foreign_keys: list[dict[str, Any]] = []
        for row in con.execute(
                f'PRAGMA foreign_key_list("{name}")').fetchall():
            # PRAGMA foreign_key_list columns:
            # (id, seq, table, from, to, on_update, on_delete, match)
            foreign_keys.append({
                "column": row[3],
                "ref_table": row[2],
                "ref_column": row[4],
            })
        tables.append({"table": name, "columns": columns,
                       "foreign_keys": foreign_keys})

    return {"op": "schema", "tables": tables}


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

    ``age_column`` is compared lexically against SQLite's ``datetime('now', ...)``
    output. This REQUIRES timestamps stored in canonical ``YYYY-MM-DD HH:MM:SS``
    format (as produced by SQLite's ``datetime()``). ISO-8601 with a ``T``
    separator or a timezone offset (e.g. ``2026-06-15T09:00:00`` or
    ``2026-06-15 09:00:00+00:00``) sorts lexically out of order versus the
    canonical form and may yield WRONG counts. Postgres uses typed timestamp
    comparison and has no such requirement. The matched value is bound as a
    parameter; the relative interval string is built from the validated integer H
    (no value interpolation of untrusted data).
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


def _fold_user_actions(rows: list[tuple[Any, Any, int]]) -> list[dict[str, Any]]:
    """Fold (user, action, count) rows into per-user nested action dicts.

    Returns a list of ``{"user_id": u, "actions": {action: count, ...},
    "total": n}`` sorted by total desc (stable for ties — preserves first-seen
    user order within an equal total). Aggregate-only: only the user-identifier,
    action label, and counts are present — never row contents.
    """
    folded: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    for user, action, count in rows:
        bucket = folded.get(user)
        if bucket is None:
            bucket = {"user_id": user, "actions": {}, "total": 0}
            folded[user] = bucket
            order.append(user)
        bucket["actions"][action] = bucket["actions"].get(action, 0) + int(count)
        bucket["total"] += int(count)
    users = [folded[u] for u in order]
    users.sort(key=lambda b: b["total"], reverse=True)
    return users


def _sqlite_audit_user_summary(con: sqlite3.Connection, table: str,
                               user_column: str,
                               action_column: str) -> dict[str, Any]:
    """Per-user action-type counts (aggregate-only GROUP BY user, action)."""
    table = _require_identifier(table, "--table")
    user_column = _require_identifier(user_column, "--user-column")
    action_column = _require_identifier(action_column, "--action-column")
    sql = (
        f'SELECT "{user_column}", "{action_column}", COUNT(*) '
        f'FROM "{table}" GROUP BY "{user_column}", "{action_column}"'
    )
    rows = [(r[0], r[1], int(r[2])) for r in con.execute(sql).fetchall()]
    return {"op": "audit_user_summary", "table": table,
            "users": _fold_user_actions(rows)}


def _sqlite_audit_velocity(con: sqlite3.Connection, table: str,
                           user_column: str, ts_column: str,
                           window_seconds: int,
                           threshold: int) -> dict[str, Any]:
    """Flag per-user event bursts using fixed-width epoch buckets.

    Each event is bucketed into ``floor(epoch(ts) / W)``; buckets with count
    >= threshold are reported. Aggregate-only: groups by user + bucket and
    selects only the user-identifier, the bucket, and the count.
    """
    table = _require_identifier(table, "--table")
    user_column = _require_identifier(user_column, "--user-column")
    ts_column = _require_identifier(ts_column, "--ts-column")
    window = _require_positive_int(window_seconds, "--window-seconds")
    thresh = _require_positive_int(threshold, "--threshold")
    # SQLite epoch via strftime('%s', ...); bucket = epoch / W (integer div).
    # window is a validated positive int — bound as a parameter all the same.
    sql = (
        f'SELECT "{user_column}", '
        f'CAST(strftime(\'%s\', "{ts_column}") AS INTEGER) / ? AS bucket, '
        f'COUNT(*) AS c '
        f'FROM "{table}" '
        f'GROUP BY "{user_column}", bucket '
        f'HAVING c >= ? '
        f'ORDER BY c DESC'
    )
    bursts: list[dict[str, Any]] = []
    for user, bucket, count in con.execute(sql, (window, thresh)).fetchall():
        bursts.append({
            "user_id": user,
            "count": int(count),
            "bucket_start_epoch": int(bucket) * window,
        })
    bursts.sort(key=lambda b: b["count"], reverse=True)
    return {"op": "audit_velocity", "table": table,
            "window_seconds": window, "threshold": thresh, "bursts": bursts}


def _sqlite_audit_fingerprint(con: sqlite3.Connection, table: str,
                              user_column: str,
                              action_column: str) -> dict[str, Any]:
    """Per-user action-type distribution (thin wrapper over user_summary data)."""
    table = _require_identifier(table, "--table")
    user_column = _require_identifier(user_column, "--user-column")
    action_column = _require_identifier(action_column, "--action-column")
    sql = (
        f'SELECT "{user_column}", "{action_column}", COUNT(*) '
        f'FROM "{table}" GROUP BY "{user_column}", "{action_column}"'
    )
    rows = [(r[0], r[1], int(r[2])) for r in con.execute(sql).fetchall()]
    users = [{"user_id": b["user_id"], "actions": b["actions"]}
             for b in _fold_user_actions(rows)]
    return {"op": "audit_fingerprint", "table": table, "users": users}


def _sqlite_audit_recency(con: sqlite3.Connection, table: str,
                          ts_column: str) -> dict[str, Any]:
    """Last-write timestamp + 24h / 7d event counts.

    Like the ``stale`` op, the 24h/7d comparisons run lexically against SQLite's
    ``datetime('now', ...)`` output. This REQUIRES timestamps stored in canonical
    ``YYYY-MM-DD HH:MM:SS`` format (as produced by SQLite's ``datetime()``).
    ISO-8601 with a ``T`` separator or a timezone offset sorts lexically out of
    order versus the canonical form and may yield WRONG counts. Postgres uses
    typed timestamp comparison and has no such requirement.
    """
    table = _require_identifier(table, "--table")
    ts_column = _require_identifier(ts_column, "--ts-column")
    last_write = con.execute(
        f'SELECT MAX("{ts_column}") FROM "{table}"').fetchone()[0]
    events_24h = int(con.execute(
        f'SELECT COUNT(*) FROM "{table}" '
        f'WHERE "{ts_column}" >= datetime(\'now\', ?)', ("-1 day",)
    ).fetchone()[0])
    events_7d = int(con.execute(
        f'SELECT COUNT(*) FROM "{table}" '
        f'WHERE "{ts_column}" >= datetime(\'now\', ?)', ("-7 days",)
    ).fetchone()[0])
    return {"op": "audit_recency", "table": table,
            "last_write_ts": last_write,
            "events_24h": events_24h, "events_7d": events_7d}


def _handle_sqlite(args: argparse.Namespace) -> dict[str, Any]:
    con = _connect_sqlite_ro(args.db)
    try:
        if args.raw is not None:
            return _sqlite_raw(con, args.raw)
        if args.op == "schema":
            table = (_require_identifier(args.table, "--table")
                     if args.table else None)
            return _sqlite_schema(con, table)
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
        if args.op == "audit_user_summary":
            return _sqlite_audit_user_summary(con, args.table, args.user_column,
                                              args.action_column)
        if args.op == "audit_velocity":
            return _sqlite_audit_velocity(con, args.table, args.user_column,
                                          args.ts_column, args.window_seconds,
                                          args.threshold)
        if args.op == "audit_fingerprint":
            return _sqlite_audit_fingerprint(con, args.table, args.user_column,
                                             args.action_column)
        if args.op == "audit_recency":
            return _sqlite_audit_recency(con, args.table, args.ts_column)
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


def _pg_schema(conn: Any, table: str | None) -> dict[str, Any]:
    """Introspect tables/columns/PKs/FKs via information_schema (plain SELECTs).

    These are ordinary read-only SELECTs against ``information_schema`` so they
    run on the normal pg query path inside the open READ ONLY transaction set up
    by ``_handle_postgres``. Schema/table FILTERS are values in information_schema
    (not identifiers), so they are bound as PARAMETERS. Metadata only — no row
    contents. Requires NO ``HC_ALLOW_RAW``.
    """
    if table is not None:
        table = _require_identifier(table, "--table")

    # 1) Tables in user schemas (optionally scoped to one table by name).
    tbl_sql = (
        "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
        "AND table_type = 'BASE TABLE'"
    )
    tbl_params: list[Any] = []
    if table is not None:
        tbl_sql += " AND table_name = %s"
        tbl_params.append(table)
    tbl_sql += " ORDER BY table_schema, table_name"
    table_rows = conn.execute(tbl_sql, tuple(tbl_params)).fetchall()

    # 2) Columns (name, type, nullability) for the in-scope tables.
    col_sql = (
        "SELECT table_schema, table_name, column_name, data_type, is_nullable "
        "FROM information_schema.columns "
        "WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
    )
    col_params: list[Any] = []
    if table is not None:
        col_sql += " AND table_name = %s"
        col_params.append(table)
    col_sql += " ORDER BY table_schema, table_name, ordinal_position"
    col_rows = conn.execute(col_sql, tuple(col_params)).fetchall()

    # 3) PRIMARY KEY columns, joined through table_constraints.
    pk_sql = (
        "SELECT tc.table_schema, tc.table_name, kcu.column_name "
        "FROM information_schema.table_constraints tc "
        "JOIN information_schema.key_column_usage kcu "
        "ON tc.constraint_name = kcu.constraint_name "
        "AND tc.table_schema = kcu.table_schema "
        "WHERE tc.constraint_type = 'PRIMARY KEY' "
        "AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')"
    )
    pk_params: list[Any] = []
    if table is not None:
        pk_sql += " AND tc.table_name = %s"
        pk_params.append(table)
    pk_rows = conn.execute(pk_sql, tuple(pk_params)).fetchall()

    # 4) FOREIGN KEY column -> referenced table.column.
    fk_sql = (
        "SELECT tc.table_schema, tc.table_name, kcu.column_name, "
        "ccu.table_name AS ref_table, ccu.column_name AS ref_column "
        "FROM information_schema.table_constraints tc "
        "JOIN information_schema.key_column_usage kcu "
        "ON tc.constraint_name = kcu.constraint_name "
        "AND tc.table_schema = kcu.table_schema "
        "JOIN information_schema.constraint_column_usage ccu "
        "ON tc.constraint_name = ccu.constraint_name "
        "AND tc.table_schema = ccu.table_schema "
        "WHERE tc.constraint_type = 'FOREIGN KEY' "
        "AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')"
    )
    fk_params: list[Any] = []
    if table is not None:
        fk_sql += " AND tc.table_name = %s"
        fk_params.append(table)
    fk_rows = conn.execute(fk_sql, tuple(fk_params)).fetchall()

    # Fold into the {table, columns[], foreign_keys[]} output shape.
    pk_set = {(r[0], r[1], r[2]) for r in pk_rows}
    tables: list[dict[str, Any]] = []
    for tschema, tname in table_rows:
        columns = [
            {"name": c[2], "type": c[3],
             "notnull": str(c[4]).upper() == "NO",
             "pk": (c[0], c[1], c[2]) in pk_set}
            for c in col_rows if c[0] == tschema and c[1] == tname
        ]
        foreign_keys = [
            {"column": f[2], "ref_table": f[3], "ref_column": f[4]}
            for f in fk_rows if f[0] == tschema and f[1] == tname
        ]
        tables.append({"table": tname, "columns": columns,
                       "foreign_keys": foreign_keys})

    return {"op": "schema", "tables": tables}


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


def _pg_audit_user_summary(conn: Any, table: str, user_column: str,
                           action_column: str) -> dict[str, Any]:
    """Per-user action-type counts (aggregate-only GROUP BY user, action)."""
    table = _require_identifier(table, "--table")
    user_column = _require_identifier(user_column, "--user-column")
    action_column = _require_identifier(action_column, "--action-column")
    sql = (
        f'SELECT "{user_column}", "{action_column}", COUNT(*) '
        f'FROM "{table}" GROUP BY "{user_column}", "{action_column}"'
    )
    rows = [(r[0], r[1], int(r[2])) for r in conn.execute(sql).fetchall()]
    return {"op": "audit_user_summary", "table": table,
            "users": _fold_user_actions(rows)}


def _pg_audit_velocity(conn: Any, table: str, user_column: str, ts_column: str,
                       window_seconds: int, threshold: int) -> dict[str, Any]:
    """Flag per-user event bursts using fixed-width epoch buckets.

    Each event is bucketed into ``floor(extract(epoch from ts) / W)``; buckets
    with count >= threshold are reported. Aggregate-only.
    """
    table = _require_identifier(table, "--table")
    user_column = _require_identifier(user_column, "--user-column")
    ts_column = _require_identifier(ts_column, "--ts-column")
    window = _require_positive_int(window_seconds, "--window-seconds")
    thresh = _require_positive_int(threshold, "--threshold")
    # window/threshold are validated positive ints — bound as parameters.
    sql = (
        f'SELECT "{user_column}", '
        f'floor(extract(epoch from "{ts_column}"))::bigint / %s AS bucket, '
        f'COUNT(*) AS c '
        f'FROM "{table}" '
        f'GROUP BY "{user_column}", bucket '
        f'HAVING COUNT(*) >= %s '
        f'ORDER BY c DESC'
    )
    bursts: list[dict[str, Any]] = []
    for user, bucket, count in conn.execute(sql, (window, thresh)).fetchall():
        bursts.append({
            "user_id": user,
            "count": int(count),
            "bucket_start_epoch": int(bucket) * window,
        })
    bursts.sort(key=lambda b: b["count"], reverse=True)
    return {"op": "audit_velocity", "table": table,
            "window_seconds": window, "threshold": thresh, "bursts": bursts}


def _pg_audit_fingerprint(conn: Any, table: str, user_column: str,
                          action_column: str) -> dict[str, Any]:
    """Per-user action-type distribution (thin wrapper over user_summary data)."""
    table = _require_identifier(table, "--table")
    user_column = _require_identifier(user_column, "--user-column")
    action_column = _require_identifier(action_column, "--action-column")
    sql = (
        f'SELECT "{user_column}", "{action_column}", COUNT(*) '
        f'FROM "{table}" GROUP BY "{user_column}", "{action_column}"'
    )
    rows = [(r[0], r[1], int(r[2])) for r in conn.execute(sql).fetchall()]
    users = [{"user_id": b["user_id"], "actions": b["actions"]}
             for b in _fold_user_actions(rows)]
    return {"op": "audit_fingerprint", "table": table, "users": users}


def _pg_audit_recency(conn: Any, table: str, ts_column: str) -> dict[str, Any]:
    """Last-write timestamp + 24h / 7d event counts (typed timestamp compare)."""
    table = _require_identifier(table, "--table")
    ts_column = _require_identifier(ts_column, "--ts-column")
    last_write = conn.execute(
        f'SELECT MAX("{ts_column}") FROM "{table}"').fetchone()[0]
    events_24h = int(conn.execute(
        f'SELECT COUNT(*) FROM "{table}" '
        f'WHERE "{ts_column}" >= now() - make_interval(days => %s)', (1,)
    ).fetchone()[0])
    events_7d = int(conn.execute(
        f'SELECT COUNT(*) FROM "{table}" '
        f'WHERE "{ts_column}" >= now() - make_interval(days => %s)', (7,)
    ).fetchone()[0])
    # MAX() of a timestamp comes back as a datetime; serialize to ISO for JSON.
    if last_write is not None and hasattr(last_write, "isoformat"):
        last_write = last_write.isoformat()
    return {"op": "audit_recency", "table": table,
            "last_write_ts": last_write,
            "events_24h": events_24h, "events_7d": events_7d}


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
        elif args.op == "schema":
            table = (_require_identifier(args.table, "--table")
                     if args.table else None)
            result = _pg_schema(conn, table)
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
        elif args.op == "audit_user_summary":
            result = _pg_audit_user_summary(conn, args.table, args.user_column,
                                            args.action_column)
        elif args.op == "audit_velocity":
            result = _pg_audit_velocity(conn, args.table, args.user_column,
                                        args.ts_column, args.window_seconds,
                                        args.threshold)
        elif args.op == "audit_fingerprint":
            result = _pg_audit_fingerprint(conn, args.table, args.user_column,
                                           args.action_column)
        elif args.op == "audit_recency":
            result = _pg_audit_recency(conn, args.table, args.ts_column)
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
                        help="Operation to run (e.g. schema, rowcount, "
                             "fk_orphans, duplicates, null_drift, stale, "
                             "audit_*). 'schema' lists tables/columns/FKs "
                             "(read-only introspection; no HC_ALLOW_RAW "
                             "needed); optional --table scopes to one table.")
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
                        help="Timestamp column to age-compare (stale). SQLite "
                             "stale requires timestamps stored in canonical "
                             "'YYYY-MM-DD HH:MM:SS' format (as produced by "
                             "SQLite's datetime()); ISO-8601 with a 'T' separator "
                             "or timezone offset will compare lexically and may "
                             "yield wrong counts. Postgres uses typed timestamp "
                             "comparison.")
    parser.add_argument("--older-than-hours", type=int, default=None,
                        help="Age threshold in hours for the stale op. Must be a "
                             "non-negative integer.")
    parser.add_argument("--user-column", default=None,
                        help="User-identifier column for audit ops "
                             "(audit_user_summary, audit_velocity, "
                             "audit_fingerprint).")
    parser.add_argument("--action-column", default=None,
                        help="Action-type column for audit_user_summary / "
                             "audit_fingerprint.")
    parser.add_argument("--ts-column", default=None,
                        help="Timestamp column for audit_velocity / "
                             "audit_recency. SQLite audit_recency requires "
                             "timestamps in canonical 'YYYY-MM-DD HH:MM:SS' "
                             "format (see --age-column note); Postgres uses "
                             "typed timestamp comparison.")
    parser.add_argument("--window-seconds", type=int, default=None,
                        help="Rolling/bucketed window width in seconds for "
                             "audit_velocity. Must be a positive integer >= 1.")
    parser.add_argument("--threshold", type=int, default=None,
                        help="Burst threshold for audit_velocity: report buckets "
                             "with event count >= this. Must be >= 1.")
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
