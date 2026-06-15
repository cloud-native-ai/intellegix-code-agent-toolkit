#!/usr/bin/env python3
"""hc_provision.py — provision-SQL *generator* for the /health-check tool.

This module supports the command's opt-in ``--provision`` mode. When a
health-check discovers missing infrastructure (no least-privilege read-only
role, no ``audit_log`` table), this module GENERATES a reviewable ``.sql``
script that the user applies manually.

HARD GUARANTEE — this module imports no database driver (psycopg / sqlite3 /
asyncpg / ...) and never opens a database connection or executes SQL. It only
generates SQL *text* via pure string builders and writes it to a file for
manual review. There is no code path here that could touch a live database.

Identifiers (role / schema / table names) are validated with a self-contained
strict regex (``^[A-Za-z_][A-Za-z0-9_]*$`` — the same rule
``hc_db.is_valid_identifier`` enforces) and quoted in the emitted SQL. Even
though the output is written to a file rather than executed, identifier
injection-safety still matters: the file is meant to be run verbatim by a
privileged role, so an unvalidated name could smuggle arbitrary SQL into the
script a reviewer trusts.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

# Self-contained identifier validator. This module deliberately does NOT import
# hc_db (which imports sqlite3 at module level) so that the no-DB-driver
# guarantee above holds at runtime, not just in source text. The regex below is
# identical to the one hc_db.is_valid_identifier enforces.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_valid_identifier(name: str) -> bool:
    """Return True if ``name`` is a safe, unquoted SQL identifier.

    Matches ``^[A-Za-z_][A-Za-z0-9_]*$`` — the same strict rule
    ``hc_db.is_valid_identifier`` enforces.

    Args:
        name: The candidate identifier.

    Returns:
        True if ``name`` is a non-empty str matching the identifier regex.
    """
    return isinstance(name, str) and bool(_IDENT_RE.match(name))


def _require_identifier(name: str, kind: str) -> str:
    """Validate ``name`` as a SQL identifier or raise ``ValueError``.

    Args:
        name: The candidate identifier (role / schema / table name).
        kind: Human-readable role of the identifier, used in the error message.

    Returns:
        The validated ``name`` unchanged.

    Raises:
        ValueError: If ``name`` is not a safe, unquoted SQL identifier.
    """
    if not isinstance(name, str) or not is_valid_identifier(name):
        raise ValueError(
            f"invalid {kind} identifier {name!r}: must match "
            r"^[A-Za-z_][A-Za-z0-9_]*$ (injection-safe identifier)"
        )
    return name


def gen_readonly_role(role: str = "healthcheck_ro", schema: str = "public") -> str:
    """Generate Postgres SQL creating a least-privilege read-only role.

    The emitted script:
      * creates ``role`` idempotently via a ``DO $$ ... $$`` block that checks
        ``pg_roles`` first (Postgres has no ``CREATE ROLE IF NOT EXISTS``), with
        a placeholder password the operator MUST change before running;
      * grants the built-in ``pg_read_all_data`` role (PG14+) for SELECT on all
        tables, then ``ALTER ROLE ... WITH BYPASSRLS`` so the audit can see
        RLS-protected rows (pg_read_all_data alone does NOT bypass RLS; on
        Supabase, RLS is on by default so without this the role reads 0 rows);
      * falls back, for PostgreSQL < 14, to explicit ``USAGE`` + ``SELECT`` +
        ``ALTER DEFAULT PRIVILEGES`` grants (also non-bypassing).

    It contains NO write grants (no INSERT/UPDATE/DELETE/ALL PRIVILEGES).

    Args:
        role: Name of the read-only role to create. Validated + quoted.
        schema: Schema to grant read access on. Validated + quoted.

    Returns:
        A Postgres SQL string (safe to write to a file for manual review).

    Raises:
        ValueError: If ``role`` or ``schema`` is not a valid SQL identifier.
    """
    _require_identifier(role, "role")
    _require_identifier(schema, "schema")

    return f"""-- Least-privilege read-only role for /health-check.
-- Creates "{role}" (idempotent) and grants SELECT-only read access.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN
        CREATE ROLE "{role}" LOGIN PASSWORD 'CHANGE_ME_BEFORE_RUNNING';
    END IF;
END
$$;

-- PG14+: pg_read_all_data grants SELECT on every table/view/sequence. Preferred over per-table grants.
GRANT pg_read_all_data TO "{role}";

-- REQUIRED for the audit to see RLS-protected rows. pg_read_all_data does NOT bypass RLS;
-- on Supabase (RLS on by default) an audit role without BYPASSRLS reads ZERO rows on protected tables.
-- Keep this role's credentials private (never in client/browser code).
ALTER ROLE "{role}" WITH BYPASSRLS;

-- Fallback for PostgreSQL < 14 (no pg_read_all_data) - explicit read grants; NOTE: these do NOT bypass RLS:
GRANT USAGE ON SCHEMA "{schema}" TO "{role}";
GRANT SELECT ON ALL TABLES IN SCHEMA "{schema}" TO "{role}";
ALTER DEFAULT PRIVILEGES IN SCHEMA "{schema}" GRANT SELECT ON TABLES TO "{role}";"""


def gen_audit_log_table(table: str = "audit_log") -> str:
    """Generate Postgres SQL creating an audit-log table (idempotent).

    The table has the columns the health-check audit ops expect:
      * ``id``        — bigserial primary key
      * ``user_id``   — text
      * ``action``    — text NOT NULL
      * ``ts``        — timestamptz NOT NULL DEFAULT now()
      * ``metadata``  — jsonb

    Two supporting indexes (``IF NOT EXISTS``) on ``(ts)`` and ``(user_id)``
    back the queries the audit ops run.

    Args:
        table: Name of the audit-log table. Validated + quoted.

    Returns:
        A Postgres SQL string (safe to write to a file for manual review).

    Raises:
        ValueError: If ``table`` is not a valid SQL identifier.
    """
    _require_identifier(table, "table")

    return f"""-- Audit-log table for /health-check audit ops.
CREATE TABLE IF NOT EXISTS "{table}" (
    id        bigserial PRIMARY KEY,
    user_id   text,
    action    text NOT NULL,
    ts        timestamptz NOT NULL DEFAULT now(),
    metadata  jsonb
);

CREATE INDEX IF NOT EXISTS "{table}_ts_idx" ON "{table}" (ts);
CREATE INDEX IF NOT EXISTS "{table}_user_id_idx" ON "{table}" (user_id);"""


def _header(app: str | None) -> str:
    """Build the prepended header comment block for a provision script."""
    suffix = f" for {app}" if app else ""
    return (
        f"-- /health-check generated provisions{suffix}\n"
        "-- REVIEW BEFORE RUNNING - never auto-applied.\n"
        "-- Apply manually against the DIRECT (non-pooler) connection with a "
        "superuser/owner role.\n"
        "-- Generated read-only; this script makes the MINIMUM changes for "
        "future health-checks."
    )


def write_provision_sql(
    parts: list[str], out_path: str | Path, app: str | None = None
) -> Path:
    """Write a reviewable provision script to disk (atomically).

    Concatenates ``parts`` (blank line between each), prepended with a header
    comment block warning the reader to review before running. Creates the
    parent directory if missing and writes atomically (temp file in the same
    directory + ``os.replace``) so a partial file is never left behind.

    This function NEVER executes anything — it only writes text.

    Args:
        parts: SQL fragments to include, in order (e.g. role + audit table).
        out_path: Destination path for the ``.sql`` file.
        app: Optional app name, mentioned in the header for traceability.

    Returns:
        The ``Path`` to the written file.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    body = "\n\n".join(parts)
    content = _header(app) + "\n\n" + body + "\n"

    # Atomic write: temp file in the same directory, then os.replace.
    fd, tmp_name = tempfile.mkstemp(
        dir=str(out.parent), prefix=".hc_provision_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        os.replace(tmp_name, out)
    except Exception:
        # Clean up the temp file on any failure so no leftover remains.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    return out
