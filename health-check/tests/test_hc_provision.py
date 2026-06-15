# tests/test_hc_provision.py
"""Tests for hc_provision.py — the provision-SQL generator.

These tests are pure (no DB, no subprocess). They import the module by file
location (mirroring the other hc_* test modules) and assert on the SQL strings
returned by the generators plus the behavior of the file writer.

A central guarantee verified here: the module NEVER imports a DB driver and the
generated read-only role grants SELECT only (no write grants).
"""
import importlib.util
import os
import subprocess
import sys

import pytest

_HC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _HC_DIR not in sys.path:
    sys.path.insert(0, _HC_DIR)
_spec = importlib.util.spec_from_file_location(
    "hc_provision", os.path.join(_HC_DIR, "hc_provision.py")
)
_hc_provision = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hc_provision)

gen_readonly_role = _hc_provision.gen_readonly_role
gen_audit_log_table = _hc_provision.gen_audit_log_table
write_provision_sql = _hc_provision.write_provision_sql


# --------------------------------------------------------------------------- #
# gen_readonly_role
# --------------------------------------------------------------------------- #
def test_readonly_role_contains_required_statements():
    sql = gen_readonly_role()
    low = sql.lower()
    assert "create role" in low
    assert 'grant usage on schema "public"' in low
    assert 'grant select on all tables in schema "public"' in low
    assert "alter default privileges in schema" in low
    # default identifiers must appear, quoted
    assert '"healthcheck_ro"' in sql
    assert '"public"' in sql


def test_readonly_role_supabase_grants_pg_read_all_data_with_bypassrls():
    """Supabase platform: grant pg_read_all_data AND ALTER ROLE ... WITH
    BYPASSRLS so the audit sees RLS-protected rows. pg_read_all_data alone does
    NOT bypass RLS, so without BYPASSRLS a Supabase audit role would read zero
    rows. The PG<14 fallback explicit grants must also be present."""
    sql = gen_readonly_role(platform="supabase")
    low = sql.lower()
    # The preferred PG14+ grant.
    assert "grant pg_read_all_data" in low
    # BYPASSRLS is required and must be emitted (the actual RLS fix).
    assert "bypassrls" in low
    # The comment must NOT falsely claim pg_read_all_data bypasses RLS.
    assert "does not bypass rls" in low or "not bypass rls" in low
    assert "postgresql < 14" in low or "pg < 14" in low or "< 14" in low
    # Fallback explicit grants present alongside the privileged path.
    assert 'grant usage on schema "public"' in low
    assert 'grant select on all tables in schema "public"' in low
    assert 'alter default privileges in schema "public"' in low


def test_readonly_role_default_and_render_are_portable_no_bypassrls():
    """BUG FIX: the default (generic) and render paths must be portable for a
    non-superuser owner — explicit GRANT SELECT, and ABSOLUTELY no
    pg_read_all_data / BYPASSRLS (those require superuser and FAIL on Render)."""
    for sql in (gen_readonly_role(), gen_readonly_role(platform="render")):
        low = sql.lower()
        assert "grant select on all tables" in low
        assert "alter default privileges" in low
        assert "grant usage on schema" in low
        # The bug fix: these privileged statements must be ABSENT. The tokens may
        # appear ONLY inside the negating explanatory comment ("no pg_read_all_data
        # / bypassrls ..."), never on an actual SQL (non-comment) line.
        sql_lines = [
            ln for ln in low.splitlines() if not ln.lstrip().startswith("--")
        ]
        sql_body = "\n".join(sql_lines)
        assert "bypassrls" not in sql_body
        assert "pg_read_all_data" not in sql_body
        # And no executable GRANT/ALTER statement form for either privileged path.
        assert "grant pg_read_all_data" not in low
        assert "with bypassrls" not in low


def test_readonly_role_database_emits_grant_connect():
    """When a database name is given on the portable path, emit GRANT CONNECT."""
    sql = gen_readonly_role(database="asr_po_system")
    assert 'grant connect on database "asr_po_system"' in sql.lower()


def test_readonly_role_creates_role_with_placeholder_password():
    """CREATE ROLE must include a LOGIN PASSWORD placeholder the operator
    is forced to change before running the script (both platforms)."""
    for sql in (gen_readonly_role(), gen_readonly_role(platform="supabase")):
        assert "CREATE ROLE" in sql
        assert "LOGIN PASSWORD 'CHANGE_ME_BEFORE_RUNNING'" in sql


def test_readonly_role_is_idempotent_do_block():
    for sql in (gen_readonly_role(), gen_readonly_role(platform="supabase")):
        low = sql.lower()
        # DO-block pattern that checks pg_roles before creating (no CREATE ROLE
        # IF NOT EXISTS exists in Postgres).
        assert "do $$" in low
        assert "pg_roles" in low


def test_readonly_role_has_no_write_grants():
    """No write grants on EITHER platform path."""
    for sql in (gen_readonly_role(), gen_readonly_role(platform="supabase")):
        low = sql.lower()
        assert "insert" not in low
        assert "update" not in low
        assert "delete" not in low
        assert "all privileges" not in low
        assert "grant all" not in low


def test_readonly_role_honors_custom_role_and_schema():
    sql = gen_readonly_role(role="reporter", schema="analytics")
    assert '"reporter"' in sql
    assert '"analytics"' in sql
    low = sql.lower()
    assert 'grant select on all tables in schema "analytics"' in low


def test_readonly_role_rejects_injection_in_role():
    with pytest.raises(ValueError):
        gen_readonly_role(role="x; DROP TABLE y")


def test_readonly_role_rejects_injection_in_schema():
    with pytest.raises(ValueError):
        gen_readonly_role(schema='public"; DROP TABLE y; --')


def test_readonly_role_rejects_injection_in_database():
    with pytest.raises(ValueError):
        gen_readonly_role(database='db"; DROP TABLE y; --')


def test_readonly_role_rejects_empty_identifier():
    with pytest.raises(ValueError):
        gen_readonly_role(role="")


# --------------------------------------------------------------------------- #
# gen_audit_log_table
# --------------------------------------------------------------------------- #
def test_audit_log_table_contains_create_and_columns():
    sql = gen_audit_log_table()
    low = sql.lower()
    assert 'create table if not exists "audit_log"' in low
    # 5 columns
    assert "user_id" in low and "text" in low
    assert "action" in low and "not null" in low
    assert "ts" in low and "timestamptz" in low and "now()" in low
    assert "metadata" in low and "jsonb" in low
    # an id primary key
    assert "primary key" in low
    assert ("bigserial" in low) or ("identity" in low)


def test_audit_log_table_contains_indexes():
    sql = gen_audit_log_table().lower()
    assert "create index if not exists" in sql
    # indexes on (ts) and (user_id)
    assert "(ts)" in sql
    assert "(user_id)" in sql


def test_audit_log_table_honors_custom_name():
    sql = gen_audit_log_table(table="events_log")
    assert '"events_log"' in sql


def test_audit_log_table_rejects_injection():
    with pytest.raises(ValueError):
        gen_audit_log_table(table="t; DROP TABLE u")


# --------------------------------------------------------------------------- #
# write_provision_sql
# --------------------------------------------------------------------------- #
def test_write_provision_sql_writes_header_and_parts(tmp_path):
    parts = [gen_readonly_role(), gen_audit_log_table()]
    out = tmp_path / "out.sql"
    result = write_provision_sql(parts, out, app="po")

    assert result == out
    assert result.exists()
    text = result.read_text(encoding="utf-8")

    # header
    assert text.startswith("--")
    assert "REVIEW BEFORE RUNNING" in text
    assert "po" in text  # app mentioned
    # all parts present
    for part in parts:
        assert part in text


def test_write_provision_sql_creates_missing_parent(tmp_path):
    out = tmp_path / "nested" / "deep" / "out.sql"
    result = write_provision_sql(["SELECT 1;"], out)
    assert result.exists()
    assert "REVIEW BEFORE RUNNING" in result.read_text(encoding="utf-8")


def test_write_provision_sql_no_app_omits_app_phrase(tmp_path):
    out = tmp_path / "out.sql"
    result = write_provision_sql(["SELECT 1;"], out)
    text = result.read_text(encoding="utf-8")
    assert "generated provisions" in text
    # no trailing " for <app>" when app is None
    assert "for None" not in text


def test_write_provision_sql_leaves_no_temp_file(tmp_path):
    out = tmp_path / "out.sql"
    write_provision_sql(["SELECT 1;"], out)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "out.sql"]
    assert leftovers == []


def test_write_provision_sql_returns_path_type(tmp_path):
    from pathlib import Path

    out = str(tmp_path / "out.sql")  # pass a str; must still return Path
    result = write_provision_sql(["SELECT 1;"], out)
    assert isinstance(result, Path)


# --------------------------------------------------------------------------- #
# module never imports a DB driver / opens a connection
# --------------------------------------------------------------------------- #
def test_module_imports_no_db_driver_source():
    """Cheap source-text check (kept for fast feedback; runtime check below is
    the authoritative one)."""
    src = open(
        os.path.join(_HC_DIR, "hc_provision.py"), encoding="utf-8"
    ).read().lower()
    for forbidden in ("import psycopg", "import sqlite3", "import asyncpg", "import pg8000"):
        assert forbidden not in src
    # no connection-opening calls
    assert ".connect(" not in src


def test_module_imports_no_db_driver():
    """Authoritative runtime check: import hc_provision in a FRESH interpreter
    and assert no DB driver landed in sys.modules transitively.

    The source-text grep above can give false confidence (e.g. importing a
    module that itself imports sqlite3). This verifies the actual runtime
    import graph instead.
    """
    code = (
        "import sys; "
        f"sys.path.insert(0, r'{_HC_DIR}'); "
        "import hc_provision; "
        "bad = sorted(m for m in sys.modules "
        "if m == 'sqlite3' or m.startswith(('psycopg', 'asyncpg', 'pg8000'))); "
        "assert not bad, bad"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"DB driver leaked into sys.modules: {proc.stderr.strip()}"
    )
