# tests/fixtures/make_demo.py
"""Synthetic SQLite app builder for the /health-check end-to-end smoke test.

``build_demo_db(path)`` creates a throwaway SQLite database that simulates a
small application, seeded with a KNOWN set of integrity problems and an
``audit_log`` so every health-check op has something deterministic to find. It
returns a dict of the expected answers so the test can assert against them by
value (not just shape).

Determinism strategy for time-relative ops:
  * ``stale`` and ``audit_recency`` compare ``column < datetime('now', ...)`` at
    query time, so the rows that must straddle a threshold are inserted relative
    to ``now`` via SQLite's ``datetime('now', '-N units')`` at INSERT time. This
    keeps the stale/recency split deterministic w.r.t. the threshold regardless
    of wall-clock time when the test runs (canonical ``YYYY-MM-DD HH:MM:SS``
    form, as required by the SQLite stale/recency ops).
  * ``audit_velocity`` buckets by a fixed-width epoch window; the burst rows use
    a fixed wall-clock minute so they all land in one 60s bucket deterministically.
    The burst is additionally seeded recent (now-relative) so audit_recency's
    24h/7d counts are also deterministic.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def build_demo_db(path: str | Path) -> dict[str, Any]:
    """Create a synthetic SQLite app DB with known integrity problems.

    Args:
        path: Filesystem path for the SQLite database file to create.

    Returns:
        A dict of expected answers keyed for the end-to-end smoke test:
            orders_rowcount, fk_orphans, null_count, stale_count,
            duplicate_groups, alice_actions, alice_total, mallory_burst_count,
            velocity_window_seconds, velocity_threshold, events_24h, events_7d.
    """
    db = Path(path)
    con = sqlite3.connect(str(db))
    try:
        cur = con.cursor()

        # --- customers -------------------------------------------------------
        cur.execute(
            "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)"
        )
        cur.executemany(
            "INSERT INTO customers (id, name) VALUES (?, ?)",
            [(1, "Acme"), (2, "Beta"), (3, "Gamma")],
        )

        # --- orders ----------------------------------------------------------
        cur.execute(
            "CREATE TABLE orders ("
            "id INTEGER PRIMARY KEY, customer_id INTEGER, status TEXT, "
            "total REAL, updated_at TEXT)"
        )

        # Rows with a valid customer_id (not orphans).
        #  - 2 'processing' rows 5 minutes ago  -> NOT stale (within 1h)
        #  - 2 'processing' rows 3 hours ago    -> STALE (older than 1h)
        cur.execute(
            "INSERT INTO orders (id, customer_id, status, total, updated_at) "
            "VALUES (1, 1, 'processing', 10.0, datetime('now', '-5 minutes'))"
        )
        cur.execute(
            "INSERT INTO orders (id, customer_id, status, total, updated_at) "
            "VALUES (2, 2, 'processing', 20.0, datetime('now', '-5 minutes'))"
        )
        cur.execute(
            "INSERT INTO orders (id, customer_id, status, total, updated_at) "
            "VALUES (3, 1, 'processing', 30.0, datetime('now', '-3 hours'))"
        )
        cur.execute(
            "INSERT INTO orders (id, customer_id, status, total, updated_at) "
            "VALUES (4, 2, 'processing', 40.0, datetime('now', '-3 hours'))"
        )
        # A 'shipped' row (valid customer) — irrelevant to the stale filter.
        cur.execute(
            "INSERT INTO orders (id, customer_id, status, total, updated_at) "
            "VALUES (5, 3, 'shipped', 50.0, datetime('now', '-3 hours'))"
        )
        # 2 rows with status NULL (valid customers) -> null_drift count = 2.
        cur.execute(
            "INSERT INTO orders (id, customer_id, status, total, updated_at) "
            "VALUES (6, 1, NULL, 60.0, datetime('now', '-5 minutes'))"
        )
        cur.execute(
            "INSERT INTO orders (id, customer_id, status, total, updated_at) "
            "VALUES (7, 2, NULL, 70.0, datetime('now', '-5 minutes'))"
        )
        # 2 ORPHAN rows: customer_id has no matching customers.id -> orphans = 2.
        cur.execute(
            "INSERT INTO orders (id, customer_id, status, total, updated_at) "
            "VALUES (8, 999, 'shipped', 80.0, datetime('now', '-5 minutes'))"
        )
        cur.execute(
            "INSERT INTO orders (id, customer_id, status, total, updated_at) "
            "VALUES (9, 998, 'shipped', 90.0, datetime('now', '-5 minutes'))"
        )

        orders_rowcount = 9
        fk_orphans = 2
        null_count = 2
        stale_count = 2  # only the 2 'processing' rows older than 1 hour

        # --- users (one duplicated email pair) -------------------------------
        cur.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT)"
        )
        cur.executemany(
            "INSERT INTO users (id, email) VALUES (?, ?)",
            [
                (1, "a@example.com"),
                (2, "b@example.com"),
                (3, "a@example.com"),  # duplicate of id 1 -> 1 duplicate group
                (4, "c@example.com"),
            ],
        )
        duplicate_groups = 1

        # --- audit_log -------------------------------------------------------
        cur.execute(
            "CREATE TABLE audit_log ("
            "id INTEGER PRIMARY KEY, user_id TEXT, action TEXT, ts TEXT)"
        )

        # alice: 3 create + 2 update, all recent (within 24h).
        for _ in range(3):
            cur.execute(
                "INSERT INTO audit_log (user_id, action, ts) "
                "VALUES ('alice', 'create', datetime('now', '-2 hours'))"
            )
        for _ in range(2):
            cur.execute(
                "INSERT INTO audit_log (user_id, action, ts) "
                "VALUES ('alice', 'update', datetime('now', '-2 hours'))"
            )

        # bob: 1 login, recent.
        cur.execute(
            "INSERT INTO audit_log (user_id, action, ts) "
            "VALUES ('bob', 'login', datetime('now', '-2 hours'))"
        )

        # mallory: 25 events inside a single fixed 60s window (a burst). They
        # are also recent (now-relative date) so they count toward 24h/7d.
        # Use a fixed second-of-minute spread 0..24 so they share one bucket for
        # window-seconds=60.
        for i in range(25):
            cur.execute(
                "INSERT INTO audit_log (user_id, action, ts) "
                "VALUES ('mallory', 'delete', "
                "strftime('%Y-%m-%d %H:%M:', datetime('now', '-1 hours')) "
                "|| substr('0' || ?, -2))",
                (i,),
            )
        mallory_burst_count = 25

        # An older event (>7 days) so 24h and 7d recency differ.
        cur.execute(
            "INSERT INTO audit_log (user_id, action, ts) "
            "VALUES ('carol', 'login', datetime('now', '-10 days'))"
        )

        # Recency expectations:
        #   within 24h: 5 alice + 1 bob + 25 mallory = 31
        #   within 7d : same 31 (the carol event at -10 days falls outside both)
        events_24h = 31
        events_7d = 31

        con.commit()
    finally:
        con.close()

    return {
        "orders_rowcount": orders_rowcount,
        "fk_orphans": fk_orphans,
        "null_count": null_count,
        "stale_count": stale_count,
        "duplicate_groups": duplicate_groups,
        "alice_actions": {"create": 3, "update": 2},
        "alice_total": 5,
        "mallory_burst_count": mallory_burst_count,
        "velocity_window_seconds": 60,
        "velocity_threshold": 20,
        "events_24h": events_24h,
        "events_7d": events_7d,
    }
