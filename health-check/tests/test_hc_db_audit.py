# tests/test_hc_db_audit.py
"""Audit-log analysis ops over sqlite fixtures (no external DB).

Covers audit_user_summary / audit_velocity / audit_fingerprint / audit_recency
plus negative tests for identifier injection and non-positive
--window-seconds / --threshold. Every assertion goes through the CLI via
subprocess, matching the style of the integrity suite.

Aggregate-only: the fixtures seed an audit_log(user_id, action, ts) table and
the ops select only the user-identifier column, the action column, timestamps,
and counts — never free-text/PII columns.
"""
import json
import os
import sqlite3
import subprocess
import sys

HC = os.path.join(os.path.dirname(__file__), "..", "hc_db.py")


def _run(args, **kw):
    return subprocess.run([sys.executable, HC, *args],
                          capture_output=True, text=True, **kw)


def _make_db(tmp_path):
    """Build audit_log + seed deterministic rows for every audit op.

    - alice: 3 creates + 2 updates -> summary {"create":3,"update":2} total 5
    - bob:   1 login               -> total 1
    - mallory: 25 events all inside one 60s window (a burst) plus a few spread
      out across other windows.
    """
    db = tmp_path / "audit.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE audit_log(user_id TEXT, action TEXT, ts TEXT)"
    )

    rows = []
    # alice: 3 creates + 2 updates, all recent (within 24h).
    for _ in range(3):
        rows.append(("alice", "create", "datetime('now', '-2 hours')"))
    for _ in range(2):
        rows.append(("alice", "update", "datetime('now', '-2 hours')"))
    # bob: 1 login, recent.
    rows.append(("bob", "login", "datetime('now', '-2 hours')"))

    for user, action, ts_expr in rows:
        con.execute(
            f"INSERT INTO audit_log(user_id, action, ts) VALUES (?, ?, {ts_expr})",
            (user, action),
        )

    # mallory: 25 events within a single fixed 60s window (a burst). Use a
    # fixed second-of-minute spread 0..24 within the same minute so they all
    # bucket together for window-seconds=60.
    burst_minute = "2026-06-15 12:30:"
    for i in range(25):
        ts = f"{burst_minute}{i:02d}"
        con.execute(
            "INSERT INTO audit_log(user_id, action, ts) VALUES (?, ?, ?)",
            ("mallory", "delete", ts),
        )
    # mallory: a few events spread out far apart (not a burst).
    for ts in ("2026-06-15 13:00:00", "2026-06-15 14:00:00",
               "2026-06-15 15:00:00"):
        con.execute(
            "INSERT INTO audit_log(user_id, action, ts) VALUES (?, ?, ?)",
            ("mallory", "delete", ts),
        )

    con.commit()
    con.close()
    return db


# --- audit_user_summary -------------------------------------------------------

def test_audit_user_summary_nested_counts_and_ordering(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db),
              "--op", "audit_user_summary", "--table", "audit_log",
              "--user-column", "user_id", "--action-column", "action"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["op"] == "audit_user_summary"
    assert out["table"] == "audit_log"
    users = {u["user_id"]: u for u in out["users"]}
    assert users["alice"]["actions"] == {"create": 3, "update": 2}
    assert users["alice"]["total"] == 5
    assert users["bob"]["actions"] == {"login": 1}
    assert users["bob"]["total"] == 1
    # Users sorted by total desc: mallory (28) first, then alice (5), bob (1).
    totals = [u["total"] for u in out["users"]]
    assert totals == sorted(totals, reverse=True)
    # alice before bob (alice total 5 > bob total 1).
    order = [u["user_id"] for u in out["users"]]
    assert order.index("alice") < order.index("bob")


# --- audit_velocity -----------------------------------------------------------

def test_audit_velocity_flags_burst_user_only(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db),
              "--op", "audit_velocity", "--table", "audit_log",
              "--user-column", "user_id", "--ts-column", "ts",
              "--window-seconds", "60", "--threshold", "20"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["op"] == "audit_velocity"
    assert out["window_seconds"] == 60
    assert out["threshold"] == 20
    burst_users = {b["user_id"]: b for b in out["bursts"]}
    assert "mallory" in burst_users
    assert burst_users["mallory"]["count"] >= 20
    assert "bucket_start_epoch" in burst_users["mallory"]
    # alice (max 5/window) and bob (1) are below the threshold.
    assert "alice" not in burst_users
    assert "bob" not in burst_users
    # Sorted by count desc.
    counts = [b["count"] for b in out["bursts"]]
    assert counts == sorted(counts, reverse=True)


# --- audit_fingerprint --------------------------------------------------------

def test_audit_fingerprint_action_distribution_shape(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db),
              "--op", "audit_fingerprint", "--table", "audit_log",
              "--user-column", "user_id", "--action-column", "action"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["op"] == "audit_fingerprint"
    assert out["table"] == "audit_log"
    users = {u["user_id"]: u for u in out["users"]}
    assert users["alice"]["actions"] == {"create": 3, "update": 2}
    assert users["mallory"]["actions"] == {"delete": 28}


# --- audit_recency ------------------------------------------------------------

def test_audit_recency_counts_and_last_write(tmp_path):
    db = tmp_path / "recency.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE audit_log(user_id TEXT, action TEXT, ts TEXT)")
    # 2 events within 24h, 1 more within 7d (but >24h), 1 ancient (>7d).
    con.execute("INSERT INTO audit_log VALUES ('a', 'x', datetime('now', '-1 hours'))")
    con.execute("INSERT INTO audit_log VALUES ('a', 'x', datetime('now', '-5 hours'))")
    con.execute("INSERT INTO audit_log VALUES ('b', 'y', datetime('now', '-3 days'))")
    con.execute("INSERT INTO audit_log VALUES ('c', 'z', datetime('now', '-30 days'))")
    con.commit()
    # The max ts is the -1 hours row.
    max_ts = con.execute(
        "SELECT MAX(ts) FROM audit_log").fetchone()[0]
    con.close()

    r = _run(["--engine", "sqlite", "--db", str(db),
              "--op", "audit_recency", "--table", "audit_log", "--ts-column", "ts"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["op"] == "audit_recency"
    assert out["table"] == "audit_log"
    assert out["events_24h"] == 2
    assert out["events_7d"] == 3  # 2 within 24h + 1 at -3 days
    assert out["last_write_ts"] == max_ts


def test_audit_recency_empty_table_null_last_write(tmp_path):
    db = tmp_path / "empty.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE audit_log(user_id TEXT, action TEXT, ts TEXT)")
    con.commit()
    con.close()
    r = _run(["--engine", "sqlite", "--db", str(db),
              "--op", "audit_recency", "--table", "audit_log", "--ts-column", "ts"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["last_write_ts"] is None
    assert out["events_24h"] == 0
    assert out["events_7d"] == 0


# --- value_distribution -------------------------------------------------------

def test_value_distribution_counts_order_and_null_group(tmp_path):
    """Known frequencies (incl. a NULL group) come back in count-desc order."""
    db = tmp_path / "dist.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE events(event_type TEXT)")
    # frequencies: 'a' x5, 'b' x3, 'c' x1, NULL x2
    rows = (["a"] * 5) + (["b"] * 3) + (["c"] * 1) + ([None] * 2)
    con.executemany("INSERT INTO events(event_type) VALUES (?)",
                    [(v,) for v in rows])
    con.commit()
    con.close()

    r = _run(["--engine", "sqlite", "--db", str(db),
              "--op", "value_distribution", "--table", "events",
              "--column", "event_type"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["op"] == "value_distribution"
    assert out["table"] == "events"
    assert out["column"] == "event_type"
    values = out["values"]
    # Count-desc order.
    counts = [v["n"] for v in values]
    assert counts == sorted(counts, reverse=True)
    # Map by value (NULL serializes to JSON null -> Python None key).
    by_val = {v["value"]: v["n"] for v in values}
    assert by_val["a"] == 5
    assert by_val["b"] == 3
    assert by_val["c"] == 1
    # The NULL group is present and informative.
    assert None in by_val
    assert by_val[None] == 2
    # Top value is 'a' (highest count).
    assert values[0]["value"] == "a"
    assert values[0]["n"] == 5


def test_value_distribution_top_truncates(tmp_path):
    """--top 2 returns only the two most-frequent groups."""
    db = tmp_path / "dist2.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE events(event_type TEXT)")
    rows = (["a"] * 5) + (["b"] * 3) + (["c"] * 1) + ([None] * 2)
    con.executemany("INSERT INTO events(event_type) VALUES (?)",
                    [(v,) for v in rows])
    con.commit()
    con.close()

    r = _run(["--engine", "sqlite", "--db", str(db),
              "--op", "value_distribution", "--table", "events",
              "--column", "event_type", "--top", "2"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert len(out["values"]) == 2
    # The two most frequent are 'a' (5) then 'b' (3).
    assert [v["value"] for v in out["values"]] == ["a", "b"]
    assert [v["n"] for v in out["values"]] == [5, 3]


def test_value_distribution_rejects_bad_column(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db),
              "--op", "value_distribution", "--table", "audit_log",
              "--column", "x; DROP"])
    assert r.returncode != 0
    combined = (r.stdout + r.stderr).lower()
    assert "identifier" in combined or "invalid" in combined


def test_value_distribution_rejects_zero_top(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db),
              "--op", "value_distribution", "--table", "audit_log",
              "--column", "action", "--top", "0"])
    assert r.returncode != 0


# --- negative tests -----------------------------------------------------------

def test_audit_user_summary_rejects_bad_user_column(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db),
              "--op", "audit_user_summary", "--table", "audit_log",
              "--user-column", "u; DROP TABLE x", "--action-column", "action"])
    assert r.returncode != 0
    combined = (r.stdout + r.stderr).lower()
    assert "identifier" in combined or "invalid" in combined


def test_audit_velocity_rejects_zero_window(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db),
              "--op", "audit_velocity", "--table", "audit_log",
              "--user-column", "user_id", "--ts-column", "ts",
              "--window-seconds", "0", "--threshold", "20"])
    assert r.returncode != 0


def test_audit_velocity_rejects_zero_threshold(tmp_path):
    db = _make_db(tmp_path)
    r = _run(["--engine", "sqlite", "--db", str(db),
              "--op", "audit_velocity", "--table", "audit_log",
              "--user-column", "user_id", "--ts-column", "ts",
              "--window-seconds", "60", "--threshold", "0"])
    assert r.returncode != 0
