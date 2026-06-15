#!/usr/bin/env python3
"""hc_history.py — rolling row-count history for the /health-check tool.

A pure-Python (no database) helper that persists a bounded history of observed
row counts to a JSON file so the command layer can detect row-count anomalies
across runs, while guaranteeing no false anomalies on the very first observation
of an ``app:table`` pair (the first-run baseline guard).

Storage format (JSON), default path ``~/.claude/health-check/row-counts.json``::

    {
      "appname:tablename": [
        {"ts": "2026-06-15T08:00:00", "count": 247},
        ...
      ]
    }

Keyed by ``"{app}:{table}"``. Each list is pruned to the last ``MAX_ENTRIES``
(30) entries — oldest dropped first. The storage path is injectable on every
public function so tests can isolate state via ``tmp_path``.

Call-order contract: detection callers should compute ``baseline(...)`` BEFORE
``record(...)`` for the current run. That way the first run for an ``app:table``
sees an empty history → ``baseline`` returns ``None`` → ``is_anomaly`` short-
circuits to ``False``. Recording the current count afterward establishes the
baseline for subsequent runs.

Writes are atomic: data is written to a temp file in the same directory and then
``os.replace``'d onto the target, so a crash mid-write cannot corrupt an
existing history file. A corrupt or empty existing file is treated as empty
rather than raising.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# Default on-disk location. Callers/tests should pass an explicit path; this is
# only the production default the command layer uses.
DEFAULT_PATH: Path = Path.home() / ".claude" / "health-check" / "row-counts.json"

# Maximum number of readings retained per app:table key (oldest dropped first).
MAX_ENTRIES: int = 30


def _key(app: str, table: str) -> str:
    """Build the storage key for an ``app``/``table`` pair."""
    return f"{app}:{table}"


def _load(path: Path) -> dict[str, list[dict]]:
    """Load the history mapping from ``path``.

    Returns an empty dict if the file is missing, empty, or corrupt. Corrupt
    files are tolerated (treated as empty) so a damaged history never crashes a
    health-check run; the next ``record`` rewrites the file cleanly.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}

    if not text.strip():
        return {}

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}

    # Guard against a well-formed JSON file that isn't the expected shape.
    if not isinstance(data, dict):
        return {}
    return data


def _atomic_write(path: Path, data: dict[str, list[dict]]) -> None:
    """Write ``data`` as JSON to ``path`` atomically.

    Creates the parent directory if needed, writes to a uniquely named temp file
    in the same directory, flushes/fsyncs it, then ``os.replace``'s it onto the
    target (an atomic rename on the same filesystem). The temp file is removed on
    any failure so no partial/leftover files accumulate.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, str(path))
    except BaseException:
        # Clean up the temp file on any error before re-raising.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def record(app: str, table: str, count: int, ts: str, path: str | Path) -> None:
    """Append a row-count reading for ``app``/``table`` and persist it.

    Loads the existing history (treating a missing/empty/corrupt file as empty),
    appends ``{"ts": ts, "count": count}`` to the ``"{app}:{table}"`` list,
    prunes that list to the most recent ``MAX_ENTRIES`` entries, and writes the
    whole mapping back atomically.
    """
    path = Path(path)
    data = _load(path)

    key = _key(app, table)
    entries = data.get(key)
    if not isinstance(entries, list):
        entries = []

    entries.append({"ts": ts, "count": count})
    # Keep only the most recent MAX_ENTRIES readings (drop oldest).
    if len(entries) > MAX_ENTRIES:
        entries = entries[-MAX_ENTRIES:]
    data[key] = entries

    _atomic_write(path, data)


def history(app: str, table: str, path: str | Path) -> list[dict]:
    """Return the stored readings for ``app``/``table`` (or ``[]`` if none)."""
    data = _load(Path(path))
    entries = data.get(_key(app, table), [])
    if not isinstance(entries, list):
        return []
    return entries


def baseline(app: str, table: str, path: str | Path) -> float | None:
    """Return the baseline (mean count) over the current history, or ``None``.

    Returns ``None`` when there are zero prior readings — the first-run signal
    that callers must treat as "no baseline yet, do not flag an anomaly". When
    called BEFORE ``record`` for the current run (the expected order), a true
    first run sees an empty history and therefore gets ``None``.
    """
    entries = history(app, table, path)
    if not entries:
        return None
    counts = [e["count"] for e in entries]
    return sum(counts) / len(counts)


def is_anomaly(count: int, base: float | None, pct: float = 0.5) -> bool:
    """Decide whether ``count`` deviates from ``base`` by more than ``pct``.

    Returns ``False`` whenever ``base`` is ``None`` (first-run guard) or ``base``
    is ``0`` (no meaningful relative change can be computed). Otherwise returns
    ``True`` iff ``abs(count - base) / base > pct`` (a fractional threshold, e.g.
    ``0.5`` == 50%).
    """
    if base is None or base == 0:
        return False
    return abs(count - base) / base > pct


if __name__ == "__main__":  # pragma: no cover
    # Minimal CLI for ad-hoc inspection; the module is primarily a library.
    import argparse

    parser = argparse.ArgumentParser(description="Row-count history helper")
    parser.add_argument("app")
    parser.add_argument("table")
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    args = parser.parse_args()

    hist = history(args.app, args.table, args.path)
    base = baseline(args.app, args.table, args.path)
    print(json.dumps({"history": hist, "baseline": base}, indent=2))
