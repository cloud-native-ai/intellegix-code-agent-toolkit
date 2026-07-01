"""Pure geometry → Bedrock-command builders for ``build_structure``.

Each primitive is expressed ONCE (run-encoded — e.g. a whole box/line/helix as a
single spec object, NOT one object per block) and expanded here into the exact
``/setblock`` / ``/fill`` command strings the WebSocket API expects. This module
is deliberately **pure** (no I/O, no network) and imports only :mod:`mcs_build`
(itself stdlib-only), so it runs in CI without ``fastmcp``/``bedrockpy`` and is
unit-tested in isolation. The live :func:`server.build_structure` tool runs the
emitted commands server-side in a tight loop — eliminating per-block model
round-trips (the dominant token cost in long builds).

Primitive types and required params (all coords absolute ints):
  box      {x1,y1,z1,x2,y2,z2, block, states?, mode?}
  line     {x1,y1,z1,x2,y2,z2, block, states?}            # 3D Bresenham
  ring     {cx,cy,cz, radius, block, states?, plane?}     # plane xz|xy|zy (def xz)
  helix    {cx,cy,cz, radius, height, block, states?, turns?, plane?}
  rail_run {path:[[x,y,z],...], powered?, rail?, power_block?}
  scatter  {x1,y1,z1,x2,y2,z2, block, count, seed?, states?}
"""
from __future__ import annotations

import math
import random
from typing import Any

import mcs_build

VALID_PRIMITIVES = {"box", "line", "ring", "helix", "rail_run", "scatter"}
_PLANES = {"xz", "xy", "zy"}


def _req_int(p: dict[str, Any], key: str) -> int:
    try:
        return int(p[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"primitive {p.get('type')!r} missing/invalid int {key!r}: {exc}") from exc


def _setblocks(coords: list[tuple[int, int, int]], block: str, states: Any) -> list[str]:
    return [mcs_build.setblock_command(x, y, z, block, states=states) for (x, y, z) in coords]


def _line_points(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """3D Bresenham — integer points from c1 to c2 inclusive."""
    x1, y1, z1 = c1
    x2, y2, z2 = c2
    dx, dy, dz = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
    sx = 1 if x2 >= x1 else -1
    sy = 1 if y2 >= y1 else -1
    sz = 1 if z2 >= z1 else -1
    pts: list[tuple[int, int, int]] = []
    x, y, z = x1, y1, z1
    if dx >= dy and dx >= dz:
        ey, ez = dx // 2, dx // 2
        for _ in range(dx + 1):
            pts.append((x, y, z))
            ey -= dy
            if ey < 0: y += sy; ey += dx
            ez -= dz
            if ez < 0: z += sz; ez += dx
            x += sx
    elif dy >= dx and dy >= dz:
        ex, ez = dy // 2, dy // 2
        for _ in range(dy + 1):
            pts.append((x, y, z))
            ex -= dx
            if ex < 0: x += sx; ex += dy
            ez -= dz
            if ez < 0: z += sz; ez += dy
            y += sy
    else:
        ex, ey = dz // 2, dz // 2
        for _ in range(dz + 1):
            pts.append((x, y, z))
            ex -= dx
            if ex < 0: x += sx; ex += dz
            ey -= dy
            if ey < 0: y += sy; ey += dz
            z += sz
    return pts


def _ring_points(cx: int, cy: int, cz: int, radius: int, plane: str) -> list[tuple[int, int, int]]:
    """A 1-block-thick circle of `radius` in the given plane, dedup'd integer points."""
    if radius < 0:
        raise ValueError("ring radius must be >= 0")
    seen: set[tuple[int, int, int]] = set()
    steps = max(8, int(2 * math.pi * max(radius, 1)) * 2)
    for i in range(steps):
        a = 2 * math.pi * i / steps
        da, db = round(radius * math.cos(a)), round(radius * math.sin(a))
        if plane == "xz":
            seen.add((cx + da, cy, cz + db))
        elif plane == "xy":
            seen.add((cx + da, cy + db, cz))
        else:  # zy
            seen.add((cx, cy + db, cz + da))
    return sorted(seen)


def _box(p: dict[str, Any]) -> list[str]:
    return mcs_build.fill_commands(
        _req_int(p, "x1"), _req_int(p, "y1"), _req_int(p, "z1"),
        _req_int(p, "x2"), _req_int(p, "y2"), _req_int(p, "z2"),
        str(p["block"]), states=p.get("states"), mode=p.get("mode", "replace"),
    )


def _line(p: dict[str, Any]) -> list[str]:
    pts = _line_points(
        (_req_int(p, "x1"), _req_int(p, "y1"), _req_int(p, "z1")),
        (_req_int(p, "x2"), _req_int(p, "y2"), _req_int(p, "z2")),
    )
    return _setblocks(pts, str(p["block"]), p.get("states"))


def _ring(p: dict[str, Any]) -> list[str]:
    plane = str(p.get("plane", "xz"))
    if plane not in _PLANES:
        raise ValueError(f"ring plane must be one of {_PLANES}, got {plane!r}")
    pts = _ring_points(_req_int(p, "cx"), _req_int(p, "cy"), _req_int(p, "cz"), _req_int(p, "radius"), plane)
    return _setblocks(pts, str(p["block"]), p.get("states"))


def _helix(p: dict[str, Any]) -> list[str]:
    cx, cy, cz = _req_int(p, "cx"), _req_int(p, "cy"), _req_int(p, "cz")
    radius, height = _req_int(p, "radius"), _req_int(p, "height")
    if height <= 0:
        raise ValueError("helix height must be > 0")
    turns = float(p.get("turns", 1.0))
    plane = str(p.get("plane", "xz"))
    if plane not in _PLANES:
        raise ValueError(f"helix plane must be one of {_PLANES}, got {plane!r}")
    seen: set[tuple[int, int, int]] = set()
    pts: list[tuple[int, int, int]] = []
    for step in range(height):
        a = 2 * math.pi * turns * (step / height)
        da, db = round(radius * math.cos(a)), round(radius * math.sin(a))
        if plane == "xz":
            pt = (cx + da, cy + step, cz + db)
        elif plane == "xy":
            pt = (cx + da, cy + step, cz)  # rises along the helix step regardless
        else:
            pt = (cx, cy + step, cz + da)
        if pt not in seen:
            seen.add(pt)
            pts.append(pt)
    return _setblocks(pts, str(p["block"]), p.get("states"))


def _rail_run(p: dict[str, Any]) -> list[str]:
    """Place a rail line along an explicit path. Powered => golden_rail on a
    redstone_block bed (Bedrock-correct; rails auto-slope on a 1:1 step)."""
    path = p.get("path")
    if not isinstance(path, list) or not path:
        raise ValueError("rail_run requires a non-empty 'path' list of [x,y,z]")
    powered = bool(p.get("powered", False))
    rail = str(p.get("rail", "golden_rail" if powered else "rail"))
    power_block = str(p.get("power_block", "redstone_block"))
    cmds: list[str] = []
    for pt in path:
        try:
            x, y, z = int(pt[0]), int(pt[1]), int(pt[2])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"rail_run path point malformed {pt!r}: {exc}") from exc
        if powered:
            cmds.append(mcs_build.setblock_command(x, y - 1, z, power_block))
        cmds.append(mcs_build.setblock_command(x, y, z, rail))
    return cmds


def _scatter(p: dict[str, Any]) -> list[str]:
    x1, y1, z1 = _req_int(p, "x1"), _req_int(p, "y1"), _req_int(p, "z1")
    x2, y2, z2 = _req_int(p, "x2"), _req_int(p, "y2"), _req_int(p, "z2")
    count = _req_int(p, "count")
    if count <= 0:
        raise ValueError("scatter count must be > 0")
    rng = random.Random(int(p.get("seed", 0)))  # fixed seed => reproducible re-runs
    xs = range(min(x1, x2), max(x1, x2) + 1)
    ys = range(min(y1, y2), max(y1, y2) + 1)
    zs = range(min(z1, z2), max(z1, z2) + 1)
    all_pts = [(x, y, z) for x in xs for y in ys for z in zs]
    if not all_pts:
        return []
    chosen = rng.sample(all_pts, min(count, len(all_pts)))
    return _setblocks(chosen, str(p["block"]), p.get("states"))


_DISPATCH = {
    "box": _box, "line": _line, "ring": _ring,
    "helix": _helix, "rail_run": _rail_run, "scatter": _scatter,
}


def primitive_commands(primitive: dict[str, Any]) -> list[str]:
    """Expand one primitive into its ordered Bedrock command strings.

    Args:
        primitive: A spec object with a ``type`` in :data:`VALID_PRIMITIVES` plus
            that type's required params.

    Returns:
        The ordered list of ``/setblock`` / ``/fill`` command strings.

    Raises:
        ValueError: If ``type`` is missing/unknown or required params are bad
            (including injection-unsafe block ids, via :mod:`mcs_build`).
    """
    if not isinstance(primitive, dict):
        raise ValueError(f"primitive must be an object, got {type(primitive).__name__}")
    ptype = primitive.get("type")
    if ptype not in _DISPATCH:
        raise ValueError(f"unknown primitive type {ptype!r}; valid: {sorted(VALID_PRIMITIVES)}")
    if "block" not in primitive and ptype != "rail_run":
        raise ValueError(f"primitive {ptype!r} requires a 'block'")
    return _DISPATCH[ptype](primitive)


def primitive_bbox(p: dict[str, Any]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Coarse bounding box ``(c1, c2)`` of a primitive — for the build-state hint."""
    t = p.get("type")
    if t in ("box", "line", "scatter"):
        return ((_req_int(p, "x1"), _req_int(p, "y1"), _req_int(p, "z1")),
                (_req_int(p, "x2"), _req_int(p, "y2"), _req_int(p, "z2")))
    if t in ("ring", "helix"):
        cx, cy, cz, r = _req_int(p, "cx"), _req_int(p, "cy"), _req_int(p, "cz"), _req_int(p, "radius")
        h = _req_int(p, "height") if t == "helix" else 0
        return ((cx - r, cy, cz - r), (cx + r, cy + h, cz + r))
    if t == "rail_run":
        pts = [(int(q[0]), int(q[1]), int(q[2])) for q in p["path"]]
        xs, ys, zs = [q[0] for q in pts], [q[1] for q in pts], [q[2] for q in pts]
        return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))
    raise ValueError(f"no bbox for primitive type {t!r}")
