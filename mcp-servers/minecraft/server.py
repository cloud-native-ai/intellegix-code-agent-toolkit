"""FastMCP server hosting a Minecraft Bedrock WebSocket listener.

Architecture
------------
ONE FastMCP server process does two jobs on a single asyncio event loop:

1. **Bedrock WebSocket listener** — a ``bedrock.server.Server`` (bedrockpy)
   accepts the connection the phone's Minecraft client makes via the in-game
   ``/connect <host>:8767`` command. bedrockpy's own ``Server.start()`` is
   *blocking* (it creates its own loop and calls ``run_forever``), which is
   incompatible with running inside FastMCP. So instead, during FastMCP's
   **lifespan** we start the underlying ``websockets`` server on FastMCP's
   already-running loop (the same thing ``Server.start`` does internally, minus
   ``run_forever``) and dispatch bedrockpy's ``ready`` server event. The serve
   handle is closed on shutdown.

2. **MCP tools** — exposed over stdio to Claude Code. This task ships only
   ``connect_status``; later tasks add ``run_command`` / ``setblock`` / ``fill``
   (the clear extension point is :func:`_make_sender` + new ``@mcp.tool``s that
   call ``_SESSION.run(...)``).

The bridge between the two is :class:`mc_session.McSession`: bedrockpy
connect/disconnect events call ``mark_connected`` / ``mark_disconnected``, and a
``sender`` coroutine (built from ``bedrock.server.Server.run``) lets tools
execute commands. ``McSession`` itself imports no heavy deps and is unit-tested
in isolation; *this* file does the live wiring and is intentionally NOT imported
by the test suite.

Run directly with ``python server.py`` to launch the MCP server (stdio).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any, Optional

from bedrock.server import Server
from fastmcp import Context, FastMCP

try:
    from fastmcp import Image  # FastMCP image-return type → model vision input
except ImportError:  # pragma: no cover - fastmcp Image location varies by version
    from fastmcp.utilities.types import Image

import vision  # CI-safe screen-capture helper (heavy deps imported lazily)
from websockets import server as wss

import mcs_build
import structures
from mc_session import McSession

logger = logging.getLogger(__name__)

# Listener bind config. 0.0.0.0 so the phone on the LAN can reach it.
# Port 8767 (NOT 8765/8766 — those are used by the browser-bridge MCP's WS hub
# and metrics endpoint respectively; 8765 collides and would break browser-bridge).
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8767

# Module-level singletons. The lifespan owns their construction; tools read the
# session via the FastMCP Context (lifespan_context) with this as a fallback so
# ``connect_status`` works regardless of how the tool is invoked.
_BEDROCK: Optional[Server] = None
_SESSION: Optional[McSession] = None

# --------------------------------------------------------------------------- #
# Build-state cache (Phase 1.5): a tiny on-disk summary of what's been placed so
# a restarted session can re-orient WITHOUT spending tokens on screenshots. It is
# a drift-tolerant HINT (blocks can be changed in-world externally) — re-verify
# on demand. Records bounding box, per-block counts, and the recent ops.
# --------------------------------------------------------------------------- #
import json
import pathlib

_BUILD_STATE_PATH = pathlib.Path(__file__).resolve().parent / "build_state.json"
_BUILD_STATE: dict[str, Any] = {"ops": 0, "by_block": {}, "bbox": None, "recent": []}


def _expand_bbox(bbox: Optional[list[int]], c1: tuple[int, int, int], c2: tuple[int, int, int]) -> list[int]:
    xs = [c1[0], c2[0]]; ys = [c1[1], c2[1]]; zs = [c1[2], c2[2]]
    if bbox is None:
        return [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)]
    return [
        min(bbox[0], *xs), min(bbox[1], *ys), min(bbox[2], *zs),
        max(bbox[3], *xs), max(bbox[4], *ys), max(bbox[5], *zs),
    ]


def _record_placement(
    kind: str, block: str, c1: tuple[int, int, int], c2: tuple[int, int, int], count: Optional[int]
) -> None:
    """Record a successful placement into the build-state cache (best-effort)."""
    try:
        if count is None:  # a fill — volume of the box
            count = (abs(c2[0] - c1[0]) + 1) * (abs(c2[1] - c1[1]) + 1) * (abs(c2[2] - c1[2]) + 1)
        _BUILD_STATE["ops"] += 1
        _BUILD_STATE["by_block"][block] = _BUILD_STATE["by_block"].get(block, 0) + count
        _BUILD_STATE["bbox"] = _expand_bbox(_BUILD_STATE["bbox"], c1, c2)
        rec = _BUILD_STATE["recent"]
        rec.append({"kind": kind, "block": block, "c1": list(c1), "c2": list(c2), "n": count})
        del rec[:-50]  # keep last 50
        _BUILD_STATE_PATH.write_text(json.dumps(_BUILD_STATE), encoding="utf-8")
    except Exception:  # noqa: BLE001 — a cache write must never break a placement
        pass


def _load_build_state() -> None:
    """Load the persisted build-state on startup (best-effort)."""
    global _BUILD_STATE
    try:
        if _BUILD_STATE_PATH.exists():
            _BUILD_STATE = json.loads(_BUILD_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass


def _make_sender(bedrock_server: Server):
    """Build the async ``sender`` coroutine injected into :class:`McSession`.

    Args:
        bedrock_server: The live bedrockpy ``Server``.

    Returns:
        An ``async def sender(command: str) -> CommandResponse`` that runs the
        command on the connected client and returns bedrockpy's typed
        ``CommandResponse`` (``McSession`` normalizes it to a ``CommandResult``).
    """

    async def sender(command: str) -> Any:
        # bedrockpy strips a single leading slash itself.
        return await bedrock_server.run(command)

    return sender


def _register_connection_events(bedrock_server: Server, session: McSession) -> None:
    """Wire bedrockpy connect/disconnect server events to the session state.

    bedrockpy dispatches a ``connect``/``disconnect`` :class:`ServerEvent` by
    matching the *decorated function's name* to the event name. The
    ``ConnectContext`` carries no player name, so we pass ``None`` — a later
    task can resolve the player via a command query if needed.

    Args:
        bedrock_server: The live bedrockpy ``Server``.
        session: The :class:`McSession` to update.
    """

    @bedrock_server.server_event
    async def connect(ctx) -> None:  # noqa: ANN001 - bedrockpy ConnectContext
        session.mark_connected(None)

    @bedrock_server.server_event
    async def disconnect(ctx) -> None:  # noqa: ANN001 - bedrockpy DisconnectContext
        session.mark_disconnected()


@contextlib.asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[McSession]:
    """FastMCP lifespan: start the Bedrock listener, yield the session.

    Starts the bedrockpy websocket listener on FastMCP's running event loop
    (without bedrockpy's blocking ``run_forever``), yields the
    :class:`McSession` as the lifespan context value, and tears the listener
    down on shutdown.

    Args:
        server: The owning :class:`FastMCP` instance (unused but required by the
            ``Callable[[FastMCP], AbstractAsyncContextManager]`` lifespan API).

    Yields:
        The :class:`McSession` shared by all tools.
    """
    global _BEDROCK, _SESSION

    bedrock_server = Server()
    session = McSession(sender=_make_sender(bedrock_server), port=LISTEN_PORT)
    _register_connection_events(bedrock_server, session)

    _BEDROCK = bedrock_server
    _SESSION = session

    # Start the websocket server on the CURRENT (FastMCP) event loop. This
    # mirrors the asyncio portion of bedrock.server.Server.start() but omits the
    # blocking run_forever so we cooperate with FastMCP's loop.
    import asyncio

    ws_server = await wss.serve(
        bedrock_server._websocket_handler,
        host=LISTEN_HOST,
        port=LISTEN_PORT,
        # Minecraft Bedrock's WS client does NOT reply to protocol-level pings,
        # so the websockets default keepalive (ping every 20s, 20s timeout)
        # force-closes the connection with 1011 "keepalive ping timeout".
        # Disable server-initiated pings; Minecraft manages its own liveness.
        ping_interval=None,
    )
    bedrock_server._loop = asyncio.get_event_loop()
    bedrock_server._dispatch_server_event(
        "ready",
        __import__("bedrock.context", fromlist=["ReadyContext"]).ReadyContext(
            bedrock_server, host=LISTEN_HOST, port=LISTEN_PORT
        ),
    )
    logger.info("Bedrock WebSocket listener up on %s:%d", LISTEN_HOST, LISTEN_PORT)

    try:
        yield session
    finally:
        ws_server.close()
        with contextlib.suppress(Exception):
            await ws_server.wait_closed()
        logger.info("Bedrock WebSocket listener stopped")
        _BEDROCK = None
        _SESSION = None


mcp: FastMCP = FastMCP(
    name="minecraft-bedrock",
    instructions=(
        "Control a Minecraft Bedrock world over its WebSocket command API. "
        "In-game, run /connect <this-machine-ip>:8767 to attach, then use the "
        "tools to inspect connection status and (in later tasks) run commands."
    ),
    lifespan=lifespan,
)


@mcp.tool
def connect_status(ctx: Context) -> dict[str, Any]:
    """Report whether a Minecraft Bedrock client is connected to the listener.

    Returns:
        ``{"connected": bool, "player": str | None, "listening": bool,
        "port": int}``. ``listening`` is True whenever the server is running;
        ``connected`` becomes True after the in-game ``/connect`` succeeds.
    """
    session = getattr(ctx, "lifespan_context", None)
    if not isinstance(session, McSession):
        session = _SESSION
    if session is None:
        # Listener not yet started (should not happen once lifespan ran).
        return {
            "connected": False,
            "player": None,
            "listening": False,
            "port": LISTEN_PORT,
        }
    return session.status()


def _require_session() -> McSession:
    """Return the live :class:`McSession`, or raise if the listener isn't up.

    Returns:
        The module-level :class:`McSession` built by the lifespan.

    Raises:
        RuntimeError: If the session is not yet constructed (lifespan not run).
    """
    if _SESSION is None:
        raise RuntimeError(
            "Bedrock listener not started; the MCP lifespan has not run yet."
        )
    return _SESSION


@mcp.tool
async def run_command(command: str) -> dict[str, Any]:
    """Run a raw Minecraft Bedrock slash-command on the connected client.

    Args:
        command: The command text (with or without a leading ``/``). Must be a
            non-blank string.

    Returns:
        ``{"ok": bool, "status_code": int, "message": str}`` from the Bedrock
        command response.

    Raises:
        ValueError: If ``command`` is blank/whitespace-only.
        RuntimeError: If no Bedrock listener session is available.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    res = await _require_session().run(command)
    return {"ok": res.ok, "status_code": res.status_code, "message": res.message}


@mcp.tool
async def setblock(
    x: int,
    y: int,
    z: int,
    block: str,
    states: Optional[dict] = None,
    relative_to_player: bool = False,
) -> dict[str, Any]:
    """Place a single block via Bedrock ``/setblock``.

    Args:
        x, y, z: Target coordinates (absolute, or player-relative when
            ``relative_to_player`` is True).
        block: Block id, e.g. ``stone`` or ``minecraft:oak_stairs``.
        states: Optional block-state mapping (e.g. ``{"weirdo_direction": 0}``).
        relative_to_player: Run relative to the connected player's position.

    Returns:
        ``{"ok": bool, "status_code": int, "message": str, "command": str}`` —
        the command response plus the exact command that was run.

    Raises:
        ValueError: If the block name or states are injection-unsafe.
        RuntimeError: If no Bedrock listener session is available.
    """
    cmd = mcs_build.setblock_command(
        x, y, z, block, states=states, relative_to_player=relative_to_player
    )
    res = await _require_session().run(cmd)
    if res.ok:
        _record_placement("setblock", block, (x, y, z), (x, y, z), 1)
        return {"ok": True}  # terse: success needs no echo (cuts per-call output tokens)
    return {"ok": False, "status_code": res.status_code, "message": res.message, "command": cmd}


@mcp.tool
async def fill(
    x1: int,
    y1: int,
    z1: int,
    x2: int,
    y2: int,
    z2: int,
    block: str,
    states: Optional[dict] = None,
    mode: str = "replace",
    relative_to_player: bool = False,
) -> dict[str, Any]:
    """Fill a 3D region via Bedrock ``/fill``, chunked to respect the fill cap.

    The region is split into sub-boxes (each within Bedrock's fill cap) and each
    resulting command is run in turn; results are aggregated.

    Args:
        x1, y1, z1: One corner of the region (absolute, or player-relative when
            ``relative_to_player`` is True).
        x2, y2, z2: The opposite corner.
        block: Block id to fill with.
        states: Optional block-state mapping (e.g. ``{"pillar_axis": "y"}``).
        mode: Fill mode — one of ``replace`` (default), ``hollow``, ``outline``,
            ``keep``, ``destroy``.
        relative_to_player: Run relative to the connected player's position.

    Returns:
        ``{"ok": bool, "chunks": int, "failed_count": int, "first_error": str|None}``
        (terse — no per-chunk list; ``first_error`` carries the failing message).

    Raises:
        ValueError: If ``mode`` is invalid or the block/states are unsafe.
        RuntimeError: If no Bedrock listener session is available.
    """
    cmds = mcs_build.fill_commands(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        block,
        states=states,
        mode=mode,
        relative_to_player=relative_to_player,
    )
    session = _require_session()

    failed = 0
    first_error: Optional[str] = None
    for cmd in cmds:
        res = await session.run(cmd)
        if not res.ok:
            failed += 1
            if first_error is None:
                first_error = res.message
    if failed == 0:
        _record_placement("fill", block, (x1, y1, z1), (x2, y2, z2), None)
    # terse: drop the full per-chunk failed[] list (it reloads every turn) —
    # first_error carries the failing region's message for recovery.
    return {"ok": not failed, "chunks": len(cmds), "failed_count": failed, "first_error": first_error}


@mcp.tool
def take_screenshot(
    window_title_substring: Optional[str] = None,
    region: Optional[tuple[int, int, int, int]] = None,
    monitor: int = 1,
    max_edge: int = vision.MAX_EDGE_DEFAULT,
    fmt: str = "jpeg",
) -> Image:
    """MILESTONE/aesthetic checks only — for structural truth use ``verify_blocks``.

    Captures the PC window showing the game (2nd Bedrock client or AirPlay mirror;
    the world renders on the phone, and the Bedrock WS returns no images).
    Downscaled to ``max_edge`` longest side (Opus vision-token control); ``fmt`` is
    wire-only. Token-expensive — screenshot at build milestones, not per block.

    Args:
        window_title_substring: Window whose title contains this (e.g. ``"Minecraft"``).
        region: ``(left, top, width, height)`` pixels instead.
        monitor: 1-based monitor index for a full-monitor capture (1 = primary).
        max_edge: Longest-edge downscale clamp (default ~1456); ``<=0`` disables.
        fmt: ``"jpeg"`` (default) or ``"png"`` — wire size only, not token count.
    """
    img = vision.capture(
        window_title_substring=window_title_substring,
        region=region,
        monitor=monitor,
        max_edge=max_edge,
        fmt=fmt,
    )
    return Image(data=img, format="jpeg" if fmt.lower() in ("jpg", "jpeg") else "png")


@mcp.tool
async def verify_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify specific blocks are actually placed — world-state "vision" over WS.

    The robust feedback channel for this phone-rendered topology: instead of
    pixels, ask the world for the truth. Runs Bedrock ``/testforblock`` per entry
    (a success means the block at that position matches the expected id), so the
    model can confirm a build landed as planned regardless of where — or whether
    — it is being rendered.

    Args:
        blocks: List (≤256) of ``{"x": int, "y": int, "z": int, "block": str}``
            entries; block id may be bare (``stone``) or namespaced
            (``minecraft:oak_stairs``).

    Returns:
        ``{"checked": int, "matched": int, "mismatched": list[dict]}`` — each
        mismatch carries its coords, the expected block, and the Bedrock message.

    Raises:
        ValueError: If ``blocks`` is empty, too large, or an entry is malformed
            or has an injection-unsafe block id.
        RuntimeError: If no Bedrock listener session is available.
    """
    import re

    if not isinstance(blocks, list) or not blocks:
        raise ValueError("blocks must be a non-empty list")
    # Raised 256 -> 1024 to cut verification round-trips ~4x. NOTE: this runs one
    # /testforblock per coord in a tight loop; if a 1024-batch ever trips Bedrock
    # command-spam / tick degradation in practice, lower this constant.
    if len(blocks) > 1024:
        raise ValueError("at most 1024 blocks per call")

    session = _require_session()
    matched = 0
    mismatched: list[dict[str, Any]] = []
    for b in blocks:
        try:
            x, y, z = int(b["x"]), int(b["y"]), int(b["z"])
            block = str(b["block"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"bad block entry {b!r}: {exc}") from exc
        if not re.fullmatch(r"[A-Za-z0-9_:]+", block):
            raise ValueError(f"injection-unsafe block id: {block!r}")
        res = await session.run(f"testforblock {x} {y} {z} {block}")
        if res.ok:
            matched += 1
        else:
            mismatched.append(
                {"x": x, "y": y, "z": z, "expected": block, "message": res.message}
            )
    return {"checked": len(blocks), "matched": matched, "mismatched": mismatched}


@mcp.tool
async def build_structure(spec: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a whole structure server-side from a run-encoded spec — NO per-block
    model round-trips (the big token win for large builds).

    Each spec entry is ONE primitive (a whole box/line/ring/helix/rail_run/scatter
    described once, not block-by-block). The server expands and runs all commands
    in a tight loop and returns a terse summary. It is **idempotent** (re-running
    is safe — setblock/fill overwrite), so on a partial failure you can fix the
    failing primitive and re-call; ``failed_primitive`` tells you WHICH one and
    where, so you don't need a screenshot to diagnose.

    Args:
        spec: List (≤256) of primitives. Types + params:
            ``box`` {x1,y1,z1,x2,y2,z2,block,states?,mode?};
            ``line`` {x1,y1,z1,x2,y2,z2,block,states?};
            ``ring`` {cx,cy,cz,radius,block,states?,plane?(xz|xy|zy)};
            ``helix`` {cx,cy,cz,radius,height,block,states?,turns?,plane?};
            ``rail_run`` {path:[[x,y,z]...],powered?,rail?,power_block?};
            ``scatter`` {x1,y1,z1,x2,y2,z2,block,count,seed?,states?}.

    Returns:
        Terse, cache-friendly (NO timestamps/volatile fields):
        ``{"placed": int, "failed_count": int, "first_error": str|None,
        "failed_primitive": {"index": int, "type": str}|None}``.

    Raises:
        ValueError: If ``spec`` is empty/too large or any primitive is malformed
            (unknown type, missing params, injection-unsafe block id).
        RuntimeError: If no Bedrock listener session is available.
    """
    if not isinstance(spec, list) or not spec:
        raise ValueError("spec must be a non-empty list of primitives")
    if len(spec) > 256:
        raise ValueError("at most 256 primitives per build_structure call")

    # Expand ALL primitives first so a malformed spec fails fast before any
    # block is placed (partial half-built structures are worse than none).
    expanded: list[tuple[int, dict[str, Any], list[str]]] = []
    for idx, prim in enumerate(spec):
        expanded.append((idx, prim, structures.primitive_commands(prim)))

    session = _require_session()
    placed = 0
    failed_count = 0
    first_error: Optional[str] = None
    failed_primitive: Optional[dict[str, Any]] = None

    for idx, prim, cmds in expanded:
        prim_ok = True
        for cmd in cmds:
            res = await session.run(cmd)
            if res.ok:
                placed += 1
            else:
                failed_count += 1
                prim_ok = False
                if first_error is None:
                    first_error = res.message
                    failed_primitive = {"index": idx, "type": prim.get("type")}
        if prim_ok:
            try:
                c1, c2 = structures.primitive_bbox(prim)
                _record_placement(str(prim.get("type")), str(prim.get("block", prim.get("rail", "rail"))), c1, c2, None)
            except Exception:  # noqa: BLE001 — build-state hint is best-effort
                pass

    return {
        "placed": placed,
        "failed_count": failed_count,
        "first_error": first_error,
        "failed_primitive": failed_primitive,
    }


@mcp.tool
def build_state() -> dict[str, Any]:
    """Re-orient on what's already been built — cheap alternative to a screenshot.

    Returns the server-side build-state cache (bounding box, per-block counts, and
    the last ~50 placement ops) so a restarted session knows WHERE it was building
    and WHAT is placed without spending vision tokens. It is a drift-tolerant
    hint — verify specific blocks with ``verify_blocks`` if certainty is needed.

    Returns:
        ``{"ops": int, "bbox": [x1,y1,z1,x2,y2,z2]|None, "by_block": {block: count},
        "recent": [{kind, block, c1, c2, n}, ...]}``.
    """
    return _BUILD_STATE


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _load_build_state()
    mcp.run()
