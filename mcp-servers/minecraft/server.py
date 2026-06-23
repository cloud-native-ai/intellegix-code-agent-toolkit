"""FastMCP server hosting a Minecraft Bedrock WebSocket listener.

Architecture
------------
ONE FastMCP server process does two jobs on a single asyncio event loop:

1. **Bedrock WebSocket listener** — a ``bedrock.server.Server`` (bedrockpy)
   accepts the connection the phone's Minecraft client makes via the in-game
   ``/connect <host>:8765`` command. bedrockpy's own ``Server.start()`` is
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
from websockets import server as wss

from mc_session import McSession

logger = logging.getLogger(__name__)

# Listener bind config. 0.0.0.0 so the phone on the LAN can reach it.
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8765

# Module-level singletons. The lifespan owns their construction; tools read the
# session via the FastMCP Context (lifespan_context) with this as a fallback so
# ``connect_status`` works regardless of how the tool is invoked.
_BEDROCK: Optional[Server] = None
_SESSION: Optional[McSession] = None


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
        bedrock_server._websocket_handler, host=LISTEN_HOST, port=LISTEN_PORT
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
        "In-game, run /connect <this-machine-ip>:8765 to attach, then use the "
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
