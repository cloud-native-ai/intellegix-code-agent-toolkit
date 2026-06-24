"""Testable Minecraft session wrapper — the unit under test for the MCP tool.

This module is deliberately **free of heavy imports**: it imports neither
``bedrockpy`` nor ``fastmcp``. The live websocket plumbing lives in
``server.py`` (which is never imported by the test suite), so CI can exercise
``McSession`` without those third-party packages installed.

``McSession`` holds the connection state of the single Minecraft client that
``/connect``s to the listener and delegates command execution to an injected
async ``sender`` coroutine. ``server.py`` builds the real ``sender`` (which
calls ``bedrock.server.Server.run``) and wires the bedrockpy connect/disconnect
events to :meth:`McSession.mark_connected` / :meth:`McSession.mark_disconnected`.

The raw result returned by the ``sender`` is normalized to a
:class:`protocol.CommandResult` so callers get a uniform shape whether the
sender hands back a raw Bedrock envelope dict, an already-typed
``CommandResult``, or a bedrockpy ``CommandResponse``-like object.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

from protocol import CommandResult, parse_response

logger = logging.getLogger(__name__)

# Sentinel status code for "no client connected" — non-zero so ``ok`` is False.
_NOT_CONNECTED_STATUS = -1

# An async callable: ``async def sender(command: str) -> Any``. ``Any`` because
# the live sender returns a bedrockpy ``CommandResponse``, while tests inject
# senders that return raw dicts or typed ``CommandResult`` objects.
Sender = Callable[[str], Awaitable[Any]]


class McSession:
    """Stateful wrapper around the single connected Minecraft Bedrock client.

    Attributes:
        port: The port the websocket listener binds to (informational; the
            actual bind happens in ``server.py``).
        listening: Always ``True`` for a constructed session — the listener is
            considered up once the server process is running.
        connected: ``True`` once a client has ``/connect``ed and the bedrockpy
            connect event fired.
        player: The connected player's name, or ``None`` when disconnected.
    """

    def __init__(self, sender: Optional[Sender] = None, port: int = 8767) -> None:
        """Create a session.

        Args:
            sender: An async callable that sends a command to the connected
                client and returns its raw response. ``None`` means no transport
                is wired (commands will be refused).
            port: The websocket listener port (informational only here).
        """
        self._sender: Optional[Sender] = sender
        self.port: int = port
        self.listening: bool = True
        self.connected: bool = False
        self.player: Optional[str] = None

    def mark_connected(self, player: Optional[str] = None) -> None:
        """Record that a Minecraft client has connected.

        Called by ``server.py`` from the bedrockpy ``connect`` server event.

        Args:
            player: The connected player's name if known (bedrockpy's
                ``ConnectContext`` does not always carry it).
        """
        self.connected = True
        self.player = player
        logger.info("Minecraft client connected (player=%r)", player)

    def mark_disconnected(self) -> None:
        """Record that the Minecraft client has disconnected.

        Called by ``server.py`` from the bedrockpy ``disconnect`` server event.
        """
        self.connected = False
        self.player = None
        logger.info("Minecraft client disconnected")

    async def run(self, command: str) -> CommandResult:
        """Run a Minecraft command against the connected client.

        Args:
            command: The slash command text (a leading ``/`` is fine; bedrockpy
                strips it).

        Returns:
            A :class:`protocol.CommandResult`. When no sender is wired or no
            client is connected, returns a not-ok result with status code -1 and
            message ``"no Minecraft client connected"`` (no network call made).
        """
        if self._sender is None or not self.connected:
            return CommandResult(
                ok=False,
                status_code=_NOT_CONNECTED_STATUS,
                message="no Minecraft client connected",
            )

        raw = await self._sender(command)
        return self._normalize(raw)

    @staticmethod
    def _normalize(raw: Any) -> CommandResult:
        """Coerce a sender's return value into a :class:`CommandResult`.

        Handles three shapes:
          * an already-typed :class:`CommandResult` (passed through unchanged);
          * a bedrockpy ``CommandResponse``-like object (has ``message`` /
            ``status`` / ``ok`` attributes but is not a dict);
          * a raw Bedrock envelope ``dict`` (delegated to ``parse_response``).

        Args:
            raw: Whatever the sender returned.

        Returns:
            A normalized :class:`CommandResult`. Unrecognized/None values yield a
            not-ok sentinel result rather than raising.
        """
        if isinstance(raw, CommandResult):
            return raw

        if isinstance(raw, dict):
            return parse_response(raw)

        # bedrockpy CommandResponse-like: duck-type on status/message.
        status = getattr(raw, "status", None)
        if status is not None:
            try:
                status_code = int(status)
            except (TypeError, ValueError):
                status_code = _NOT_CONNECTED_STATUS
            message = getattr(raw, "message", "")
            if not isinstance(message, str):
                message = "" if message is None else str(message)
            return CommandResult(
                ok=(status_code == 0), status_code=status_code, message=message
            )

        # Unknown / None response — report failure rather than raise.
        logger.warning("sender returned unrecognized result: %r", type(raw))
        return CommandResult(
            ok=False,
            status_code=_NOT_CONNECTED_STATUS,
            message="unrecognized command response",
        )

    def status(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the session state.

        Returns:
            ``{"connected": bool, "player": str | None, "listening": bool,
            "port": int}``.
        """
        return {
            "connected": self.connected,
            "player": self.player,
            "listening": self.listening,
            "port": self.port,
        }
