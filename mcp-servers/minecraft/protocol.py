"""Minecraft Bedrock WebSocket protocol framing + response parsing.

Pure stdlib (``uuid`` only) — NO third-party imports, so the raw protocol
stays usable in CI and independently of ``bedrockpy``.

The Bedrock WebSocket API exchanges JSON envelopes of the shape
``{"header": {...}, "body": {...}}``. This module builds outgoing
command/subscribe packets and parses incoming command responses.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

# Sentinel status code used when a response carries no parseable statusCode.
# Any non-zero value works; -1 is conventional for "unknown/failed".
_MISSING_STATUS_SENTINEL = -1


@dataclass
class CommandResult:
    """Parsed result of a Bedrock ``commandResponse`` packet.

    Attributes:
        ok: True only when ``status_code == 0`` (Bedrock success).
        status_code: The body's ``statusCode`` (0 = success). Falls back to a
            non-zero sentinel when missing/malformed so ``ok`` stays False.
        message: The body's ``statusMessage`` (empty string when absent).
    """

    ok: bool
    status_code: int
    message: str


def build_command_packet(command: str, request_id: str | None = None) -> dict:
    """Build an outgoing Bedrock ``commandRequest`` envelope.

    Args:
        command: The slash-command text. Exactly one leading ``/`` is stripped
            (``"//foo"`` -> ``"/foo"``). Must be a non-empty, non-whitespace str.
        request_id: Correlation id. A uuid4 string is generated when None.

    Returns:
        A ``{"header": {...}, "body": {...}}`` dict matching the Bedrock
        protocol (header version 1, body version 1, origin type ``player``).

    Raises:
        ValueError: If ``command`` is not a string, or is empty/whitespace-only.
    """
    if not isinstance(command, str):
        raise ValueError("command must be a string")
    if not command.strip():
        raise ValueError("command must be a non-empty string")

    # Strip exactly ONE leading slash (not all): "//foo" -> "/foo".
    command_line = command[1:] if command.startswith("/") else command

    rid = request_id if request_id is not None else str(uuid.uuid4())

    return {
        "header": {
            "version": 1,
            "requestId": rid,
            "messageType": "commandRequest",
            "messagePurpose": "commandRequest",
        },
        "body": {
            "version": 1,
            "commandLine": command_line,
            "origin": {"type": "player"},
        },
    }


def build_subscribe_packet(event_name: str, request_id: str | None = None) -> dict:
    """Build an outgoing Bedrock event ``subscribe`` envelope.

    Args:
        event_name: The Bedrock event name (e.g. ``"PlayerMessage"``). Must be
            a non-empty, non-whitespace str.
        request_id: Correlation id. A uuid4 string is generated when None.

    Returns:
        A ``{"header": {...}, "body": {...}}`` dict with header
        ``messagePurpose == "subscribe"`` and body ``{"eventName": ...}``.

    Raises:
        ValueError: If ``event_name`` is not a non-empty string.
    """
    if not isinstance(event_name, str) or not event_name.strip():
        raise ValueError("event_name must be a non-empty string")

    rid = request_id if request_id is not None else str(uuid.uuid4())

    return {
        "header": {
            "version": 1,
            "requestId": rid,
            "messageType": "commandRequest",
            "messagePurpose": "subscribe",
        },
        "body": {
            "eventName": event_name,
        },
    }


def parse_response(packet: dict) -> CommandResult:
    """Parse an incoming Bedrock ``commandResponse`` envelope defensively.

    Missing ``body`` or missing keys must NOT raise; instead the result is
    reported as not-ok with a sentinel status code.

    Args:
        packet: The incoming ``{"header": {...}, "body": {...}}`` dict.

    Returns:
        A :class:`CommandResult`. ``ok`` is True only when the body's
        ``statusCode`` is exactly 0.
    """
    body = packet.get("body") if isinstance(packet, dict) else None
    if not isinstance(body, dict):
        body = {}

    raw_code = body.get("statusCode", _MISSING_STATUS_SENTINEL)
    try:
        status_code = int(raw_code)
    except (TypeError, ValueError):
        status_code = _MISSING_STATUS_SENTINEL

    message = body.get("statusMessage", "")
    if not isinstance(message, str):
        message = str(message)

    return CommandResult(ok=(status_code == 0), status_code=status_code, message=message)
