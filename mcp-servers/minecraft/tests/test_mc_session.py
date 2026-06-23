"""Unit tests for :mod:`mc_session`.

These tests import ONLY ``mc_session`` (and ``protocol``) — never ``server`` —
so the suite runs in CI without ``fastmcp``/``bedrockpy`` installed.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mc_session import McSession  # noqa: E402
from protocol import CommandResult  # noqa: E402


def test_run_delegates_to_sender_and_parses():
    async def fake_send(command):
        # sender returns the raw bedrock response packet (or dict)
        return {
            "header": {"messagePurpose": "commandResponse"},
            "body": {"statusCode": 0, "statusMessage": "ok"},
        }

    s = McSession(sender=fake_send)
    s.mark_connected("Steve")
    res = asyncio.run(s.run("say hi"))
    assert isinstance(res, CommandResult) and res.ok


def test_status_disconnected_initially():
    s = McSession(sender=None)
    st = s.status()
    assert st["connected"] is False and st["listening"] is True


def test_run_when_disconnected_returns_not_ok():
    s = McSession(sender=None)
    res = asyncio.run(s.run("say hi"))
    assert isinstance(res, CommandResult) and not res.ok  # no game connected


def test_status_reports_player_and_port_after_connect():
    s = McSession(sender=lambda c: None, port=9999)
    s.mark_connected("Alex")
    st = s.status()
    assert st == {
        "connected": True,
        "player": "Alex",
        "listening": True,
        "port": 9999,
    }


def test_mark_disconnected_clears_player():
    async def fake_send(command):
        return {"body": {"statusCode": 0, "statusMessage": "ok"}}

    s = McSession(sender=fake_send)
    s.mark_connected("Steve")
    s.mark_disconnected()
    st = s.status()
    assert st["connected"] is False and st["player"] is None
    # run must now refuse since no client is connected
    res = asyncio.run(s.run("say hi"))
    assert not res.ok


def test_run_accepts_already_typed_command_result():
    typed = CommandResult(ok=True, status_code=0, message="already-typed")

    async def fake_send(command):
        return typed

    s = McSession(sender=fake_send)
    s.mark_connected("Steve")
    res = asyncio.run(s.run("say hi"))
    assert res is typed and res.ok


def test_run_accepts_bedrock_commandresponse_like_object():
    class FakeResponse:
        """Mimics bedrock.response.CommandResponse (message/status/ok props)."""

        message = "done"
        status = 0
        ok = True

    async def fake_send(command):
        return FakeResponse()

    s = McSession(sender=fake_send)
    s.mark_connected("Steve")
    res = asyncio.run(s.run("setblock 0 0 0 stone"))
    assert isinstance(res, CommandResult)
    assert res.ok and res.status_code == 0 and res.message == "done"
