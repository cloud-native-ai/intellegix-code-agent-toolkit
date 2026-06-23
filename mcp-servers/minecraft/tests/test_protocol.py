import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from protocol import build_command_packet, build_subscribe_packet, parse_response, CommandResult


def test_command_packet_shape():
    p = build_command_packet("setblock 1 2 3 stone", request_id="fixed-uuid")
    assert p["header"]["messagePurpose"] == "commandRequest"
    assert p["header"]["messageType"] == "commandRequest"
    assert p["header"]["version"] == 1
    assert p["header"]["requestId"] == "fixed-uuid"
    assert p["body"]["commandLine"] == "setblock 1 2 3 stone"
    assert p["body"]["origin"]["type"] == "player"


def test_strips_one_leading_slash():
    assert build_command_packet("/say hi", request_id="x")["body"]["commandLine"] == "say hi"


def test_generates_uuid_when_none():
    p = build_command_packet("say hi")
    rid = p["header"]["requestId"]
    assert isinstance(rid, str) and len(rid) >= 32   # uuid4 string


def test_subscribe_packet():
    p = build_subscribe_packet("PlayerMessage", request_id="x")
    assert p["header"]["messagePurpose"] == "subscribe"
    assert p["body"]["eventName"] == "PlayerMessage"


def test_parse_response_ok():
    r = parse_response({"header": {"messagePurpose": "commandResponse"}, "body": {"statusCode": 0, "statusMessage": "Block placed."}})
    assert isinstance(r, CommandResult) and r.ok and r.status_code == 0 and "placed" in r.message.lower()


def test_parse_response_error():
    r = parse_response({"header": {"messagePurpose": "commandResponse"}, "body": {"statusCode": -2147352576, "statusMessage": "Syntax error"}})
    assert not r.ok and r.status_code == -2147352576 and "syntax" in r.message.lower()


def test_parse_response_missing_body_is_safe():
    r = parse_response({"header": {"messagePurpose": "commandResponse"}, "body": {}})
    assert isinstance(r, CommandResult)  # no crash; default status handling


# --- additional coverage ---

def test_empty_command_raises():
    with pytest.raises(ValueError):
        build_command_packet("")


def test_whitespace_only_command_raises():
    with pytest.raises(ValueError):
        build_command_packet("   ")


def test_only_one_leading_slash_stripped():
    # "//foo" -> "/foo" (exactly one slash removed)
    assert build_command_packet("//foo", request_id="x")["body"]["commandLine"] == "/foo"


def test_non_string_command_rejected():
    with pytest.raises(ValueError):
        build_command_packet(123)  # type: ignore[arg-type]


def test_subscribe_generates_uuid_when_none():
    p = build_subscribe_packet("PlayerMessage")
    rid = p["header"]["requestId"]
    assert isinstance(rid, str) and len(rid) >= 32


def test_subscribe_header_version_and_messagetype():
    p = build_subscribe_packet("PlayerMessage", request_id="x")
    assert p["header"]["version"] == 1
    assert p["header"]["messageType"] == "commandRequest"


def test_command_body_version_is_one():
    p = build_command_packet("say hi", request_id="x")
    assert p["body"]["version"] == 1


def test_empty_subscribe_event_rejected():
    with pytest.raises(ValueError):
        build_subscribe_packet("")


def test_parse_response_missing_body_key_is_safe():
    # No "body" key at all must not raise.
    r = parse_response({"header": {"messagePurpose": "commandResponse"}})
    assert isinstance(r, CommandResult) and not r.ok


def test_parse_response_missing_status_defaults_not_ok():
    r = parse_response({"body": {"statusMessage": "no code here"}})
    assert isinstance(r, CommandResult) and not r.ok and r.status_code != 0


def test_parse_response_empty_dict_is_safe():
    r = parse_response({})
    assert isinstance(r, CommandResult) and not r.ok
