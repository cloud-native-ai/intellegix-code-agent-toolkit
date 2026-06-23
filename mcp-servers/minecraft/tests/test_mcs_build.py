"""Unit tests for the pure command-string builders in ``mcs_build``.

These tests import ONLY ``mcs_build`` (which itself imports only ``blockstate``
and ``chunking``) — never ``server.py`` — so they run in CI without
fastmcp / bedrockpy installed.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcs_build import fill_commands, setblock_command  # noqa: E402


# --------------------------------------------------------------------------- #
# setblock
# --------------------------------------------------------------------------- #
def test_setblock_absolute():
    assert setblock_command(1, 2, 3, "stone") == "setblock 1 2 3 stone"


def test_setblock_with_states():
    assert (
        setblock_command(1, 2, 3, "oak_stairs", {"weirdo_direction": 0})
        == 'setblock 1 2 3 oak_stairs["weirdo_direction"=0]'
    )


def test_setblock_relative_to_player_wraps_execute():
    assert (
        setblock_command(1, 2, 3, "stone", relative_to_player=True)
        == "execute as @p at @s run setblock ~1 ~2 ~3 stone"
    )


def test_setblock_relative_negative_offset():
    assert (
        setblock_command(-1, 0, -2, "stone", relative_to_player=True)
        == "execute as @p at @s run setblock ~-1 ~0 ~-2 stone"
    )


def test_setblock_invalid_block_raises():
    # Uppercase block names fail serialize_block's injection-safe regex.
    with pytest.raises(ValueError):
        setblock_command(0, 0, 0, "Stone")


# --------------------------------------------------------------------------- #
# fill
# --------------------------------------------------------------------------- #
def test_fill_single_chunk_replace():
    assert fill_commands(0, 0, 0, 9, 9, 9, "stone") == ["fill 0 0 0 9 9 9 stone"]


def test_fill_mode_appended():
    assert fill_commands(0, 0, 0, 1, 1, 1, "glass", mode="hollow") == [
        "fill 0 0 0 1 1 1 glass hollow"
    ]


def test_fill_replace_mode_omitted_default():
    # 'replace' is the default Bedrock behavior — omit it from the command.
    assert fill_commands(0, 0, 0, 1, 1, 1, "stone", mode="replace") == [
        "fill 0 0 0 1 1 1 stone"
    ]


def test_fill_with_states():
    assert fill_commands(0, 0, 0, 1, 1, 1, "oak_log", {"pillar_axis": "y"}) == [
        'fill 0 0 0 1 1 1 oak_log["pillar_axis"="y"]'
    ]


def test_fill_large_region_chunked():
    cmds = fill_commands(0, 0, 0, 63, 63, 63, "stone")
    assert len(cmds) > 1 and all(c.startswith("fill ") for c in cmds)


def test_fill_relative_to_player():
    cmds = fill_commands(0, 0, 0, 1, 1, 1, "stone", relative_to_player=True)
    assert cmds == ["execute as @p at @s run fill ~0 ~0 ~0 ~1 ~1 ~1 stone"]


def test_fill_rejects_bad_mode():
    with pytest.raises(ValueError):
        fill_commands(0, 0, 0, 1, 1, 1, "stone", mode="banana")


# --------------------------------------------------------------------------- #
# Extra coverage
# --------------------------------------------------------------------------- #
def test_fill_invalid_block_raises():
    with pytest.raises(ValueError):
        fill_commands(0, 0, 0, 1, 1, 1, "Stone")


def test_fill_chunked_each_command_well_formed():
    # 64^3 = 262144 blocks > FILL_CAP(32768) -> must split into >1 chunk.
    cmds = fill_commands(0, 0, 0, 63, 63, 63, "stone")
    assert len(cmds) > 1
    for c in cmds:
        parts = c.split()
        # "fill" + 6 coords + 1 block token = 8 tokens, no mode (default replace).
        assert parts[0] == "fill"
        assert len(parts) == 8
        assert parts[-1] == "stone"
        coords = [int(p) for p in parts[1:7]]  # all must parse as ints
        assert all(0 <= v <= 63 for v in coords)


def test_fill_large_relative_wraps_each_chunk():
    cmds = fill_commands(0, 0, 0, 63, 63, 63, "stone", relative_to_player=True)
    assert len(cmds) > 1
    for c in cmds:
        assert c.startswith("execute as @p at @s run fill ~")


def test_fill_chunks_cover_region_exactly():
    # Sanity: union of chunk volumes equals the whole 64^3 region, no overlap.
    cmds = fill_commands(0, 0, 0, 63, 63, 63, "stone")
    covered = set()
    for c in cmds:
        p = c.split()
        ax, ay, az, bx, by, bz = (int(v) for v in p[1:7])
        for x in range(ax, bx + 1):
            for y in range(ay, by + 1):
                for z in range(az, bz + 1):
                    key = (x, y, z)
                    assert key not in covered  # no overlap
                    covered.add(key)
    assert len(covered) == 64 * 64 * 64
