"""Pure command-string builders for Minecraft Bedrock /setblock and /fill.

These functions turn structured arguments into the exact Bedrock command text
the WebSocket API expects. They are deliberately *pure* — no I/O, no network —
and import ONLY :mod:`blockstate` and :mod:`chunking` (both stdlib-only), so
they run in CI without ``fastmcp`` / ``bedrockpy`` installed and are unit-tested
in isolation.

Two cross-cutting behaviors:

* **Relative coordinates** — when ``relative_to_player`` is True, every
  coordinate is emitted as ``~<v>`` (e.g. ``~1``, ``~-2``) and the whole command
  is wrapped in ``execute as @p at @s run <cmd>`` so it runs at the player's
  position. For chunked fills, EACH chunk command is wrapped independently.
* **Block serialization** — block name + optional states are serialized via
  :func:`blockstate.serialize_block`, which validates everything for
  injection-safety (so an uppercase or hostile block name raises ``ValueError``).
"""

from __future__ import annotations

import blockstate
import chunking

#: Valid ``/fill`` modes. ``replace`` is Bedrock's default and is OMITTED from
#: the emitted command; every other mode is appended as a trailing token.
VALID_FILL_MODES: set[str] = {"replace", "hollow", "outline", "keep", "destroy"}

#: Prefix that wraps a command so it executes at the connected player's position.
_EXECUTE_PREFIX = "execute as @p at @s run "


def _coord(value: int, relative: bool) -> str:
    """Format a single coordinate, with a ``~`` prefix when relative.

    Args:
        value: The integer coordinate.
        relative: If True, emit ``~<value>`` (no space, so ``-1`` -> ``~-1``);
            otherwise emit the plain decimal string.

    Returns:
        The formatted coordinate token.
    """
    return f"~{value}" if relative else str(value)


def setblock_command(
    x: int,
    y: int,
    z: int,
    block: str,
    states: dict | None = None,
    relative_to_player: bool = False,
) -> str:
    """Build a Bedrock ``/setblock`` command string.

    Args:
        x, y, z: Target block coordinates (absolute, or relative to the player
            when ``relative_to_player`` is True).
        block: Block id, e.g. ``stone`` or ``minecraft:oak_stairs``. Validated by
            :func:`blockstate.serialize_block`.
        states: Optional block-state mapping (e.g. ``{"weirdo_direction": 0}``).
        relative_to_player: When True, coordinates become ``~`` relative and the
            command is wrapped in ``execute as @p at @s run ...``.

    Returns:
        The command string, e.g. ``setblock 1 2 3 stone`` or
        ``execute as @p at @s run setblock ~1 ~2 ~3 stone``.

    Raises:
        ValueError: If the block name or any state is injection-unsafe.
    """
    serialized = blockstate.serialize_block(block, states)
    coords = (
        f"{_coord(x, relative_to_player)} "
        f"{_coord(y, relative_to_player)} "
        f"{_coord(z, relative_to_player)}"
    )
    cmd = f"setblock {coords} {serialized}"
    return f"{_EXECUTE_PREFIX}{cmd}" if relative_to_player else cmd


def fill_commands(
    x1: int,
    y1: int,
    z1: int,
    x2: int,
    y2: int,
    z2: int,
    block: str,
    states: dict | None = None,
    mode: str = "replace",
    relative_to_player: bool = False,
) -> list[str]:
    """Build one or more Bedrock ``/fill`` command strings for a region.

    The region is split into sub-boxes that each respect Bedrock's fill cap via
    :func:`chunking.chunk_fill`, so a region larger than the cap yields multiple
    commands. The default ``replace`` mode is omitted from the command (it is
    Bedrock's default); any other mode is appended as a trailing token.

    Args:
        x1, y1, z1: One corner of the region (absolute, or player-relative when
            ``relative_to_player`` is True).
        x2, y2, z2: The opposite corner.
        block: Block id. Validated by :func:`blockstate.serialize_block`.
        states: Optional block-state mapping (e.g. ``{"pillar_axis": "y"}``).
        mode: Fill mode — one of :data:`VALID_FILL_MODES`. ``replace`` is the
            default and is omitted from the output.
        relative_to_player: When True, EACH chunk command uses ``~`` coordinates
            and is wrapped in ``execute as @p at @s run ...``.

    Returns:
        A list of command strings whose covered regions together equal the
        requested box (one entry when it fits under the fill cap, more when it
        must be chunked).

    Raises:
        ValueError: If ``mode`` is not a valid fill mode, or the block/states
            are injection-unsafe.
    """
    if mode not in VALID_FILL_MODES:
        raise ValueError(
            f"Invalid fill mode {mode!r}; expected one of {sorted(VALID_FILL_MODES)}"
        )

    serialized = blockstate.serialize_block(block, states)
    mode_suffix = "" if mode == "replace" else f" {mode}"

    commands: list[str] = []
    for ax, ay, az, bx, by, bz in chunking.chunk_fill(x1, y1, z1, x2, y2, z2):
        coords = (
            f"{_coord(ax, relative_to_player)} "
            f"{_coord(ay, relative_to_player)} "
            f"{_coord(az, relative_to_player)} "
            f"{_coord(bx, relative_to_player)} "
            f"{_coord(by, relative_to_player)} "
            f"{_coord(bz, relative_to_player)}"
        )
        cmd = f"fill {coords} {serialized}{mode_suffix}"
        commands.append(f"{_EXECUTE_PREFIX}{cmd}" if relative_to_player else cmd)

    return commands
