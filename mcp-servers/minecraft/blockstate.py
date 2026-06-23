"""Pure-Python Bedrock block-state serialization (stdlib-only).

Turns a block name and optional state dict into the exact Minecraft Bedrock
command syntax ``block["state"=value]`` (NO space before the ``[``).

This string is injected into a live Bedrock command, so block names and state
keys are strictly validated to be injection-safe.
"""

import re

_BLOCK_RE = re.compile(r"^(minecraft:)?[a-z0-9_]+$")
_STATE_KEY_RE = re.compile(r"^[a-z0-9_:]+$")


def _serialize_value(value: bool | int | str) -> str:
    """Serialize a single state value to Bedrock syntax.

    bool -> ``true``/``false`` (lowercase), int -> bare digits, str -> double-quoted.
    Note: ``bool`` is checked before ``int`` because ``bool`` is a subclass of ``int``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    raise ValueError(f"Unsupported state value type: {type(value).__name__}")


def serialize_block(block: str, states: dict | None = None) -> str:
    """Serialize a block name plus optional states into Bedrock command syntax.

    Args:
        block: The block id, e.g. ``stone`` or ``minecraft:oak_stairs``. Must match
            ``^(minecraft:)?[a-z0-9_]+$`` (lowercase, injection-safe).
        states: Optional mapping of state name -> value. Values may be bool, int,
            or str. State keys must match ``^[a-z0-9_:]+$`` (colon allowed for the
            ``minecraft:`` prefix). Falsy/empty -> block is returned unchanged.

    Returns:
        Either ``block`` unchanged, or ``block["k1"=v1,"k2"=v2,...]`` with keys
        sorted for determinism, NO space before ``[``, each key double-quoted,
        bools lowercased, ints bare, and strings double-quoted.

    Raises:
        ValueError: If the block name or any state key fails validation.
    """
    if not _BLOCK_RE.match(block):
        raise ValueError(f"Invalid block name (injection-unsafe): {block!r}")

    if not states:
        return block

    parts = []
    for key in sorted(states):
        if not _STATE_KEY_RE.match(key):
            raise ValueError(f"Invalid state key (injection-unsafe): {key!r}")
        parts.append(f'"{key}"={_serialize_value(states[key])}')

    return f"{block}[{','.join(parts)}]"
