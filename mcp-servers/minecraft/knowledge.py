"""Bedrock building knowledge base.

Pure-stdlib loader for the Minecraft Bedrock block reference (``data/blocks.json``)
plus the world/build limit constants used by the MCP server. This module has ZERO
third-party dependencies so it imports cleanly in CI without bedrockpy/fastmcp.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

#: Lowest buildable Y coordinate in a modern Bedrock Overworld.
Y_MIN: int = -64
#: Highest buildable Y coordinate in a modern Bedrock Overworld.
Y_MAX: int = 319
#: Maximum number of blocks a single fill operation may affect.
FILL_CAP: int = 32768

_BLOCKS_PATH = os.path.join(os.path.dirname(__file__), "data", "blocks.json")
_BLOCKS_CACHE: Dict[str, Any] | None = None


def load_blocks() -> Dict[str, Any]:
    """Load and return the Bedrock block reference.

    Reads ``data/blocks.json`` relative to this file (so it works regardless of
    the current working directory) and returns the parsed mapping of block
    short-name -> ``{"id", "states", "notes"}``. The result is cached after the
    first call.

    Returns:
        A dict mapping block short-name to its reference entry.
    """
    global _BLOCKS_CACHE
    if _BLOCKS_CACHE is None:
        with open(_BLOCKS_PATH, encoding="utf-8") as fh:
            _BLOCKS_CACHE = json.load(fh)
    return _BLOCKS_CACHE


def is_known_block(name: str) -> bool:
    """Return whether ``name`` is a known Bedrock block.

    A leading ``minecraft:`` namespace prefix is tolerated and stripped before
    lookup, so both ``"stone"`` and ``"minecraft:stone"`` resolve.

    Args:
        name: Block short-name, optionally namespaced.

    Returns:
        True if the (de-namespaced) name is present in the block reference.
    """
    key = name[len("minecraft:"):] if name.startswith("minecraft:") else name
    return key in load_blocks()
