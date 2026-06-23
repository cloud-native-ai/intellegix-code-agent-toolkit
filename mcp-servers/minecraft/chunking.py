"""Split a large 3D fill region into sub-boxes that respect Bedrock's /fill cap.

Minecraft Bedrock's ``/fill`` command rejects any region larger than
``FILL_CAP`` (32768) blocks. This module computes a grid of axis-aligned,
inclusive sub-boxes that together cover the requested region exactly: their
union is the whole box, they never overlap, and each is within the cap.

Pure standard library -- no third-party imports -- so it runs in CI without
bedrockpy / fastmcp installed.
"""

from __future__ import annotations

FILL_CAP: int = 32768  # Bedrock /fill hard limit, in blocks


def _chunk_spans(length: int, max_span: int) -> list[tuple[int, int]]:
    """Return (start, end) inclusive offset pairs tiling ``[0, length-1]``.

    Each span is at most ``max_span`` long; the final span may be shorter.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + max_span, length) - 1
        spans.append((start, end))
        start = end + 1
    return spans


def chunk_fill(
    x1: int, y1: int, z1: int, x2: int, y2: int, z2: int
) -> list[tuple[int, int, int, int, int, int]]:
    """Split an inclusive 3D box into sub-boxes each <= ``FILL_CAP`` blocks.

    Args:
        x1, y1, z1: One corner of the region (any ordering vs. the other).
        x2, y2, z2: The opposite corner.

    Returns:
        A list of ``(ax, ay, az, bx, by, bz)`` inclusive sub-boxes with
        ``min <= max`` on every axis. Their union equals the normalized input
        box, they do not overlap, and each has a block volume <= ``FILL_CAP``.

    The region is normalized first (reversed corners are handled). If it fits
    within the cap a single box is returned. Otherwise per-axis chunk sizes
    (sx, sy, sz) are chosen so ``sx * sy * sz <= FILL_CAP`` while keeping the
    sub-boxes as large and cubic as possible: each axis span starts at its full
    length, then the currently-largest span is halved (ceiling) repeatedly until
    the product fits under the cap. The grid of sub-boxes is then emitted.
    """
    # Normalize so each axis runs low -> high.
    ax, bx = (x1, x2) if x1 <= x2 else (x2, x1)
    ay, by = (y1, y2) if y1 <= y2 else (y2, y1)
    az, bz = (z1, z2) if z1 <= z2 else (z2, z1)

    len_x = bx - ax + 1
    len_y = by - ay + 1
    len_z = bz - az + 1

    if len_x * len_y * len_z <= FILL_CAP:
        return [(ax, ay, az, bx, by, bz)]

    # Choose per-axis chunk spans. Start full-size, halve the largest axis until
    # the product fits under the cap. Ceiling-halving keeps spans as even as
    # possible (avoids tiny remainder slivers).
    spans = [len_x, len_y, len_z]
    while spans[0] * spans[1] * spans[2] > FILL_CAP:
        largest = max(range(3), key=lambda i: spans[i])
        spans[largest] = (spans[largest] + 1) // 2

    sx, sy, sz = spans

    boxes: list[tuple[int, int, int, int, int, int]] = []
    for ox1, ox2 in _chunk_spans(len_x, sx):
        for oy1, oy2 in _chunk_spans(len_y, sy):
            for oz1, oz2 in _chunk_spans(len_z, sz):
                boxes.append(
                    (
                        ax + ox1,
                        ay + oy1,
                        az + oz1,
                        ax + ox2,
                        ay + oy2,
                        az + oz2,
                    )
                )
    return boxes
