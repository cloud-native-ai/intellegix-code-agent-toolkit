import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chunking import chunk_fill, FILL_CAP


def _vol(b):
    return (abs(b[3] - b[0]) + 1) * (abs(b[4] - b[1]) + 1) * (abs(b[5] - b[2]) + 1)


def _within_bounds(b, lo, hi):
    """Every chunk lies fully inside the normalized bounding box."""
    return (
        lo[0] <= b[0] <= b[3] <= hi[0]
        and lo[1] <= b[1] <= b[4] <= hi[1]
        and lo[2] <= b[2] <= b[5] <= hi[2]
    )


def test_cap_constant():
    assert FILL_CAP == 32768


def test_small_box_single_chunk():
    boxes = chunk_fill(0, 0, 0, 9, 9, 9)  # 1000 blocks
    assert len(boxes) == 1 and boxes[0] == (0, 0, 0, 9, 9, 9)


def test_exactly_cap_single_chunk():
    boxes = chunk_fill(0, 0, 0, 31, 31, 31)  # 32768 exactly
    assert len(boxes) == 1


def test_large_box_splits_under_cap_full_coverage():
    boxes = chunk_fill(0, 0, 0, 63, 63, 63)  # 262144 blocks
    assert len(boxes) > 1
    assert all(_vol(b) <= FILL_CAP for b in boxes)
    assert sum(_vol(b) for b in boxes) == 64 * 64 * 64  # full coverage, no overlap


def test_handles_reversed_coords():
    boxes = chunk_fill(9, 9, 9, 0, 0, 0)
    assert all(_vol(b) <= FILL_CAP for b in boxes)
    assert sum(_vol(b) for b in boxes) == 1000


def test_flat_wide_region():
    boxes = chunk_fill(0, 0, 0, 199, 0, 199)  # 40000 blocks, thin slab
    assert all(_vol(b) <= FILL_CAP for b in boxes)
    assert sum(_vol(b) for b in boxes) == 200 * 1 * 200


def test_100x40x100_region_no_overlap():
    """A 100x40x100 box: every chunk <=cap, summed volume == bounding volume
    (proves full coverage AND no overlap together with the within-bounds check),
    and every chunk lies within bounds."""
    lo = (0, 0, 0)
    hi = (99, 39, 99)
    boxes = chunk_fill(0, 0, 0, 99, 39, 99)  # 400000 blocks
    bounding_vol = 100 * 40 * 100
    assert all(_vol(b) <= FILL_CAP for b in boxes)
    assert all(_within_bounds(b, lo, hi) for b in boxes)
    # If all chunks are inside the bounds AND their volumes sum to the bounding
    # volume, they cannot overlap (overlap would push the sum above the bound)
    # and must fully cover it (a gap would push the sum below).
    assert sum(_vol(b) for b in boxes) == bounding_vol
    assert len(boxes) > 1


def test_negative_coordinates_sum_correctly():
    # -10,-64,-10 to 10,-50,10 -> 21 x 15 x 21 = 6615 blocks
    boxes = chunk_fill(-10, -64, -10, 10, -50, 10)
    assert all(_vol(b) <= FILL_CAP for b in boxes)
    assert sum(_vol(b) for b in boxes) == 21 * 15 * 21


def test_negative_coords_large_no_overlap():
    # large negative-spanning box that must split
    lo = (-50, -64, -50)
    hi = (50, 5, 50)  # 101 x 70 x 101 = 714070 blocks
    boxes = chunk_fill(-50, -64, -50, 50, 5, 50)
    assert all(_vol(b) <= FILL_CAP for b in boxes)
    assert all(_within_bounds(b, lo, hi) for b in boxes)
    assert sum(_vol(b) for b in boxes) == 101 * 70 * 101
