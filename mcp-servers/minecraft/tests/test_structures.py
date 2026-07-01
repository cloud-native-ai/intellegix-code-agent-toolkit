"""Tests for structures.py — pure geometry → command expansion (CI-safe)."""
import pytest

import structures


def test_box_delegates_to_fill():
    cmds = structures.primitive_commands(
        {"type": "box", "x1": 0, "y1": 0, "z1": 0, "x2": 2, "y2": 0, "z2": 2, "block": "stone"}
    )
    assert cmds and all("fill" in c for c in cmds)


def test_line_is_contiguous_bresenham():
    cmds = structures.primitive_commands(
        {"type": "line", "x1": 0, "y1": 0, "z1": 0, "x2": 3, "y2": 0, "z2": 0, "block": "oak_planks"}
    )
    # 4 points along x (0..3), one setblock each
    assert len(cmds) == 4
    assert all("setblock" in c for c in cmds)


def test_line_diagonal_point_count():
    cmds = structures.primitive_commands(
        {"type": "line", "x1": 0, "y1": 0, "z1": 0, "x2": 5, "y2": 5, "z2": 0, "block": "stone"}
    )
    assert len(cmds) == 6  # max delta + 1


def test_ring_points_are_within_radius():
    pts = structures._ring_points(0, 0, 0, 5, "xz")
    assert pts
    for (x, y, z) in pts:
        assert y == 0
        r = (x * x + z * z) ** 0.5
        assert 4.0 <= r <= 6.0  # ~radius 5, rounded


def test_helix_rises_one_y_per_step_and_no_dupes():
    cmds = structures.primitive_commands(
        {"type": "helix", "cx": 0, "cy": -60, "cz": 0, "radius": 3, "height": 10, "block": "smooth_quartz"}
    )
    assert len(cmds) == 10  # one block per height step
    # setblock <x> <y> <z> <block> → y is column index 2
    ys = [int(c.split()[2]) for c in cmds if c.startswith("setblock")]
    assert ys == sorted(ys)  # monotonically rising
    assert ys == list(range(-60, -50))  # cy=-60, one step per height


def test_rail_run_powered_lays_bed_plus_rail():
    cmds = structures.primitive_commands(
        {"type": "rail_run", "path": [[0, -60, 0], [1, -60, 0]], "powered": True}
    )
    # each point => redstone_block bed + golden_rail = 2 cmds
    assert len(cmds) == 4
    assert any("redstone_block" in c for c in cmds)
    assert any("golden_rail" in c for c in cmds)


def test_rail_run_unpowered_is_plain_rail():
    cmds = structures.primitive_commands({"type": "rail_run", "path": [[0, 0, 0]], "powered": False})
    assert len(cmds) == 1
    assert "rail" in cmds[0] and "golden_rail" not in cmds[0]


def test_scatter_is_seeded_and_bounded():
    p = {"type": "scatter", "x1": 0, "y1": 0, "z1": 0, "x2": 9, "y2": 0, "z2": 9, "block": "poppy", "count": 5, "seed": 42}
    a = structures.primitive_commands(p)
    b = structures.primitive_commands(p)
    assert a == b  # reproducible
    assert len(a) == 5


def test_scatter_count_capped_to_volume():
    cmds = structures.primitive_commands(
        {"type": "scatter", "x1": 0, "y1": 0, "z1": 0, "x2": 1, "y2": 0, "z2": 0, "block": "stone", "count": 99}
    )
    assert len(cmds) == 2  # only 2 cells exist


def test_unknown_primitive_raises():
    with pytest.raises(ValueError):
        structures.primitive_commands({"type": "pyramid", "block": "stone"})


def test_missing_block_raises():
    with pytest.raises(ValueError):
        structures.primitive_commands({"type": "box", "x1": 0, "y1": 0, "z1": 0, "x2": 1, "y2": 1, "z2": 1})


def test_injection_unsafe_block_raises_via_mcs_build():
    with pytest.raises(ValueError):
        structures.primitive_commands(
            {"type": "line", "x1": 0, "y1": 0, "z1": 0, "x2": 1, "y2": 0, "z2": 0, "block": "stone; kill @a"}
        )


def test_primitive_bbox_box_and_helix():
    assert structures.primitive_bbox(
        {"type": "box", "x1": 0, "y1": 0, "z1": 0, "x2": 5, "y2": 2, "z2": 3}
    ) == ((0, 0, 0), (5, 2, 3))
    assert structures.primitive_bbox(
        {"type": "helix", "cx": 10, "cy": -60, "cz": 10, "radius": 3, "height": 8}
    ) == ((7, -60, 7), (13, -52, 13))
