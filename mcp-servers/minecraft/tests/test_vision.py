"""Tests for vision.py — only the pure, dependency-free parts.

vision.py lazy-imports mss/pygetwindow inside its functions, so importing the
module (and testing resolve_region) needs no display and no capture deps — it
stays green in CI exactly like the other pure helpers in this repo.
"""
import pytest

import vision


def test_resolve_region_builds_mss_dict():
    # Arrange / Act
    grab = vision.resolve_region(100, 200, 640, 480)
    # Assert
    assert grab == {"left": 100, "top": 200, "width": 640, "height": 480}


def test_resolve_region_coerces_to_int():
    grab = vision.resolve_region(1.0, 2.0, 3.0, 4.0)  # type: ignore[arg-type]
    assert grab == {"left": 1, "top": 2, "width": 3, "height": 4}
    assert all(isinstance(v, int) for v in grab.values())


@pytest.mark.parametrize("w,h", [(0, 10), (10, 0), (-1, 10), (10, -5)])
def test_resolve_region_rejects_nonpositive_dims(w, h):
    with pytest.raises(ValueError):
        vision.resolve_region(0, 0, w, h)
