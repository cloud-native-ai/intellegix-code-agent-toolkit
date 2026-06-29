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


# --- _downscale_and_encode: needs Pillow; skips cleanly in CI without it ----- #
def _decode_size(data: bytes) -> tuple[int, int]:
    Image = pytest.importorskip("PIL.Image")
    import io

    return Image.open(io.BytesIO(data)).size


def _rgb(w: int, h: int) -> bytes:
    return bytes([120, 180, 90]) * (w * h)  # solid color, w*h pixels


def test_downscale_clamps_longest_edge_preserving_aspect():
    pytest.importorskip("PIL")
    out = vision._downscale_and_encode(_rgb(1920, 1080), (1920, 1080), max_edge=1456, fmt="png")
    w, h = _decode_size(out)
    assert max(w, h) <= 1456
    # 16:9 preserved within rounding
    assert abs((w / h) - (1920 / 1080)) < 0.02


def test_downscale_never_upscales_small_image():
    pytest.importorskip("PIL")
    out = vision._downscale_and_encode(_rgb(640, 480), (640, 480), max_edge=1456, fmt="png")
    assert _decode_size(out) == (640, 480)


def test_max_edge_zero_disables_downscale():
    pytest.importorskip("PIL")
    out = vision._downscale_and_encode(_rgb(2000, 100), (2000, 100), max_edge=0, fmt="png")
    assert _decode_size(out) == (2000, 100)


def test_jpeg_and_png_both_decode():
    pytest.importorskip("PIL")
    for fmt in ("jpeg", "png"):
        out = vision._downscale_and_encode(_rgb(800, 600), (800, 600), max_edge=1456, fmt=fmt)
        assert _decode_size(out) == (800, 600)
        assert isinstance(out, bytes) and len(out) > 0


def test_downscaled_1080p_stays_under_vision_token_budget():
    """⌈w/28⌉×⌈h/28⌉ for the downscaled image must stay ≤ ~1568 tokens."""
    pytest.importorskip("PIL")
    import math

    out = vision._downscale_and_encode(_rgb(1920, 1080), (1920, 1080), max_edge=1456, fmt="jpeg")
    w, h = _decode_size(out)
    tokens = math.ceil(w / 28) * math.ceil(h / 28)
    assert tokens <= 1568
