"""Screen-capture helper for the Minecraft Bedrock vision tools.

TOPOLOGY NOTE (important): in this MCP the Minecraft world renders on the
**iPhone** (Bedrock on iOS connected via ``/connect`` to the PC's WebSocket
server). The PC only runs the MCP server — there is *no Minecraft window on the
PC by default*. The Bedrock WebSocket protocol cannot return screenshots, so to
get pixels the PC must show the game in SOME window first:

  - **Recommended**: run a second Bedrock client on the PC (Windows 10/11 Edition)
    joined to the same world, then capture *that* window; or
  - mirror the iPhone screen to the PC via an AirPlay receiver, then capture the
    mirror window.

This module captures a target window (by title substring), an explicit pixel
region, or a whole monitor, and returns PNG bytes. The heavy capture deps
(``mss``, ``pygetwindow``) are imported lazily *inside* the functions so this
module stays import-safe in CI (matching the repo's pure-helper convention —
nothing here imports a runtime-only dependency at module load).
"""
from __future__ import annotations

import io
import logging
from typing import Optional

log = logging.getLogger(__name__)

# Default max longest-edge for downscaling. Claude vision tokens are geometric
# (ceil(w/28) * ceil(h/28), format-independent); a 1456px longest edge caps an
# image at ~1568 tokens even on Opus 4.7+ (which does NOT auto-downscale a 1080p
# capture and would otherwise bill ~2691 tokens). Kept a config knob because the
# exact per-model threshold tracks Anthropic's vision pipeline.
MAX_EDGE_DEFAULT = 1456


def resolve_region(left: int, top: int, width: int, height: int) -> dict[str, int]:
    """Build an ``mss`` grab dict from a window/region rect. Pure + testable.

    Args:
        left: Left pixel coordinate of the capture rect.
        top: Top pixel coordinate of the capture rect.
        width: Width in pixels (must be > 0).
        height: Height in pixels (must be > 0).

    Returns:
        An ``mss``-compatible monitor dict ``{"left","top","width","height"}``.

    Raises:
        ValueError: If ``width`` or ``height`` is not positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"region width/height must be positive, got {width}x{height}")
    return {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}


def _find_window_rect(title_substring: str) -> dict[str, int]:
    """Resolve the first visible window whose title contains ``title_substring``.

    Args:
        title_substring: Case-insensitive substring matched against window titles
            (e.g. ``"Minecraft"`` for a PC Bedrock client, or the AirPlay receiver
            app's window title for a mirrored iPhone).

    Returns:
        An ``mss`` grab dict for that window's on-screen rect.

    Raises:
        RuntimeError: If ``pygetwindow`` is unavailable or no visible window
            matches; the error lists a few current window titles to help the
            caller pick the right substring.
    """
    try:
        import pygetwindow as gw  # lazy — Windows/macOS only
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "window targeting needs pygetwindow (pip install pygetwindow); "
            f"import failed: {exc}"
        ) from exc

    sub = title_substring.lower()
    candidates = [
        w for w in gw.getAllWindows()
        if (w.title or "") and sub in w.title.lower() and w.width > 0 and w.height > 0
    ]
    if not candidates:
        visible = [w.title for w in gw.getAllWindows() if (w.title or "").strip()][:12]
        raise RuntimeError(
            f"no visible window title contains {title_substring!r}. "
            f"Open windows: {visible}"
        )
    w = candidates[0]
    return resolve_region(w.left, w.top, w.width, w.height)


def _downscale_and_encode(
    rgb: bytes, size: tuple[int, int], max_edge: int, fmt: str
) -> bytes:
    """Downscale a raw RGB grab to ``max_edge`` longest side and encode.

    Uses Pillow (lazy-imported to stay CI-safe). LANCZOS resampling is chosen
    deliberately: Minecraft is high-contrast pixel-art on a regular block grid,
    and BILINEAR/BICUBIC blur block boundaries → the model misreads block types
    → costly re-screenshot recovery loops. Never upscales (``thumbnail`` only
    shrinks). ``fmt`` (jpeg|png) affects WIRE size only — Claude's vision token
    count is geometric and format-independent — so JPEG is the default purely to
    cut MCP message bytes.

    Args:
        rgb: Raw RGB bytes from an ``mss`` grab (``shot.rgb``).
        size: ``(width, height)`` of the grab (``shot.size``).
        max_edge: Longest-edge clamp in pixels; ``<=0`` disables downscaling.
        fmt: ``"jpeg"`` or ``"png"``.

    Returns:
        Encoded image bytes in ``fmt``.

    Raises:
        RuntimeError: If Pillow is unavailable.
    """
    try:
        from PIL import Image  # lazy — keeps module import-safe in CI
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "image downscaling needs Pillow (pip install pillow); "
            f"import failed: {exc}"
        ) from exc

    img = Image.frombytes("RGB", size, rgb)
    w, h = img.size
    if max_edge and max_edge > 0 and max(w, h) > max_edge:
        img.thumbnail((max_edge, max_edge), Image.LANCZOS)  # shrink-only, aspect-safe
        log.info("[vision] downscaled %dx%d -> %dx%d (max_edge=%d)", w, h, img.width, img.height, max_edge)
    else:
        log.info("[vision] capture %dx%d <= max_edge=%d, no downscale", w, h, max_edge)

    buf = io.BytesIO()
    if fmt.lower() in ("jpg", "jpeg"):
        img.save(buf, format="JPEG", quality=80, optimize=True)
    else:
        img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def capture(
    window_title_substring: Optional[str] = None,
    region: Optional[tuple[int, int, int, int]] = None,
    monitor: int = 1,
    max_edge: int = MAX_EDGE_DEFAULT,
    fmt: str = "jpeg",
) -> bytes:
    """Capture the screen/window/region, downscale, and return encoded bytes.

    Precedence: ``window_title_substring`` > ``region`` > full ``monitor``.

    Args:
        window_title_substring: Capture the first visible window whose title
            contains this substring (the game window — a PC Bedrock client or an
            iPhone AirPlay-mirror window).
        region: ``(left, top, width, height)`` pixel rect to capture instead.
        monitor: 1-based monitor index for a full-monitor capture when neither of
            the above is given (``mss`` index; 1 = primary). Falls back to primary
            if out of range.
        max_edge: Longest-edge clamp for the returned image (default
            :data:`MAX_EDGE_DEFAULT`). ``<=0`` disables downscaling. This is the
            single biggest vision-token lever for Opus.
        fmt: ``"jpeg"`` (default) or ``"png"`` — affects wire size only, NOT the
            vision token count Claude bills.

    Returns:
        Encoded image bytes (downscaled to ``max_edge``).

    Raises:
        RuntimeError: If ``mss``/Pillow is unavailable, or window targeting fails.
        ValueError: If ``region`` has a non-positive dimension.
    """
    try:
        import mss  # noqa: F401  (mss.tools no longer needed — Pillow encodes)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "screen capture needs mss (pip install mss); "
            f"import failed: {exc}"
        ) from exc

    with mss.mss() as sct:
        if window_title_substring:
            grab = _find_window_rect(window_title_substring)
        elif region is not None:
            left, top, width, height = region
            grab = resolve_region(left, top, width, height)
        else:
            monitors = sct.monitors  # [0]=all, [1]=primary, ...
            idx = monitor if 0 <= monitor < len(monitors) else 1
            grab = monitors[idx]
        shot = sct.grab(grab)
        return _downscale_and_encode(shot.rgb, shot.size, max_edge, fmt)
