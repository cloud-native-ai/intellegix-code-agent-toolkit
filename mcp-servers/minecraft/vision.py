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

from typing import Optional


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


def capture(
    window_title_substring: Optional[str] = None,
    region: Optional[tuple[int, int, int, int]] = None,
    monitor: int = 1,
) -> bytes:
    """Capture the screen/window/region and return PNG bytes.

    Precedence: ``window_title_substring`` > ``region`` > full ``monitor``.

    Args:
        window_title_substring: Capture the first visible window whose title
            contains this substring (the game window — a PC Bedrock client or an
            iPhone AirPlay-mirror window).
        region: ``(left, top, width, height)`` pixel rect to capture instead.
        monitor: 1-based monitor index for a full-monitor capture when neither of
            the above is given (``mss`` index; 1 = primary). Falls back to primary
            if out of range.

    Returns:
        PNG-encoded image bytes.

    Raises:
        RuntimeError: If ``mss`` is unavailable, or window targeting fails.
        ValueError: If ``region`` has a non-positive dimension.
    """
    try:
        import mss
        import mss.tools
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
        # mss ships its own PNG encoder — avoids a Pillow dependency.
        return mss.tools.to_png(shot.rgb, shot.size)
