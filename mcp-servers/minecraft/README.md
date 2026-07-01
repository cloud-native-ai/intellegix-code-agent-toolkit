# Minecraft Bedrock MCP Server

A Python [FastMCP](https://github.com/jlowin/fastmcp) server that lets Claude build
inside **Minecraft Bedrock Edition** over a LAN WebSocket connection. It runs a
`bedrockpy` WS listener on the PC; your phone (or any Bedrock client) runs Minecraft,
joins a world with cheats, and `/connect`s to the PC. Claude then drives the world
through MCP tools (`setblock`, `fill`, raw commands).

---

## Install

```bash
pip install -r requirements.txt
```

Python 3.11+ recommended. `bedrockpy` is **pinned to 1.0.1** because `server.py`
couples to its internals (see `requirements.txt` for the rationale) — do not bump it
without re-verifying `server.py`.

---

## Register in Claude

Add the server to your Claude MCP config (same shape as `browser-bridge`):

```json
{
  "mcpServers": {
    "minecraft": {
      "command": "python",
      "args": ["<abs path>/mcp-servers/minecraft/server.py"]
    }
  }
}
```

Replace `<abs path>` with the absolute path to this repo on your machine.

---

## One-time PC setup

### 1. Windows firewall — allow inbound on 8767

The server binds `0.0.0.0:8767`. Open that port for inbound TCP on your **Private**
network profile (run PowerShell as Administrator):

```powershell
New-NetFirewallRule -DisplayName "MC Bedrock WS" -Direction Inbound -Protocol TCP -LocalPort 8767 -Action Allow -Profile Private
```

### 2. Find the PC's LAN IP

```powershell
ipconfig
```

Use the IPv4 address of the adapter on the same WiFi/LAN as your phone (shown below as
`<PC-LAN-IP>`).

---

## Minecraft world setup

- **Enable cheats** when creating/editing the world (commands require cheats).
- In **Settings → turn OFF "Require Encrypted Websockets"** — the bridge speaks plain
  (unencrypted) WS on the LAN. If this is left on, `/connect` will fail.

---

## iOS client setup (mandatory)

On the iPhone/iPad:

- **Settings → Privacy & Security → Local Network → Minecraft = ON.**

This is **required**. Without Local Network permission, iOS silently blocks the
connection and `/connect` fails with no useful error.

---

## In-game connect

Make sure the phone and PC are on the **same WiFi/SSID** with **no AP/client
isolation** (guest networks often isolate clients — use the main network).

Start the MCP server (Claude launches it, or run `python server.py` to test), then in
the Minecraft chat:

```
/connect <PC-LAN-IP>:8767/ws
```

Get `<PC-LAN-IP>` from `ipconfig`. On success the world is now driveable from Claude.

---

## Tools

| Tool | What it does |
|------|--------------|
| `connect_status` | Reports whether a Bedrock client is currently connected to the WS listener. |
| `run_command` | Sends a raw slash-command to the connected world and returns its response. |
| `setblock` | Places a single block at a coordinate (or relative to the player). |
| `fill` | Fills a cuboid region with a block; auto-chunks past the 32768-block per-fill cap. |
| `verify_blocks` | World-state "vision": checks that specific blocks are actually placed (`/testforblock` per entry) — confirms a build matches the plan, no rendering required. |
| `take_screenshot` | Captures a window/region/monitor and returns it to Claude as an image (visual feedback). See **Vision** below. |

**`relative_to_player` note:** `setblock` and `fill` accept a `relative_to_player`
flag so you can build *at the player* without knowing absolute coordinates — the tool
wraps the command in `execute as @p at @s run ... ~ ~ ~ ...`. Use it for "build here".

---

## Vision (giving Claude eyes)

The Bedrock WebSocket protocol **cannot return screenshots** — and in this setup the
world renders on your **iPhone**, not the PC, so there is nothing for the PC to capture
by default. Two channels are provided:

- **`verify_blocks` (recommended primary)** — asks the world for the truth via
  `/testforblock`. Works regardless of where (or whether) the world is rendered, and is
  never stale or obscured by mobile UI. Use it to confirm a build landed: pass a list of
  `{x, y, z, block}` and it reports matched vs. mismatched.

- **`take_screenshot` (actual pixels)** — captures a **window on the PC** that is showing
  the game. Since the game is on the phone, first make the game visible on the PC by
  **either**:
  1. **(Recommended)** running a **second Bedrock client on the PC** (Windows 10/11
     Edition) joined to the same world, then pass `window_title_substring="Minecraft"`; or
  2. **AirPlay-mirroring the iPhone** to the PC with a receiver app, then pass that app's
     window title (e.g. `window_title_substring="LonelyScreen"`).

  You can also pass an explicit `region=(left, top, width, height)` or capture a whole
  `monitor` (1 = primary). Capture deps (`mss`, `pygetwindow`) install from
  `requirements.txt`; they're lazy-imported so the rest of the server (and CI) runs
  without them.

  > **Gotcha:** a screenshot can show the *wrong* thing — a stale mirror, a mobile UI
  > overlay, or an observer client looking the wrong way — while `verify_blocks` stays
  > truthful. Prefer `verify_blocks` for "did it build correctly"; use `take_screenshot`
  > for final aesthetic inspection.

---

## Troubleshooting

**"Could not connect" / `/connect` does nothing** — check, in order:

1. **iOS Local Network permission** is ON for Minecraft (most common cause).
2. **Firewall** inbound rule for TCP 8767 exists (and you're on the Private profile).
3. **Server is running** and bound to `0.0.0.0:8767` (not `127.0.0.1`).
4. **"Require Encrypted Websockets" is OFF** in the world settings.
5. **Same SSID, no AP isolation** — phone and PC on the same (non-guest) network.
6. The IP in `/connect` matches the PC's current `ipconfig` IPv4 (it can change on
   DHCP renewal).
