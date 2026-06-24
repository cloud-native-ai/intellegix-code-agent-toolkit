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

**`relative_to_player` note:** `setblock` and `fill` accept a `relative_to_player`
flag so you can build *at the player* without knowing absolute coordinates — the tool
wraps the command in `execute as @p at @s run ... ~ ~ ~ ...`. Use it for "build here".

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
