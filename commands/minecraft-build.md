# /minecraft-build — Research-Grounded Minecraft Build (v2 SCAFFOLD — NOT YET WIRED)

> **⚠️ v2 SCAFFOLD — NOT ACTIVE FOR v1.** This skill is a placeholder describing the
> intended research-grounded build flow. It is **not yet wired** to the minecraft MCP
> server and must not be invoked as a working command yet. For v1, drive builds
> directly via the minecraft MCP tools (`connect_status`, `setblock`, `fill`,
> `run_command`) plus `knowledge/building-principles.md`. Remove this banner when the
> flow below is implemented and tested.

Build a structure in Minecraft Bedrock by grounding the design in real-world
architectural references (via Perplexity) before placing a single block.

---

## Input

`$ARGUMENTS` = a natural-language build request, e.g. `a medieval stone watchtower` or
`a cozy 2-story oak cottage`.

---

## Intended flow (v2)

1. **Research the design.** Call `research_query` (Perplexity) for design references for
   the requested build — real-world **proportions**, **material/color palette**, and
   **construction technique** (e.g. "typical proportions and stone palette of a
   medieval watchtower; wall-to-height ratio, crenellation spacing, window placement").
2. **Translate to a parametric plan.** Claude converts the research into a concrete
   parametric plan using `mcp-servers/minecraft/knowledge/building-principles.md` —
   footprint dimensions, palette (primary/secondary/accent/trim), the
   footprint → walls → openings → roof → towers → battlements → details/lighting
   decomposition, and exact block ids/states.
3. **Execute via the minecraft MCP tools.** `connect_status` first (abort if no client
   connected), then emit the plan as ordered `setblock`/`fill` calls. Use
   `relative_to_player` for "build here" (no absolute coords needed); auto-chunk any
   fill over the 32768-block cap.
4. **Verify.** Read each tool's command response to confirm blocks placed (and re-issue
   any failed step), then summarize the finished build.

---

## Not-active checklist (before flipping to active)

- [ ] Wire `research_query` → plan translation step.
- [ ] Map plan steps to live `setblock`/`fill`/`run_command` calls with error isolation.
- [ ] Add a connect-status gate and a dry-run preview mode.
- [ ] Verify against `knowledge/building-principles.md` and `data/blocks.json` ids.
- [ ] Remove the SCAFFOLD banner above.

---

## Token efficiency (MANDATORY — a Max account was burned in 5h without this)

Research-verified policy (`/extended-research`, Anthropic-doc-cited). Follow it on every build:

1. **Prefer `build_structure(spec)` over per-block calls.** Express a whole box/line/ring/helix/rail_run/scatter as ONE run-encoded primitive; the server places it with **no per-block model round-trips** (the dominant token cost). Only fall back to individual `setblock`/`fill` for irregular detail the primitives can't express. `build_structure` is idempotent — on a partial failure it returns `failed_primitive` (index+type), so fix and re-call; no screenshot needed to diagnose.
2. **`verify_blocks` is the PRIMARY correctness channel, not screenshots.** It's cheap text truth (`/testforblock`, up to 1024/call). Use it to confirm builds landed.
3. **Screenshots are token-expensive — EVENT-GATE them.** `take_screenshot` only at *milestones* or for explicit *aesthetic* judgement, never to confirm structure and never per block. It's downscaled to ~1456px (the Opus vision-token lever) but every screenshot still invalidates the messages prompt-cache, so each one is costly. Aim camera with `tp @p <x> <y> <z> facing …` and ensure the PC Bedrock window is foregrounded (minimized → blank capture).
4. **Mechanical turns run with extended thinking OFF.** Reserve Opus extended-thinking for high-level planning and error recovery — not for emitting fill/verify calls (hidden thinking tokens are a silent multiplier).
5. **Re-orient with `build_state()` after a restart** instead of screenshotting — it returns the bounding box + per-block counts + recent ops from the server-side cache.
6. **Prompt caching:** register the minecraft MCP toolset with a `cache_control: {type:"ephemeral"}` marker so the tool schemas + tool-results auto-cache (~90% on re-reads); keep screenshots gated (step 3) or the messages-cache benefit is lost.
