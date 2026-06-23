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
