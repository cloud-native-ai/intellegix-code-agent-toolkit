# Bedrock Building Principles — AI Build Assistant Reference

> **TODO:** enrich/verify against Microsoft Bedrock docs via the deferred Perplexity
> Run 3 (e.g. exact slab states, current stair-state names, mossy/cracked variants).
> Treat every block-state detail below as "best-known", not authoritative, until that
> pass lands.

A precise, rule-first reference for an AI assistant building in Minecraft **Bedrock
Edition** via the minecraft MCP tools (`setblock`, `fill`, `run_command`). Prefer rules
and recipes over prose. Block names referenced here are validated against
`data/blocks.json` in this server.

---

## 1. Command toolkit

### `/fill`
`/fill <from x y z> <to x y z> <block> [mode]`

| Mode | Effect |
|------|--------|
| `replace` | (default) replace every block in the region. Optional trailing filter block: `replace <oldBlock>` only swaps that block. |
| `hollow` | fill the outer shell, set the **interior to air** (makes a hollow box). |
| `outline` | fill only the outer shell, **leave the interior untouched**. |
| `keep` | only fill positions that are currently **air** (never overwrite existing blocks). |
| `destroy` | replace blocks and **drop** the old blocks as items (plays break effects). |

- **32768-block per-fill cap:** Bedrock rejects any single `/fill` whose volume exceeds
  32768 blocks (`32 × 32 × 32`). The `fill` MCP tool **auto-chunks** larger regions
  into compliant sub-cuboids — but when emitting raw `/fill` via `run_command`, keep
  each call under the cap yourself.

### `/setblock`
`/setblock <x y z> <block> [mode]` — places one block. `mode` is `replace` (default),
`keep`, or `destroy`. Use for single accents, lights, doors, detail work.

### Block-state syntax
`block["state"=value]` — **no space** before the `[`, states quoted.
- Correct: `minecraft:oak_stairs["weirdo_direction"=2,"upside_down_bit"=0]`
- Wrong: `oak_stairs ["weirdo_direction"=2]` (space breaks it).
- Multiple states are comma-separated inside the one bracket.

### `/clone` vs `/structure` vs `/fill`
- `/fill` — solid/hollow geometric regions of **one** block. Walls, floors, frames.
- `/clone <begin> <end> <dest> [maskMode] [cloneMode]` — duplicate an **existing**
  built region elsewhere (repeat a window module, mirror a tower). Use when a detailed
  unit already exists in-world and you want copies.
- `/structure save|load` — save a named structure then stamp it repeatedly; best for
  reusable prefab modules across sessions (a fully-detailed turret, a stair set).
  Prefer over many manual `setblock`s when the same complex unit recurs.

---

## 2. Relative positioning

- **Build at the player** (the tools' `relative_to_player` flag): wrap commands as
  `execute as @p at @s run <... ~ ~ ~ ...>` so coordinates anchor on the player's
  position. This is how "build here" works without absolute coords.
- **Tilde `~`** = relative to the execution position (`~ ~ ~` = here; `~5 ~ ~-3` =
  +5 X, same Y, −3 Z from here).
- **Caret `^`** = relative to the execution **facing** direction (left/up/forward).
  Rarely needed for static builds; `~` is the workhorse.

---

## 3. Limits & coordinates

- **Y range (overworld):** −64 (bedrock floor) to 319 (build ceiling). Stay within.
- **Axes:** **X+ = east**, **X− = west**; **Z+ = south**, **Z− = north**; **Y+ = up**.
- A footprint laid on X (width) × Z (depth) and raised on Y (height) is the standard
  mental model: pick a corner, extend +X / +Z / +Y.

---

## 4. Bedrock gotchas

- **Powered rail id is `golden_rail`**, not `powered_rail` (that's Java).
- **No space before `[`** in block-state syntax (repeat of §1, common mistake).
- **One command per chat line** when issuing manually — Bedrock has no `;` chaining.
  (The MCP `fill` tool batches internally, but raw `run_command` is one command each.)
- **Color is part of the block id**, not a state: `pink_wool`, `pink_stained_glass`,
  `lime_concrete` — there is no `wool["color"=...]` in modern Bedrock for most colored
  blocks.
- **Stairs use `weirdo_direction` + `upside_down_bit`**, not Java's `facing`/`half`.
- **`grass_block`** is the dirt-with-grass block; plain `grass`/`short_grass` is the
  plant. **`stone_bricks`** is plural; mossy/cracked/chiseled are separate ids, not
  states.

---

## 5. Professional build rules

These separate a "box" from a build that reads as architecture:

1. **No flat single-texture walls.** Break large faces with **depth/relief**: recess
   windows, add `outline`/inset bands, pop pillars proud of the wall plane, add a
   trim course at top and bottom.
2. **Palette = primary + secondary + accent + trim.** Pick four roles and stick to
   them. Good pairings:
   - `stone_bricks` (primary) + `deepslate` (secondary) + `polished_andesite`
     (accent) + dark trim → a clean grey castle/keep palette.
   - `oak_planks` (primary) + `spruce_planks`/`spruce_log` (trim) + `glass` (windows)
     + `stone` foundation → a warm cottage palette.
3. **Proportion & scale (human-scale).** Doorways 1 wide × 2 tall (min); windows start
   ~1 block above floor; ceilings ≥3 interior height so the space isn't claustrophobic.
4. **Pitched roofs via stairs.** Step `*_stairs` upward/inward course by course rather
   than a flat slab lid — gables and ridges read far better than a flat roof.
5. **Detailing.** Corner **pillars** (full-block or log), **window depth** (set the
   glass one block back from the wall face), horizontal **banding** (a contrasting
   course every few rows), and **trim** along eaves and base.
6. **Interior lighting** to suppress mob spawns: `glowstone`, `sea_lantern`, `torch`,
   `lantern`. Keep interior light level up so nothing spawns inside.
7. **Foundations & landscaping** so builds don't float: extend the base one block
   below grade, skirt with `stone`/`cobblestone`, blend into terrain with `grass_block`
   /`dirt` — never leave a structure hovering on a flat plate.
8. **Symmetry vs intentional asymmetry.** Default to symmetry for keeps/temples; use
   *deliberate* asymmetry (an off-center tower, an added wing) for organic builds — but
   make it look chosen, not accidental.

---

## 6. Parametric decomposition recipe

General order for any structure — each step maps to `fill`/`setblock`:

**footprint → walls → openings → roof → towers → battlements → details/lighting**

### Castle (ordered)
1. **Footprint:** `fill` a flat foundation slab (primary stone), one course below grade.
2. **Walls:** `fill ... outline` (or `hollow` for a closed keep) the perimeter up to
   wall height (e.g. 6–8) using primary; band with a secondary course mid-height.
3. **Openings:** carve a gatehouse (`fill` air for the arch), `setblock` `oak_door`
   pair; recess arrow-slit windows (1×2 air insets), glass optional.
4. **Roof:** flat fighting platform for the curtain wall; pitched `*_stairs` roofs on
   inner halls.
5. **Towers:** at each corner, `fill outline` a taller cylinder/square of primary;
   `/clone` or `/structure` one finished tower to the other corners.
6. **Battlements:** crenellations along wall tops — alternate `setblock` block / air
   in a `_._._` merlon pattern (or `fill` then carve every other top block to air).
7. **Details/lighting:** corner pillars (`deepslate` accent), trim course at the
   parapet, `glowstone`/`sea_lantern` along walkways and gate, `torch` on tower faces.

### House (ordered)
1. **Footprint:** `fill` a `stone`/`cobblestone` foundation pad (e.g. 9×7), one below
   grade.
2. **Walls:** `fill outline` the perimeter in `oak_planks` to height ~4; add
   `oak_log`/`spruce_log` corner pillars via `setblock` (pillar_axis = `y`).
3. **Openings:** front `oak_door` (1×2 air + door); windows as 1×1 or 2×1 `glass`
   set one block back from the face (window depth).
4. **Roof:** pitched gable with `oak_stairs` (or `spruce_stairs`) stepping up to a
   ridge; fill the gable ends with planks.
5. **Towers:** (optional) a small stair-turret or chimney of `stone`/`cobblestone`.
6. **Battlements:** N/A for a house — substitute a trim/eave course of stairs/slabs.
7. **Details/lighting:** spruce trim band under the eave, flower-box accents,
   interior `lantern`/`torch`, a `stone` path of `grass_block` landscaping out front.

### Tower (ordered)
1. **Footprint:** small square/circle base (e.g. 5×5) on a `stone` foundation.
2. **Walls:** `fill outline` straight up to height (e.g. 12) in primary; band every
   4th course with secondary.
3. **Openings:** spiral-stagger arrow-slit windows up the shaft (1×2 air insets).
4. **Roof:** conical/pyramidal cap with `*_stairs` stepping inward course by course.
5. **Towers:** N/A (this *is* the tower) — optionally corbel the top out one block for
   a machicolation overhang.
6. **Battlements:** crenellate the parapet (merlon/air alternation) before the roof
   cap.
7. **Details/lighting:** `deepslate` accent base course, `sea_lantern`/`glowstone` at
   the top room and a beacon-style light, `torch` ring under the overhang.
