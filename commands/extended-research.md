# /extended-research — Exhaustive Multi-Pass Artifact Verification via Perplexity

Runs 5–40 iterative `research_query` passes against any artifact (architectural blueprint, implementation plan, debugging trace, refactor proposal, single function) and stops when iterations converge — no new findings, no new contradictions, no new options worth exploring. Produces a definitive verdict + exhaustive list of every gotcha, gap, and flaw, with the most-optimal option chosen per finding.

**CRITICAL — PLAN SYNTHESIS + VERIFICATION IS MANDATORY ON TERMINAL RUNS. After the runner completes with verdict `CONVERGED`, `CAP-HIT`, or `STRUCTURAL-UNRESOLVABLE`, you MUST chain Steps 7 → 8 → 9 below: enter plan mode, synthesize a two-tier plan from `report.md`, critique that plan via a second `research_query` call, revise once if needed, then call `ExitPlanMode`. NEVER call `ExitPlanMode` without completing Step 8. NEVER stop after the Step 6 chat summary unless one of the documented guards fires (INTERRUPTED, zero-findings short-circuit). This applies regardless of plan size or apparent completeness of the run. NO EXCEPTIONS.**

## Risk Register

Known limitations of this skill, surfaced by the E2E run on 2026-05-27 against this
file's own plan-synthesis tail (recursive self-verification, 8 passes, CAP-HIT
verdict). Each item below was consciously deferred during implementation — the
mitigations listed are partial and DO NOT close the underlying gap.

| ID | Severity | Status | Notes |
|---|---|---|---|
| F002 | HIGH | Partial mitigation | Step 6.6 second-invocation guard uses filename heuristics. Strict Python regex (added 2026-05-27) eliminates false positives from arbitrary plan-file names. Does NOT solve the underlying gap: no formal approval-state model exists, so "unapproved" still cannot be determined reliably. Full resolution requires plan-file frontmatter with `approved: true\|false` or a sidecar state file — DEFERRED, exceeds skill-file-only scope. |
| F003 | HIGH | Partial mitigation | Step 7.1 stale-context read is shallow (8 files × 200 lines, regex extraction). `[FILE NOT FOUND]` tagging (added 2026-05-27) catches paths that definitively don't exist. Does NOT catch silently-fabricated-but-plausible paths. Full resolution requires content-hash verification of the artifact's referenced files or mandatory re-index — DEFERRED, requires a Python helper. |
| F006 | LOW | Deferred | No regression test scaffolding for the plan-synthesis tail. Future edits to report.md format or runner output could silently break Steps 6.5/7.2 without detection. Tracked separately; requires test infra outside skill-file scope. |
| Runner-FRESH_OBSERVER-collision | MED | Mitigated by user-side advice | `extended_research_runner.py:select_next_pass_type` schedules FRESH_OBSERVER (rail 4: passes 8/14/20/...) BEFORE reserving FINAL_VERDICT at `pass == max_passes` (rail 1). At `max_passes=8` these collide, FRESH_OBSERVER wins, no FINAL_VERDICT runs. Mitigations: (a) `# TODO(F-runner)` comment in `extended_research_runner.py` at the pass-scheduling logic; (b) recommend `--max-passes >= 9` for runs needing FINAL_VERDICT. Full fix requires reordering safety rails in the runner — outside skill-file scope. |

When invoking this skill, if a user-visible issue traces to one of these IDs,
reference the row above rather than re-deriving the limitation.

**Sequential & agentic (2026-05-19):** Passes run strictly one-at-a-time. Each pass sees a running narrative brief synthesizing all prior passes, and each pass (except DECOMPOSE and FINAL_VERDICT) recommends its own next move via the `recommended_next_pass` field — choosing among TARGETED_PROBE, EXPLORATORY_BRANCH, BLUEPRINT, GUIDANCE, ADVERSARIAL, INTEGRATION, or CRITIQUE. The orchestrator honors that recommendation unless safety rails (AND-gate convergence, max_passes cap, adversarial floor, postmortem trigger, fresh-observer schedule) override.

**Design rationale (5 design passes, 2026-05-18):** Pass 1 draft → Pass 2 adversarial critique (9 bugs) → Pass 3 refined v2 → Pass 4 final critique (8 more bugs) → Pass 5 conditional approval. All 17 bugs integrated. Logs at `~/.claude/council-cache/`.

## Usage

```
/extended-research [--max-passes N] [--mode whole|per-phase] [--resume slug] [--min-passes M]
                   [--perplexity-advisory-runner]
<artifact text or path>
```

**Flags:**
- `--max-passes N` — hard cap (overrides dynamic formula). Default: `4 + N_phases + ceil(N_phases/2) + 2`, computed after DECOMPOSE.
- `--mode whole|per-phase` — `per-phase` (default) drills each phase separately; `whole` treats artifact as one unit.
- `--resume <slug>` — resume an interrupted run. Compares artifact SHA-256 — warns on drift.
- `--min-passes M` — floor for short artifacts. Default: 5. Forces ≥M passes even if convergence fires earlier.
- `--perplexity-advisory-runner` *(opt-in, 2026-05-20)* — switches every prompt from rigid JSON schema to `/research-perplexity`'s natural 8-section advisory format (CURRENT STATE / PROGRESS VS PLAN / SCRUTINY / IMMEDIATE NEXT STEPS / BLOCKERS / TECHNICAL DEBT / STRATEGIC RECOMMENDATIONS / RISKS & MITIGATIONS / CODEBASE FIT). Convergence honors the model-emitted `VERDICT:` line (CONVERGED / NEEDS_MORE_PASSES / NEEDS_ADVERSARIAL / TERMINATE_ON_STRUCTURAL_LIMIT) with safety rails (min_passes floor, adversarial floor). **When to use**: any time you want the runner output to read like a /research-perplexity advisory rather than a per-finding JSON tree. **Why it's opt-in**: legacy JSON mode remains the default during validation; switch to advisory mode for new artifacts and verify behavior on a few real runs before flipping it on universally. Phase 4 (report-format rewrite) is deferred — report.md still uses the legacy per-finding tables; the per-pass advisory prose is preserved in `passes.jsonl` and `salvaged-responses.md`.

## What Claude Does When Invoked

### Step 0 — Routing Pre-Flight (MANDATORY, runs before any other step)

Before doing ANY other work, evaluate whether this invocation matches the right tool. Score the FOCUS AREA + compiled session context against the 5 signals below. Each YES = +1 point.

| # | Signal | YES condition |
|---|---|---|
| S1 | Large output | Expected artifact > 2K tokens / 8KB |
| S2 | Multi-component | Touches ≥ 3 interacting components/services |
| S3 | Open trade-off | Multiple defensible answers exist (not a known-right lookup) |
| S4 | High blast radius | Wrong answer = broken architecture for months (auth, schema, contract design, async/distributed boundaries) |
| S5 | Adversarial divergence | A second expert could plausibly disagree with the first answer |

**Routing rules (this tool: /extended-research):**

| Invoked tool | Score | Action |
|---|---|---|
| /research-perplexity | ≥ 3 | AUTO-SWITCH UP to /extended-research |
| /research-perplexity | = 2 | ASK USER: basic or extended? |
| /research-perplexity | ≤ 1 | Proceed as called |
| /extended-research | ≥ 2 | Proceed as called |
| /extended-research | ≤ 1 | ASK USER: switch down to basic? (NEVER auto-down) |

**Asymmetric override:**
- UP-override allowed: if you spot a specific architectural risk not captured by the 5 signals (e.g., touches auth, payments, multi-service state), you MAY force UP-switch. You MUST name the risk in one sentence before proceeding.
- DOWN-override FORBIDDEN: you may NEVER auto-down-switch from /extended-research. Only the user can.

**Visibility (always emit one of these lines before next step):**
- Auto-switch: `[routing] score=N — auto-switching to /OTHER-TOOL (signals: S1, S2, ...)`
- Ask: `[routing] score=N (signals: ...) — basic or extended?` then wait for user reply
- Override UP: `[routing] override UP — risk: {one-sentence risk}. Switching.`
- Proceed: `[routing] score=N — proceeding with /extended-research.`

**On UP-switch from /research-perplexity:** invoke /extended-research workflow starting at its Step 1 (Parse and stage). Pass the original FOCUS AREA as the artifact. Add a one-line note in the DECOMPOSE prompt: `[ROUTED-FROM-BASIC: pre-flight detected this needs multi-pass — DECOMPOSE more aggressively]`. The basic tool never executed its synthesis pass, so there's no prior output to feed.

**On DOWN-switch from /extended-research (user-confirmed only):** when user replies "yes, basic" / "switch down" / equivalent to the ASK prompt, invoke /research-perplexity workflow starting at its Step 1 (Compile Session Context). Pass the original FOCUS AREA. Emit one-line audit notice: `[routing] user confirmed down-switch — using /research-perplexity instead.` Steps 1-9 below do NOT execute.

### Step 1 — Parse and stage

Claude extracts the artifact (everything after flags) and the flag values. If the artifact looks like a file path (`./blueprint.md` or absolute), Claude `Read`s it. Otherwise treats it as inline text.

```
SLUG = first-8-words-sanitized + sha256(artifact)[:8]
WORKDIR = ~/.claude/extended-research-logs/{SLUG}/
```

If `--resume` is set, Claude reads `{WORKDIR}/ledger.json` and compares stored `artifact_hash` vs current artifact SHA-256. **On hash mismatch:** prints both hashes and asks the user to confirm `--force-resume` (continues with DRIFT-WARNING tag on every finding) or restart fresh.

### Step 2 — Bootstrap dependencies

```bash
pip install -q -r ~/.claude/council-automation/requirements.txt 2>&1 | tail -3
```

Pinned: `jsonschema>=4.0`, `filelock>=3.13` (both pure-Python wheels, no admin needed).

### Step 3 — Write artifact + launch runner async

```bash
mkdir -p $WORKDIR
# Write artifact with SHA-256 + ISO timestamp header
python -c "import hashlib,sys,pathlib;t=pathlib.Path(sys.argv[1]).read_text() if pathlib.Path(sys.argv[1]).exists() else sys.argv[1];h=hashlib.sha256(t.encode()).hexdigest();open(sys.argv[2],'w').write(f'HASH:sha256:{h}\nVERSION:1\n---\n{t}')" "$ARTIFACT" "$WORKDIR/artifact.txt"

# Shell out to runner — DETACHED so Claude returns immediately
nohup python ~/.claude/council-automation/extended_research_runner.py \
  --workdir "$WORKDIR" \
  --mode "$MODE" \
  ${MAX_PASSES:+--max-passes $MAX_PASSES} \
  ${MIN_PASSES:+--min-passes $MIN_PASSES} \
  ${RESUME:+--resume} \
  > "$WORKDIR/runner.log" 2>&1 &
echo $! > "$WORKDIR/runner.pid"
```

On Windows-bash where `nohup` may be absent: use `(python ... > runner.log 2>&1 &)` — the subshell + `&` is equivalent.

### Step 4 — Return control to user immediately

Claude prints (and stops, returning the conversation to the user):

```
Extended research started.
  Slug:    {SLUG}
  Workdir: ~/.claude/extended-research-logs/{SLUG}/
  PID:     {PID}
  Status:  Running DECOMPOSE (pass 1)...

Estimated runtime: 5–40 minutes (depends on phase count + convergence).
This is a background job. Continue using Claude normally.

To check status:
  /extended-research-status {SLUG}                 (Claude reads ledger + last_heartbeat_ts)
  tail -f ~/.claude/extended-research-logs/{SLUG}/runner.log    (live stream from terminal)

When done, the runner writes runner.log.done. Ask Claude "is my research run done?" or wait
for the file to appear, then ask Claude to summarize the report.
```

**Claude does NOT poll for completion proactively** (Claude Code has no background-task mechanism). The user must come back and ask, OR run a status check.

### Step 5 — Status check (when user asks)

When the user asks "status on my research run" or runs `/extended-research-status <slug>`:

```bash
cat ~/.claude/extended-research-logs/{SLUG}/ledger.json | python -m json.tool
ls -la ~/.claude/extended-research-logs/{SLUG}/runner.log.done 2>/dev/null && echo "DONE" || echo "RUNNING"
```

Claude reads:
- `passes_completed`, `max_passes`, `adversarial_pass_count`
- `last_heartbeat_ts` — if `now - heartbeat > 5min`, print `⚠️ STALE — runner may be stuck`
- If `runner.log.done` exists, read `report.md` and summarize

### Step 6 — On completion (passive trigger via user)

When `runner.log.done` exists:

1. Claude reads `report.md` in full.
2. Claude summarizes to the user in chat:
   - **Verdict:** `CONVERGED` / `CAP-HIT(N HIGH open)` / `INTERRUPTED`
   - **Termination Reason:** one sentence
   - **Top 3 findings** (by severity, with `STRUCTURAL-UNRESOLVABLE` tagged if applicable)
   - **Recommended option** (highest-scored, per FINAL_VERDICT)
   - **Full report path:** `~/.claude/extended-research-logs/{SLUG}/report.md`
3. **Dispatch:**
   - If `ledger["status"] == "INTERRUPTED"`: append the resume hint
     `Run --resume {SLUG} to continue from pass {interrupted_at_pass + 1}.` and STOP. Do NOT
     proceed to Steps 6.5/6.6/7/8/9 — partial findings do not justify a plan.
   - Otherwise (`CONVERGED`, `CAP-HIT`, `STRUCTURAL-UNRESOLVABLE`): proceed
     immediately to Step 6.5. Do NOT stop after the chat summary.

### Step 6.5 — Structural-health gate + zero-findings short-circuit

Run a single Python block that performs four checks in order:

```bash
python - <<'PY' "$WORKDIR"
import json, re, sys
from pathlib import Path
try:
    import jsonschema
except ImportError:
    print("Step 6.5: jsonschema not installed — run pip install -r ~/.claude/council-automation/requirements.txt")
    sys.exit(2)

workdir = Path(sys.argv[1])
ledger_p = workdir / "ledger.json"
report_p = workdir / "report.md"

# Check 1 — ledger MISSING
if not ledger_p.exists():
    print(f"Step 6.5 FAIL: ledger MISSING — {ledger_p} not found")
    sys.exit(3)

# Check 2 — ledger SCHEMA-VALID
LEDGER_SCHEMA = {
    "type": "object",
    "required": ["status", "findings", "passes_completed"],
    "properties": {
        "status": {"type": "string", "enum": ["COMPLETED", "INTERRUPTED", "ABORTED"]},
        "findings": {"type": "array"},
        "passes_completed": {"type": "integer", "minimum": 1},
    },
}
ledger = json.loads(ledger_p.read_text(encoding="utf-8"))
try:
    jsonschema.validate(ledger, LEDGER_SCHEMA)
except jsonschema.ValidationError as e:
    print(f"Step 6.5 FAIL: ledger SCHEMA-INVALID at {list(e.absolute_path) or 'root'} — {e.message}")
    sys.exit(3)

# Check 3 — report.md has required sections
if not report_p.exists():
    print(f"Step 6.5 FAIL: report MISSING — {report_p} not found")
    sys.exit(3)
report_text = report_p.read_text(encoding="utf-8")
for section in ("## Executive Summary", "Termination Reason"):
    if not re.search(re.escape(section), report_text):
        print(f"Step 6.5 FAIL: report PRESENT but missing required section: {section!r}")
        sys.exit(3)

# Check 4 — count open HIGH/MED (after structural checks pass)
n_high_med = sum(
    1 for f in ledger.get("findings", [])
    if f.get("status") == "OPEN"
    and str(f.get("effective_severity", f.get("severity", ""))).upper() in ("HIGH", "MEDIUM")
)
if not ledger["findings"]:
    print(f"Step 6.5 WARN: ledger SCHEMA-VALID but findings[] empty — verify research depth (verdict={ledger.get('status')})")
    print(f"OPEN_HIGH_MED=0")
else:
    print(f"OPEN_HIGH_MED={n_high_med}")
PY
```

Diagnostic state mapping:
- Exit code 2 (`jsonschema not installed`): abort with install hint; do NOT proceed.
- Exit code 3 (`ledger MISSING` / `ledger SCHEMA-INVALID` / `report MISSING` / `report PRESENT but missing required section`): emit the diagnostic message and STOP. Do NOT enter plan mode. The run artifacts are unsafe for synthesis.
- Stdout `OPEN_HIGH_MED=0` (with or without the WARN line above): emit
  `No open HIGH/MED findings — no implementation plan generated. Verdict: {verdict}.`
  Do NOT enter plan mode. Skip Steps 7–9.
- Stdout `OPEN_HIGH_MED=N` where N ≥ 1: proceed to Step 6.6.

### Step 6.6 — Second-invocation guard

Before entering plan mode, check for stale unapproved extended-research plans.
The pattern must be strict — a permissive shell glob like `*-extended-research-*.md`
matches any plan with "extended-research" in its name (including artifact source
files), producing false positives. Use a Python regex anchored on the actual
timestamp + literal `extended-research-` infix + the current SLUG:

```bash
python - <<'PY' "$SLUG"
import glob, re, sys
from pathlib import Path
slug = sys.argv[1]
pattern = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{4}-extended-research-[A-Za-z0-9_\-]+\.md$"
)
matches = []
for p in glob.glob(str(Path("~/.claude/plans").expanduser() / "*.md")):
    name = Path(p).name
    if pattern.match(name) and slug not in name:
        matches.append(p)
for m in matches[:5]:
    print(m)
PY
```

If any matches print, surface them to the user:

```
Found N actual /extended-research output plans that may be unapproved:
  {list of filenames}
Continuing will create a new plan in plan mode alongside these. If a prior plan
is still awaiting your approval, dismiss it (ExitPlanMode rejected, or move/delete
the file) before proceeding. Reply with "continue" to proceed anyway, or take action
on the prior plans first.
```

Wait for the user to type `continue` (or equivalent ack). Only then proceed to Step 7.
If no matches print, proceed directly to Step 7.

**Known limitation (Risk Register F002):** this guard cannot reliably distinguish
"approved" from "unapproved" plans without an explicit approval-state model. The
strict regex eliminates false positives from arbitrary plan-file names but does
not address the underlying state-tracking gap. See the Risk Register near the top
of this skill file.

### Step 7 — Synthesize Plan (MANDATORY for CONVERGED / CAP-HIT / STRUCTURAL-UNRESOLVABLE)

Call `EnterPlanMode` to enter plan mode.

#### Step 7.1 — Stale-context safety read

Before writing any file-path sub-plans, verify Claude's codebase context is current.
This is critical when the user asks "is my run done?" hours after launching the
runner — Claude's session may have zero prior reads of the relevant codebase.

- Scan `report.md` Findings + `running_brief.md` for file paths and function names.
- For each unique path (up to 8 distinct files), `Read` the file (or `Glob` if the
  reference is a directory hint). Cap reads at ~200 lines per file.
- Skip this only if the artifact's source files were freshly edited earlier this same session.

**Missing-file tagging:** if `Read` returns "file not found" for a path extracted
from `report.md`, do NOT silently proceed with that path in any sub-plan. Instead,
tag every reference to that file in the Tier 2 sub-plan with:

```
[FILE NOT FOUND — verify before execution: {path}]
```

The path may have been renamed, deleted, or fabricated by the research model. The
tag makes the discrepancy visible to the user at plan-review time so they can
correct or skip the affected sub-plan before approving.

**Known limitation (Risk Register F003):** this stale-context read is shallow
(8 files × 200 lines, regex-extracted paths). It cannot fully eliminate the risk
of fabricated paths in sub-plans — only flag the cases where the file definitively
does not exist. See the Risk Register near the top of this skill file.

#### Step 7.2 — Write the two-tier plan

Write to:
```
~/.claude/plans/{YYYY-MM-DD_HHmm}-extended-research-{SLUG}.md
```

Use the SAME `{SLUG}` the runner used — do NOT generate a fresh slug. Pull
`{YYYY-MM-DD_HHmm}` from current time.

**Tier 1 — Master Plan**: one numbered **Phase** per open HIGH/MED finding. Phase
header carries:

- Title
- 1-line goal
- Complexity: S / M / L
- Prerequisite phases (e.g. `prereq: Phase 1`)
- Source finding IDs (e.g. `[F003, F007]`)

Phase ordering: blockers → dependencies → independent work → polish → memory update → commit & push.
Group findings touching the same files into the same Phase.

**Tier 2 — Sub-Plans** (one per Phase):

- Specific files to create/modify (absolute paths)
- Code changes (what, not line-diffs)
- Acceptance criteria — how to verify this phase is done
- Risk mitigations sourced from `report.md`'s `Contradiction Log` + `Recommended Next Actions`
- Dependencies on other phases
- When the source finding has an `Option Comparison` table: cite the selected best option
  by label + score and explain WHY (highest correctness weight, lowest blast radius, etc.)

**STRUCTURAL-UNRESOLVABLE handling:** these findings do NOT become standalone phases.
Embed each as a **Risk Callout** subsection inside the implementation Phase it most
affects:

```
> ⚠️ STRUCTURAL-UNRESOLVABLE: {finding claim}.
> Cannot be fixed by this plan — design around it.
```

A phase that can't be executed is a plan defect; representing it as a risk inside an
executable phase is the correct shape.

**Mandatory final phases** (verbatim from `/research-perplexity` Step 5):

- **Second-to-last Phase: Update project memory** — follow these 6 rules:
  1. MEMORY.md stays under 150 lines — move implementation details to memory/*.md topic files
  2. No duplication between MEMORY.md and CLAUDE.md — behavioral rules belong in CLAUDE.md only
  3. New session-learned patterns go in MEMORY.md; implementation details go to topic files
  4. Delete outdated entries rather than accumulating
  5. If adding a new topic file, add a 1-line entry to the Topic File Index in MEMORY.md
  6. Topic file naming: kebab-case.md
- **Final Phase: Commit & Push** — commit all changes and push to remote.

**Edge case — Empty/minimal report.md (FINAL_VERDICT didn't fire):** if `report.md`
lacks a FINAL_VERDICT section (rare — happens on cap-hit without INTEGRATION pass),
the master plan opens with:

```
## Phase 1: Re-run with higher --max-passes [complexity: S]
Goal: Obtain a complete verdict before implementing.
```

before any synthesis-driven phases. This prevents the user from following a plan
derived from a truncated run without knowing it was truncated.

Cover ALL open HIGH/MED findings. Do NOT filter — unless the long-tail consolidation rule below fires.

**Long-tail consolidation rule (large-output scoping):** when
`count(open HIGH/MED findings) > 8`, the master plan auto-consolidates lower-priority
findings into a single "Deferred lower-priority findings" phase to keep the plan
scannable. Ranking:

1. Primary key: `effective_severity` descending (HIGH before MEDIUM).
2. Secondary key: `findings_history[i].targeted_probe_count` descending (more
   probes = better-supported finding — implies the runner spent more attention
   on it, so it warrants standalone treatment).
3. Tertiary tiebreak: finding ID ascending (stable ordering).

After ranking:

- **Top 8 findings** → standalone detailed Phases (one per finding, as normal).
- **Findings ranked 9 and below** → a single phase titled
  `Phase N: Deferred lower-priority findings (M findings)` with one bullet per
  finding referencing `ledger.json` ID and a one-line summary. No detailed
  sub-plan. The user can later promote any of these to a standalone phase by
  re-running plan synthesis.

If the long-tail-consolidation fires, surface it in the chat summary preceding
EnterPlanMode: `Plan auto-consolidates {M} lower-priority findings (re-run with
--max-passes >= {N+4} if every finding warrants a standalone phase).`

Inputs (all from `~/.claude/extended-research-logs/{SLUG}/`):

- `report.md` — authoritative finding/option/verdict source
- `running_brief.md` — narrative arc of the research (12-entry tail)
- `ledger.json` — `findings[]`, `findings_history[]`, `contradictions[]`, `options[]`

Write the full plan, then proceed to **Step 8**. Do NOT call `ExitPlanMode` yet.

### Step 8 — Verify Plan via Second Perplexity Pass (MANDATORY, NO EXCEPTIONS)

**This is the hard gate before `ExitPlanMode`. NEVER skip it. NEVER call `ExitPlanMode`
without completing this step. Skipping verification is a protocol violation — applies
regardless of plan size, complexity, or apparent correctness.**

#### Step 8.1 — Build the verification query

The Step 8 critique must evaluate the **PLAN**, not re-evaluate the research. The
query payload must include all three sections below:

1. **Plan to Review** — full plan text from Step 7. If total length exceeds ~5000 chars,
   truncate at 5000 with marker `... [{N} additional phases truncated for query size]`.
2. **Research Context** — ~3000-char extract from `report.md`:
   - Executive Summary
   - Termination Reason
   - Top-3 HIGH findings by severity
   - `Option Comparison` table rows for any option referenced in the plan
3. **Codebase Context** — file snippets from Step 7.1's stale-context safety reads.

Combined target: ~8000–9000 chars total query payload.

Query format starts with the **MANDATORY CONTEXT PREAMBLE** block (verbatim from
`research-perplexity.md`):

```
[ENVIRONMENT CONTEXT — READ FIRST]
This project is being developed using Claude Code, Anthropic's official CLI tool for Claude (claude.ai/claude-code). The developer uses a Claude Max subscription and works entirely in the terminal via the `claude` CLI command. Claude Code is an agentic coding assistant that reads/writes files, runs terminal commands, searches codebases, and executes multi-step development tasks autonomously. All code generation, refactoring, debugging, and project management happens through Claude Code's conversation interface — there is no IDE or GUI involved. Responses should account for this workflow: recommend CLI-compatible tools, terminal-based solutions, and approaches that work well with an AI coding agent operating in a command-line environment.
[END ENVIRONMENT CONTEXT]

You are a senior software architect reviewing an implementation plan. Critically evaluate this plan for correctness, completeness, and feasibility.

## Plan to Review
{plan text from Step 7, truncated to 5000 chars if needed}

## Research Context
{~3000-char extract from report.md}

## Codebase Context
{file snippets from Step 7.1}

Please evaluate:
1. LOGICAL ERRORS: Are there contradictions, circular dependencies, or impossible sequences?
2. MISSING EDGE CASES: What scenarios does the plan fail to address?
3. FILE PATH ACCURACY: Do referenced files/paths actually exist in the codebase?
4. DEPENDENCY ORDERING: Are phases ordered correctly given their prerequisites?
5. SCOPE CREEP: Does the plan include unnecessary work beyond what was researched?
6. FEASIBILITY: Are estimated complexities realistic? Are there hidden costs?
7. RISK GAPS: What risks are unmitigated?
8. VERDICT: APPROVED (proceed as-is) or REVISE (with specific changes needed)
```

#### Step 8.2 — Run verification

Call `research_query` MCP tool with:

- `query`: the critique prompt from 8.1
- `includeContext`: `true`

**Submission_lock note:** if the runner is somehow still active (rare — `runner.log.done`
should already exist), the `research_query` call will queue behind `submission_lock`
for up to 240s. This is expected serialization, NOT a deadlock. If Step 8 takes
> 5 minutes, surface a status message: "Waiting for submission_lock — runner may still
be writing to the lock."

#### Step 8.3 — Revise plan (if needed)

- If the critique returns **APPROVED**: proceed to Step 9 as-is.
- If the critique returns **REVISE**: apply revisions per the **scope limiter** below.
  Maximum 1 revision pass — do NOT re-verify after revision.

**Scope limiter (revision):** if the critique flags issues in more than 3 phases,
revise ONLY the flagged phases. Annotate untouched phases with `(unchanged after critique)`.
Rewriting 10–20 phases in one shot loses coherence — targeted revision preserves it.

#### Step 8.4 — Error handling

If `research_query` fails:

1. Retry once.
2. If the retry also fails: append a note to the plan file stating the failure reason
   ("Step 8 verification failed twice — plan not externally critiqued"), then proceed
   to Step 9. The verification attempt is mandatory; its success is not.

### Step 9 — ExitPlanMode + deferred TaskCreate

Only after Step 8.3 completes, call `ExitPlanMode` for user approval.

**Do NOT auto-`TaskCreate` on `ExitPlanMode`.** Extended-research plans can have
10–20 Phases; bulk-creating that many tasks before the user has accepted the plan
pollutes the task tracker if they reject or heavily revise.

**Post-approval execution** (only after the user explicitly approves — types
"approve", "go ahead", "looks good", or begins discussing Phase 1):

- `TaskCreate` one task per Phase from the master plan.
- Set dependencies with `addBlockedBy` matching phase prerequisites.
- Each task description carries the full sub-plan for that phase.
- Begin executing the first unblocked task.

## Sequential Refinement (Agentic Mode)

Each pass is shaped by every prior pass — there is no "first pass repeated 20 times." Two mechanisms make this work:

**1. Running brief.** After every pass the runner appends a 3–6 line narrative summary to `{WORKDIR}/running_brief.md` (mirrored into `ledger["running_brief"]`). The brief carries the *conversation so far*: pass type, headline finding, contradictions raised, open question. The next pass's prompt receives this brief as its primary context — not just a JSON list of finding IDs. Last 12 entries retained, hard-capped at ~2000 tokens. FRESH_OBSERVER passes intentionally skip the brief to stay unanchored.

**2. Perplexity-recommended next moves.** Every pass except DECOMPOSE (Pass 1) and FINAL_VERDICT (terminal) ends by populating a `recommended_next_pass` object:

```json
{
  "pass_type": "BLUEPRINT" | "GUIDANCE" | "EXPLORATORY_BRANCH" | "TARGETED_PROBE" |
               "ADVERSARIAL" | "INTEGRATION" | "CRITIQUE",
  "target_finding_id": "F003" | null,
  "question": "the specific research question to ask next",
  "rationale": "why this is the highest-value next step"
}
```

The orchestrator's `select_next_pass_type` reads this from the last completed pass and uses it for the next iteration. Safety rails come first, in order:

1. **Final-pass reservations** — `max_passes - 1` → INTEGRATION; `max_passes` → FINAL_VERDICT
2. **Adversarial deficit** — convergence math demands at least `ceil(N/2)` adversarial passes; if blocked only by this, ADVERSARIAL is forced
3. **POSTMORTEM trigger** — fires once when every HIGH/MEDIUM open finding has ≥1 TARGETED_PROBE
4. **FRESH_OBSERVER schedule** — passes 8, 14, 20, ... (anti-anchoring)
5. **Agentic recommendation** — if no rail above fired and the last pass's recommendation is valid, the runner honors it. Logged as `[AGENTIC] honoring Perplexity recommendation: {type} — {rationale}`
6. **Deterministic fallback** — TARGETED_PROBE on highest-severity open finding with fewest prior probes

If a recommendation references a `target_finding_id` that no longer exists or is closed, the runner falls back to deterministic targeting for that pass. Invalid pass_types are logged and ignored.

## The 12 Pass Types (executed by the runner)

The runner orchestrates these. Claude doesn't run them inline — but here's what each does so the user understands what their run is doing:

| Pass | Type | When fires | Purpose |
|---|---|---|---|
| 1 | DECOMPOSE | Always first | Break artifact into N phases + line_start/line_end ranges. Emits one HIGH finding per phase (central question). |
| 2 | CRITIQUE | After DECOMPOSE | Search literature for gaps in each phase's claims. Produces findings + contradictions. |
| 3 | ADVERSARIAL | After CRITIQUE | Hostile expert mode. Strongest possible attacks against the artifact. |
| 4 | OPTIONS_SWEEP | After ADVERSARIAL | Enumerate 3–6 distinct solution paths for the central problem. |
| 5+ | TARGETED_PROBE | Recommended or default fallback | One pass per open HIGH/MEDIUM finding. Tries to disprove the finding. Updates in-place via `findings_history[]`. |
| 5+ | BLUEPRINT | Recommended | Perplexity returns a complete architectural blueprint for a named problem: component layout, data flow, failure modes, 2-3 alternative architectures, implementation steps. |
| 5+ | GUIDANCE | Recommended | Perplexity returns 2–4 ranked research routes for a stated blocker, ordered by expected information gain per cost. No direct evidence — strategic routing. |
| 5+ | EXPLORATORY_BRANCH | Recommended | Branch outward from an interesting/anomalous finding into adjacent territory we haven't investigated. Produces net-new findings (and can spawn new phases). |
| event | POSTMORTEM | Fires once when all HIGH/MED have ≥1 TARGETED_PROBE | Compares against real-world failures in domain. Uses `DOMAIN-POSTMORTEM-UNAVAILABLE` flag for niche domains. |
| every 6 from 8 | FRESH_OBSERVER | Pass 8, 14, 20... | Claude-generated 2k-token summary + finding titles only. "Identify only what is missing." Intentionally skips the running brief. |
| N−1 | INTEGRATION | One pass before final | Cross-phase synthesis. Identifies seams that break when phases combine. |
| N | FINAL_VERDICT | Final pass | Per-phase verdict (CONFIRMED / REFUTED / INCONCLUSIVE / STRUCTURAL-UNRESOLVABLE) + ranked options. |

## Convergence Rules (AND-gate)

The run terminates when **ALL** of these hold:
1. **3 consecutive passes** with zero new findings AND
2. **3 consecutive passes** with zero new contradictions AND
3. **All open findings are effective_severity LOW** (effective_severity = `min(raw_severity, MEDIUM)` if `source_flag=ANALOGOUS`) for 2 consecutive passes AND
4. **`adversarial_pass_count ≥ ceil(N/2)`** (forces at least one adversarial pass for any artifact, more for multi-phase)

**Or** the hard cap (`max_passes`) fires.

**Forced-ADVERSARIAL rule:** if (1)+(2)+(3) would fire but adversarial deficit blocks (4), the next pass MUST be ADVERSARIAL. The runner injects it explicitly. No silent burn of budget on dead-target probes.

## Termination Reasons (always in report.md)

- `CONVERGED` — all 4 AND-gate conditions met. Verdict trustworthy.
- `CAP-HIT (N open HIGH, M unresolved contradictions)` — hard cap fired. Findings remain.
- `STRUCTURAL-UNRESOLVABLE` — HIGH findings persisted across ≥3 consecutive passes (auto-tagged). Verdict treats them as inherent constraints, not gaps to close.
- `INTERRUPTED at pass K` — SIGINT during run. `--resume {slug}` continues from K+1.

## Report Format (`{WORKDIR}/report.md`)

```markdown
# Extended Research Report: {slug}
**Date:** {date} | **Passes:** {N}/{max} | **Verdict:** {verdict}

## Executive Summary
One paragraph. What the artifact is, verdict, single most important finding.

## Termination Reason
CONVERGED | CAP-HIT(...) | STRUCTURAL-UNRESOLVABLE | INTERRUPTED — with one-sentence why.

## Findings by Phase
### Phase 1: {name}
- **F001 [HIGH] [PRIMARY|ANALOGOUS]** — {claim}
  - Status: OPEN | RESOLVED ({resolution}) | STRUCTURAL-UNRESOLVABLE
  - Targeted probes: {N}, history: {findings_history list}
  - Best option: {option_label} (score: X.X)

## Integration Seams
{cross-phase issues from INTEGRATION pass}

## Exhaustive Flaw Table
| ID | Phase | Severity (raw → effective) | Source | Status | Resolution |

## Option Comparison
For each finding with multiple options:
| Option | Correctness (30%) | Simplicity (20%) | Blast (20%) | Reversibility (15%) | Speed (15%) | **Score** |
Highest-scoring option selected. Score margins < 0.5 flagged as judgment-call.

## Contradiction Log
All detected contradictions, paired resolutions.

## Recommended Next Actions
One concrete sentence per open HIGH finding.

## Metadata
- adversarial_pass_count: {N}
- fresh_observer_passes: [{list of pass numbers}]
- POSTMORTEM domain: {domain or "DOMAIN-POSTMORTEM-UNAVAILABLE"}
- last_heartbeat_ts: {iso}
```

## Examples

### Example 1 — Verify an architectural blueprint

```
/extended-research --mode per-phase
[paste the contents of ~/.claude/plans/my-multi-phase-plan.md here]
```

Runner extracts phases from the plan (looking for `## Phase` / `## Edit` / numbered sections), drills each. Returns per-phase verdict + integration seams + scored alternatives for any HIGH findings.

### Example 2 — Stress-test a single function

```
/extended-research --max-passes 10
[paste a 50-line Python function]
```

DECOMPOSE finds N=2 (logic-correctness phase + edge-case-handling phase). Formula gives max_passes=9. Convergence likely by pass 5–7. Output: ranked alternatives for any logic gaps found.

### Example 3 — Resume after Ctrl-C

```
/extended-research --resume gpt4-safety-alignment-a3f92c1d
```

Loads `~/.claude/extended-research-logs/gpt4-safety-alignment-a3f92c1d/ledger.json`, verifies artifact hash matches (or prompts), continues from `interrupted_at_pass + 1`.

## Files Touched

- `~/.claude/council-automation/extended_research_runner.py` — the orchestrator (long-running, async-spawned)
- `~/.claude/council-automation/requirements.txt` — `jsonschema`, `filelock`
- `~/.claude/extended-research-logs/{slug}/` — per-run workdir (artifact.txt, ledger.json, passes.jsonl, report.md, runner.log, runner.log.done, fresh_observer_summary.txt, **running_brief.md**)
  - `running_brief.md` — live narrative state, appended after every pass. Tail this to watch the synthesis evolve. Also mirrored into `ledger["running_brief"]` for atomic JSON-only reads.

## Coordination With Other Commands

- **`/research-perplexity`** — both commands share the `submission_lock` (`~/.claude/council-automation/submission_lock.py`). If `/research-perplexity` is mid-call, the runner's lock acquire blocks up to 180s, then proceeds. Single-Claude-session-level serialization is automatic; cross-session via the file-based lock.
- **`/council-refine`** — independent. Different prompt path through Perplexity. No conflicts.
- **`/solve-perplexity`** — `/solve-perplexity` is for problem-solving (1–5 iterations with contradiction tracking); `/extended-research` is for VERIFICATION (5–40 iterations until convergence). Use solve for "how do I X?", use extended-research for "is this proposed X actually correct?"

## Cost & Time

- **Cost:** $0 — uses the existing Perplexity Pro login session via Playwright (no API key needed).
- **Time:** ~60–120s per pass. Typical runs: 5–40 minutes wall time. The artifact + complexity + convergence pattern determines actual run length.
- **Concurrency:** the runner serializes its own Perplexity calls via `submission_lock`. The user can keep using `/research-perplexity` in the same Claude Code session — calls will queue at the lock.
