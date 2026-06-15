# /health-check — Read-Only 3-Layer Production Health Check

Perform a comprehensive, **read-only** end-to-end review of any portfolio app across
three layers — **Frontend (E2E), Database integrity, and Audit Log & User Activity** —
and produce a structured, observations-only report. Optionally, on explicit opt-in,
generate (never apply) the infrastructure provisions needed so future health-checks are
possible.

**Architecture:** P0 (Target & Safety Gate, one confirmation) → P1 (Frontend E2E, read-only,
request-capped) → P2 (Database integrity via `hc_db.py`) → P3 (Audit log & user activity) →
P4 (Report). Then, only with `--provision`: a separate plan-mode path that writes a local
`provision.sql` for manual review.

> **TRUST CONTRACT — READ FIRST.**
> The **default run is 100% read-only. It never mutates production.** Every database query
> flows through `hc_db.py`, the single code-enforced read-only boundary (SQLite opened
> `mode=ro`; Postgres runs every query inside a server-enforced `READ ONLY` transaction and
> hard-refuses if it cannot establish one). The frontend layer only *navigates* — it never
> submits a form, clicks a destructive control, or logs in as a real user. The `--provision`
> path is the ONLY thing that produces change, and even then it merely **writes a local `.sql`
> file for you to review and apply manually** — it never connects to a database and never
> auto-applies anything.

**Cost: Free** — local tools + browser-bridge MCP + optional Perplexity login session. No API keys required.

**Guiding principle:** *health check, not a sprint.* Full visibility and peace of mind — not change.

---

## Input

`$ARGUMENTS` = `<repo-path-or-app-name> [flags]`

| Flag | Description | Default |
|------|-------------|---------|
| `--provision` | Opt-in: enter plan mode and generate a local `provision.sql` (read-only role + `audit_log` table as needed). Never applies it. | off |
| `--deep` | Exact row counts (`--exact`) + full sequential scans instead of approximate `pg_stat` estimates. **Recommend off-peak** — heavier DB load. | off (approximate) |
| `--skip-frontend` | Skip P1 (frontend E2E). | run |
| `--skip-db` | Skip P2 (database integrity). | run |
| `--skip-audit` | Skip P3 (audit log & user activity). | run |
| `--sentinel` | Force the Sentinel cross-reference in P3 (otherwise auto-detected). | auto-detect |
| `--url <override>` | Override the auto-detected deployed/prod URL. | auto-detect |
| `--db <override>` | Override the auto-detected `DATABASE_URL`. | auto-detect |
| `--off-peak-ack` | Acknowledge that you are knowingly running against the busy app **now** (skips the off-peak nag for the PO system). | off |

---

## Silence rule

**Run silently from after the P0 confirmation prompt straight through to the P4 report.**
The ONLY user interaction in a default run is the single P0 confirmation (and, for the busy
PO system, the off-peak acknowledgement). After the gate passes, work autonomously — do not
narrate, do not ask follow-up questions — and present the full report at the end. (Mirror
`/frontend-e2e`'s silence rule.)

## Per-check error isolation (MANDATORY, everywhere)

Every individual check is wrapped in its own error handling. If any single check fails or
errors, record it as ⚠️ or ❌ with the error detail and **continue to the next check**.
NEVER abort the whole run because one check failed. This applies in P1 (per page/probe),
P2 (per DB op), and P3 (per audit op).

---

## P0 — Target & Safety Gate (ONE confirmation, then silent)

### P0.1 Resolve the repo

Parse `$ARGUMENTS` for the repo path (or app name) and flags. If given an app name rather
than a path, resolve it against the portfolio (`~/.claude/portfolio/PORTFOLIO.md`) or known
dev roots (e.g. `C:\dev\<app>`). If the repo cannot be resolved, STOP and ask the user for
the path.

### P0.2 Detect the stack (Glob signature files)

Use Glob to detect the stack. Do NOT short-circuit on the first match — collect all that apply
(an app can be multi-typed, e.g. monorepo + Next.js + FastAPI).

| Signature | Stack |
|-----------|-------|
| `next.config.{js,ts,mjs}` or `app/` + `pages/` | Next.js |
| `vite.config.{js,ts}` | Vite |
| `package.json` with `react-scripts` | Create React App |
| `remix.config.js` or `app/root.tsx` + `app/routes/` | Remix |
| `astro.config.mjs` | Astro |
| `svelte.config.js` (`@sveltejs/kit`) | SvelteKit |
| `nuxt.config.{js,ts}` | Nuxt |
| `main.py` + `requirements.txt`/`pyproject.toml` with `fastapi` | FastAPI |
| `app.py` + `requirements.txt` with `flask` | Flask |
| `manage.py` + `settings.py` | Django |
| `pnpm-workspace.yaml` / `lerna.json` / `nx.json` / `turbo.json` | Monorepo |

Record `STACK` (may be multi-valued).

### P0.3 Read config for the deployed URL + DB connection

Read environment config in this **priority order** (first found wins per key):

1. `.env.production`
2. `.env.local`
3. `.env`

Also read `vercel.json` and `render.yaml` for the deployed URL.

- **Deployed URL:** from `--url` if provided; else from `vercel.json` / `render.yaml` /
  the `.env*` file. If unresolved, P1 is skipped with a noted reason.
- **`render.yaml` with MULTIPLE services:** ASK which is the production web service, or
  prefer the paid/standard tier (not a free/preview service). Do not guess silently.
- **`DATABASE_URL`:** from `--db` if provided; else parse `DATABASE_URL` from the
  priority-ordered `.env*`. If unresolved, P2/P3 DB checks are skipped with a noted reason.
- **Supabase pooler vs direct — PREFER THE DIRECT CONNECTION.** Always prefer the **DIRECT**
  connection (port `5432`, host `db.<project>.supabase.co`). If the resolved DSN is the
  **pooler** (port `6543` or host containing `pooler.supabase.com`), this is a
  **transaction-mode pooler**: session-level settings the audit relies on (`READ ONLY`,
  `search_path`) may NOT persist across the pooler's per-statement connection reuse. **WARN**
  the user and recommend switching to the direct URL. If the user insists on the pooler,
  proceed — but still strictly read-only — and note the caveat in the report. **`--provision`
  DDL ALWAYS requires the DIRECT (non-pooler) URL.** (`hc_db.py` surfaces pooler detection as
  `"pooler": true` with a note.)
- **TLS:** recommend `sslmode=require` on the DSN (append `?sslmode=require` if absent) so the
  connection to a real production DB is encrypted.

**NEVER print the password or full DSN.** When you reference the DB, show only the host
(and redact credentials).

### P0.4 Read-only probe (capture safety metadata)

Run a quick read-only probe through `hc_db.py` to confirm the connection and capture the
read-only metadata. For Postgres, a metadata-bearing `rowcount` against a known table works
(any table the audit will touch is fine):

```bash
python ~/.claude/health-check/hc_db.py --engine postgres --db "<DATABASE_URL>" \
    --op rowcount --table <some_known_table>
```

The JSON for a Postgres op carries `transaction_read_only` (true when the READ ONLY txn was
established), `writable_role` (advisory — whether the role *could* write), and `pooler`
(plus a `note` when pooled). Capture these.

- If `hc_db.py` hard-refuses (exit non-zero with
  `{"error": "refused: could not establish read-only transaction"}`), the read-only
  precondition was not met — surface that prominently in the gate and do NOT proceed with DB
  checks.
- If `writable_role` is `true`, WARN clearly that the credential is over-privileged — but the
  **`READ ONLY` transaction is still the hard guarantee**, so the run remains safe.
- For SQLite, the file is opened `mode=ro` (OS-level read-only); there is no writable-role
  concept.

### P0.4.5 Connection-saturation preflight (Postgres — protect a real user's slot)

Before touching the busy production DB, check how close it is to connection saturation with
the dedicated `connections` op:

```bash
python ~/.claude/health-check/hc_db.py --engine postgres --db "<DATABASE_URL>" --op connections
```

Output: `{"op":"connections","current":N,"max":M,"available":M-N}` (under the READ ONLY txn).

- **If `available < 5`: ABORT the run.** Do not risk taking the last connection slot away from
  a real user on the busiest prod DB. Surface the current/max/available numbers in the gate and
  recommend retrying **off-peak**.
- Otherwise proceed. (SQLite returns `{"op":"connections","note":"n/a for sqlite"}` — no pool
  to saturate; skip this gate for SQLite.)

### P0.4.6 RLS sanity note (Postgres — make sure the audit can actually see rows)

Recommend running the audit with a role that can read all rows. **Important: `pg_read_all_data`
alone does NOT bypass RLS** — the role also needs `ALTER ROLE ... WITH BYPASSRLS` (the
`--provision` output includes this). On Supabase, RLS is on by default, so an audit role
without BYPASSRLS reads **zero rows** on protected tables. The table owner / `postgres` also
sees all rows. Row-Level Security otherwise silently hides rows, making a healthy DB look empty.

After schema discovery (P2.1), **verify a known non-empty table reports rows** — its
`n_live_tup`/`reltuples` (or `--exact` count) should be `> 0`. **If every table reads as
empty, WARN that RLS may be hiding rows from the audit role** and recommend re-running with a
`BYPASSRLS`/owner role (generate one via `--provision`).

### P0.5 Single confirmation prompt (the ONE gate)

Present a single confirmation showing (credentials redacted):

```
/health-check — Target confirmation
  App:              <app name>
  Repo:             <repo path>
  Stack:            <detected stack(s)>
  Deployed URL:     <resolved URL or "none — frontend skipped">
  DB host:          <host only — NO credentials/password>
  Engine:           <sqlite | postgres>
  Pooler:           <yes (DDL needs direct URL) | no | n/a>
  Read-only txn:    <established | could-not-establish | n/a (sqlite mode=ro)>
  Writable role:    <no | YES — over-privileged, but txn is READ ONLY>

  This run is READ-ONLY. It will not mutate production. Proceed?
```

**Busy-app guard (PO / Purchase Order system).** If the target is the user's busiest app —
the **PO (Purchase Order) system** — ALSO recommend running off-peak and **require** either
the `--off-peak-ack` flag OR an explicit "yes" acknowledging the current load before
proceeding. (The PO system is the highest-traffic production app; the request cap and
approximate counts protect it, but off-peak is still safest.)

After this gate passes, **run silently** through to the P4 report.

---

## P1 — Frontend E2E (read-only) — skip if `--skip-frontend`

Use **browser-bridge MCP only** (per the global browser-automation rule — never
`claude-in-chrome`). Start with `mcp__browser-bridge__browser_get_tabs` to confirm the
extension is connected; if it is not, note "browser-bridge not connected — P1 skipped" and
continue to P2.

### HARD RULES (state and obey — anti-DoS on the busy app)

- **Read-only navigation ONLY.** NEVER submit a form, never click a destructive control,
  never trigger a mutating request.
- **Dedicated test account only.** If authenticated probing is needed, use a dedicated test
  account — NEVER a real user's credentials.
- **≥ 500 ms between requests.** Pace every navigation/probe.
- **Realistic user-agent.** Do not present as an obvious bot.
- **HARD CAP: ≤ 20 HTTP requests for the entire P1 phase.** Count every navigation and probe.
  Once you hit 20, stop P1 and report what was covered.

### Checks (each isolated; record ✅/⚠️/❌ with the observed value)

1. **Homepage 200** — `browser_navigate` to the deployed URL; confirm it loads (HTTP 200,
   `document.body` present).
2. **Auth boundary** — navigate to a protected route (e.g. `/dashboard`, `/admin`,
   `/settings`) **unauthenticated**; expect a **401 or a redirect to login — NOT a 200 with
   real content**. A protected route returning 200 unauthenticated is a ❌.
3. **JS console errors on load** — `browser_console_messages` (level error) on the homepage;
   filter known noise (favicon 404, devtools, extension chatter). Report the count.
4. **Health endpoints + latency** — probe known health endpoints (`/health`, `/healthz`,
   `/api/health`, `/status`); record status + response latency.
5. **Web-vitals proxy** — via `browser_evaluate`, read the Performance API
   (`loadEventEnd - navigationStart`, LCP if available); WARN on slow loads (>3 s).

Use `mcp__browser-bridge__browser_close_session` when P1 finishes.

---

## P2 — Database integrity (read-only) — skip if `--skip-db`

All DB access goes through `hc_db.py`. Skip cleanly (note the reason) if no `DATABASE_URL`
was resolved. Engine is `sqlite` or `postgres` per P0.

### P2.1 Schema introspection (read-only)

Discover tables, columns, and FK relationships with the dedicated `schema` op — **no
`HC_ALLOW_RAW` needed**:

```
python health-check/hc_db.py --engine <e> --db <db> --op schema
```

Add `--table <t>` to scope to a single table. Output shape (metadata only — never row
contents):

```json
{"op":"schema","tables":[
  {"table":"orders",
   "columns":[{"name":"id","type":"INTEGER","notnull":true,"pk":true}, ...],
   "foreign_keys":[{"column":"customer_id","ref_table":"customers","ref_column":"id"}, ...]}
  , ...]}
```

Under the hood this is read-only introspection:

- **Postgres:** plain `SELECT`s against `information_schema.tables` /
  `information_schema.columns` / `information_schema.table_constraints` joined to
  `key_column_usage` / `constraint_column_usage` (PK + FK), run inside the server-enforced
  `READ ONLY` transaction. Schema/table filters are bound as **parameters**.
- **SQLite:** `sqlite_master` for table names, then `PRAGMA table_info` / `PRAGMA
  foreign_key_list` per table over the `mode=ro` connection. This is a dedicated code path
  that does NOT go through `is_safe_sql` or the `--raw` gate; every interpolated table name
  is identifier-validated and quoted.

Use `--op schema` to drive the integrity ops below (FK pairs, unique/NOT NULL columns).
**Do not** use the gated `--raw` hatch for introspection — `schema` covers it without
un-gating row dumps. The `--raw` escape hatch remains **GATED** (refused unless
`HC_ALLOW_RAW=1`) and is for diagnostics only; never use it to dump table row contents.

### P2.2 Integrity ops (aggregate counts only — never row contents)

Run these `hc_db.py` ops over the introspected tables/columns. Each op prints a single JSON
object to stdout. Exact flag names and output keys:

| Check | Command | Output keys |
|-------|---------|-------------|
| Schema | `--op schema` (add `--table <t>` to scope; **no `HC_ALLOW_RAW`**) | `{"op":"schema","tables":[{"table","columns":[{"name","type","notnull","pk"}],"foreign_keys":[{"column","ref_table","ref_column"}]}]}` |
| Row count | `--op rowcount --table <t>` (add `--exact` only on `--deep`) | `{"table","count","approximate","source"}` (Postgres `source` ∈ `n_live_tup`\|`reltuples`\|`unanalyzed`\|`exact`; an unanalyzed table reports `count: null`; SQLite is always exact) |
| Connections | `--op connections` | `{"op":"connections","current","max","available"}` (Postgres; `{"op":"connections","note":"n/a for sqlite"}` for SQLite) |
| FK orphans | `--op fk_orphans --child <t> --column <fk> --parent <pt> --parent-column <pc>` | `{"op":"fk_orphans","child","column","parent","orphans"}` |
| Duplicates | `--op duplicates --table <t> --column <unique_col>` | `{"op":"duplicates","table","column","duplicate_groups"}` |
| NULL drift | `--op null_drift --table <t> --column <not_null_col>` | `{"op":"null_drift","table","column","nulls"}` |
| Stale rows | `--op stale --table <t> --column <status_col> --value <transitional> --age-column <ts_col> --older-than-hours <H>` | `{"op":"stale","table","column","value","stale"}` |

Apply each to the right targets:

- **fk_orphans** — for each FK relationship discovered in P2.1.
- **duplicates** — on columns that should be unique (unique-indexed columns).
- **null_drift** — on `NOT NULL` columns (count rows that slipped through as NULL).
- **stale** — on tables with a transitional/in-progress status (e.g. `status = 'processing'`
  / `'pending'`) using the row's age column.

**Counts default to approximate and NEVER auto-run `COUNT(*)`.** The Postgres approximate path
reads `n_live_tup` (from `pg_stat_user_tables`) first, falls back to `pg_class.reltuples` if
that is NULL/0, and — if the table was never analyzed (`reltuples` is `-1`/`0`) — reports
`count: null` with `source: "unanalyzed"` **rather than scanning the table**. A full
sequential `COUNT(*)` only ever runs on the explicit `--deep` / `--exact` path
(`source: "exact"`) — **recommend off-peak** because it is heavier on the busiest prod DB.
Every Postgres op runs inside the server-enforced `READ ONLY` transaction with
`SET LOCAL statement_timeout = '3s'`, `SET LOCAL work_mem = '4MB'`, and
`SET LOCAL search_path = public, extensions` already applied by `hc_db.py` — you do not set
these yourself.

### P2.3 Row-count anomaly (baseline-before-record — CRITICAL ORDER)

For each table you `rowcount`, use `hc_history.py` to detect drift across runs. **Call
`baseline()` BEFORE `record()` for the current run** — the first-run guard depends on this
ordering:

```python
import sys; sys.path.insert(0, r"C:\Users\AustinKidwell\.claude\health-check")
import hc_history
from datetime import datetime

PATH = r"C:\Users\AustinKidwell\.claude\health-check\row-counts.json"
app, table, count = "<app>", "<table>", <count_from_rowcount>

base = hc_history.baseline(app, table, PATH)          # 1) read PRIOR mean FIRST
hc_history.record(app, table, count, datetime.now().isoformat(), PATH)  # 2) THEN record
flagged = hc_history.is_anomaly(count, base, pct=0.5)  # 3) compare current vs prior
```

- **First run:** `baseline()` returns `None` (empty history) → `is_anomaly(..., None)` is
  **always `False`** → record the baseline, do NOT flag an anomaly. (Reversing the order would
  poison the first baseline with the current count and defeat the first-run guard.)
- **Subsequent runs:** `is_anomaly` is `True` when the count deviates from the rolling mean by
  more than 50%. History is pruned to the last 30 readings.

### P2.4 PII safety (reiterate)

P2 reports **aggregate counts only** — it NEVER dumps row contents. (`--raw` row-dump is gated
behind `HC_ALLOW_RAW=1` and must not be used to read row bodies.) `statement_timeout` and
`work_mem` caps are enforced inside `hc_db.py` so a heavy scan can never run away on prod.

---

## P3 — Audit Log & User Activity — skip if `--skip-audit`

### P3.1 Detect Sentinel wiring

Check whether the app is instrumented with Sentinel:

- a vendored Sentinel client (e.g. `@asr/sentinel-client`, a Python Sentinel client),
- `SENTINEL_*` env vars (`SENTINEL_URL`, `SENTINEL_APP`, `SENTINEL_INGEST_KEY`),
- a `scripts/sentinel.py` file (dropped by `/sentinel-add`).

**If Sentinel is wired OR `--sentinel` is passed:** read Sentinel events for this app AND the
app's own `audit_log` table (via the `hc_db.py` audit ops below), then **diff** the two:

- **Capture gaps** — events present in one source but not the other.
- **Silent windows** — Sentinel saw activity for a window but the app's `audit_log` was empty
  for that same window (the app failed to log). Flag these.

**If Sentinel is NOT wired and `--sentinel` was not passed:** use the app's `audit_log` alone,
and record **"Sentinel not wired"** as a MISSING PROVISION (surfaced in P4 + actionable via
`--provision` / `/sentinel-add`).

### P3.2 Audit ops via `hc_db.py` (aggregate counts only)

| Check | Command | Output keys |
|-------|---------|-------------|
| Per-user summary | `--op audit_user_summary --table <audit> --user-column <uc> --action-column <ac>` | `{"op":"audit_user_summary","table","users":[{"user_id","actions":{...},"total"}]}` |
| Velocity bursts | `--op audit_velocity --table <audit> --user-column <uc> --ts-column <tc> --window-seconds <W> --threshold <N>` | `{"op":"audit_velocity","table","window_seconds","threshold","bursts":[{"user_id","count","bucket_start_epoch"}]}` |
| Action fingerprint | `--op audit_fingerprint --table <audit> --user-column <uc> --action-column <ac>` | `{"op":"audit_fingerprint","table","users":[{"user_id","actions":{...}}]}` |
| Recency + rate | `--op audit_recency --table <audit> --ts-column <tc>` | `{"op":"audit_recency","table","last_write_ts","events_24h","events_7d"}` |

Use these to:

- **Summarize every user's actions** (`audit_user_summary` — per-user action-type counts +
  totals).
- **Flag velocity spikes** (`audit_velocity` — e.g. "47 actions in 2 minutes": set
  `--window-seconds 120 --threshold 20` and report flagged bursts).
- **Detect action-type fingerprint shifts** (`audit_fingerprint` — e.g. a user whose mix
  shifts from mostly GET/read to DELETE/destructive is suspicious).
- **Confirm logging coverage & error rate** (`audit_recency` — last write timestamp + 24h/7d
  event counts; a stale `last_write_ts` or a zeroed `events_24h` despite live frontend traffic
  is a coverage gap).

> **SQLite timestamp note:** `audit_velocity` / `audit_recency` / `stale` on SQLite require
> timestamps in canonical `YYYY-MM-DD HH:MM:SS` format (as produced by SQLite's `datetime()`).
> ISO-8601 with a `T` separator or a timezone offset compares lexically and may yield wrong
> counts. Postgres uses typed timestamp comparison and has no such requirement.

---

## P4 — Report

Assemble a structured, sectioned, **observations-only** report. State **what you found** and
**any anomalies/concerns** — do **NOT** emit action items unless something is critically broken.
Per-check error isolation: each check shows ✅ (healthy) / ⚠️ (concern / stale / minor) /
❌ (broken / failed) with the observed value. Frame each section as "current state" then
"anomalies / concerns."

End with a **"PROVISIONS MISSING"** list (e.g. no dedicated read-only DB role, no `audit_log`
table, app not Sentinel-wired) and the closing line:

> Run `/health-check <app> --provision` to generate a reviewable provision.sql (never auto-applied).

### Sample report layout

```
══════════════════════════════════════════════════════════
 /health-check  <App Name>  <YYYY-MM-DD HH:MM>
══════════════════════════════════════════════════════════
 Stack: <stack>   URL: <url>   DB: <host> (<engine>, pooler=<y/n>)
 Mode: read-only   Flags: <flags>
══════════════════════════════════════════════════════════

## FRONTEND
  Current state:
    ✅ Homepage 200 (412 ms)
    ✅ Auth boundary: /dashboard → 302 redirect to /login (correct)
    ✅ Console errors on load: 0
    ⚠️ /health: 200 in 1.9 s (slow)
    ✅ Web-vitals: load 1.1 s
  Anomalies/concerns:
    - /health latency trending high (1.9 s)
  (P1 used 8 / 20 request budget)

## DATABASE
  Current state:
    ✅ orders: 12,438 rows (approximate; within 50% of rolling mean 12,001)
    ✅ FK orphans orders.customer_id → customers.id: 0
    ✅ Duplicates users.email: 0 groups
    ⚠️ null_drift orders.status: 3 NULLs (column is NOT NULL by intent)
    ⚠️ stale orders.status='processing' >24h: 17 rows
  Anomalies/concerns:
    - 3 NULL statuses on a should-be-NOT-NULL column
    - 17 orders stuck in 'processing' over 24h

## AUDIT
  Current state:
    ✅ Sentinel wired; audit_log present
    ✅ last_write_ts: 2026-06-15T09:41 | events_24h: 1,204 | events_7d: 9,876
    ✅ Capture diff (Sentinel vs audit_log): 0 gaps
    ⚠️ Velocity: user 42 — 47 actions in 120 s window
    ⚠️ Fingerprint: user 17 mix shifted toward DELETE
  Anomalies/concerns:
    - user 42 velocity burst (47/2min) — review
    - user 17 GET→DELETE fingerprint shift — review

══════════════════════════════════════════════════════════
 PROVISIONS MISSING
   - (none) — read-only role present, audit_log present, Sentinel wired
══════════════════════════════════════════════════════════
Run `/health-check <App Name> --provision` to generate a reviewable
provision.sql (never auto-applied).
```

---

## `--provision` mode (separate, opt-in — writes a local file only)

When `--provision` is passed:

1. **Enter plan mode.** Provisioning is a change-shaped action; present a plan, do not
   silently produce side effects beyond writing the reviewable file.
2. Using `hc_provision.py`, generate the parts the health-check found missing:
   - `gen_readonly_role(role="healthcheck_ro", schema="public", platform=...)` — a
     least-privilege SELECT-only role (no write grants), if no dedicated read-only role
     exists. **The role SQL is chosen by detected platform:**
     - **Supabase** → `platform="supabase"`: the `pg_read_all_data` + `ALTER ROLE ... WITH
       BYPASSRLS` variant (RLS is on by default on Supabase, so the audit role needs
       BYPASSRLS to read protected rows; pg_read_all_data alone does not bypass RLS). **Run
       this in the Supabase dashboard SQL editor**, which executes as a privileged role.
     - **Render / generic managed Postgres** → `platform="render"` (or the default
       `"generic"`): the portable explicit-`GRANT SELECT` variant (`GRANT USAGE` + `GRANT
       SELECT ON ALL TABLES` + `ALTER DEFAULT PRIVILEGES`). **No `BYPASSRLS` / no
       `pg_read_all_data`** — those require a superuser and would FAIL for a non-superuser
       DB owner (the typical Render connecting user); RLS is off by default there so neither
       is needed. Pass `database="<db>"` to also emit `GRANT CONNECT ON DATABASE`.
   - `gen_audit_log_table(table="audit_log")` — an `audit_log` table with
     `id / user_id / action / ts / metadata` + supporting indexes, if no audit table exists.
3. Write the concatenated script via
   `write_provision_sql(parts, out_path, app="<app>")` to
   `~/.claude/health-check/out/<app>-provision.sql`. The file is prepended with a
   `-- REVIEW BEFORE RUNNING — never auto-applied` header.
4. Print the output path, the full SQL, and **manual-apply instructions**: apply it against
   the **DIRECT (non-pooler) connection URL** with an **owner/superuser** role (the pooler URL
   cannot run DDL; the read-only role obviously cannot create roles/tables).
5. If Sentinel was found missing in P3, also suggest `/sentinel-add <repo>` to wire the app.

**NEVER auto-apply. NEVER connect to the database in `--provision` mode.** `hc_provision.py`
imports no database driver and only writes text — preserve that guarantee at the command layer
by never piping its output into a DB client.

---

## How it works / safety model

- **`hc_db.py` is the code-enforced read-only boundary** — not a convention. SQLite is opened
  `file:<db>?mode=ro` (OS-level read-only). Postgres runs every query inside a server-enforced
  `READ ONLY` transaction and **hard-refuses** (exit non-zero) if it cannot establish one;
  `is_safe_sql` is defense-in-depth on top (SELECT/WITH/EXPLAIN only, no stacked statements, no
  data-modifying CTEs, `EXPLAIN ANALYZE` rejected). `statement_timeout` + `work_mem` caps bound
  any query. The `--raw` row-dump hatch is gated behind `HC_ALLOW_RAW=1`. Output is
  aggregate/scalar JSON — never row contents.
- **Provisions are write-to-file only** — `hc_provision.py` never opens a DB connection; it
  emits reviewable SQL text for the user to apply manually.
- **Frontend is navigation-only** — capped at ≤ 20 requests, ≥ 500 ms apart, never submitting
  forms, never using real credentials.
- **First-run baseline guard** — `hc_history.baseline()` returns `None` on first observation so
  the first run never raises a false anomaly.

## Files

- `~/.claude/commands/health-check.md` — this command.
- `~/.claude/health-check/hc_db.py` — code-enforced read-only DB/audit introspector
  (sqlite3 stdlib + lazy psycopg for Postgres); aggregate-only JSON output.
- `~/.claude/health-check/hc_history.py` — rolling row-count history
  (`~/.claude/health-check/row-counts.json`); first-run baseline guard.
- `~/.claude/health-check/hc_provision.py` — provision SQL generator (write-to-file only;
  no DB driver, no connection).
