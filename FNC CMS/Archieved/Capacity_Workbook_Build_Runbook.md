# Capacity Workbook — Build Runbook (Config-Driven Architecture)

**Applies to:** Mercury, FNC CMS (Phone + Email/Web + Tier 2), Mercury Integrations
(Email/Web only), and any new Platforms client built on the same pattern.

**Supersedes** the linear-tab and Config sections of the prior runbooks. The old design
baked capacity policy and the FTE math into the SQL/linear tab. The current design moves
**all policy to a single Config tab** and keeps SQL to a raw data layer. Everything on
Summary and Config is a live formula — the only manual cells are the yellow Config inputs.

---

## 1. Architecture at a glance

Three layers, one direction of flow:

```
  SQL data layer (tabs, refreshed from BigQuery)
        │   ticket volumes · calendar · erlang FTE · workday weights · T2 ratio
        ▼
  Config tab (single engine)
        │   owns policy (occ, shrink, attrition, op-hours, AHT, T2 factor)
        │   computes Email/Web FTE  +  Tier 2 FTE  per month
        ▼
  Summary tab (surface)
        │   Phone (Erlang) + Email/Web (Config) + Tier 2 (Config)
        │   headcount, gaps, KPIs, planning rollups
```

Rules that hold everywhere:
- **No hardcoded numbers on Summary or Config** except the yellow Config inputs. (If you
  type a number into a formula cell to "fix" a month, you've created a stale override —
  see §9, the CMS Tier-1 headcount lesson.)
- **One Config tab per workbook.** It drives both Email/Web and Tier 2.
- **Phone (Erlang) is independent** of the linear refactor — it keeps its own pre-built
  FTE tab and is only surfaced on Summary.

---

## 2. Tab inventory

Order left→right: **Summary**, data tabs, **Config** (last).

| Tab | Source | Role |
|---|---|---|
| `Summary` | formulas | The deliverable. Reads Config + data tabs. |
| `<client>_ticket_volumes` | `gbq_<client>_linear_data.sql` | Email/Web demand (volumes + calendar). |
| `erlang_<client>_FTE_Summary` | erlang model | Phone FTE need (pre-computed). |
| `<client>_erlang_hist_interval` | erlang hist | Phone actual volume/handle, interval grain. |
| `cx_nontax_erlang_fx_interval` / `<client>_erlang_fx_interval` | erlang forecast | Phone forecast volume, interval grain. |
| `Workday` | `gbq_<client>_workday_headcount.sql` | Roster with T1/T2 weights baked in. |
| `T2 Ratio` | `gbq_<client>_t2_ratio.sql` | Monthly escalation ratios (history only). |
| `Config` | formulas + yellow inputs | The engine (Email/Web + Tier 2). |

Email/Web-only clients (e.g. Mercury Integrations) omit the Phone, T2 Ratio, and erlang
tabs and use the simplified Config/Summary in §8.

---

## 3. Date & key formats — read this first

This is the single most common reason a rebuild "silently goes blank." Each lookup keys on
a specific type; the source must match it exactly.

| Tab / column | Stored as | Why |
|---|---|---|
| `*_ticket_volumes` `call_month` (A) | **TEXT `yyyy-mm-dd`** | Looked up with `TEXT(C$3,"yyyy-mm-dd")`. Emit via `FORMAT_DATE('%Y-%m-%d', …)` in SQL. |
| `Workday` `report_month` (A) | **TEXT `yyyy-mm-dd`** | `SUMIFS` criterion is the text key. |
| `T2 Ratio` `created_month` (A) | **TEXT `yyyy-mm-dd`** | Looked up with `TEXT(C$3,"yyyy-mm-dd")`. |
| `erlang_*_FTE_Summary` `month` (A) | **real DATE** | Looked up with `MATCH(C$3, …)` (date↔date). |
| `*_erlang_*_interval` timestamp (A) | **TEXT `yyyy-mm-dd hh:mm:ss`** | Bucketed by `LEFT(A,7)=TEXT(C$3,"yyyy-mm")`. |
| Summary/Config `date_key` (row 3) | **real DATE** | Master spine. Every lookup derives its key from here. |

If a refresh delivers `call_month` as a real date instead of text (or vice-versa), the
text-criterion lookups return nothing and the dependent rows blank out. **Verify column A
types after every refresh.**

Other conventions:
- **Missing values** arrive as the literal text `(null)` or as empty cells. All numeric
  pulls are wrapped `IFERROR(1*INDEX(…),"")` — the `1*` coerces `(null)`→error→`""`. Never
  strip this guard.
- **`bu_adj_vol` is a signed delta and is ADDED** to the base forecast
  (`net_forecast_vol = GREATEST(base + bu_adj, 0)`). Positive raises, negative lowers.
  Actuals are never adjusted.

---

## 4. The spine (Summary & Config rows 3–4)

- **Contiguous**, columns **C → AF = 30 months**, Jan-2025 → Jun-2027. No blank
  year-separator columns. Column landmarks used by the rollups:
  `C = Jan-25 · O = Jan-26 · S = May-26 · T = Jun-26 · Z = Dec-26 · AF = Jun-27`.
- **Summary** owns the dates: `C3..AF3` are real first-of-month dates, formatted
  `yyyy-mm-dd`. **Config** mirrors them: `Config!C3 = =Summary!C3`.
- Row 4 is the display label: `=TEXT(C3,"mmm")&CHAR(10)&TEXT(C3,"yyyy")`.
- Freeze panes at **C5**; freeze columns A–B (A = labels, B = thin spacer).
- `ActualsCutoff` — a workbook-scope defined name:
  `=DATE(YEAR(TODAY()),MONTH(TODAY()),1)`. Gates actuals to closed months; auto-rolls.

---

## 5. Data tabs — exact columns

### 5a. `<client>_ticket_volumes`  (Email/Web demand)
Headers row 1, data row 2+. Output of `gbq_<client>_linear_data.sql`.

| Col | Field | Notes |
|---|---|---|
| A | `call_month` | TEXT `yyyy-mm-dd` |
| B | `business_unit` | |
| C | `client` | |
| D | `client_id` | |
| E | `groups` | |
| F | `origin` | `email/web` |
| G | `total_days_in_month` | calendar fact |
| H | `weekday_count` | |
| I | `weekday_holiday_count` | weekday holidays only |
| J | `capacity_days` | weekdays − weekday holidays — **Config reads this** |
| K | `actual_volume` | unadjusted — **Config reads this** |
| L | `base_fx_vol` | |
| M | `bu_adj_vol` | signed |
| N | `net_forecast_vol` | `GREATEST(base + bu_adj, 0)` — **Config reads this** |

### 5b. `erlang_<client>_FTE_Summary`  (Phone, pre-computed)
Key `month` (A) is a **real DATE**. Columns Summary consumes:
`C` actual SL · `M` total_calendar_days · `P` capacity_days_net · `R` productive_hrs_per_fte ·
`S/T` actual FTE dom/telus · `V/W` target FTE dom/telus · `Y/Z` forecast FTE dom/telus.

### 5c. Erlang interval tabs
`A` timestamp TEXT, `F` call_volume; hist also has `G` total_handle_secs. Aggregated by
month prefix via `SUMPRODUCT`.

### 5d. `Workday`  (headcount with weights)
`A` report_month TEXT, `J` `T1_weight`, `K` `T2_weight`. **The per-employee weighting rules
live in the SQL** (managers/team-leads zeroed, explicit emp_id overrides, partial weights,
Associates routed by job code). Summary just sums the weights — never re-derive headcount in
the sheet.

### 5e. `T2 Ratio`  (escalation ratios, history only)
`A` created_month TEXT, `G` `t2_esc_non_cancelled` (T2 actual volume), `H` `t2_esc_ratio`.
Only closed months exist here; future ratios come from the trailing average (§6, row 27).

---

## 6. Config tab — full row map

Single engine. Edit only the yellow input cells (`C6:C13`). Every other cell is a formula
filled **C→AF**. `{m}` = the current month column.

**Assumptions (inputs)**
| Row | Label | Cell | Value (CMS / Mercury / MI) |
|---|---|---|---|
| 6 | Daily Operating Hours | C6 | 12 / 12 / 7.5 |
| 7 | Occupancy | C7 | 0.70 |
| 8 | Shrinkage | C8 | 0.30 |
| 9 | Attrition (hiring buffer) | C9 | 0.10 |
| 10 | Email/Web AHT (sec) | C10 | **867 / 381 / 2151** |
| 12 | T2 AHT factor (× EW AHT) | C12 | 1.00 |
| 13 | Forecast ratio lookback (months) | C13 | 6 |
| 14 | Tier 2 AHT (sec) *(derived)* | C14 | `=IFERROR($C$10*$C$12,"")` |

**Shared calendar + productive hours** (filled C→AF)
| Row | Label | Formula |
|---|---|---|
| 16 | Capacity Days | `=IFERROR(1*INDEX(vol!$J$2:$J$34,MATCH(TEXT({m}$3,"yyyy-mm-dd"),vol!$A$2:$A$34,0)),"")` |
| 17 | Productive Hrs / FTE | `=IFERROR({m}16*$C$6*$C$7*(1-$C$8),"")` |

**Email/Web FTE engine — Domestic** (filled C→AF)
| Row | Label | Formula |
|---|---|---|
| 19 | Actual Volume | `=IFERROR(1*INDEX(vol!$K$2:$K$34,MATCH(TEXT({m}$3,"yyyy-mm-dd"),vol!$A$2:$A$34,0)),"")` |
| 20 | Forecast Volume (net) | `…INDEX(vol!$N$2:$N$34…)` |
| 21 | Actual Net FTE | `=IFERROR({m}19*$C$10/3600/{m}17,"")` |
| 22 | **Actual Gross FTE** → Summary | `=IFERROR({m}21*(1+$C$9),"")` |
| 23 | Forecast Net FTE | `=IFERROR({m}20*$C$10/3600/{m}17,"")` |
| 24 | **Forecast Gross FTE** → Summary | `=IFERROR({m}23*(1+$C$9),"")` |

**Tier 2 volume build** (filled C→AF)
| Row | Label | Formula |
|---|---|---|
| 26 | Escalation ratio — same-month | `…INDEX('T2 Ratio'!$H$2:$H$500…)` |
| 27 | Escalation ratio — go-forward | `=IFERROR(AVERAGE(OFFSET('T2 Ratio'!$H$1,MAX(COUNT('T2 Ratio'!$H$2:$H$500)-$C$13,0)+1,0,MIN($C$13,COUNT('T2 Ratio'!$H$2:$H$500)),1)),"")` |
| 28 | Escalation ratio — used | `=IF({m}26<>"",{m}26,{m}27)` |
| 29 | Phone forecast volume | `=IFERROR(Summary!{m}60,"")` |
| 30 | Email/Web forecast volume | `={m}20` |
| 31 | Total forecast contact volume | `=IFERROR({m}29+{m}30,"")` |
| 32 | **Tier 2 actual volume** → Summary | `…INDEX('T2 Ratio'!$G$2:$G$500…)` |
| 33 | **Tier 2 forecast volume** → Summary | `=IFERROR({m}28*{m}31,"")` |

**Tier 2 FTE engine** — reuses EW Domestic policy + Tier 2 AHT (filled C→AF)
| Row | Label | Formula |
|---|---|---|
| 35 | Actual workload hrs | `=IFERROR({m}32*$C$14/3600,"")` |
| 36 | Actual Net FTE | `=IFERROR({m}35/{m}17,"")` |
| 37 | **Actual Gross FTE** → Summary | `=IFERROR({m}36*(1+$C$9),"")` |
| 38 | Forecast workload hrs | `=IFERROR({m}33*$C$14/3600,"")` |
| 39 | Forecast Net FTE | `=IFERROR({m}38/{m}17,"")` |
| 40 | **Forecast Gross FTE** → Summary | `=IFERROR({m}39*(1+$C$9),"")` |

> `vol!` = the `<client>_ticket_volumes` tab. Adjust the `$…$34` row bound to the actual
> last data row. Row 29 reads Summary's phone-forecast KPI — this is **not** a circular
> reference: Summary!60 depends only on the fx-interval source, not on Config.

---

## 7. Summary tab — full row map

Real-date spine (row 3). Formulas fill C→AF. `gate(x)` =
`=IF({m}$3<ActualsCutoff,x,"")` (blanks the cell for the current and future months).
`vol/erl/wd/hist/fx/cfg` = the respective tabs.

**Operations Calendar**
| Row | Label | Source |
|---|---|---|
| 6 | Total Calendar Days | `erl!M` (date key) |
| 7 | Capacity Days | `erl!P` |
| 8 | Productive Hrs / FTE (phone) | `erl!R` |

**Active Headcount** (Domestic live from Workday; Telus = add source)
| Row | Label | Formula |
|---|---|---|
| 10 | HC — Domestic Tier 1 | `=SUMIFS(wd!$J$2:$J$500,wd!$A$2:$A$500,TEXT({m}$3,"yyyy-mm-dd"))` |
| 11 | HC — Domestic Tier 2 | `=SUMIFS(wd!$K$2:$K$500,…)` |
| 12 | HC — Domestic Total | `=SUM({m}10:{m}11)` |
| 13 | HC — Telus | *(yellow input — blank)* |
| 14 | Total Headcount | `=SUM({m}12:{m}13)` |

**Headcount Gaps** (green = surplus / red = deficit; conditional format `C16:AF19`)
| Row | Label | Formula |
|---|---|---|
| 16 | Gap: T1 Dom vs Actual | `=IF({m}$3<ActualsCutoff,IFERROR({m}10-({m}25+{m}35),""),"")` |
| 17 | Gap: T1 Dom vs Forecast | `=IFERROR({m}10-{m}21,"")` |
| 18 | Gap: T2 Dom vs Actual | `=IF({m}$3<ActualsCutoff,IFERROR({m}11-{m}49,""),"")` |
| 19 | Gap: T2 Dom vs Forecast | `=IFERROR({m}11-{m}45,"")` |

**Tier 1 FTE Need — Phone + Email/Web (forecast rollup)**
| Row | Formula |
|---|---|
| 21 Domestic | `=SUM({m}31,{m}41)` |
| 22 Telus | `=SUM({m}32,{m}42)` |
| 23 Total | `=SUM({m}21:{m}22)` |

**Tier 1 Phone FTE — Erlang** (rows 25–33)
Actual `gate(erl!S/T)`, Target `gate(erl!V/W)`, Forecast `erl!Y/Z` (ungated), each with a
Domestic / TELUS / Total triplet (`SUM` for totals).

**Email/Web FTE — Linear (from Config)** (rows 35–43)
| Row | Formula |
|---|---|
| 35 Actual Dom | `=IF({m}$3<ActualsCutoff,IFERROR(cfg!{m}22,""),"")` |
| 36 Actual Telus | *(blank placeholder)* |
| 37 Actual Total | `=SUM({m}35:{m}36)` |
| 38–40 Actual(Target) | `={m}35` / blank / `SUM` |
| 41 Forecast Dom | `=IFERROR(cfg!{m}24,"")` |
| 42 Forecast Telus | *(blank)* |
| 43 Forecast Total | `=SUM({m}41:{m}42)` |

**Tier 2 FTE Linear (forecast rollup)** (rows 45–47): `=SUM({m}55)` / `=SUM({m}56)` / `=SUM`.

**Tier 2 FTE Need — Linear on escalations (engine: Config)** (rows 49–57)
| Row | Formula |
|---|---|
| 49 Actual Dom | `=IF({m}$3<ActualsCutoff,IFERROR(cfg!{m}37,""),"")` |
| 52 Target Dom | `={m}49` |
| 55 Forecast Dom | `=IFERROR(cfg!{m}40,"")` |
| 50/53/56 Telus | *(blank)* · 51/54/57 Total `=SUM` |

**KPIs** (rows 59–69)
| Row | Label | Formula |
|---|---|---|
| 59 | Phone Vol — Actual | `gate(SUMPRODUCT((LEFT(hist!$A$2:$A$4237,7)=TEXT({m}$3,"yyyy-mm"))*hist!$F$2:$F$4237))` |
| 60 | Phone Vol — Forecast | same on `fx!F` (ungated) — **Config row 29 reads this** |
| 61 | Phone AHT (sec) — Actual | `gate(Σ(hist G)/Σ(hist F))` |
| 62 | Phone SL % — Actual | `gate(erl!C)` |
| 63–64 | Chat — Actual / Forecast | *(blank placeholders)* |
| 65 | Email/Web Vol — Actual | `gate(INDEX(vol!K…))` |
| 66 | Email/Web Vol — Forecast | `=IFERROR(INDEX(vol!N…),"")` |
| 67 | Tier 2 Esc — Actual | `=IF({m}$3<ActualsCutoff,IFERROR(cfg!{m}32,""),"")` |
| 68 | Tier 2 Esc — Forecast | `=IFERROR(cfg!{m}33,"")` |
| 69 | Tier 2 Esc Ratio — used | `=IFERROR(cfg!{m}28,"")` |

**Planning Rollups** (single cells in column C; contiguous spine landmarks)
| Row | Label | Formula |
|---|---|---|
| 73 | T1 Current Gap (Jan–May 26, vs Actual) | `=IFERROR(AVERAGE(O16:S16),"")` |
| 74 | T1 Forecast Gap (Jun–Dec 26, vs Forecast) | `=IFERROR(AVERAGE(T17:Z17),"")` |
| 76 | T2 Current Gap (Jan–May 26, vs Actual) | `=IFERROR(AVERAGE(O18:S18),"")` |
| 77 | T2 Forecast Gap (Jun–Dec 26, vs Forecast) | `=IFERROR(AVERAGE(T19:Z19),"")` |

---

## 8. Variant — Email/Web-only client (Mercury Integrations)

No Phone, no Tier 2, no erlang/T2-Ratio tabs. Config keeps only rows 6–10 (EW assumptions),
16–17 (calendar/productive hrs), 19–24 (EW engine), and a per-location split if Telus is in
play. Summary keeps Operations Calendar, Headcount (manual or Workday), the EW FTE block,
the volume KPIs, and the rollups. Headcount with no Workday source becomes a manual Config
input. Same date/key rules, same `IFERROR(1*INDEX…)` and gating discipline.

---

## 9. Number formats, styling, gotchas

**Number formats** (third section blanks zeros/empties):
- FTE / Headcount → `#,##0.00;(#,##0.00);""`
- Gaps → `#,##0.00;(#,##0.00);"–"`
- Volume → `#,##0;(#,##0);""` · Days → `#,##0;(#,##0);""`
- Productive hrs → `#,##0.0;(#,##0.0);""` · AHT → `#,##0;;""`
- SL % → `0.0%;;""` · Ratio → `0.00%;;""`
- Occupancy / Shrinkage / Attrition / Split → `0%`

**Styling:** Arial; navy band rows `#1F3864` white bold; yellow inputs `#FFF2CC` with blue
font `#0000FF`; date row `#EEF3FB`; month-label row `#D9E1F2`. Gridlines off, freeze C5,
landscape fit-to-width.

**Gotchas / lessons:**
- **Never hardcode a formula cell.** A real example: CMS Tier-1 headcount had 11 manually
  typed month values (5,5,6…7) that drifted from the live Workday weights (8,8,9). The fix
  is the `SUMIFS` everywhere — if a month looks wrong, correct the **Workday weighting / cost
  centre membership**, not the Summary cell.
- **`(null)` is text.** A pull that loses the `1*INDEX/IFERROR` guard turns those into
  `#VALUE!`. Keep the guard on every numeric lookup.
- **Don't reintroduce year-separator spacer columns.** The spine is contiguous; the rollup
  ranges (`O:S`, `T:Z`) assume it.
- **Phone is not part of the linear refactor.** If phone FTE looks off, it's the erlang tab,
  not Config.

**Validation (do this every refresh):**
1. Recalculate the whole workbook; confirm **0 formula errors**.
2. Spot-check 3–4 closed months against the previous build — Email/Web and Tier 2 FTE should
   match to the penny unless the underlying volume/calendar changed.
3. Confirm actuals blank for the current and future months, and that `ActualsCutoff` resolved
   to the first of the current month.
4. Confirm column A of every keyed tab is the **expected type** (text vs date per §3).

---

## 10. Build / refresh procedure

1. **Refresh SQL tabs** from BigQuery: `*_ticket_volumes` (from `gbq_<client>_linear_data.sql`),
   `Workday`, `T2 Ratio`, the erlang FTE summary, and the two erlang interval tabs. Paste
   values, headers in row 1.
2. **Check key types** (§3) — especially that `call_month`, `report_month`, `created_month`
   are TEXT `yyyy-mm-dd` and the erlang `month` is a real date.
3. **Config**: confirm the yellow inputs (op-hours, occ, shrink, attrition, AHT, T2 factor,
   lookback). Adjust `$…$34` row bounds if the data length changed.
4. **Summary**: spine dates are the master; everything else recalculates.
5. **Recalculate and validate** (§9).
6. The Config inputs are the only knobs — change an assumption once and every FTE figure,
   gap, and rollup updates across the horizon.
