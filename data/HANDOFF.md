# HANDOFF — Data pipeline

**Owner:** [MEMBER 2 NAME]

## What you need from others
| From | What | Contract | By |
|------|------|----------|----|
| Lead ([LEAD NAME]) | Frozen contract; confirmation of `hotspots.csv` event labels and required dtypes | A, D | H+2 |
| Lead | Which rainfall windows the backtest needs (event dates for 2015 and 2023) | D | H+3 |
| Lead | Feedback on sample files (missing columns, wrong types) | D | H+5 |

## What others need from you
| To | What | By |
|----|------|----|
| Lead (ml/) | ~50-row samples of `cells.geojson`, `features.parquet`, `hotspots.csv`, `rainfall.parquet` | H+3 |
| Backend, frontend | `cells.geojson` sample (cell ids and geometry for mock data) | H+3 |
| Lead (ml/) | Full processed files + `data/SOURCES.md` | H+8 |
| Lead (ml/) | Refreshed forecast `rainfall.parquet` for demo day | H+14 and again before H+23 |
| Everyone | Short README in `data/` on how to regenerate files | H+10 |

**Delivery note:** `data/processed/*` and `data/raw/*` are git-ignored (except `.gitkeep`). Share samples and full files via a shared drive link or by temporarily sending them in the team chat; the lead will agree the channel at H+2. Scripts (committed) must regenerate everything.

## Mocking plan if blocked
- Rainfall API down: fall back to a small cached CSV of historical hourly rainfall, or generate synthetic storm curves clearly labelled as synthetic.
- DEM download slow: use a coarser DEM first, or fetch a bounding-box tile only; keep the same column names.
- OSM extraction slow: start with roads only, add waterways next; leave dist columns filled with documented defaults until done (and say so).
- Flood-location sources thin: begin with 30 well-documented points (news/government reports), expand later.

## Integration checkpoints that involve you
- **H+2** contract frozen — confirm bounding box and file schemas.
- **H+3** samples delivered (not a formal checkpoint but everyone waits on it).
- **H+10** first end-to-end run — processed files must be loadable by ml/.
- **H+14** real model wired in — forecast rainfall must be fresh.
- **H+20** feature freeze — no schema changes after this.
