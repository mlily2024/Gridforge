# Climate data — UKCP18 deltas

This directory ships small reference data tables used by the GridForge
climate-overlay module (`gridforge.data.ukcp18`).

## Current contents

### `ukcp18_placeholder_deltas.csv`

**PLACEHOLDER values.** Do not cite these numbers in any document or paper.

12 NUTS-1 UK regions × {RCP2.6, RCP4.5, RCP8.5} × {2030, 2050, 2080} =
108 rows. Each row carries:

| Column            | Meaning                                                    |
|-------------------|------------------------------------------------------------|
| `region_code`     | NUTS-1 code (UKC, UKD, ..., UKN)                           |
| `region_name`     | Human-readable region name                                 |
| `scenario`        | UKCP18 emission scenario (`RCP2.6` / `RCP4.5` / `RCP8.5`)  |
| `period`          | Target year — central year of a 20-year period             |
| `delta_ambient_C` | Mean ambient warming above the 1981-2010 baseline (°C)     |
| `delta_moisture`  | Mean summer soil moisture change vs baseline (m³/m³)       |

Values are derived from clear constants documented in
`_generate_placeholder.py`. They follow the rough patterns reported in
the UKCP18 Land Projections Science Report (e.g. southern regions warm
more than northern; London has an urban-heat-island uplift; RCP8.5 ≈ 1.7
× RCP4.5 warming by 2080) but are not the real UKCP18 numbers.

To regenerate the file:

```bash
cd gridforge/         # the inner subpackage root (where gridforge/, scripts/, data/ live)
python data/climate/_generate_placeholder.py
```

### Why placeholder for now?

The full UKCP18 land projections are hosted on CEDA and require a few
hundred MB of NetCDF download + per-region spatial averaging. M5 of the
F-009 climate-overlay implementation replaces this CSV with the real
extracted values via `scripts/09_extract_ukcp18_deltas.py`. The
placeholder lets the rest of the pipeline (M3 orchestrator, M4 REST
endpoint) be developed and tested without that download as a
dependency.

## When the real data lands (M5)

- `_generate_placeholder.py` becomes archival; the script is kept for
  the audit trail showing how the placeholder was constructed.
- A new file `ukcp18_real_deltas.csv` is committed alongside.
- The `ukcp18.py` module switches its `DATA_FILE` constant from the
  placeholder file to the real file.
- All tests in `test_ukcp18.py` should still pass; only the absolute
  magnitudes will tighten to match the published UKCP18 reports.

## Data source for the real version (planned)

| Property            | Value                                                      |
|---------------------|------------------------------------------------------------|
| Dataset             | UKCP18 Land Projections                                    |
| Host                | CEDA (Centre for Environmental Data Analysis)              |
| Variables           | tas (near-surface air temperature), mrso (total soil moisture) |
| Spatial resolution  | 12 km Regional Climate Model                                |
| Temporal resolution | Monthly mean → 20-year-period mean → period midpoint year   |
| Baseline period     | 1981-2010                                                  |
| Scenarios           | RCP2.6, RCP4.5, RCP8.5                                     |
| Aggregation         | Mean over each NUTS-1 polygon                              |
