# UKCP18 Climate Data — Download Guide

Step-by-step instructions for downloading the UKCP18 climate-projection
data used by the GridOptima AI climate-conditional risk overlay (F-009).

We acquire the same projections from **two independent sources** so we
can cross-check that our extraction is producing the right numbers:

| Stage | Source | Purpose | Download path |
|---|---|---|---|
| **Stage 1** | Met Office UKCP UI (CSV) | Headline values, easy to start | This guide §1 |
| **Stage 2** | CEDA archive (NetCDF) | Raw underlying data | This guide §2 |
| **Stage 3** | Comparison script | Confidence in extraction | (No download — runs locally) |

Either stage on its own is sufficient to run the climate overlay. Doing
both adds a useful cross-check that improves academic defensibility.

---

## 1. Stage 1 — Met Office UKCP User Interface (CSV, easiest)

### 1.1 Register

1. Open https://ukclimateprojections-ui.metoffice.gov.uk/ in any browser.
2. Click **Sign up** (top-right corner).
3. Fill in your name, email address, choose a password.
4. Verify the confirmation email Met Office sends to your inbox.
5. Log in.

Total time: about three minutes. The account is free and there is no
data-access fee.

### 1.2 Find the dataset

After log-in, from the main menu:

1. Click **Products**.
2. Look for the entry named one of these (the wording occasionally
   changes between site refreshes):
   - "UKCP18 Probabilistic Projections — Land — Administrative Region"
   - "Probabilistic land projections (administrative regions)"
   - "Marine and Land Probabilistic Regional Projections"
3. Click it. You should land on an order-form page with several
   dropdowns and check-box panels.

### 1.3 Configure the first download (temperature)

Set the form fields as follows:

| Field | Value |
|---|---|
| **Spatial type / area** | Administrative regions |
| **Regions** | Select **all** (the "select all" checkbox if there is one; otherwise tick each of the 16 boxes) |
| **Variable** | **Mean temperature anomaly** (sometimes labelled `tasAnom`) |
| **Baseline** | 1981–2000 (this is the only baseline the dataset offers) |
| **Future time periods** | Tick `2020–2039`, `2040–2059`, `2070–2089`. These 20-year averages report as `2030`, `2050`, `2080` in our internal naming. |
| **Emission scenarios** | RCP 2.6, RCP 4.5, RCP 8.5 (do not tick RCP 6.0 or SRES-A1B) |
| **Percentile** | **50th** (the median; we will add 10th and 90th later for uncertainty bands) |
| **Frequency** | Annual mean |
| **Output format** | CSV |

Click **Submit** or **Download**. A CSV file will be generated; depending
on browser settings it may download immediately or land in your *My
Downloads* tab on the UI.

### 1.4 Configure the second download (precipitation)

Repeat the order form **but change only the Variable**:

| Field | Value |
|---|---|
| **Variable** | **Precipitation anomaly** (`prAnom`) |
| (everything else) | Same as §1.3 |

Submit. You now have a second CSV.

### 1.5 Place the files on disk

Create the folder

```
C:\Users\lylli\gridoptima-work\gridforge\data\raw\ukcp18_metoffice\
```

if it does not already exist. This path is already gitignored under
the existing `data/raw/` rule, so the file stays on your local
machine only — never reaches GitHub.

Save both CSVs into that folder. Any filename is fine; the extraction
script reads the column structure, not the file name. Suggested names
for tidiness:

```
gridforge/data/raw/ukcp18_metoffice/
├── tasAnom_administrative_regions.csv
└── prAnom_administrative_regions.csv
```

### 1.6 Tell the dev tooling

Once both files are in that folder, run from the repo root:

```
python gridforge/scripts/09_extract_ukcp18_metoffice.py
```

(this script is written after Stage 1 download — see §3 below).

The script:
- Reads both CSVs
- Maps the 16 Met Office admin regions onto our 12 NUTS-1 codes
- Computes per-region per-scenario per-period deltas
- Writes `gridforge/data/climate/ukcp18_metoffice_deltas.csv`
- The `gridforge.data.ukcp18` lookup module will then prefer the real
  file over the placeholder

The placeholder CSV stays in the repo for the audit trail showing
how the project shipped before real data was available.

---

## 2. Stage 2 — CEDA NetCDF (raw data, deeper but more setup)

Do this stage after Stage 1 is working. The CEDA data lets us
independently extract the same numbers and cross-check that our
processing of the Met Office CSV is faithful.

### 2.1 Register at CEDA

1. Open https://services.ceda.ac.uk/cedasite/register/start/
2. Click **Register**.
3. Fill in name, email, institution (use "Independent researcher" if
   not affiliated), reason for access ("UKCP18 land projections for
   public-good cable-asset climate-resilience research" is true and
   sufficient).
4. Verify the email confirmation link.
5. Log in. Your CEDA username will be the one you chose at sign-up.

Total time: ~10 minutes including email verification. Free, no fee.

### 2.2 Navigate to the dataset

The same dataset as Stage 1 is at:

```
https://catalogue.ceda.ac.uk/uuid/8eca5b80ee244d9486162e699c5197f5/
```

Underlying file tree (you reach this after logging in):

```
https://data.ceda.ac.uk/badc/ukcp18/data/land-prob/uk/region/
```

### 2.3 Download the subset you need

You do **not** need the full 53 GB. The path structure is:

```
land-prob/uk/region/sample/{rcp26,rcp45,rcp85}/{tasAnom,prAnom}/mon/
```

Inside each `mon/` directory there are several NetCDF files (one per
percentile, one per time-period band). For our extraction we need:

- All files under `rcp26/tasAnom/mon/`
- All files under `rcp45/tasAnom/mon/`
- All files under `rcp85/tasAnom/mon/`
- All files under `rcp26/prAnom/mon/`
- All files under `rcp45/prAnom/mon/`
- All files under `rcp85/prAnom/mon/`

You can either download each file via the web UI, or use a tool such
as `wget` or `curl` with your CEDA credentials. A wget recipe is in
§2.5 below.

### 2.4 Place the files on disk

Mirror the CEDA folder layout under:

```
C:\Users\lylli\gridoptima-work\gridforge\data\raw\ukcp18_ceda\
```

Folder layout you should produce:

```
gridforge/data/raw/ukcp18_ceda/
├── rcp26/
│   ├── tasAnom/mon/*.nc
│   └── prAnom/mon/*.nc
├── rcp45/
│   ├── tasAnom/mon/*.nc
│   └── prAnom/mon/*.nc
└── rcp85/
    ├── tasAnom/mon/*.nc
    └── prAnom/mon/*.nc
```

Path is gitignored. Files stay local.

### 2.5 Optional — bulk-download recipe

If you prefer one command over clicking through the web UI, the CEDA
archive supports `wget` with your account credentials. From the
`gridforge/data/raw/ukcp18_ceda/` folder:

```bash
wget --user='YOUR_CEDA_USERNAME' --password='YOUR_CEDA_PASSWORD' \
     --no-check-certificate \
     --recursive --no-parent --reject "index.html*" \
     --accept "*.nc" \
     --cut-dirs=6 \
     https://data.ceda.ac.uk/badc/ukcp18/data/land-prob/uk/region/sample/rcp45/tasAnom/mon/
```

Repeat for each `{rcp26,rcp45,rcp85} × {tasAnom,prAnom}` combination.
Total download: roughly 200–400 MB once filtered to the variables and
scenarios we care about.

### 2.6 Run the second extractor

Once the files are on disk:

```
python gridforge/scripts/10_extract_ukcp18_ceda.py
```

Produces `gridforge/data/climate/ukcp18_ceda_deltas.csv` in the same
schema as the Met Office output.

---

## 3. Stage 3 — Cross-validate the two extractions

Once both CSVs exist, run:

```
python gridforge/scripts/11_compare_ukcp18_sources.py
```

This script produces `gridforge/data/climate/ukcp18_dataset_comparison.csv`
showing for each (region, scenario, period) cell:

- Met Office value
- CEDA value
- Absolute difference
- Relative difference (percent)
- A traffic-light flag: green if within 5 percent, amber 5–15 percent,
  red >15 percent

A handful of amber or red cells are normal (the two routes use slightly
different aggregation conventions). A pattern of large mismatches would
indicate an extraction bug worth investigating.

The comparison CSV ships with the repo as evidence of the
cross-validation.

---

## 4. After download — verify the climate overlay still works

Once any of the real-data CSVs is in place, run:

```
RUN_TESTS.bat
```

The whole pipeline should still pass. The placeholder CSV stays for
the audit trail; the `gridforge.data.ukcp18` lookup module prefers the
real data when it is present.

The climate overlay demo will now produce numbers anchored in the real
UKCP18 projections rather than the placeholder:

```
python backend/demo_climate_overlay.py
```

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Met Office UI says "no data available for selection" | A region or scenario was deselected by mistake | Restart the order form; re-tick the "select all regions" box and the 3 RCP boxes |
| Met Office download is `.zip` not `.csv` | They sometimes wrap multi-region exports in a zip | Unzip into the target folder; the extraction script reads `.csv` inside subfolders |
| CEDA wget says "401 Unauthorised" | Credentials or institutional verification step still pending | Wait for the CEDA registration email approval; can take up to 24 hours for an "Independent researcher" record |
| Extraction script fails with "no .nc files found" | Files not in the expected sub-folder structure | Check the `data/raw/ukcp18_ceda/` tree matches §2.4 exactly |
| Demo still produces placeholder-looking numbers | Lookup module is still pointing at the placeholder | Check `gridforge/data/climate/` for the new CSV; the lookup auto-prefers the real one if both are present |

---

## 6. Re-downloading later (if UKCP refreshes)

UKCP18 was published in 2018. A refresh ("UKCP21" / "UKCP24") would
eventually replace these projections. To switch to a new release:

1. Re-run §1 or §2 against the new dataset
2. Update the schema-mapping at the top of the extraction script if
   column names changed
3. Re-run the comparison script
4. Update the README in `gridforge/data/climate/` to reflect the new
   citation source

Everything downstream of the CSV (`ukcp18.py` lookup, the climate
overlay service, the REST endpoint) keeps working unchanged — only the
extraction-script ingestion changes.
