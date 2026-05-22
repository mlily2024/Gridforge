"""
Regenerate `ukcp18_placeholder_deltas.csv` from constants.

This script is here ONLY so the placeholder values stay reproducible from
source. The real UKCP18 extraction script lands in M5 (planned filename
`gridforge/scripts/09_extract_ukcp18_deltas.py`). After that lands, the
file that ships with the package is the real-data one and this generator
becomes archival.

Usage (from the gridforge subpackage root):

    python data/climate/_generate_placeholder.py
"""

from __future__ import annotations

import csv
from pathlib import Path


# NUTS-1 UK regions (code, name, ambient-warming offset [degC], drying multiplier)
# Offsets reflect rough latitude / urban-heat-island patterns reported in
# UKCP18 land projection summaries; numbers are illustrative.
REGIONS: list[tuple[str, str, float, float]] = [
    ("UKC", "North East",                 -0.2, 0.6),
    ("UKD", "North West",                 -0.1, 0.7),
    ("UKE", "Yorkshire and the Humber",   -0.1, 0.8),
    ("UKF", "East Midlands",               0.0, 0.9),
    ("UKG", "West Midlands",               0.0, 0.9),
    ("UKH", "East of England",             0.1, 1.1),
    ("UKI", "London",                      0.3, 1.2),
    ("UKJ", "South East",                  0.2, 1.2),
    ("UKK", "South West",                  0.1, 1.1),
    ("UKL", "Wales",                       0.0, 0.8),
    ("UKM", "Scotland",                   -0.3, 0.5),
    ("UKN", "Northern Ireland",           -0.2, 0.7),
]

# UK-mean warming above the 1981-2010 baseline period (illustrative)
BASE_WARMING_C: dict[int, float] = {2030: 0.7, 2050: 1.4, 2080: 2.4}

# UK-mean summer theta reduction below the 1981-2010 baseline (m3/m3)
BASE_DRYING: dict[int, float] = {2030: 0.02, 2050: 0.06, 2080: 0.12}

# RCP intensity multiplier
SCENARIO_MULT: dict[str, float] = {"RCP2.6": 0.7, "RCP4.5": 1.0, "RCP8.5": 1.7}


def main() -> int:
    out_path = Path(__file__).resolve().parent / "ukcp18_placeholder_deltas.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "region_code", "region_name", "scenario", "period",
            "delta_ambient_C", "delta_moisture",
        ])
        for code, name, dT_off, dry_mult in REGIONS:
            for scenario, smul in SCENARIO_MULT.items():
                for period, base_T in BASE_WARMING_C.items():
                    delta_T = base_T * smul + dT_off
                    delta_moist = -BASE_DRYING[period] * smul * dry_mult
                    writer.writerow([
                        code, name, scenario, period,
                        f"{delta_T:+.3f}",
                        f"{delta_moist:+.4f}",
                    ])
    n_rows = len(REGIONS) * len(SCENARIO_MULT) * len(BASE_WARMING_C)
    print(f"wrote {n_rows} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
