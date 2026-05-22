"""
UKCP18 climate-delta lookup for the GridForge climate overlay.

This module loads a small CSV of per-region per-scenario per-period
climate deltas (ambient temperature and soil moisture) relative to a
1981-2010 baseline, and exposes a typed lookup function used by the
climate-overlay service in the GridOptima backend.

The CSV that ships with the repo at `gridforge/data/climate/` is
currently **placeholder** values — illustrative magnitudes consistent
with UKCP18 land-projection patterns, but not the real numbers.
M5 of the F-009 climate-overlay implementation replaces it with values
extracted from the real UKCP18 dataset. The module API is identical
either way, so callers don't need to change when the real data lands.

Public surface:

  * `ClimateDeltas`            dataclass with the looked-up values
  * `get_climate_deltas(...)`  the lookup function
  * `list_regions()`           available NUTS-1 region codes + names
  * `list_scenarios()`         available RCP scenarios
  * `list_periods()`           available target periods
  * `KNOWN_SCENARIOS`          tuple of valid scenario strings
  * `KNOWN_PERIODS`            tuple of valid target periods

Errors:

  * Unknown region / scenario / period raises `ValueError` with a clear
    message listing available values.
  * The CSV is parsed lazily on first call and cached. If the file is
    missing or malformed, the first call raises `FileNotFoundError` or
    `ValueError` accordingly.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Resolve the data file: this module lives at
#   gridforge/gridforge/data/ukcp18.py
# The CSV lives at
#   gridforge/data/climate/ukcp18_placeholder_deltas.csv
# i.e. up two parent dirs from this file's dir, then down into data/climate.
DATA_FILE: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "data" / "climate" / "ukcp18_placeholder_deltas.csv"
)

KNOWN_SCENARIOS: Final[tuple[str, ...]] = ("RCP2.6", "RCP4.5", "RCP8.5")
KNOWN_PERIODS: Final[tuple[int, ...]] = (2030, 2050, 2080)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClimateDeltas:
    """Looked-up climate deltas for one (region, scenario, period) triple."""

    region_code: str
    region_name: str
    scenario: str
    period: int
    delta_ambient_C: float
    delta_moisture: float


# ---------------------------------------------------------------------------
# Loader (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_table() -> dict[tuple[str, str, int], ClimateDeltas]:
    """Parse the CSV once into a (region, scenario, period) -> ClimateDeltas map.

    Cached for the lifetime of the process. To force a reload (e.g. in
    tests after modifying the CSV), call `_load_table.cache_clear()`.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"UKCP18 deltas file not found at {DATA_FILE}. "
            "Run `python data/climate/_generate_placeholder.py` from the "
            "gridforge subpackage root to regenerate the placeholder."
        )

    table: dict[tuple[str, str, int], ClimateDeltas] = {}
    with DATA_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {
            "region_code", "region_name", "scenario", "period",
            "delta_ambient_C", "delta_moisture",
        }
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(
                f"UKCP18 deltas CSV has unexpected columns. "
                f"Expected: {sorted(expected)}; "
                f"got: {sorted(reader.fieldnames or [])}"
            )
        for row in reader:
            entry = ClimateDeltas(
                region_code=row["region_code"],
                region_name=row["region_name"],
                scenario=row["scenario"],
                period=int(row["period"]),
                delta_ambient_C=float(row["delta_ambient_C"]),
                delta_moisture=float(row["delta_moisture"]),
            )
            key = (entry.region_code, entry.scenario, entry.period)
            if key in table:
                raise ValueError(f"duplicate row in UKCP18 deltas CSV: {key!r}")
            table[key] = entry
    return table


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_climate_deltas(
    region_code: str,
    scenario: str,
    period: int,
) -> ClimateDeltas:
    """Look up climate deltas for one (region, scenario, period) triple.

    Args:
        region_code:  NUTS-1 code, e.g. "UKI" for London.
        scenario:     Emission scenario, one of `KNOWN_SCENARIOS`.
        period:       Target central year, one of `KNOWN_PERIODS`.

    Returns:
        A `ClimateDeltas` with `delta_ambient_C` (degC) and
        `delta_moisture` (m3/m3).

    Raises:
        ValueError: if any of the three keys is outside the available set.
    """
    if scenario not in KNOWN_SCENARIOS:
        raise ValueError(
            f"unknown scenario {scenario!r}; available: {list(KNOWN_SCENARIOS)}"
        )
    if period not in KNOWN_PERIODS:
        raise ValueError(
            f"unknown period {period!r}; available: {list(KNOWN_PERIODS)}"
        )

    table = _load_table()
    key = (region_code, scenario, period)
    if key not in table:
        # Distinguish "unknown region" from "everything else missing"
        known_regions = sorted({r for (r, _, _) in table})
        if region_code not in known_regions:
            raise ValueError(
                f"unknown region {region_code!r}; "
                f"available: {known_regions}"
            )
        # Region known but combination missing — data file is partial
        raise ValueError(
            f"no row in UKCP18 deltas for {key!r} — data file may be incomplete"
        )
    return table[key]


def list_regions() -> list[tuple[str, str]]:
    """All available (region_code, region_name) pairs, sorted by code."""
    table = _load_table()
    seen: dict[str, str] = {}
    for entry in table.values():
        seen.setdefault(entry.region_code, entry.region_name)
    return sorted(seen.items())


def list_scenarios() -> tuple[str, ...]:
    """All available scenario strings."""
    return KNOWN_SCENARIOS


def list_periods() -> tuple[int, ...]:
    """All available target periods (central year)."""
    return KNOWN_PERIODS


__all__ = [
    "ClimateDeltas",
    "DATA_FILE",
    "KNOWN_SCENARIOS",
    "KNOWN_PERIODS",
    "get_climate_deltas",
    "list_regions",
    "list_scenarios",
    "list_periods",
]
