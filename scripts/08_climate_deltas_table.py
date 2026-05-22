"""
Print the UKCP18 placeholder climate-deltas table to stdout.

Format: one block per scenario, rows are regions, columns are periods.
Two tables: ambient warming (degC) and soil-moisture change (m3/m3).

Run from the gridforge subpackage root:

    python scripts/08_climate_deltas_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridforge.data.ukcp18 import (
    KNOWN_PERIODS,
    KNOWN_SCENARIOS,
    get_climate_deltas,
    list_regions,
)


def _print_block(scenario: str, metric: str) -> None:
    """One scenario block — rows are regions, columns are periods."""
    print()
    print(f"=== {scenario} — {metric} ===")
    header = f"{'Region':<28}" + "".join(f" {p:>10}" for p in KNOWN_PERIODS)
    print(header)
    print("-" * len(header))
    for code, name in list_regions():
        row = f"{code} {name:<24}"
        for period in KNOWN_PERIODS:
            d = get_climate_deltas(code, scenario, period)
            if metric == "delta_ambient_C":
                row += f" {d.delta_ambient_C:>+10.2f}"
            else:
                row += f" {d.delta_moisture:>+10.4f}"
        print(row)


def main() -> int:
    print()
    print("UKCP18 placeholder climate deltas")
    print("=" * 60)
    print("PLACEHOLDER VALUES — see data/climate/README.md")
    print("12 NUTS-1 regions x 3 RCPs x 3 periods")
    print()

    for scenario in KNOWN_SCENARIOS:
        _print_block(scenario, "delta_ambient_C")

    print("\n" + "-" * 60)
    print()
    for scenario in KNOWN_SCENARIOS:
        _print_block(scenario, "delta_moisture")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
