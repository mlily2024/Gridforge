"""Tests for the UKCP18 climate-delta lookup module."""

from __future__ import annotations

import pytest

from gridforge.data.ukcp18 import (
    DATA_FILE,
    KNOWN_PERIODS,
    KNOWN_SCENARIOS,
    ClimateDeltas,
    _load_table,
    get_climate_deltas,
    list_periods,
    list_regions,
    list_scenarios,
)


# ---------------------------------------------------------------------------
# Loader basics
# ---------------------------------------------------------------------------


class TestLoader:
    """The CSV must exist and parse without error."""

    def test_data_file_exists(self) -> None:
        assert DATA_FILE.exists(), (
            f"placeholder CSV missing at {DATA_FILE} — run "
            "data/climate/_generate_placeholder.py"
        )

    def test_table_loads(self) -> None:
        _load_table.cache_clear()
        table = _load_table()
        assert isinstance(table, dict)
        assert len(table) > 0

    def test_row_count_matches_expected_grid(self) -> None:
        # 12 NUTS-1 UK regions x 3 scenarios x 3 periods = 108 entries.
        _load_table.cache_clear()
        table = _load_table()
        assert len(table) == 108, f"expected 108 rows, got {len(table)}"


# ---------------------------------------------------------------------------
# Single-row lookup
# ---------------------------------------------------------------------------


class TestSingleLookup:
    """Looking up a known triple returns a populated ClimateDeltas."""

    def test_known_triple_returns_dataclass(self) -> None:
        d = get_climate_deltas("UKI", "RCP8.5", 2080)
        assert isinstance(d, ClimateDeltas)
        assert d.region_code == "UKI"
        assert d.region_name == "London"
        assert d.scenario == "RCP8.5"
        assert d.period == 2080
        assert d.delta_ambient_C > 0.0  # warming, not cooling
        assert d.delta_moisture < 0.0   # drying, not wetting

    def test_repeated_lookup_is_cached(self) -> None:
        # Two calls should return identical objects (same id via lru_cache).
        a = get_climate_deltas("UKM", "RCP4.5", 2050)
        b = get_climate_deltas("UKM", "RCP4.5", 2050)
        assert a is b


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------


class TestDomainValidation:
    """Out-of-domain keys raise ValueError with informative messages."""

    def test_unknown_region_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown region 'UKZ'"):
            get_climate_deltas("UKZ", "RCP4.5", 2050)

    def test_unknown_scenario_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown scenario 'RCP9.9'"):
            get_climate_deltas("UKI", "RCP9.9", 2050)

    def test_unknown_period_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown period 1995"):
            get_climate_deltas("UKI", "RCP4.5", 1995)


# ---------------------------------------------------------------------------
# Physical monotonicity sanity checks
# ---------------------------------------------------------------------------


class TestMonotonicity:
    """Same region: later periods warmer; higher RCP warmer."""

    @pytest.mark.parametrize("scenario", KNOWN_SCENARIOS)
    @pytest.mark.parametrize("region_code", ["UKI", "UKM"])
    def test_warming_monotone_in_period(self, region_code: str, scenario: str) -> None:
        d_2030 = get_climate_deltas(region_code, scenario, 2030).delta_ambient_C
        d_2050 = get_climate_deltas(region_code, scenario, 2050).delta_ambient_C
        d_2080 = get_climate_deltas(region_code, scenario, 2080).delta_ambient_C
        assert d_2030 < d_2050 < d_2080, (
            f"{region_code}/{scenario} not monotone in period: "
            f"{d_2030:.2f} -> {d_2050:.2f} -> {d_2080:.2f}"
        )

    @pytest.mark.parametrize("period", KNOWN_PERIODS)
    @pytest.mark.parametrize("region_code", ["UKI", "UKM"])
    def test_warming_monotone_in_scenario(self, region_code: str, period: int) -> None:
        d_26 = get_climate_deltas(region_code, "RCP2.6", period).delta_ambient_C
        d_45 = get_climate_deltas(region_code, "RCP4.5", period).delta_ambient_C
        d_85 = get_climate_deltas(region_code, "RCP8.5", period).delta_ambient_C
        assert d_26 < d_45 < d_85, (
            f"{region_code}/{period} not monotone in scenario: "
            f"RCP2.6={d_26:.2f}, RCP4.5={d_45:.2f}, RCP8.5={d_85:.2f}"
        )

    @pytest.mark.parametrize("scenario", KNOWN_SCENARIOS)
    @pytest.mark.parametrize("period", KNOWN_PERIODS)
    def test_southern_regions_warmer_than_northern(
        self, scenario: str, period: int,
    ) -> None:
        # London (UKI) should warm more than Scotland (UKM) under all
        # scenarios and periods.
        london = get_climate_deltas("UKI", scenario, period).delta_ambient_C
        scotland = get_climate_deltas("UKM", scenario, period).delta_ambient_C
        assert london > scotland, (
            f"{scenario}/{period}: London ({london:.2f}) should warm more than "
            f"Scotland ({scotland:.2f})"
        )

    @pytest.mark.parametrize("scenario", KNOWN_SCENARIOS)
    @pytest.mark.parametrize("period", KNOWN_PERIODS)
    def test_moisture_changes_are_non_positive(
        self, scenario: str, period: int,
    ) -> None:
        # Placeholder model: every region dries (or stays the same) in
        # the projected mean. None should wet.
        for region_code, _ in list_regions():
            d = get_climate_deltas(region_code, scenario, period).delta_moisture
            assert d <= 0.0, (
                f"{region_code}/{scenario}/{period}: moisture delta "
                f"{d:+.4f} should be <= 0"
            )


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


class TestDiscoveryHelpers:
    """list_regions / list_scenarios / list_periods return what we expect."""

    def test_list_regions_returns_12(self) -> None:
        regions = list_regions()
        assert len(regions) == 12
        codes = {code for code, _ in regions}
        assert codes == {
            "UKC", "UKD", "UKE", "UKF", "UKG", "UKH",
            "UKI", "UKJ", "UKK", "UKL", "UKM", "UKN",
        }

    def test_list_regions_is_sorted_by_code(self) -> None:
        regions = list_regions()
        codes = [code for code, _ in regions]
        assert codes == sorted(codes)

    def test_list_scenarios_returns_three_rcps(self) -> None:
        assert list_scenarios() == ("RCP2.6", "RCP4.5", "RCP8.5")

    def test_list_periods_returns_three(self) -> None:
        assert list_periods() == (2030, 2050, 2080)
