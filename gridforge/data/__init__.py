"""Synthetic dataset generation — load profiles, weather, failure modes,
cable-year simulator, multi-cable dataset assembler."""

from .load_profiles import (
    PROFILES,
    LoadSpec,
    commercial,
    industrial,
    mixed,
    residential,
)
from .weather import (
    SOIL_TEMP_AMPLITUDE_K,
    SOIL_TEMP_MEAN_C,
    WeatherSpec,
    soil_ambient_C,
    soil_moisture_index,
)
from .failure_modes import (
    MODES,
    AcceleratedDielectricMode,
    FailureMode,
    HealthyMode,
    ThermalAgeingMode,
    WaterIngressMode,
    make_failure_mode,
)
from .cable_year import (
    CableYearResult,
    CableYearSpec,
    simulate_cable_year,
)
from .dataset import (
    DEFAULT_SPLIT_RATIOS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TELEMETRY_COLUMNS,
    DatasetSummary,
    assemble_dataset,
    assign_split,
)

__all__ = [
    # load profiles
    "PROFILES",
    "LoadSpec",
    "residential",
    "commercial",
    "industrial",
    "mixed",
    # weather
    "SOIL_TEMP_MEAN_C",
    "SOIL_TEMP_AMPLITUDE_K",
    "WeatherSpec",
    "soil_ambient_C",
    "soil_moisture_index",
    # failure modes
    "MODES",
    "FailureMode",
    "HealthyMode",
    "WaterIngressMode",
    "ThermalAgeingMode",
    "AcceleratedDielectricMode",
    "make_failure_mode",
    # cable year
    "CableYearResult",
    "CableYearSpec",
    "simulate_cable_year",
    # dataset
    "DEFAULT_SPLIT_RATIOS",
    "SPLIT_TRAIN",
    "SPLIT_VAL",
    "SPLIT_TEST",
    "TELEMETRY_COLUMNS",
    "DatasetSummary",
    "assemble_dataset",
    "assign_split",
]
