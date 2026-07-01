"""Synthetic dataset generation — load profiles, weather, condition modes,
cable-year simulator, multi-cable dataset assembler."""

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
from .conditions import (
    MODES,
    AcceleratedDielectricMode,
    ConditionMode,
    HealthyMode,
    ThermalAgeingMode,
    WaterIngressMode,
    make_condition,
)
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
    # condition modes
    "MODES",
    "ConditionMode",
    "HealthyMode",
    "WaterIngressMode",
    "ThermalAgeingMode",
    "AcceleratedDielectricMode",
    "make_condition",
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
