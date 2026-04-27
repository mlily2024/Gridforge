"""GridForge — physics-informed digital twin and benchmark for UK 11kV cables."""

__version__ = "0.0.2"

from .physics.thermal import (
    CableGeometry,
    CableMaterials,
    InstallationConditions,
    ThermalResistances,
    ThermalSolution,
    solve_steady_state,
    thermal_resistance_T1,
    thermal_resistance_T3,
    thermal_resistance_T4,
    ac_resistance_at_temp,
    dielectric_loss_per_phase,
)
from .physics.cable_archetype import (
    UK_11KV_240MM2_XLPE_3CORE,
    UK_TYPICAL_INSTALLATION,
)
from .physics.electrical import (
    average_e_field,
    conductor_outer_radius_m,
    insulation_outer_radius_m,
    max_e_field,
    min_e_field,
    radial_e_field,
)
from .physics.transient import (
    TransientResult,
    cable_thermal_capacitance,
    simulate_transient,
    total_thermal_resistance,
)
from .physics.ageing import (
    CrineParameters,
    cumulative_damage,
    damage_rate,
    life_at_constant_stress,
    remaining_useful_life,
    remaining_useful_life_years,
)

__all__ = [
    "__version__",
    # thermal (steady state)
    "CableGeometry",
    "CableMaterials",
    "InstallationConditions",
    "ThermalResistances",
    "ThermalSolution",
    "solve_steady_state",
    "thermal_resistance_T1",
    "thermal_resistance_T3",
    "thermal_resistance_T4",
    "ac_resistance_at_temp",
    "dielectric_loss_per_phase",
    # archetypes
    "UK_11KV_240MM2_XLPE_3CORE",
    "UK_TYPICAL_INSTALLATION",
    # electrical
    "average_e_field",
    "conductor_outer_radius_m",
    "insulation_outer_radius_m",
    "max_e_field",
    "min_e_field",
    "radial_e_field",
    # transient
    "TransientResult",
    "cable_thermal_capacitance",
    "simulate_transient",
    "total_thermal_resistance",
    # ageing
    "CrineParameters",
    "cumulative_damage",
    "damage_rate",
    "life_at_constant_stress",
    "remaining_useful_life",
    "remaining_useful_life_years",
]
