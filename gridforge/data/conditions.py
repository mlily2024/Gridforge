"""
Failure-mode injectors for synthetic cable-year generation.

Four canonical scenarios. Each is a small dataclass that returns:

  - a per-time stress multiplier on the electric field (`field_multiplier(t_s)`)
  - a per-time temperature offset added to the conductor (`temp_offset_C(t_s)`)
  - a per-time partial-discharge rate scalar (`pd_rate_multiplier(t_s)`)

These multipliers are folded into the cable-year simulator's Crine-damage
integration, so different scenarios produce visibly different failure-time
distributions even with identical load and weather.

Failure-mode design follows the published taxonomy in Mazzanti & Marzinotto
(2013), *Extruded Cables for High-Voltage Direct-Current Transmission*, IEEE
Press, Chapter 5 on degradation modes — adapted to the medium-voltage XLPE
distribution context.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SECONDS_PER_YEAR: float = 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class ConditionMode:
    """Base specification for a synthetic condition mode."""

    name: str
    description: str
    seed: int = 0

    def field_multiplier(self, t_s: float) -> float:
        return 1.0

    def temp_offset_C(self, t_s: float) -> float:
        return 0.0

    def pd_rate_multiplier(self, t_s: float) -> float:
        return 1.0


@dataclass(frozen=True)
class HealthyMode(ConditionMode):
    """Nominal operation. No accelerated degradation."""

    name: str = "healthy"
    description: str = "nominal — no defects, design-life Crine kinetics"


@dataclass(frozen=True)
class WaterIngressMode(ConditionMode):
    """Progressive water-tree growth at one cable joint.

    Locally lifted permittivity reduces the effective insulation thickness,
    raising E_max by a factor that grows linearly with simulation time. PD
    activity rises as water trees develop.
    """

    name: str = "water_ingress"
    description: str = "progressive water-tree degradation at a joint"
    onset_year: float = 0.5
    saturation_year: float = 8.0
    max_field_boost: float = 1.6
    max_pd_boost: float = 8.0

    def field_multiplier(self, t_s: float) -> float:
        years = t_s / SECONDS_PER_YEAR
        if years < self.onset_year:
            return 1.0
        progress = min(
            1.0,
            (years - self.onset_year) / (self.saturation_year - self.onset_year),
        )
        return float(1.0 + (self.max_field_boost - 1.0) * progress)

    def pd_rate_multiplier(self, t_s: float) -> float:
        years = t_s / SECONDS_PER_YEAR
        if years < self.onset_year:
            return 1.0
        progress = min(
            1.0,
            (years - self.onset_year) / (self.saturation_year - self.onset_year),
        )
        return float(1.0 + (self.max_pd_boost - 1.0) * progress * progress)


@dataclass(frozen=True)
class ThermalAgeingMode(ConditionMode):
    """Sustained overload — conductor runs hot every day.

    Adds a constant offset to the conductor temperature on top of what the
    transient solver predicts, simulating high-utilisation feeders that run
    consistently above their nameplate rating.
    """

    name: str = "thermal_ageing"
    description: str = "sustained overload — conductor runs hot all year"
    overheat_offset_C: float = 25.0

    def temp_offset_C(self, t_s: float) -> float:
        return self.overheat_offset_C


@dataclass(frozen=True)
class AcceleratedDielectricMode(ConditionMode):
    """Switching-transient bombardment — frequent surge events.

    Modelled as discrete impulse events superposed on the field history. Each
    event boosts E_max by a factor for a short window. Crine damage accumulates
    much faster as a result. PD rate rises in step with the impulse train.
    """

    name: str = "accelerated_dielectric"
    description: str = "switching-transient bombardment — frequent surges"
    impulse_per_year: float = 180.0
    impulse_field_boost: float = 2.5
    impulse_duration_s: float = 60.0

    def field_multiplier(self, t_s: float) -> float:
        # Deterministic Poisson-like impulse train via hash-based jitter
        rng = np.random.default_rng(self.seed * 32_452_843 + int(t_s) // 3600)
        prob_impulse_this_hour = self.impulse_per_year / (365.25 * 24.0)
        if rng.random() < prob_impulse_this_hour:
            return float(self.impulse_field_boost)
        return 1.0

    def pd_rate_multiplier(self, t_s: float) -> float:
        return self.field_multiplier(t_s) ** 3


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODES: dict[str, type[ConditionMode]] = {
    "healthy": HealthyMode,
    "water_ingress": WaterIngressMode,
    "thermal_ageing": ThermalAgeingMode,
    "accelerated_dielectric": AcceleratedDielectricMode,
}


def make_condition(name: str, seed: int = 0, **kwargs) -> ConditionMode:
    """Construct a condition mode by name with overrides."""
    if name not in MODES:
        raise KeyError(f"unknown condition mode: {name}; choose from {list(MODES)}")
    cls = MODES[name]
    return cls(seed=seed, **kwargs)
