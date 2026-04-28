"""
Canonical UK 11 kV underground cable archetypes.

Each archetype bundles geometry + material properties for one cable
configuration commonly found in UK distribution networks. Values are sourced
from BS EN 60228 (conductor sizes), BS 7870 (UK distribution cable practice),
manufacturer datasheets that conform to those standards, and IEC 60287-2-1
material tables.

Adding a new archetype: instantiate `CableGeometry` and `CableMaterials`
with documented values; do not derive parameters from any non-public source.
Each archetype must also be registered in `ARCHETYPES` below so the dataset
generator can resolve it by name.
"""

from __future__ import annotations

from .thermal import (
    CableGeometry,
    CableMaterials,
    InstallationConditions,
    RHO_T_HDPE,
    RHO_T_PILC,
    RHO_T_SOIL_TYPICAL_UK,
    RHO_T_XLPE,
)


# ---------------------------------------------------------------------------
# 1) 11 kV 240 mm^2 Cu XLPE 3-core, individually screened cores, HDPE jacket
# ---------------------------------------------------------------------------
#
# Geometry sources:
#   d_c   18.5 mm   compacted circular Cu, 240 mm^2 Class 2 stranded (BS EN 60228)
#   t_i    3.4 mm   XLPE wall, 11 kV grade (BS 7870-3.10)
#   t_j    2.5 mm   HDPE outer sheath (BS 7870-3.10)
#   D_e   70   mm   typical overall diameter for 240 mm^2 11 kV 3-core XLPE
#
# Electrical sources:
#   R_dc(20 C)  7.55e-5 ohm/m   IEC 60228, Class 2 Cu
#   alpha       3.93e-3  /K     IEC 60228 Cu temperature coefficient
#   R_ac/R_dc   1.02            240 mm^2 Cu @ 50 Hz, IEC 60287-1-1 §2.1
#   C           2.0e-10 F/m     ~0.20 uF/km, typical 11 kV XLPE
#   tan_delta   1.0e-3          XLPE dielectric loss factor

UK_11KV_240MM2_XLPE_3CORE_GEOM = CableGeometry(
    d_c_mm=18.5,
    t_i_mm=3.4,
    t_j_mm=2.5,
    D_e_mm=70.0,
    n_conductors=3,
)

UK_11KV_240MM2_XLPE_3CORE_MAT = CableMaterials(
    rho_t_insulation_KmW=RHO_T_XLPE,
    rho_t_jacket_KmW=RHO_T_HDPE,
    R_dc_20C_ohm_per_m=7.55e-5,
    alpha_per_C=3.93e-3,
    R_ac_dc_ratio=1.02,
    capacitance_F_per_m=2.0e-10,
    tan_delta=1.0e-3,
)

UK_11KV_240MM2_XLPE_3CORE = (
    UK_11KV_240MM2_XLPE_3CORE_GEOM,
    UK_11KV_240MM2_XLPE_3CORE_MAT,
)


# ---------------------------------------------------------------------------
# 2) 11 kV 95 mm^2 Cu XLPE 3-core — smaller distribution feeder
# ---------------------------------------------------------------------------
#
# Common on radial feeders supplying ~600 kVA secondary substations.
#
# Geometry sources (BS EN 60228, BS 7870-3.10, manufacturer data conformant):
#   d_c   11.5 mm   compacted Cu, 95 mm^2 Class 2 stranded
#   t_i    3.4 mm   XLPE wall, 11 kV grade
#   t_j    2.0 mm   HDPE outer sheath
#   D_e   50   mm   typical overall diameter for 95 mm^2 3-core XLPE
#
# Electrical:
#   R_dc(20 C)  1.93e-4 ohm/m   IEC 60228 Class 2 Cu
#   R_ac/R_dc   1.005            95 mm^2 — minimal skin effect at 50 Hz
#   C           1.6e-10 F/m     thinner conductor, smaller capacitance

UK_11KV_95MM2_CU_XLPE_3CORE_GEOM = CableGeometry(
    d_c_mm=11.5,
    t_i_mm=3.4,
    t_j_mm=2.0,
    D_e_mm=50.0,
    n_conductors=3,
)

UK_11KV_95MM2_CU_XLPE_3CORE_MAT = CableMaterials(
    rho_t_insulation_KmW=RHO_T_XLPE,
    rho_t_jacket_KmW=RHO_T_HDPE,
    R_dc_20C_ohm_per_m=1.93e-4,
    alpha_per_C=3.93e-3,
    R_ac_dc_ratio=1.005,
    capacitance_F_per_m=1.6e-10,
    tan_delta=1.0e-3,
)

UK_11KV_95MM2_CU_XLPE_3CORE = (
    UK_11KV_95MM2_CU_XLPE_3CORE_GEOM,
    UK_11KV_95MM2_CU_XLPE_3CORE_MAT,
)


# ---------------------------------------------------------------------------
# 3) 11 kV 300 mm^2 Cu XLPE single-core — larger primary feeder
# ---------------------------------------------------------------------------
#
# Single-core trefoil installations are common on urban primaries between
# bulk-supply points. We model the cable as a single isolated buried entity
# (n_conductors = 1); mutual heating from adjacent phase cores is a
# follow-up correction documented in the IEC 60287-2-1 group-rating annex.
#
# Geometry sources:
#   d_c   20.5 mm   compacted Cu, 300 mm^2 Class 2 stranded
#   t_i    3.4 mm   XLPE wall, 11 kV grade
#   t_j    2.5 mm   HDPE outer sheath
#   D_e   40   mm   single-core overall diameter (no triplex assembly)
#
# Electrical:
#   R_dc(20 C)  6.01e-5 ohm/m   IEC 60228 Class 2 Cu
#   R_ac/R_dc   1.04             300 mm^2 — modest skin effect at 50 Hz
#   C           2.4e-10 F/m     larger conductor diameter, slightly higher capacitance

UK_11KV_300MM2_CU_XLPE_1CORE_GEOM = CableGeometry(
    d_c_mm=20.5,
    t_i_mm=3.4,
    t_j_mm=2.5,
    D_e_mm=40.0,
    n_conductors=1,
)

UK_11KV_300MM2_CU_XLPE_1CORE_MAT = CableMaterials(
    rho_t_insulation_KmW=RHO_T_XLPE,
    rho_t_jacket_KmW=RHO_T_HDPE,
    R_dc_20C_ohm_per_m=6.01e-5,
    alpha_per_C=3.93e-3,
    R_ac_dc_ratio=1.04,
    capacitance_F_per_m=2.4e-10,
    tan_delta=1.0e-3,
)

UK_11KV_300MM2_CU_XLPE_1CORE = (
    UK_11KV_300MM2_CU_XLPE_1CORE_GEOM,
    UK_11KV_300MM2_CU_XLPE_1CORE_MAT,
)


# ---------------------------------------------------------------------------
# 4) 11 kV 240 mm^2 Cu PILC 3-core — older paper-insulated lead-covered cable
# ---------------------------------------------------------------------------
#
# Paper-insulated lead-covered (PILC) cables predominate in older parts of
# UK 11 kV networks. They are still being retrofitted out as XLPE replaces
# them, so a meaningful share (>20 %) of in-service cable is PILC. PILC
# uses paper impregnated with viscous oil as insulation, with a continuous
# lead sheath and steel-wire armour. Key differences vs XLPE:
#
#   higher rho_T_insulation  (5.0 vs 3.5 K.m/W) — paper conducts heat worse
#   higher tan_delta          (3.5e-3 vs 1.0e-3) — more dielectric loss
#   lower thermal capacity    (paper + oil less massive than XLPE per volume)
#
# Geometry approximated to match XLPE 240 mm^2 dimensions (the cable
# physical envelope is similar; the differences are in the materials).
#
# Sources:
#   IEC 60287-2-1 Table 1 (paper rho_T = 5.0 K.m/W for impregnated paper)
#   IEEE Std 400.3 (typical tan_delta for in-service PILC at 11 kV)

UK_11KV_240MM2_CU_PILC_3CORE_GEOM = CableGeometry(
    d_c_mm=18.5,
    t_i_mm=3.4,
    t_j_mm=2.5,
    D_e_mm=72.0,    # slightly larger due to lead sheath thickness
    n_conductors=3,
)

UK_11KV_240MM2_CU_PILC_3CORE_MAT = CableMaterials(
    rho_t_insulation_KmW=RHO_T_PILC,
    rho_t_jacket_KmW=RHO_T_HDPE,    # outer serving over the lead sheath
    R_dc_20C_ohm_per_m=7.55e-5,
    alpha_per_C=3.93e-3,
    R_ac_dc_ratio=1.02,
    capacitance_F_per_m=2.5e-10,    # paper has higher relative permittivity
    tan_delta=3.5e-3,
)

UK_11KV_240MM2_CU_PILC_3CORE = (
    UK_11KV_240MM2_CU_PILC_3CORE_GEOM,
    UK_11KV_240MM2_CU_PILC_3CORE_MAT,
)


# ---------------------------------------------------------------------------
# Typical UK installation conditions for direct-buried 11 kV distribution cable
# ---------------------------------------------------------------------------

UK_TYPICAL_INSTALLATION = InstallationConditions(
    burial_depth_m=0.8,
    soil_thermal_resistivity_KmW=RHO_T_SOIL_TYPICAL_UK,
    ambient_soil_temp_C=15.0,
)


# ---------------------------------------------------------------------------
# Registry — canonical name -> (geom, mat) lookup
# ---------------------------------------------------------------------------
#
# Used by `gridforge.data.cable_year.simulate_cable_year` to resolve an
# archetype string given on a CableYearSpec.

ARCHETYPES: dict[str, tuple[CableGeometry, CableMaterials]] = {
    "11kV_240mm2_Cu_XLPE_3c": UK_11KV_240MM2_XLPE_3CORE,
    "11kV_95mm2_Cu_XLPE_3c": UK_11KV_95MM2_CU_XLPE_3CORE,
    "11kV_300mm2_Cu_XLPE_1c": UK_11KV_300MM2_CU_XLPE_1CORE,
    "11kV_240mm2_Cu_PILC_3c": UK_11KV_240MM2_CU_PILC_3CORE,
}


def archetype_by_name(name: str) -> tuple[CableGeometry, CableMaterials]:
    """Resolve an archetype name to its (geom, mat) tuple.

    Raises KeyError with the available names if `name` is not registered.
    """
    if name not in ARCHETYPES:
        raise KeyError(
            f"unknown archetype '{name}'; available: {sorted(ARCHETYPES)}"
        )
    return ARCHETYPES[name]
