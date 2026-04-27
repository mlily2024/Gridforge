"""
Canonical UK 11 kV underground cable archetypes.

Each archetype bundles geometry + material properties for one cable
configuration commonly found in UK distribution networks. Values are sourced
from BS EN 60228 (conductor sizes), BS 7870 (UK distribution cable practice),
manufacturer datasheets that conform to those standards, and IEC 60287-2-1
material tables.

Adding a new archetype: instantiate `CableGeometry` and `CableMaterials`
with documented values; do not derive parameters from any non-public source.
"""

from __future__ import annotations

from .thermal import (
    CableGeometry,
    CableMaterials,
    InstallationConditions,
    RHO_T_HDPE,
    RHO_T_SOIL_TYPICAL_UK,
    RHO_T_XLPE,
)


# ---------------------------------------------------------------------------
# 11 kV 240 mm^2 Cu XLPE 3-core, individually screened cores, HDPE outer jacket
# ---------------------------------------------------------------------------
#
# Geometry sources:
#   d_c   18.5 mm   compacted circular Cu, 240 mm^2 Class 2 stranded (BS EN 60228)
#   t_i    3.4 mm   XLPE wall, 11 kV grade (BS 7870-3.10)
#   t_j    2.5 mm   HDPE outer sheath (BS 7870-3.10)
#   D_e   70   mm   typical overall diameter for 240 mm^2 11 kV 3-core XLPE
#                   (manufacturer datasheets conforming to BS 7870-3.10)
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
# Typical UK installation conditions for direct-buried 11 kV distribution cable
# ---------------------------------------------------------------------------
#
#   burial depth      0.8 m    BS 7671 / DNO standard practice
#   soil rho_t        1.0 K.m/W  typical UK clay/loam at temperate moisture
#                                (BS 7870 reference; ranges 0.7-1.5 in service)
#   ambient soil      15 C     UK summer mean ground temperature at 0.8 m

UK_TYPICAL_INSTALLATION = InstallationConditions(
    burial_depth_m=0.8,
    soil_thermal_resistivity_KmW=RHO_T_SOIL_TYPICAL_UK,
    ambient_soil_temp_C=15.0,
)
