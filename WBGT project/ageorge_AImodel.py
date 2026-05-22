"""
WBGT Calculator — Liljegren et al. (2008) Model
================================================
Computes outdoor WBGT from user-entered measurements.
Globe temperature is measured directly (150 mm standard globe).
Natural wet bulb temperature is solved from the wick energy balance.

Reference: Liljegren, Carhart, Lawday, Tschopp & Sharp (2008),
  "Modeling the Wet Bulb Globe Temperature Using Standard
   Meteorological Measurements", JOEH 5:10, 645-655.

WBGT = 0.7·Tnwb + 0.2·Tg + 0.1·Ta           [Eq. 1]
"""

import math
from scipy.optimize import brentq

# ── Physical constants ──────────────────────────────────────────────
SIGMA   = 5.67e-8       # Stefan-Boltzmann  (W m⁻² K⁻⁴)
M_AIR   = 28.97e-3      # Molar mass dry air (kg mol⁻¹)
M_H2O   = 18.015e-3     # Molar mass water   (kg mol⁻¹)
R_GAS   = 8.314         # Universal gas const (J mol⁻¹ K⁻¹)
C_P     = 1003.5        # Specific heat dry air (J kg⁻¹ K⁻¹)
R_AIR   = R_GAS / M_AIR # Specific gas const dry air

# ── Wick geometry (QUESTemp 34, per Liljegren) ──────────────────────
D_WICK  = 0.007          # diameter  7 mm
L_WICK  = 0.0254         # length   25.4 mm

# ── Wick radiative / optical properties ─────────────────────────────
EMIS_WICK  = 0.95        # emissivity of wet cotton wick
ALPHA_WICK = 0.4         # surface albedo (white cotton, per Liljegren)

# ── Minimum wind speed (natural-convection floor, Liljegren) ───────
V_MIN = 0.13             # m s⁻¹


# ════════════════════════════════════════════════════════════════════
#  Thermophysical property functions
# ════════════════════════════════════════════════════════════════════

def e_sat(Tc):
    """Saturation vapour pressure (Pa) — Buck (1981)."""
    return 611.21 * math.exp((18.678 - Tc / 234.5) * Tc / (257.14 + Tc))


def viscosity(Tk):
    """Dynamic viscosity of air (Pa·s) — Sutherland."""
    return 1.827e-5 * (291.15 + 120.0) / (Tk + 120.0) * (Tk / 291.15) ** 1.5


def thermal_cond(Tk):
    """Thermal conductivity of air (W m⁻¹ K⁻¹)."""
    return 0.02624 * (Tk / 300.0) ** 0.8646


def diff_h2o(Tk, Pa):
    """Diffusivity of water vapour in air (m² s⁻¹)."""
    return 2.471e-5 * (Tk / 273.15) ** 1.81 * (101325.0 / Pa)


def density(Tk, Pa, ea):
    """Moist-air density (kg m⁻³) via virtual temperature."""
    Tv = Tk / (1.0 - (ea / Pa) * (1.0 - M_H2O / M_AIR))
    return Pa / (R_AIR * Tv)


def latent_heat(Tc):
    """Latent heat of vaporisation (J kg⁻¹)."""
    return (2501.0 - 2.37 * Tc) * 1000.0


# ════════════════════════════════════════════════════════════════════
#  Convective-transfer helpers (cylinder in crossflow)
# ════════════════════════════════════════════════════════════════════

def _hilpert(Re, third_power):
    """Hilpert-type C·Re^m for cylinder crossflow, returns C*Re^m."""
    if Re < 4:
        return 0.891 * Re ** 0.330 * third_power
    elif Re < 40:
        return 0.821 * Re ** 0.385 * third_power
    elif Re < 4000:
        return 0.615 * Re ** 0.466 * third_power
    elif Re < 40000:
        return 0.174 * Re ** 0.618 * third_power
    else:
        return 0.024 * Re ** 0.805 * third_power


def h_conv_wick(Tf, Pa, ea, V):
    """Convective heat-transfer coefficient for the wick (W m⁻² K⁻¹)."""
    rho = density(Tf, Pa, ea)
    mu  = viscosity(Tf)
    Re  = rho * V * D_WICK / mu
    Pr  = mu * C_P / thermal_cond(Tf)
    Nu  = max(_hilpert(Re, Pr ** (1.0 / 3.0)), 0.1)
    return Nu * thermal_cond(Tf) / D_WICK


def h_mass_wick(Tf, Pa, ea, V):
    """Mass-transfer coefficient for the wick (m s⁻¹, concentration-based).
    Via Sherwood number using same Hilpert correlation with Sc."""
    rho  = density(Tf, Pa, ea)
    mu   = viscosity(Tf)
    Re   = rho * V * D_WICK / mu
    Dv   = diff_h2o(Tf, Pa)
    nu_k = mu / rho                        # kinematic viscosity
    Sc   = nu_k / Dv                        # Schmidt number
    Sh   = max(_hilpert(Re, Sc ** (1.0 / 3.0)), 0.1)
    return Sh * Dv / D_WICK


# ════════════════════════════════════════════════════════════════════
#  Longwave radiation estimates
# ════════════════════════════════════════════════════════════════════

def lw_down(Tc, rh):
    """Downwelling longwave irradiance (W m⁻²) — Brutsaert (1975)."""
    Tk   = Tc + 273.15
    ea   = (rh / 100.0) * e_sat(Tc) / 100.0     # vapour pressure in hPa
    emis = min(1.24 * (ea / Tk) ** (1.0 / 7.0), 1.0)
    return emis * SIGMA * Tk ** 4


def lw_up(Tc):
    """Upwelling longwave from ground ≈ black-body at air temperature."""
    Tk = Tc + 273.15
    return 0.999 * SIGMA * Tk ** 4


# ════════════════════════════════════════════════════════════════════
#  Solve Natural Wet-Bulb Temperature  (Liljegren wick energy balance)
# ════════════════════════════════════════════════════════════════════
#
#  Energy balance on the wick (Eq. in paper):
#
#    h_c·(Ta − Tw)                               convective heating
#  + ε·[ (LW↓ + LW↑)/2 − σ·Tw⁴ ]               net longwave
#  + (1−α)·S/π                                   absorbed solar (proj. area / surf. area = 1/π)
#  − L·h_m·M_w·(e_w − e_a) / (R·T_film)         evaporative cooling
#  = 0
#

def solve_Tnwb(Ta_C, rh, Pa_hPa, V_ms, S_Wm2):
    """
    Solve for the natural wet-bulb temperature (°C).

    Parameters
    ----------
    Ta_C    : dry-bulb temperature (°C)
    rh      : relative humidity (%)
    Pa_hPa  : barometric pressure (hPa)
    V_ms    : wind speed (m s⁻¹)
    S_Wm2   : global solar irradiance (W m⁻²)
    """
    Pa  = Pa_hPa * 100.0                        # Pa
    Ta  = Ta_C + 273.15                          # K
    ea  = (rh / 100.0) * e_sat(Ta_C)            # Pa
    V   = max(V_ms, V_MIN)
    LWd = lw_down(Ta_C, rh)
    LWu = lw_up(Ta_C)

    def residual(Tw_C):
        Tw   = Tw_C + 273.15
        Tf   = 0.5 * (Ta + Tw)                  # film temperature
        ew   = e_sat(Tw_C)                       # saturation vap. press. at wick
        hc   = h_conv_wick(Tf, Pa, ea, V)
        hm   = h_mass_wick(Tf, Pa, ea, V)
        Lv   = latent_heat(Tw_C)

        Q_conv  = hc * (Ta_C - Tw_C)
        Q_lw    = EMIS_WICK * ((LWd + LWu) / 2.0 - SIGMA * Tw ** 4)
        Q_solar = (1.0 - ALPHA_WICK) * S_Wm2 / math.pi
        Q_evap  = Lv * hm * M_H2O * (ew - ea) / (R_GAS * Tf)

        return Q_conv + Q_lw + Q_solar - Q_evap

    # Bracket: Tnwb ∈ [well below dew-point, slightly above Ta]
    lo, hi = -50.0, Ta_C + 5.0
    # Widen if needed
    if residual(lo) * residual(hi) > 0:
        lo, hi = -60.0, Ta_C + 10.0

    try:
        return brentq(residual, lo, hi, xtol=1e-4, maxiter=500)
    except ValueError:
        # Fallback — Stull (2011) psychrometric approximation
        T = Ta_C
        return (T * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
                + math.atan(T + rh) - math.atan(rh - 1.676331)
                + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
                - 4.686035)


# ════════════════════════════════════════════════════════════════════
#  WBGT (outdoor)
# ════════════════════════════════════════════════════════════════════

def wbgt_outdoor(Tg_C, Ta_C, rh, Pa_hPa, V_ms, S_Wm2):
    """
    Outdoor WBGT (°C) = 0.7·Tnwb + 0.2·Tg + 0.1·Ta

    Tg_C   : measured 150 mm globe temperature (°C)
    Ta_C   : dry-bulb temperature (°C)
    rh     : relative humidity (%)
    Pa_hPa : barometric pressure (hPa)
    V_ms   : wind speed (m s⁻¹)
    S_Wm2  : global solar irradiance (W m⁻²)
    """
    Tnwb = solve_Tnwb(Ta_C, rh, Pa_hPa, V_ms, S_Wm2)
    WBGT = 0.7 * Tnwb + 0.2 * Tg_C + 0.1 * Ta_C
    return WBGT, Tnwb


def heat_category(wbgt):
    """U.S. Army heat-stress category (TB MED 507, Table 3-1)."""
    if   wbgt < 25.6: return "Below Category 1"
    elif wbgt <= 27.7: return "Cat 1 (Green)   25.6–27.7 °C"
    elif wbgt <= 29.3: return "Cat 2 (Yellow)  27.8–29.3 °C"
    elif wbgt <= 31.0: return "Cat 3 (Red)     29.4–31.0 °C"
    elif wbgt <= 32.2: return "Cat 4 (Dk Red)  31.1–32.2 °C"
    else:              return "Cat 5 (Black)   > 32.2 °C"


# ════════════════════════════════════════════════════════════════════
#  Interactive main
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 56)
    print("  WBGT Calculator  –  Liljegren et al. (2008)")
    print("  Globe: 150 mm standard black globe (measured)")
    print("=" * 56)

    Ta   = float(input("  Dry Bulb Temperature     Ta   (°C)   : "))
    rh   = float(input("  Relative Humidity        RH   (%)    : "))
    Pa   = float(input("  Barometric Pressure      P    (hPa)  : "))
    Tg   = float(input("  Globe Temperature        Tg   (°C)   : "))
    S    = float(input("  Solar Radiation          S    (W/m²) : "))
    V    = float(input("  Wind Speed               V    (m/s)  : "))

    WBGT, Tnwb = wbgt_outdoor(Tg, Ta, rh, Pa, V, S)

    print("\n" + "─" * 56)
    print("  RESULTS")
    print("─" * 56)
    print(f"  Natural Wet Bulb Temp  Tnwb = {Tnwb:7.2f} °C")
    print(f"  Globe Temperature      Tg   = {Tg:7.2f} °C  (measured)")
    print(f"  Dry Bulb Temperature   Ta   = {Ta:7.2f} °C")
    print(f"  ──────────────────────────────────────")
    print(f"  Outdoor WBGT                = {WBGT:7.2f} °C")
    print(f"  Heat Category               : {heat_category(WBGT)}")
    print("─" * 56)
