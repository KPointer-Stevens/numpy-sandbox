import numpy as np
from scipy.optimize import brentq

# Input variables:
WINDSPEED_MPS   = 0.6
DRYBULBTEMP_C   = 35
GLOBETEMP_C     = 50
HUMIDITY_PERCENT= 35
PRESSURE_MBAR   = 1020
IRRADIANCE_WPM2 = 650
ZENITHANGLE_DEG = 35

# Universal constants:
pi = 3.14159265359                      # ratio of a circle's circumference to its diameter
sigma = 0.0000000567                    # Stefan-Boltzmann constant, W/m^2*K^4
M_h2o = 0.018015                        # molecular weight of water, kg/mol
M_air = 0.02897                         # molecular weight of air, kg/mol
mu_0 = 0.00001716                       # Sutherland reference constant for air viscosity at reference temp, kg/ms
t_0K = 273.15                           # reference temperature for mu_0, Kelvin
S_K = 110.4                             # Sutherland temperature, Kelvin
R_gas = 8.31446                         # molar gas constant, J/mol*K
R_air = R_gas/M_air                     # specific gas constant for air, J
c_p = 1005.0                            # specific heat capacity of dry air, J/K*kg
P_0 = 101325                            # pressure constant?, Pascals
D_0 = 0.0000226                         # diffusivity constant for water vapor in air, m^2/s
g = 9.81                                # Earth gravitational constant, m/s^2
S_0 = 1367                              # maximum solar irradiance sans atmosphere, W/m^2

# a, b, c values prescribed in Liljegren:
a = 0.56
b = 0.281
c = 0.4

# Experimental constants (experimentally-fixed variables, as prescribed in Liljegren):
D = 0.007                               # wick diameter, m
L = 0.0254                              # wick length, m
A = pi*D*L                              # wick surface area, m^2
E_wick = 0.95                           # wick emissivity, ratio
E_atm = 0.85                            # atmospheric emissivity, ratio
alpha_w = 0.95                          # wick albedo, ratio
alpha_sfc = 0.45                        # surface albedo, ratio


# Returns saturation vapor pressure in Pascals from temperature
def e_sat(temp_C):
    if temp_C >= 0:
        return 100 * (6.1121 * np.exp((18.678 - (temp_C / 234.5)) * (temp_C / (257.14 + temp_C))) )
    return 100 * (6.1121 * np.exp((23.036 - (temp_C / 333.7)) * (temp_C / (279.82 + temp_C))) )

# Returns "ΔF_net/A" term using Eq.(12) from Liljegren:
def deltaFnetPerA(zenithAngle, S_rad, temp_aC, temp_wC):
    S_max = 0
    f_dir = 0
    if zenithAngle <= 89.5:
        S_max = S_0 * np.cos(np.deg2rad(zenithAngle))
        Sstar = S_rad/S_max
        f_dir = np.exp(3-1.34*Sstar-1.65/Sstar)
    deltaFNetperA_T1 = sigma * E_wick * (0.5 * (1 + E_atm) * np.float_power(temp_aC,4) - np.float_power(temp_wC,4))
    deltaFNetperA_T2 = (1 - alpha_w) * S_rad * ((1 - f_dir) * (1 + (D / (4 * L))) + f_dir * ((np.tan(np.deg2rad(zenithAngle)) / pi) + (D / (4 * L)) + alpha_sfc))
    return deltaFNetperA_T1 + deltaFNetperA_T2

# Returns natural wet bulb temp (t_nwb) in °C from atmospheric parameters, as per Liljegren (2008):
def wet_bulb_liljegren(t_aC, rh_percent, p_mBar, windspeed_mps, radiance_Wpm2, zenithAngle_deg):
    # Final target value, wet bulb temperature:
    t_wCguess = t_aC                         # ambient (dry bulb) temp in Celsius, used as starting guess for wet bulb
    t_wDiff = 0
    prevDiff = 0
    dir = -1
    incr = 10

    # input conversions:
    t_aK = t_aC + 273.15                # Celsius -> Kelvin
    rh = rh_percent/100.0               # percent -> ratio
    P = p_mBar*100                      # millibar -> Pascal
    V = windspeed_mps                   # N/A; easier variable name
    S = radiance_Wpm2                   # N/A; easier variable name
    s_za = zenithAngle_deg              # approx. avg. solar zenith angle, degrees TODO: get actual data from lat/long/time?

    e_a = rh * e_sat(t_aC)                                          # partial water vapor pressure at ambient temperature, Pascals
    k = 0.0241 * np.float_power(t_aK/273.15,0.9)                    # thermal conductivity of air, W/m*k
    q = 0.622 * e_a / (P-0.378*e_a)                                 # specific humidity, ratio
    rho = P / (R_air*t_aK*(1+0.61*q))                               # fluid density of air, kg/m^3
    mu = mu_0*np.float_power(t_aK/t_0K,3/2)*(t_0K+S_K)/(t_aK+S_K)   # fluid viscosity of air (Sutherland eq.), kg/m*s
    D_v = D_0 * np.float_power(t_aK/273.15,1.75) * P_0/P            # water vapor diffusivity in air, m^2/s

    Re = (rho*V*D)/mu                                               # Reynolds number (convection strength), unitless
    Pr = (c_p*mu)/k                                                 # Prandtl number
    Sc = mu/(rho*D_v)                                               # Schmidt number
    
    h = (k/D)*b*np.float_power(Re,1-c)*np.float_power(Pr,1-a)       # convective heat coefficient, W/m^2*K

    safetyCheck = 1000
    while safetyCheck > 0:
        safetyCheck -= 1
        # Calculate new values:
        e_w = e_sat(t_wCguess)                              # saturation vapor pressure at wet bulb temperature, Pascals
        DFnetPerA = deltaFnetPerA(s_za, S, t_aC, t_wCguess) # Eq.(12) result
        deltaH = 2500900 - 2439*t_wCguess                   # latent heat of vaporization at wet bulb temp, J/kg

        # Eq.(9):
        t_wCResult = t_aC - (deltaH/c_p)*(M_h2o/M_air)*np.float_power(Pr/Sc,a)*((e_w-e_a)/(P-e_w)) + (DFnetPerA/h)
        prevDiff = t_wDiff
        t_wDiff = t_wCResult - t_wCguess

        if abs(t_wDiff) < 0.02: break

        if (t_wDiff > 0 and prevDiff < 0) or (t_wDiff < 0 and prevDiff > 0):
            dir *= -1
            incr /= 10
        elif t_wCguess < -273.15 and dir == -1:
            dir = 1
        
        t_wCguess += incr * dir

    if safetyCheck <= 0:
        print("ERROR: NO ROOTS FOUND. RETURNED NAN.")
        return np.nan

    return t_wCguess

if __name__ == "__main__":
    WBT = wet_bulb_liljegren(DRYBULBTEMP_C, HUMIDITY_PERCENT, PRESSURE_MBAR, WINDSPEED_MPS, IRRADIANCE_WPM2, ZENITHANGLE_DEG)
    WBGT = 0.7*WBT + 0.2*GLOBETEMP_C + 0.1*DRYBULBTEMP_C
    print()
    print(f"windspeed={WINDSPEED_MPS:0.2f}, ambient={DRYBULBTEMP_C:0.2f}, globe={GLOBETEMP_C:0.2f}, humidity={HUMIDITY_PERCENT:0.2f}, pressure={PRESSURE_MBAR:0.2f}, solarrad={IRRADIANCE_WPM2:0.2f}, solarangle={ZENITHANGLE_DEG:0.2f}\n")
    print(f"WBT = \t{WBT:.5f}")
    print(f"WBGT = \t{WBGT:.5f}\n")
