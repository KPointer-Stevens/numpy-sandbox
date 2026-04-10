import os
import numpy as np
import sympy as smp
import matplotlib.pyplot as plt


# Returns dew point from ambient temp and relative humidity:
def dew_pt(temp_aC, rh_percent):
    t_a = temp_aC
    h = rh_percent/100
    # avoid divide-by-zero:
    if t_a == -243.04:  t_a = -243.041
    gamma = np.log(h) + (17.625*t_a) / (243.04+t_a)
    # avoid another divide-by-zero:
    if gamma == 17.625:  gamma = 17.6251
    return (243.04*gamma) / (17.625-gamma)

# Returns median radiant temperature from globe temp, air velocity, and dry bulb temp:
def mean_radiant_from_globe(GT, AV, DB):
    return np.float_power(np.float_power(GT+273,4)+(((110000000*np.float_power(AV,0.6))/(0.95*np.float_power(0.15,0.4)))*(GT-DB)),0.25)-273

# Returns partial vapor pressure in hPa:
def Buck_equation(temp_C):
    t = temp_C
    if t == -257.14: t = -257.141
    return 6.1121 * np.exp((18.678-(t/234.5))*(t/(257.14+t)))

# Wrapper converting Buck equation result to kiloPascals:
def Buck_eq_kPa(temp_C):
    return Buck_equation(temp_C) / 10.0

# Wrapper converting Buck equation result to Pascals:
def Buck_eq_Pa(temp_C):
    return 100 * Buck_equation(temp_C)

# Approximates wet bulb temp from air velocity, ambient temp, globe temp, and relative humidity:
def approximate_wet_bulb(v, a, g, h, p):
    if v < 0.01:    # do not factor wind if near-zero to avoid artifacts inherent to ISO model
        print("wind speed ~0; using psychrometric...\n")
        return psychrometric_wet_bulb(a, p, h)

    # Arbitrary starting values:
    w_guess = dew_pt(a, h)
    p_w = Buck_equation(w_guess)
    p_a = Buck_equation(a)
    r = mean_radiant_from_globe(g, v, a)
    dir = 1
    incr = 10
    result = 1
    result_prev = 1
    safetyCheck = 1000

    while (abs(result) > 0.05):
        if safetyCheck > 0:
            safetyCheck -= 1
        else:
            print("ERROR! LOOPED TOO LONG!")
            print("dumped results: ", result_prev, ", ", result)
            return w_guess
        
        w_guess = w_guess + incr * dir
        p_w = Buck_equation(w_guess)

        term1 = 4.18*np.float_power(v,0.444)*(a-w_guess) + 0.00000001*((np.float_power(r+273,4))-(np.float_power(w_guess+273,4)))
        term2 = 77.1*np.float_power(v,0.421)*(p_w-((h/100)*p_a))

        result_prev = result
        result = term1-term2

        if (abs(result) < 0.02):
                return w_guess
        if (result < 0 and result_prev > 0) or (result_prev < 0 and result > 0):
            dir = dir * -1
            incr = incr/10

    return w_guess


# Approximates wet bulb temp from t_a, baro pressure, and relative humidity:
def psychrometric_wet_bulb(a, p, h):
    p_s = Buck_equation(a)
    P = p_s * (h/100)

    p_diff = 1
    prev_p_diff = p_diff
    w_guess = dew_pt(a,h)
    dir = 1
    incr = 10

    print(f"p_s={p_s:.6f}, P={P:.6f}, dew_pt={w_guess:.6f}\n")

    safetycheck = 1000
    while abs(p_diff) > 0.02:
        safetycheck = safetycheck - 1
        if safetycheck <= 0:
            print("ERROR! PSYCHRO. LOOPED TOO LONG!")
            print("dumped values: p_diff=", p_diff, ", prev_p_diff=", prev_p_diff, ", w=", w_guess)
            return w_guess
        prev_p_diff = p_diff
        p_wguess = Buck_equation(w_guess)
        P_guess = p_wguess - 0.000667 * (p/10.0) * (a - w_guess)
        p_diff = P - P_guess
        # print(f"w_guess={w_guess:.3f}\t\tp_wguess={p_wguess:.3f}\t\tP_guess={P_guess:.3f}\t\tprev_p_diff={prev_p_diff:.3f}\t\tp_diff={p_diff:.3f}\t\tincr={incr:.3f}")

        if abs(p_diff) <= 0.02:
            # print(f"END.\tp_diff = {p_diff:0.4f}\t\tw_guess = {w_guess:0.4f}\n")
            break
        if (p_diff < 0 and prev_p_diff > 0) or (p_diff > 0 and prev_p_diff < 0):
            # print("target passed, turning around...\n")
            dir = dir * -1
            incr = incr / 10.0
        w_guess = w_guess + incr * dir
    
    return w_guess

# Returns "film temperature"--average of ambient and wet bulb temp, in Kelvin:
def temp_fK(temp_aK, temp_wK):
    return (temp_aK+temp_wK)/2

def wet_bulb_liljegren(t_aC, rh_percent, p_mBar, windspeed_mps, radiance_Wpm2):
    # Final target value, wet bulb temperature:
    t_wC = dew_pt(t_aC, rh_percent)     # use dew point as starting guess, Celsius

    # input unit conversions:
    t_aK = t_aC + 273.15                # Celsius -> Kelvin
    t_wK = t_wC + 273.15                # Celsius -> Kelvin
    rh = rh_percent/100.0               # percent -> ratio
    P = p_mBar*100                      # millibar -> Pascal
    u = windspeed_mps                   # N/A; better variable name
    S_dir = radiance_Wpm2               # N/A; better variable name

    # Constants:
    g = 9.81                            # gravitational constant, m/s^2
    sigma = 0.0000000567                # Stefan-Boltzmann constant, W/m^2*K^4
    R_d = 287.05                        # ???, J/kg*K TODO
    mu_0 = 0.00001716                   # Sutherland reference constant for air viscosity at reference temp, kg/ms
    t_0K = 273.15                       # reference temperature for mu_0, Kelvin
    S_K = 110.4                         # Sutherland temperature, Kelvin
    c_p = 1005                          # specific heat of air, J/kg*K
    P_0 = 101325                        # pressure constant?, Pascals  TODO: verify
    D_0 = 0.0000226                     # diffusivity constant?, m^2/s  TODO: verify

    # Experimental constants (experimentally-fixed variables):
    D = 0.007                           # wick diameter, m
    E_wick = 0.95                       # wick emissivity, ratio
    E_atm = 0.85                        # atmospheric emissivity, ratio TODO: double-check approximation
    alpha = 0.4                         # wick albedo, ratio
    # theta_z = 45                        # Solar zenith angle, degrees
    # theta_zRad = np.deg2rad(theta_z)    # Solar zenith angle, radians

    # Arbitrary starting values:
    dir = 1
    incr = 10
    Q_net = 1
    Q_net_prev = Q_net

    i = 1000
    while i > 0:
        i -= 1
        # Calculate values:
        t_wK = t_wC + 273.15                                                # Celsius -> Kelvin
        t_fK = temp_fK(t_aK, t_wK)                                          # film temperature, Kelvin
        e_sa = Buck_eq_Pa(t_aC)                                             # saturation vapor pressure at ambient temperature, Pascals
        e_a = rh * e_sa                                                     # partial water vapor pressure at ambient temperature, Pascals
        e_sw = Buck_eq_Pa(t_wC)                                             # saturation vapor pressure at wet bulb temperature, Pascals
        q = 0.622 * e_a / (P-0.378*e_a)                                     # specific humidity, ratio
        rho = P / (R_d*t_fK*(1+0.61*q))                                     # air density, kg/m^3
        mu = mu_0 * np.float_power(t_fK/t_0K,3/2) * (t_0K+S_K)/(t_fK+S_K)   # dynamic viscosity of air (Sutherland eq.), kg/m*s
        k = 0.0241 * np.float_power(t_fK/273.15,0.9)                        # thermal conductivity of air, W/m*k
        D_v = D_0 * np.float_power(t_fK/273.15,1.75) * P_0/P                # water vapor diffusivity in air, m^2/s
        Pr = (c_p*mu)/k                                                     # Prandtl number, unitless TODO: verify
        Sc = mu/(rho*D_v)                                                   # Schmidt number (evaporation efficiency), unitless TODO: verify
        Re = (rho*u*D)/mu                                                   # Reynolds numebr (convection strength), unitless TODO: verify
        beta = 1/t_fK                                                       # coefficient of thermal expansion, Kelvin^-1 TODO
        v = mu/rho                                                          # kinematic viscosity, m^2/s
        Gr = (g*beta*abs(t_wK-t_aK)*np.float_power(D,3))/np.float_power(v,2)    # Grashof number, unitless TODO: verify
        C = 0.48                                                            # ???, ??? TODO
        n1 = 1/4                                                            # ???, ??? TODO
        n2 = 3                                                              # ???, ??? TODO
        # Nusselt convection components—Forced, Natural, Combined:
        Nu_f = 0.3 + (0.62*np.float_power(Re,1/2)*np.float_power(Pr,1/3))/(np.float_power(1 + np.float_power(0.4/Pr,2/3),1/4)) * np.float_power(1+np.float_power(Re/282000,5/8),4/5)
        Nu_n = C * np.float_power(Gr*Pr,n1)
        Nu = np.float_power(np.float_power(Nu_f,n2) + np.float_power(Nu_n,n2),1/n2)
        h_c = (Nu*k)/D                                          # convective heat coefficient, W/m^2*K
        Sh = Nu * np.float_power(Sc/Pr,1/3)                     # Sherwood number, unitless
        h_e = (Sh*D_v)/D                                        # mass transfer coefficient, m/s
        L_v = 2501000 - 2370*t_wC                               # latent heat of vaporization at wet bulb temp, J/kg
        # F_dir = np.sin(np.deg2rad(theta_z))                     # geometric factor from zenith angle, unitless
        # F_eff = 1/np.pi + np.sin(np.deg2rad(theta_z))           # effective geometric zenith angle factor, unitless
        F_est = 0.5                                             # estimated all-purpose geometric zenith angle factor, unitless
        # F_estH = 0.6                                            # estimated geometric zenith angle factor for middle-day, unitless
        # F_estL = 0.35                                           # estimated geometric zenith angle factor for early- or late-day, unitless
        # R_short = (1-alpha)*(S_dir*F_eff)                       # shortwave radiation component, W/m^2
        R_short = (1-alpha)*(S_dir*F_est)                       # shortwave radiation component, W/m^2
        R_longIn = E_atm*sigma*np.float_power(t_aK,4)           # incoming longwave radiation component, W/m^2
        R_longOut = E_wick*sigma*np.float_power(t_wK,4)         # outgoing longwave radiation component, ?
        
        # Calculate fundamental heat flux components:
        Q_evap = rho*L_v*h_e * (e_sw-e_a)/(P)                   # evaporative heat flux, W/m^2
        Q_rad = R_short + R_longIn - R_longOut                  # radiative heat flux, W/m^2
        Q_conv = h_c * (t_aC - t_wC)                            # convective heat flux, W/m^2

        # Calculate final net heat flux:
        Q_net_prev = Q_net
        Q_net = Q_evap + Q_rad + Q_conv                         # total net heat flux: when Q_net ≈ 0, t_w is correct
        print(Q_net_prev, Q_net)

        if abs(Q_net) < 0.02:
            return t_wC
        
        if (Q_net_prev > 0 and Q_net < 0) or (Q_net_prev < 0 and Q_net > 0):
            print("passed, turning around...")
            dir *= -1
            incr /= 10
        
        t_wC += incr * dir                                      # new wet bulb guess
    
    return t_wC


# SCRIPT BODY START:
ambientTemp = 24.44
globeTemp = 36.12
humidity = 35.27
windSpeed = 1.19
baroPressure = 1005.79
solarRadiance = 1028.91
print("\nambientTemp=\t", ambientTemp, "\nbaroPressure=\t", baroPressure, "\nglobeTemp=\t", globeTemp, "\nhumidity=\t", humidity, "\nwindSpeed=\t", windSpeed, "\nsolarRadiance=\t", solarRadiance, "\n")

wet_bulb = approximate_wet_bulb(windSpeed, ambientTemp, globeTemp, humidity, baroPressure)
print(f"WBT={wet_bulb:.5f}")

wbgt = 0.7*wet_bulb + 0.2*globeTemp + 0.1*ambientTemp
print(f"WBGT={wbgt:.5f}\n")

wb_liljegren = wet_bulb_liljegren(ambientTemp, humidity, baroPressure, windSpeed, solarRadiance)
print(f"WBT={wb_liljegren:.5f}")

wbgt = 0.7*wb_liljegren + 0.2*globeTemp + 0.1*ambientTemp
print(f"WBGT={wbgt:.5f}\n")
