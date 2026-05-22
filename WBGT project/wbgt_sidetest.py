import os
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import multiprocessing
import ageorge_AImodel as AG


HEATMAPS = False
GEN_NEW = True

# Standard inputs for single gen:
V_A = 0.5
T_A = 35
T_G = 45
REL_H = 75
BARO = 1020
RAD = 850

# Universal constants:
pi = 3.14159265359                  # ratio of a circle's circumference to its diameter
sigma = 0.0000000567                # Stefan-Boltzmann constant, W/m^2*K^4
M_h2o = 0.018015                    # molecular weight of water, kg/mol
M_air = 0.02897                     # molecular weight of air, kg/mol
mu_0 = 0.00001716                   # Sutherland reference constant for air viscosity at reference temp, kg/ms
t_0K = 273.15                       # reference temperature for mu_0, Kelvin
S_K = 110.4                         # Sutherland temperature, Kelvin
R_gas = 8.31446                     # molar gas constant, J/mol*K
R_air = R_gas/M_air                 # specific gas constant for air, J
c_p = 1005.0                        # specific heat capacity of dry air, J/K*kg
P_0 = 101325                        # pressure constant?, Pascals  TODO: verify
D_0 = 0.0000226                     # diffusivity constant for water vapor in air?, m^2/s  TODO: verify
g = 9.81                            # Earth gravitational constant, m/s^2
S_0 = 1367                          # maximum solar irradiance sans atmosphere, W/m^2
# Experimental constants (experimentally-fixed variables):
D_wick = 0.007                      # wick diameter, m
L_wick = 0.0254                     # wick length, m
A_wick = pi*D_wick*L_wick           # wick surface area, as prescribed in Liljegren (m^2)
E_wick = 0.95                       # wick emissivity, ratio
E_atm = 0.85                        # atmospheric emissivity, ratio TODO: double-check approximation
alpha_wick = 0.95                   # wick albedo, ratio
alpha_sfc = 0.45                    # estimated albedo of surface, ratio

# Scaling for params; Count = number of points per param; max idx value = Count-1
aCount = 51
am = 1.0
ab = 10.0
hCount = 2
hm = 10.0
hb = 65.0
pCount = 1
pm = 10.0
pb = 1020.0
vCount = 51
vm = 0.03
vb = 0.0
rCount = 3
rm = 50.0
rb = 850.0
gCount = 2
gm = 10.0
gb = 45.0


# Returns dew point from ambient temp and relative humidity:
def dew_pt(temp_aC, rh_percent):
    t_a = temp_aC
    h = rh_percent/100
    # avoid divide-by-zero:
    if t_a == -243.04:  t_a = -243.041
    gamma = np.log(h) + (17.625*t_a)/(243.04+t_a)
    # avoid another divide-by-zero:
    if gamma == 17.625:  gamma = 17.6251
    return (243.04*gamma) / (17.625-gamma)

# Returns median radiant temperature from globe temp, air velocity, and dry bulb temp:
def mean_radiant_from_globe(GT, AV, DB):
    return np.float_power(np.float_power(GT+273,4)+(((110000000*np.float_power(AV,0.6))/(0.95*np.float_power(0.15,0.4)))*(GT-DB)),0.25)-273

# Returns partial vapor pressure in hPa:
def Buck_equation(temp_C):
    # if temp_C < 273: return temp_C # erroneous temp value
    # if temp_C < 0: return 6.1115 * np.exp((23.036-(temp_C/333.7))*(temp_C/(279.82+temp_C)))
    t = temp_C
    if t == -257.14: t = -257.141
    return 6.1121 * np.exp((18.678-(t/234.5))*(t/(257.14+t)))

# Wrapper converting Buck equation result to kiloPascals:
# def Buck_eq_kPa(temp_C):
#     return Buck_equation(temp_C) / 10.0

# Wrapper converting Buck equation result to Pascals:
def Buck_eq_Pa(temp_C):
    return 100 * Buck_equation(temp_C)

# Approximates wet bulb temp from v_a, t_a, t_g, and RH:
def approximate_wet_bulb_ISO(v, a, g, h):#, p):
    # w_psy = 0
    # if v == 0:
    #     return psychrometric_wet_bulb(a, p, h)
    # if v < 0.01:    # do not factor wind if near-zero to avoid artifacts inherent to ISO model
    #     w_psy = psychrometric_wet_bulb(a, p, h)

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

    while abs(result) > 0.05:
        if safetyCheck > 0:
            safetyCheck -= 1
        else:
            print("ERROR! LOOPED TOO LONG!")
            print("dumped results: ", result_prev, ", ", result)
            # return w_guess
            break
        
        w_guess = w_guess + incr * dir
        p_w = Buck_equation(w_guess)

        term1 = 4.18*np.float_power(v,0.444)*(a-w_guess) + 0.00000001*((np.float_power(r+273,4))-(np.float_power(w_guess+273,4)))
        term2 = 77.1*np.float_power(v,0.421)*(p_w-((h/100)*p_a))

        result_prev = result
        result = term1-term2

        if (abs(result) < 0.02):
                # return w_guess
                break
        if (result < 0 and result_prev > 0) or (result_prev < 0 and result > 0):
            dir = dir * -1
            incr = incr/10
    
    # if v < 0.01:
    #     return w_psy*(1-(v/0.01)) + w_guess*(v/0.01)
    #     # return w_psy*(1-(v/0.5)) + w_guess*(v/0.5)

    return w_guess

# Estimates wet bulb temperature without factoring wind:
def psychrometric_wet_bulb(a, p, h):
    p_s = Buck_equation(a)
    P = p_s * (h/100)

    p_diff = 1
    prev_p_diff = p_diff
    w_guess = dew_pt(a,h)
    dir = 1
    incr = 10

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

        if abs(p_diff) <= 0.02:
            break
        if (p_diff < 0 and prev_p_diff > 0) or (p_diff > 0 and prev_p_diff < 0):
            dir = dir * -1
            incr = incr / 10.0
        w_guess = w_guess + incr * dir
    
    return w_guess

# Returns "film temperature"--average of ambient and wet bulb temp, in Kelvin:
def temp_fK(temp_aK, temp_wK):
    return (temp_aK+temp_wK)/2

def wet_bulb_liljegren(t_aC, rh_percent, p_mBar, windspeed_mps, radiance_Wpm2):
    # Final target value, wet bulb temperature:
    t_wC = t_aC                         # use ambient temp as starting guess, Celsius

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
    R_a = 287.05                        # ???, J/kg*K TODO
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
    # alpha = 0.4                         # wick albedo, ratio
    alpha = 0.95                        # wick albedo, ratio
    # theta_z = 45                        # Solar zenith angle, degrees
    # theta_zRad = np.deg2rad(theta_z)    # Solar zenith angle, radians

    # Arbitrary starting values:
    dir = -1                            # t_aC > t_wC, so start from ambient and decrease
    incr = 10
    Q_net = 1
    Q_net_prev = Q_net

    i = 1000
    while i > 0:
        # print(f"i={i}, t_wguess={t_wC:.3f}")
        i -= 1
        # Calculate values:
        t_wK = t_wC + 273.15                                                # Celsius -> Kelvin
        # t_fK = temp_fK(t_aK, t_wK)                                          # film temperature, Kelvin
        e_sa = Buck_eq_Pa(t_aC)                                             # saturation vapor pressure at ambient temperature, Pascals
        e_a = rh * e_sa                                                     # partial water vapor pressure at ambient temperature, Pascals
        e_sw = Buck_eq_Pa(t_wC)                                             # saturation vapor pressure at wet bulb temperature, Pascals
        q = 0.622 * e_a / (P-0.378*e_a)                                     # specific humidity, ratio
        rho = P / (R_a*t_aK*(1+0.61*q))                                     # air density, kg/m^3
        mu = mu_0 * np.float_power(t_aK/t_0K,3/2) * (t_0K+S_K)/(t_aK+S_K)   # dynamic viscosity of air (Sutherland eq.), kg/m*s
        k = 0.0241 * np.float_power(t_aK/273.15,0.9)                        # thermal conductivity of air, W/m*k
        D_v = D_0 * np.float_power(t_aK/273.15,1.75) * P_0/P                # water vapor diffusivity in air, m^2/s
        Pr = (c_p*mu)/k                                                     # Prandtl number, unitless TODO: verify
        Sc = mu/(rho*D_v)                                                   # Schmidt number (evaporation efficiency), unitless TODO: verify
        Re = (rho*u*D)/mu                                                   # Reynolds numebr (convection strength), unitless TODO: verify
        beta = 1/t_aK                                                       # coefficient of thermal expansion, Kelvin^-1 TODO
        v = mu/rho                                                          # kinematic viscosity, m^2/s
        Gr = (g*beta*abs(t_wK-t_aK)*np.float_power(D,3))/np.float_power(v,2)    # Grashof number, unitless TODO: verify
        C = 0.48                                                            # ???, ??? TODO
        n1 = 1/4                                                            # ???, ??? TODO
        n2 = 3                                                              # ???, ??? TODO
        # Nusselt convection components—Forced, Natural, Combined:
        Nu_f = 0.3 + (0.62*np.float_power(Re,1/2)*np.float_power(Pr,1/3))/(np.float_power(1 + np.float_power(0.4/Pr,2/3),1/4)) * np.float_power(1+np.float_power(Re/282000,5/8),4/5)
        Nu_n = C * np.float_power(Gr*Pr,n1)
        Nu = np.float_power(np.float_power(Nu_f,n2) + np.float_power(Nu_n,n2),1/n2)
        # Nu = 2.0 + 0.6*np.float_power(Re,1/2)*np.float_power(Pr,1/3)
        h_c = (Nu*k)/D                                          # convective heat coefficient, W/m^2*K
        Sh = Nu * np.float_power(Sc/Pr,1/3)                     # Sherwood number, unitless
        h_e = (Sh*D_v)/D                                        # mass transfer coefficient, m/s
        L_v = 2501000 - 2370*t_wC                               # latent heat of vaporization at wet bulb temp, J/kg
        # F_dir = np.sin(np.deg2rad(theta_z))                     # geometric factor from zenith angle, unitless
        # R_short = (1-alpha)*(S_dir*F_dir)                       # shortwave radiation component, W/m^2
        F_est = 0.5                                             # estimated all-purpose geometric zenith angle factor, unitless
        R_short = (1-alpha)*(S_dir*F_est)                       # shortwave radiation component, W/m^2
        R_longIn = E_atm*sigma*np.float_power(t_aK,4)           # incoming longwave radiation component, W/m^2
        R_longOut = E_wick*sigma*np.float_power(t_wK,4)         # outgoing longwave radiation component, ?
        
        # Calculate fundamental heat flux components:
        Q_evap = rho*L_v*h_e * (e_sw-e_a)/(P)                   # evaporative heat flux, W/m^2
        Q_rad = R_short + R_longIn - R_longOut                  # radiative heat flux, W/m^2
        Q_conv = h_c * (t_aC - t_wC)                            # convective heat flux, W/m^2

        # Calculate final net heat flux:
        Q_net_prev = Q_net
        Q_net = Q_rad + Q_conv - Q_evap                         # total net heat flux: when Q_net ≈ 0, t_w is correct
        # print(Q_net_prev, Q_net)

        if abs(Q_net) < 0.02:
            if t_wC < -10:
                print(f"Low value detected: t_wC={t_wC:.3f}, Qnet={Q_net:.5f}, Qevap={Q_evap:.5f}, Qrad={Q_rad:.5f}, Qconv={Q_conv:.5f}")
            return t_wC
        
        if (Q_net_prev > 0 and Q_net < 0) or (Q_net_prev < 0 and Q_net > 0):
            dir *= -1
            incr /= 10
        
        t_wC += incr * dir                                      # new wet bulb guess
        if t_wC < 273.15:
            dir = 1
            t_wC += incr * dir
        elif t_wC > 100:
            dir = -1
            t_wC += incr * dir
    
    # print(f"ERROR: Looped too long! dumping values: t_wC={t_wC:.3f}, Qnet={Q_net:.5f}, Qevap={Q_evap:.5f}, Qrad={Q_rad:.5f}, Qconv={Q_conv:.5f}")
    return t_wC

# Returns ΔF_net/A term using Eq.(12) from Liljegren:
def deltaFnetPerA(zenithAngle, S_rad, temp_aC, temp_wC):
    S_max = 0
    f_dir = 0
    if zenithAngle <= 89.5:
        S_max = S_0 * np.cos(np.deg2rad(zenithAngle))
        Sstar = S_rad/S_max
        f_dir = np.exp(3-1.34*Sstar-1.65/Sstar)
    deltaFNetperA_T1 = sigma * E_wick * (0.5 * (1 + E_atm) * np.float_power(temp_aC,4) - np.float_power(temp_wC,4))
    deltaFNetperA_T2 = (1 - alpha_wick) * S_rad * ((1 - f_dir) * (1 + (D_wick / (4 * L_wick))) + f_dir * ((np.tan(np.deg2rad(zenithAngle)) / pi) + (D_wick / (4 * L_wick)) + alpha_sfc))
    return deltaFNetperA_T1 + deltaFNetperA_T2

def paper_wet_bulb_liljegren(t_aC, rh_percent, p_mBar, windspeed_mps, radiance_Wpm2, zenithAngle_deg):
    # Final target value, wet bulb temperature:
    t_wC = t_aC                         # use ambient temp as starting guess, Celsius

    # input unit conversions:
    t_aK = t_aC + 273.15                # Celsius -> Kelvin
    rh = rh_percent/100.0               # percent -> ratio
    P = p_mBar*100                      # millibar -> Pascal
    u = windspeed_mps                   # N/A; better variable name
    S = radiance_Wpm2                   # N/A; better variable name
    s_za = zenithAngle_deg              # solar zenith angle, degrees TODO: estimate mock data? get actual data from lat/long/time?

    e_sa = Buck_eq_Pa(t_aC)             # saturation vapor pressure at ambient temperature, Pascals
    e_a = rh * e_sa                     # partial water vapor pressure at ambient temperature, Pascals
    A = A_wick                          # wick surface area as prescribed in Liljegren, m^2
    k = 0.0241 * np.float_power(t_aK/273.15,0.9)                    # thermal conductivity of air, W/m*k
    q = 0.622 * e_a / (P-0.378*e_a)                                 # specific humidity, ratio TODO: should either/both e_a be e_w instead?
    rho = P / (R_air*t_aK*(1+0.61*q))                               # fluid density of air, kg/m^3
    mu = mu_0*np.float_power(t_aK/t_0K,3/2)*(t_0K+S_K)/(t_aK+S_K)   # fluid viscosity of air (Sutherland eq.), kg/m*s
    D_v = D_0 * np.float_power(t_aK/273.15,1.75) * P_0/P            # water vapor diffusivity in air, m^2/s

    Re = (rho*u*D_wick)/mu                                          # Reynolds number (convection strength), unitless TODO: verify?
    Pr = (c_p*mu)/k                                                 # Prandtl number
    Sc = mu/(rho*D_v)                                               # Schmidt number

    # a,b,c prescribed by Liljegren:
    a = 0.56
    b = 0.281
    c = 0.4
    h = (k/D_wick)*b*np.float_power(Re,1-c)*np.float_power(Pr,1-a)  # convective heat coefficient, W/m^2*K

    def computeEq9(t_wCguess):
        # Calculate new values:
        e_sw = Buck_eq_Pa(t_wC)                             # saturation vapor pressure at wet bulb temperature, Pascals
        e_w = rh * e_sw                                     # partial water vapor pressure at wet bulb temperature, Pascals
        DFnetPerA = deltaFnetPerA(s_za, S, t_aC, t_wC)      # Eq.(12) result
        deltaH = 2500900 - 2439*t_wCguess                   # latent heat of vaporization at wet bulb temp, J/kg

        # Eq.(9):
        t_wCResult = t_aC-(deltaH/c_p) * (M_h2o/M_air) * np.float_power(Pr/Sc,a) * ((e_w-e_a)/(P-e_w)) + (DFnetPerA/h)
        t_wDiff = t_wCResult - t_wCguess
        return t_wDiff                                      # difference between guess and calculation; guess is correct when t_wDiff≈0
    return brentq(computeEq9, dew_pt(t_aC,rh)-50.0, 100.0, xtol=0.02, maxiter=1000)

def new_wet_bulb_liljegren(t_aC, rh_percent, p_mBar, windspeed_mps, radiance_Wpm2):
    # Final target value, wet bulb temperature:
    t_wC = t_aC                         # use ambient temp as starting guess, Celsius

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
    R_a = 287.05                        # ???, J/kg*K TODO
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
    alpha = 0.95                        # wick albedo, ratio

    def balance_Q(t_wCguess):
        # Calculate values:
        t_wK = t_wCguess + 273.15                                                # Celsius -> Kelvin
        e_sa = Buck_eq_Pa(t_aC)                                             # saturation vapor pressure at ambient temperature, Pascals
        e_a = rh * e_sa                                                     # partial water vapor pressure at ambient temperature, Pascals
        e_sw = Buck_eq_Pa(t_wCguess)                                             # saturation vapor pressure at wet bulb temperature, Pascals
        q = 0.622 * e_a / (P-0.378*e_a)                                     # specific humidity, ratio
        rho = P / (R_a*t_aK*(1+0.61*q))                                     # air density, kg/m^3
        mu = mu_0 * np.float_power(t_aK/t_0K,3/2) * (t_0K+S_K)/(t_aK+S_K)   # dynamic viscosity of air (Sutherland eq.), kg/m*s
        k = 0.0241 * np.float_power(t_aK/273.15,0.9)                        # thermal conductivity of air, W/m*k
        D_v = D_0 * np.float_power(t_aK/273.15,1.75) * P_0/P                # water vapor diffusivity in air, m^2/s
        Pr = (c_p*mu)/k                                                     # Prandtl number, unitless TODO: verify
        Sc = mu/(rho*D_v)                                                   # Schmidt number (evaporation efficiency), unitless TODO: verify
        Re = (rho*u*D)/mu                                                   # Reynolds numebr (convection strength), unitless TODO: verify
        beta = 1/t_aK                                                       # coefficient of thermal expansion, Kelvin^-1 TODO
        v = mu/rho                                                          # kinematic viscosity, m^2/s
        Gr = (g*beta*abs(t_wK-t_aK)*np.float_power(D,3))/np.float_power(v,2)    # Grashof number, unitless TODO: verify
        C = 0.48                                                            # ???, ??? TODO
        # Nusselt convection components—Forced, Natural, Combined:
        Nu_f = 0.3 + (0.62*np.float_power(Re,1/2)*np.float_power(Pr,1/3))/(np.float_power(1 + np.float_power(0.4/Pr,2/3),1/4)) * np.float_power(1+np.float_power(Re/282000,5/8),4/5)
        Nu_n = C * np.float_power(Gr*Pr,1/4)
        Nu = np.float_power(np.float_power(Nu_f,3) + np.float_power(Nu_n,3),1/3)
        h_c = (Nu*k)/D                                          # convective heat coefficient, W/m^2*K
        Sh = Nu * np.float_power(Sc/Pr,1/3)                     # Sherwood number, unitless
        h_e = (Sh*D_v)/D                                        # mass transfer coefficient, m/s
        # L_v = 2501000 - 2370*t_wCguess                          # latent heat of vaporization at wet bulb temp, J/kg
        L_v = 2500900 - 2445*t_wCguess                          # latent heat of vaporization at wet bulb temp, J/kg
        F_est = 0.5                                             # estimated all-purpose geometric zenith angle factor, unitless
        R_short = (1-alpha)*(S_dir*F_est)                       # shortwave radiation component, W/m^2
        R_longIn = E_atm*sigma*np.float_power(t_aK,4)           # incoming longwave radiation component, W/m^2
        R_longOut = E_wick*sigma*np.float_power(t_wK,4)         # outgoing longwave radiation component, ?
        
        # Calculate fundamental heat flux components:
        Q_evap = rho*L_v*h_e * (e_sw-e_a)/(P)                   # evaporative heat flux, W/m^2
        Q_rad = R_short + R_longIn - R_longOut                  # radiative heat flux, W/m^2
        Q_conv = h_c * (t_aC - t_wCguess)                       # convective heat flux, W/m^2

        return Q_rad + Q_conv - Q_evap                          # total net heat flux: when Q_net ≈ 0, t_w is correct
    
    return brentq(balance_Q, dew_pt(t_aC,rh)-50.0, 100.0, xtol=0.01, maxiter=1000)

# # Returns mu, dynamic viscosity, given temperature (K)
# def viscosity(temp_K):
#     mu_0 = 0.00001716                   # Sutherland reference constant for air viscosity at reference temp, kg/ms
#     t_0K = 273.15                       # reference temperature for mu_0, Kelvin
#     S_K = 110.4                         # Sutherland temperature, Kelvin
#     return mu_0 * np.float_power(temp_K/t_0K,3/2) * (t_0K+S_K)/(temp_K+S_K)

# # Returns k, thermal conductivity (W/m*k) given temperature (K)
# def thermal_conductivity(temp_K):
#     return 0.0241 * np.float_power(temp_K/273.15,0.9)

# # Returns D_v, water vapor diffusivity in air (m^2/s), given ambient temperature (K)
# def diffusivity(temp_aK, baro_Pa):
#     D_0 = 0.0000226                     # diffusivity constant?, m^2/s  TODO: verify
#     P_0 = 101325                        # pressure constant?, Pascals  TODO: verify
#     return D_0 * np.float_power(temp_aK/273.15,1.75) * P_0/baro_Pa

# # Returns Prandtl number (unitless), given  TODO: verify
# def Pr(temp_K):
#     mu = viscosity(temp_K)
#     k = thermal_conductivity(temp_K)
#     return (c_p*mu)/k

# # Returns Nusselt number (unitless), from 
# def Nusselt_number(Re, Pr, Gr):
#     C = 0.48                            # ???, ??? TODO
#     # forced convection:
#     Nu_f = 0.3 + (0.62*np.float_power(Re,1/2)*np.float_power(Pr,1/3))/(np.float_power(1 + np.float_power(0.4/Pr,2/3),1/4)) * np.float_power(1+np.float_power(Re/282000,5/8),4/5)
#     # natural convection:
#     Nu_n = C * np.float_power(Gr*Pr,1/4)
#     # combined:
#     return 2.0 + 0.6*np.float_power(Re,1/2)*np.float_power(Pr,1/3)

# def old_new_wet_bulb_liljegren(t_aC, rh_percent, p_mBar, windspeed_mps, radiance_Wpm2):
#     # Arguments:
#     #   t_aC : ambient (dry bulb) air temperature, °C
#     #   rh_percent : relative humidity, %
#     #   p_mBar : barometric pressure, millibar
#     #   windspeed_mps : air velocity, m/s
#     #   radiance_Wpm2 : direct solar irradiance, W/m^2

#     # Calculated/converted values:
#     #   t_aK : ambient (dry bulb) air temperature, °K
#     #   t_dewC : dew point temperature, °C
#     #   P_hPa : barometric pressure, hectoPascals (equivalent to millibars; updated var name for clarity)
#     #   P : barometric pressure, Pascals
#     #   v_a : air velocity, m/s (shorter var name for cleaner equations)
#     #   S : direct solar irradiance, W/m^2 (shorter var name for cleaner equations)
#     #   f_dir : fraction of solar irradiance due to solar zenith angle
#     #   c_za : cosine of solar zenith angle

#     t_aK = t_aC + 273.15
#     t_wC = t_aC
#     t_wK = t_wC + 273.15

#     rh = rh_percent
#     u = windspeed_mps
#     P_mbar = p_mBar
#     P = p_mBar * 100
#     P_hPa = p_mBar
#     v_a = windspeed_mps
#     S = radiance_Wpm2
#     f_dir = 0.5

#     # # Universal constants:
#     # sigma = 0.0000000567                # Stefan-Boltzmann constant, W/m^2*K^4
#     # M_air = 0.02897                     # molecular weight of air, kg/mol
#     # M_h2o = 0.018015                    # molecular weight of water, kg/mol
#     # R_gas = 8.31446                     # molar gas constant, J/mol*K
#     # R_air = R_gas/M_air                 # specific gas constant for air, J
#     # c_p = 1005.0                        # specific heat capacity of air, J/K*kg
#     # g = 9.81                            # Earth gravitational constant, m/s^2

#     # # Experimental constants (experimentally-fixed variables):
#     # D_wick = 0.007                      # wick diameter, m
#     # L_wick = 0.0254                     # wick length, m
#     # E_wick = 0.95                       # wick emissivity, ratio
#     # E_atm = 0.85                        # atmospheric emissivity, ratio TODO: double-check approximation
#     # alpha = 0.95                        # wick albedo, ratio

#     # calculated values
#     t_dewC = dew_pt(t_aC, rh_percent)
#     e_sa = Buck_eq_Pa(t_aC)             # saturation vapor pressure at ambient temperature, Pascals
#     e_a = rh * e_sa                     # partial water vapor pressure at ambient temperature, Pascals
#     e_sw = Buck_eq_Pa(t_wC)             # saturation vapor pressure at wet bulb temperature, Pascals
#     q = 0.622 * e_a / (P-0.378*e_a)     # specific humidity, ratio
#     rho = P / (R_air*t_aK*(1+0.61*q))   # air density, kg/m^3
#     mu = viscosity(t_aK)                # dynamic viscosity of air (Sutherland eq.), kg/m*s
#     Re = (rho*u*D_wick)/mu              # Reynolds number (convection strength), unitless TODO: verify
#     L_v = 2501000 - 2370*t_wC           # latent heat of vaporization at wet bulb temp, J/kg


#     # # R_a = 287.05                        # ???, J/kg*K TODO
#     # mu_0 = 0.00001716                   # Sutherland reference constant for air viscosity at reference temp, kg/ms
#     # t_0K = 273.15                       # reference temperature for mu_0, Kelvin
#     # S_K = 110.4                         # Sutherland temperature, Kelvin
#     # c_p = 1005                          # specific heat of air, J/kg*K
#     # P_0 = 101325                        # pressure constant?, Pascals  TODO: verify
#     # D_0 = 0.0000226                     # diffusivity constant?, m^2/s  TODO: verify

#     # # Experimental constants (experimentally-fixed variables):
#     # D_wick = 0.007                      # wick diameter, m
#     # L_wick = 0.0254                     # wick length, m
#     # E_wick = 0.95                       # wick emissivity, ratio

#     # E_atm = 0.85                        # atmospheric emissivity, ratio TODO: double-check approximation
#     # # alpha = 0.4                         # wick albedo, ratio
#     # alpha = 0.95                        # wick albedo, ratio
#     # # theta_z = 45                        # Solar zenith angle, degrees
#     # # theta_zRad = np.deg2rad(theta_z)    # Solar zenith angle, radians



#     #OLD:
#     # e_sa = Buck_eq_Pa(t_aC)                                             # saturation vapor pressure at ambient temperature, Pascals
#     # e_a = rh * e_sa                                                     # partial water vapor pressure at ambient temperature, Pascals
#     # e_sw = Buck_eq_Pa(t_wC)                                             # saturation vapor pressure at wet bulb temperature, Pascals
#     # q = 0.622 * e_a / (P-0.378*e_a)                                     # specific humidity, ratio
#     # rho = P / (R_a*t_aK*(1+0.61*q))                                     # air density, kg/m^3
#     # mu = mu_0 * np.float_power(t_aK/t_0K,3/2) * (t_0K+S_K)/(t_aK+S_K)   # dynamic viscosity of air (Sutherland eq.), kg/m*s
#     # k = 0.0241 * np.float_power(t_aK/273.15,0.9)                        # thermal conductivity of air, W/m*k
#     # D_v = D_0 * np.float_power(t_aK/273.15,1.75) * P_0/P                # water vapor diffusivity in air, m^2/s
#     # Pr = (c_p*mu)/k                                                     # Prandtl number, unitless TODO: verify
#     # Sc = mu/(rho*D_v)                                                   # Schmidt number (evaporation efficiency), unitless TODO: verify
#     # Re = (rho*u*D)/mu                                                   # Reynolds number (convection strength), unitless TODO: verify
#     # beta = 1/t_aK                                                       # coefficient of thermal expansion, Kelvin^-1 TODO
#     # v = mu/rho                                                          # kinematic viscosity, m^2/s
#     # Gr = (g*beta*abs(t_wK-t_aK)*np.float_power(D,3))/np.float_power(v,2)    # Grashof number, unitless TODO: verify
#     # C = 0.48                                                            # ???, ??? TODO
#     # n1 = 1/4                                                            # ???, ??? TODO
#     # n2 = 3                                                              # ???, ??? TODO
#     # # Nusselt convection components—Forced, Natural, Combined:
#     # Nu_f = 0.3 + (0.62*np.float_power(Re,1/2)*np.float_power(Pr,1/3))/(np.float_power(1 + np.float_power(0.4/Pr,2/3),1/4)) * np.float_power(1+np.float_power(Re/282000,5/8),4/5)
#     # Nu_n = C * np.float_power(Gr*Pr,n1)
#     # Nu = np.float_power(np.float_power(Nu_f,n2) + np.float_power(Nu_n,n2),1/n2)
#     # Nu = 2.0 + 0.6*np.float_power(Re,1/2)*np.float_power(Pr,1/3)
#     # h_c = (Nu*k)/D                                          # convective heat coefficient, W/m^2*K
#     # Sh = Nu * np.float_power(Sc/Pr,1/3)                     # Sherwood number, unitless
#     # h_e = (Sh*D_v)/D                                        # mass transfer coefficient, m/s
#     # L_v = 2501000 - 2370*t_wC                               # latent heat of vaporization at wet bulb temp, J/kg
#     # # F_dir = np.sin(np.deg2rad(theta_z))                     # geometric factor from zenith angle, unitless
#     # # R_short = (1-alpha)*(S_dir*F_dir)                       # shortwave radiation component, W/m^2
#     # F_est = 0.5                                             # estimated all-purpose geometric zenith angle factor, unitless
#     # R_short = (1-alpha)*(S_dir*F_est)                       # shortwave radiation component, W/m^2
#     # R_longIn = E_atm*sigma*np.float_power(t_aK,4)           # incoming longwave radiation component, W/m^2
#     # R_longOut = E_wick*sigma*np.float_power(t_wK,4)         # outgoing longwave radiation component, ?
    
#     # # Calculate fundamental heat flux components:
#     # Q_evap = rho*L_v*h_e * (e_sw-e_a)/(P)                   # evaporative heat flux, W/m^2
#     # Q_rad = R_short + R_longIn - R_longOut                  # radiative heat flux, W/m^2
#     # Q_conv = h_c * (t_aC - t_wC)                            # convective heat flux, W/m^2

#     # # Calculate final net heat flux:
#     # Q_net_prev = Q_net
#     # Q_net = Q_evap + Q_rad + Q_conv                         # total net heat flux: when Q_net ≈ 0, t_w is correct

def calc_WBGT(idx_a, idx_h, idx_p, idx_v, idx_r, idx_g):
    global ab, am, hb, hm, pb, pm, vb, vm, rb, rm, gb, gm
    a = ab + am * idx_a
    h = hb + hm * idx_h
    p = pb + pm * idx_p
    v = vb + vm * idx_v
    r = rb + rm * idx_r
    g = gb + gm * idx_g
    # w = wet_bulb_liljegren(a, h, p, v, r)
    w = approximate_wet_bulb_ISO(v, a, g, h)
    return 0.7*w + 0.2*g + 0.1*a

# SCRIPT START:
if __name__ == "__main__":
    ambientTemp = 38.0
    globeTemp = 50.0
    humidity = 50.0
    windSpeed = 0.001
    baroPressure = 1020.0
    solarRadiance = 650.0
    zenithAngle = 55.0
    print("\nambientTemp=", ambientTemp, " baroPressure=", baroPressure, " globeTemp=", globeTemp, " humidity=", humidity, ", windSpeed=", windSpeed, ", solarRadiance=", solarRadiance, ", zenithAngle=", zenithAngle, "\n")

    wet_bulb_psy = psychrometric_wet_bulb(ambientTemp, baroPressure, humidity)
    print(f"WBT_psy=\t{wet_bulb_psy:.5f}")
    wbgt = 0.7*wet_bulb_psy + 0.2*globeTemp + 0.1*ambientTemp
    print(f"WBGT_psy=\t{wbgt:.5f}\n")

    wet_bulb_ISO = approximate_wet_bulb_ISO(windSpeed, ambientTemp, globeTemp, humidity)#, baroPressure)
    print(f"WBT_ISO=\t{wet_bulb_ISO:.5f}")
    wbgt = 0.7*wet_bulb_ISO + 0.2*globeTemp + 0.1*ambientTemp
    print(f"WBGT_ISO=\t{wbgt:.5f}\n")

    wb_liljegren = new_wet_bulb_liljegren(ambientTemp, humidity, baroPressure, windSpeed, solarRadiance)
    print(f"WBT_liljegren=\t{wb_liljegren:.5f}")
    wbgt = 0.7*wb_liljegren + 0.2*globeTemp + 0.1*ambientTemp
    print(f"WBGT_liljegren=\t{wbgt:.5f}\n")

    wb_liljegren2 = AG.solve_Tnwb(ambientTemp, humidity, baroPressure, windSpeed, solarRadiance)
    print(f"WBT_liljegren2=\t{wb_liljegren2:.5f}")
    wbgt2 = 0.7*wb_liljegren2 + 0.2*globeTemp + 0.1*ambientTemp
    print(f"WBGT_liljegren2={wbgt2:.5f}\n")

    wb_liljegren3 = paper_wet_bulb_liljegren(ambientTemp, humidity, baroPressure, windSpeed, solarRadiance, zenithAngle)
    print(f"WBT_liljegren3=\t{wb_liljegren3:.8f}")
    wbgt3 = 0.7*wb_liljegren3 + 0.2*globeTemp + 0.1*ambientTemp
    print(f"WBGT_liljegren3={wbgt3:.5f}\n")

    if HEATMAPS:
        if GEN_NEW:
            WBGT_LIL = np.zeros((aCount, hCount, pCount, vCount, rCount, gCount))
            aArr = range(int(aCount))
            hArr = range(int(hCount))
            pArr = range(int(pCount))
            vArr = range(int(vCount))
            rArr = range(int(rCount))
            gArr = range(int(gCount))
            Args = []
            
            for a in aArr:
                for h in hArr:
                    for p in pArr:
                        for v in vArr:
                            for r in rArr:
                                for g in gArr:
                                    Args.append((a,h,p,v,r,g))

            # count = 0
            # total = aCount*hCount*pCount*vCount*rCount*gCount
            # with multiprocessing.Pool(8) as pool:
            #     pool.starmap(calc_WBGT, Args)

            results = []
            with multiprocessing.Pool(8) as pool:
                results = pool.starmap(calc_WBGT, ((a,h,p,v,r,g) for g in range(gCount) for r in range(rCount) for v in range(vCount) for p in range(pCount) for h in range(hCount) for a in range(aCount)))

            for idx, item in enumerate(Args):
                WBGT_LIL[Args[idx][0],Args[idx][1],Args[idx][2],Args[idx][3],Args[idx][4],Args[idx][5]] = results[idx]
                print("WBGT[",Args[idx][0],Args[idx][1],Args[idx][2],Args[idx][3],Args[idx][4],Args[idx][5],"]=",WBGT_LIL[Args[idx][0],Args[idx][1],Args[idx][2],Args[idx][3],Args[idx][4],Args[idx][5]])

            np.save("testObj.npy", WBGT_LIL)
        else:
            WBGT_LIL = np.load("testObj.npy")

        # count = 0
        # total = tgMax*taMax*vaMax*rhMax
        # for Tg in range (0, tgMax):
        #     g = gm*Tg+gb
        #     for Ta in range (0, taMax):
        #         a = am*Ta+ab
        #         for Va in range (0, vaMax):
        #             v = vm*Va+vb
        #             for Hum in range(0,rhMax):
        #                 h = hm*Hum+hb
        #                 w_LIL = wet_bulb_liljegren(a, h, BARO, v, RAD)
        #                 WBGT_LIL[Tg, Ta, Va, Hum] = 0.7*w_LIL + 0.2*g + 0.1*a
        #                 # if WBGT_LIL[Tg, Ta, Va, Hum] < 0 or w_LIL < 0:
        #                 #     print(f"Low value detected: WBT={w_LIL:.3f}, WBGT={WBGT_LIL[Tg, Ta, Va, Hum]:.3f}, g={g:.3f}, a={a:.3f}, v={v:.3f}, h={h:.3f}")
        #                 count = count + 1
        #         # os.system("cls")
        #         # print(f"Building matrix; Progress: {count}/{total} ({(100*count/total):.3f}%)")
        #     # os.system("cls")
        #     # print(f"Building matrix; Progress: {count}/{total} ({(100*count/total):.3f}%)")

        fig, axes = plt.subplot_mosaic([['plot1']], layout='tight')
        # plt.imshow(WBGT_LIL[0,:,:,0].transpose(), cmap='coolwarm', origin='lower')
        print(WBGT_LIL[:,0,0,:,0,0].transpose())
        plt.imshow(WBGT_LIL[:,0,0,:,0,0].transpose(), cmap='coolwarm', origin='lower')
        plt.setp(axes['plot1'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(10,ab+am*(aCount-1),11).round(3), yticks=np.linspace(0,50,6), yticklabels=np.linspace(0,vb+vm*(vCount-1),6).round(3))
        # ambientTemp, humidity, baroPressure, windSpeed, solarRadiance
        plt.title(f"Liljegren Model: DB temp by air velocity")
        plt.xlabel("Dry bulb temp (°C)")
        plt.ylabel("Air velocity (m/s)")
        plt.colorbar()

        plt.show()
