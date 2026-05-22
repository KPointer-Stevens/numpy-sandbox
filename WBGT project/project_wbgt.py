import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from scipy.optimize import brentq

GENERATE_NEW = True         # Generate new matrices (vs load previously-generated ones)
SINGLE_MATRIX = False        # Generate/load only one heatmap per model
RUN_PSYCHRO = False         # Process heatmap for Psychrometric model
RUN_ISO = False              # Process heatmap for ISO model
RUN_LILJEGREN = True        # Process heatmap for Liljegren model
RUN_3D = False               # Display 3D scatter plot(s)
GENERATE_ONLY = True        # Do not graph results

COLORMAP = mpl.colors.ListedColormap(mpl.colormaps['coolwarm'](np.linspace(0.0, 1.0, 256)))
COLORMAP_ALT = mpl.colors.ListedColormap(mpl.colormaps['RdYlGn'](np.linspace(1.0, 0.0, 256)))

ERROR_COUNT = 0

# Standard inputs for single gen:
V_A = 0.5
T_A = 35
T_G = 45
REL_H = 75
BARO = 1020
RAD = 850
Z_ANGLE = 35


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

# Calculates rough dew point, for starting guess:
def dew_pt(DB, RH):
    t_a = DB
    if t_a == -243.04:
        print("Avoiding divide-by-zero...")
        t_a = t_a + 0.001
    gamma = np.log(RH/100)+(17.625*t_a)/(243.04+t_a)
    if gamma == 17.625:
        print("Avoiding divide-by-zero...")
        gamma = gamma + 0.0001
    return (243.04*gamma)/(17.625-gamma)

# Returns median radiant temperature:
def mean_radiant_from_globe(GT, AV, DB):
    return np.float_power(np.float_power(GT+273,4)+(((110000000*np.float_power(AV,0.6))/(0.95*np.float_power(0.15,0.4)))*(GT-DB)),0.25)-273

# Returns partial vapor pressure in hPa, given temperature in C:
def Buck_equation(temp_C):
    if temp_C >= 0:
        return 6.1121 * np.exp((18.678 - (temp_C / 234.5)) * (temp_C / (257.14 + temp_C)))
    return 6.1121 * np.exp((23.036 - (temp_C / 333.7)) * (temp_C / (279.82 + temp_C)))

# Wrapper converting Buck equation result to Pascals:
def e_sat(temp_C):
    return 100 * Buck_equation(temp_C)

# Approximates wet bulb temp from v_a, t_a, t_g, RH, and pressure:
def approximate_wet_bulb(v, a, g, h):
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
            break
        
        w_guess = w_guess + incr * dir
        p_w = Buck_equation(w_guess)

        term1 = 4.18*np.float_power(v,0.444)*(a-w_guess) + 0.00000001*((np.float_power(r+273,4))-(np.float_power(w_guess+273,4)))
        term2 = 77.1*np.float_power(v,0.421)*(p_w-((h/100)*p_a))

        result_prev = result
        result = term1-term2

        if (abs(result) < 0.02):
                break
        if (result < 0 and result_prev > 0) or (result_prev < 0 and result > 0):
            dir = dir * -1
            incr = incr/10

    return w_guess

# Approximates wet bulb temp from v_a, t_a, t_g, and RH:
def approximate_wet_bulb_ISO(v, a, g, h):
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
            break
        
        w_guess = w_guess + incr * dir
        p_w = Buck_equation(w_guess)

        term1 = 4.18*np.float_power(v,0.444)*(a-w_guess) + 0.00000001*((np.float_power(r+273,4))-(np.float_power(w_guess+273,4)))
        term2 = 77.1*np.float_power(v,0.421)*(p_w-((h/100)*p_a))

        result_prev = result
        result = term1-term2

        if (abs(result) < 0.02):
                break
        if (result < 0 and result_prev > 0) or (result_prev < 0 and result > 0):
            dir = dir * -1
            incr = incr/10

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

    e_sa = e_sat(t_aC)             # saturation vapor pressure at ambient temperature, Pascals
    e_a = rh * e_sa                     # partial water vapor pressure at ambient temperature, Pascals
    A = alpha_w                          # wick surface area as prescribed in Liljegren, m^2
    k = 0.0241 * np.float_power(t_aK/273.15,0.9)                    # thermal conductivity of air, W/m*k
    q = 0.622 * e_a / (P-0.378*e_a)                                 # specific humidity, ratio TODO: should either/both e_a be e_w instead?
    rho = P / (R_air*t_aK*(1+0.61*q))                               # fluid density of air, kg/m^3
    mu = mu_0*np.float_power(t_aK/t_0K,3/2)*(t_0K+S_K)/(t_aK+S_K)   # fluid viscosity of air (Sutherland eq.), kg/m*s
    D_v = D_0 * np.float_power(t_aK/273.15,1.75) * P_0/P            # water vapor diffusivity in air, m^2/s

    Re = (rho*u*D)/mu                                          # Reynolds number (convection strength), unitless TODO: verify?
    Pr = (c_p*mu)/k                                                 # Prandtl number
    Sc = mu/(rho*D_v)                                               # Schmidt number

    # a,b,c prescribed by Liljegren:
    a = 0.56
    b = 0.281
    c = 0.4
    h = (k/D)*b*np.float_power(Re,1-c)*np.float_power(Pr,1-a)  # convective heat coefficient, W/m^2*K

    def computeEq9(t_wCguess):
        # Calculate new values:
        e_sw = e_sat(t_wC)                             # saturation vapor pressure at wet bulb temperature, Pascals
        e_w = rh * e_sw                                     # partial water vapor pressure at wet bulb temperature, Pascals
        DFnetPerA = deltaFnetPerA(s_za, S, t_aC, t_wC)      # Eq.(12) result
        deltaH = 2500900 - 2439*t_wCguess                   # latent heat of vaporization at wet bulb temp, J/kg

        # Eq.(9):
        t_wCResult = t_aC-(deltaH/c_p) * (M_h2o/M_air) * np.float_power(Pr/Sc,a) * ((e_w-e_a)/(P-e_w)) + (DFnetPerA/h)
        t_wDiff = t_wCResult - t_wCguess
        return t_wDiff                                      # difference between guess and calculation; guess is correct when t_wDiff≈0
    return brentq(computeEq9, dew_pt(t_aC,rh)-50.0, 100.0, xtol=0.02, maxiter=1000)

# AI Slop liljegren model:
def OLD_wet_bulb_liljegren(t_aC, rh_percent, p_mBar, windspeed_mps, radiance_Wpm2):
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
    alpha = 0.4                         # wick albedo, ratio
    # theta_z = 45                        # Solar zenith angle, degrees
    # theta_zRad = np.deg2rad(theta_z)    # Solar zenith angle, radians

    # Arbitrary starting values:
    dir = -1                            # t_aC > t_wC, so start from ambient and decrease
    incr = 10
    Q_net = 1
    Q_net_prev = Q_net

    i = 1000
    while i > 0:
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
        Nu = 2.0 + 0.6*np.float_power(Re,1/2)*np.float_power(Pr,1/3)
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
            return t_wC
        
        if (Q_net_prev > 0 and Q_net < 0) or (Q_net_prev < 0 and Q_net > 0):
            dir *= -1
            incr /= 10
        
        t_wC += incr * dir                                      # new wet bulb guess
    
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
    deltaFNetperA_T2 = (1 - alpha_w) * S_rad * ((1 - f_dir) * (1 + (D / (4*L))) + f_dir * ((np.tan(np.deg2rad(zenithAngle)) / pi) + (D / (4*L)) + alpha_sfc))
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
    # V = max(0.9, V)
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
        # print("ERROR: NO ROOTS FOUND. RETURNED NAN.")
        return np.nan

    return t_wCguess

    
def map_all_dims(fullMatrix):
    # select default index:
    i = 13
    fig, axes = plt.subplot_mosaic([['p1x2', 'p1x3', 'p1x4', 'p1x5', 'p1x6'],['p2x3', 'p2x4', 'p2x5', 'p2x6', 'p3x4'],['p3x5', 'p3x6', 'p4x5', 'p4x6', 'p5x6']], layout='tight')
    plt.subplot(351)
    # WBGT_LIL[iA, iH, iP, iG, iV, iR]
    plt.imshow(fullMatrix[:,:,i,i,i,i].transpose(), cmap=COLORMAP_ALT, origin='lower')
    plt.setp(axes['p1x2'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,11), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),11).round(3))
    plt.title(f"Psychrometric Model: Ta x RH; P,RH={gm*i+gb:.1f},{hm*i+hb:.1f}")
    plt.xlabel("Dry bulb temp")
    plt.ylabel("Air velocity")
    plt.colorbar()
    plt.subplot(352)
    plt.imshow(fullMatrix[:,i,:,i,i,i].transpose(), cmap=COLORMAP_ALT, origin='lower')
    plt.setp(axes['p1x3'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,11), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),11).round(3))
    plt.title(f"ISO Model: Ta x Va; Tg,RH={gm*i+gb:.1f},{hm*i+hb:.1f}")
    plt.xlabel("Dry bulb temp")
    plt.ylabel("Air velocity")
    plt.colorbar()
    plt.subplot(353)
    plt.imshow(fullMatrix[:,i,i,:,i,i].transpose(), cmap=COLORMAP_ALT, origin='lower')
    plt.setp(axes['p1x4'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,11), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),11).round(3))
    plt.title(f"Liljegren Model: Ta x Va; Tg,RH={gm*i+gb:.1f},{hm*i+hb:.1f}")
    plt.xlabel("Dry bulb temp")
    plt.ylabel("Air velocity")
    plt.colorbar()
    plt.subplot(354)
    plt.imshow(fullMatrix[:,i,i,i,:,i].transpose(), cmap=COLORMAP, origin='lower')
    plt.setp(axes['p1x5'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,11), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),11).round(3))
    plt.title(f"Psychrometric Model: Ta x Va; Tg,RH={gm*i+gb:.1f},{hm*i+hb:.1f}")
    plt.xlabel("Dry bulb temp")
    plt.ylabel("Air velocity")
    plt.colorbar()
    plt.subplot(355)
    plt.imshow(fullMatrix[:,i,i,i,i,:].transpose(), cmap=COLORMAP, origin='lower')
    plt.setp(axes['p1x6'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,11), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),11).round(3))
    plt.title(f"ISO Model: Ta x Va; Tg,RH={gm*i+gb:.1f},{hm*i+hb:.1f}")
    plt.xlabel("Dry bulb temp")
    plt.ylabel("Air velocity")
    plt.colorbar()
    plt.subplot(356)
    # WBGT_LIL[iA, iH, iP, iG, iV, iR]
    plt.imshow(fullMatrix[i,:,:,i,i,i].transpose(), cmap=COLORMAP, origin='lower')
    plt.setp(axes['p2x3'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
    plt.title(f"Liljegren Model: Ta x Va; Tg,RH={gm*i+gb:.1f},{hm*i+hb:.1f}")
    plt.xlabel("Dry bulb temp")
    plt.ylabel("Air velocity")
    plt.colorbar()
    plt.subplot(357)
    plt.imshow(fullMatrix[i,:,i,:,i,i].transpose(), cmap=COLORMAP, origin='lower')
    plt.setp(axes['p2x4'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
    plt.title(f"Liljegren Model: Ta x Va; Tg,RH={gm*i+gb:.1f},{hm*i+hb:.1f}")
    plt.xlabel("Dry bulb temp")
    plt.ylabel("Air velocity")
    plt.colorbar()
    plt.subplot(358)
    plt.imshow(fullMatrix[i,:,i,i,:,i].transpose(), cmap=COLORMAP, origin='lower')
    plt.setp(axes['p2x5'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
    plt.title(f"Liljegren Model: Ta x Va; Tg,RH={gm*i+gb:.1f},{hm*i+hb:.1f}")
    plt.xlabel("Dry bulb temp")
    plt.ylabel("Air velocity")
    plt.colorbar()
    plt.subplot(359)
    plt.imshow(fullMatrix[i,:,i,i,i,:].transpose(), cmap=COLORMAP, origin='lower')
    plt.setp(axes['p2x6'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
    plt.title(f"Liljegren Model: Ta x Va; Tg,RH={gm*i+gb:.1f},{hm*i+hb:.1f}")
    plt.xlabel("Dry bulb temp")
    plt.ylabel("Air velocity")
    plt.colorbar()
    plt.subplot(3510)
    plt.imshow(fullMatrix[i,i,:,:,i,i].transpose(), cmap=COLORMAP, origin='lower')
    plt.setp(axes['p3x4'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
    plt.title(f"Liljegren Model: Ta x Va; Tg,RH={gm*i+gb:.1f},{hm*i+hb:.1f}")
    plt.xlabel("Dry bulb temp")
    plt.ylabel("Air velocity")
    plt.colorbar()
    plt.subplot(3511)
    # WBGT_LIL[iA, iH, iP, iG, iV, iR]
    plt.imshow(fullMatrix[i,i,:,i,:,i].transpose(), cmap=COLORMAP, origin='lower')
    plt.setp(axes['p3x5'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
    plt.title(f"Liljegren Model: Ta x Va; Tg,RH={gm*i+gb:.1f},{hm*i+hb:.1f}")
    plt.xlabel("Dry bulb temp")
    plt.ylabel("Air velocity")
    plt.colorbar()
    plt.subplot(3512)
    plt.imshow(fullMatrix[i,i,:,i,i,:].transpose(), cmap=COLORMAP, origin='lower')
    plt.setp(axes['p3x6'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
    plt.title(f"Liljegren Model: Ta x Va; Tg,RH={gm*i+gb:.1f},{hm*i+hb:.1f}")
    plt.xlabel("Dry bulb temp")
    plt.ylabel("Air velocity")
    plt.colorbar()
    plt.subplot(3513)
    plt.imshow(fullMatrix[i,i,i,:,:,i].transpose(), cmap=COLORMAP, origin='lower')
    plt.setp(axes['p4x5'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
    plt.title(f"Liljegren Model: Ta x Va; Tg,RH={gm*i+gb:.1f},{hm*i+hb:.1f}")
    plt.xlabel("Dry bulb temp")
    plt.ylabel("Air velocity")
    plt.colorbar()
    plt.subplot(3514)
    plt.imshow(fullMatrix[i,i,i,:,i,:].transpose(), cmap=COLORMAP, origin='lower')
    plt.setp(axes['p4x6'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
    plt.title(f"Liljegren Model: Ta x Va; Tg,RH={gm*i+gb:.1f},{hm*i+hb:.1f}")
    plt.xlabel("Dry bulb temp")
    plt.ylabel("Air velocity")
    plt.colorbar()
    plt.subplot(3515)
    plt.imshow(fullMatrix[i,i,i,i,:,:].transpose(), cmap=COLORMAP, origin='lower')
    plt.setp(axes['p5x6'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
    plt.title(f"Liljegren Model: Ta x Va; Tg,RH={gm*i+gb:.1f},{hm*i+hb:.1f}")
    plt.xlabel("Dry bulb temp")
    plt.ylabel("Air velocity")
    plt.colorbar()
    plt.show()


#####################
### SCRIPT START: ###
#####################
if __name__ == '__main__':
    tgMax = 0
    gm = 0
    gb = 0
    taMax = 0
    am = 0
    ab = 0
    vaMax = 0
    vm = 0
    vb = 0
    rhMax = 0
    hm = 0
    hb = 0
    bpMax = 0
    pm = 0
    pb = 0
    rsMax = 0
    rm = 0
    rb = 0

    if GENERATE_NEW:
        taMax = 26
        am = 1
        ab = 20.0
        rhMax = 21
        hm = 5.0
        hb = 0.0
        bpMax = 5
        pm = 10
        pb = 980
        tgMax = 26
        gm = 1
        gb = 30.0
        vaMax = 16
        vm = 0.1
        vb = 0.1
        rsMax = 6
        rm = 200
        rb = 300
    if SINGLE_MATRIX:
        taMax = 11
        am = 1
        ab = 25.0
        rhMax = 11
        hm = 10
        hb = 0.0
        bpMax = 11
        pm = 10
        pb = 950
        tgMax = 11
        gm = 1
        gb = 35.0
        vaMax = 10
        vm = 0.1
        vb = 0.1
        rsMax = 11
        rm = 100
        rb = 100
        if RUN_3D:
            taMax = 11
            am = 5.0
            ab = 10.0
            rhMax = 11
            hm = 10.0
            hb = 0.0
            bpMax = 1
            pm = 10
            pb = 1020.0
            tgMax = 1
            gm = 10.0
            gb = 50.0
            vaMax = 10
            vm = 0.12
            vb = 0.12
            rsMax = 1
            rm = 50
            rb = 650.0
    elif RUN_3D:
        taMax = 5
        am = 10.0
        ab = 15.0
        rhMax = 5
        hm = 25.0
        hb = 0.0
        bpMax = 5
        pm = 25.0
        pb = 950.0
        tgMax = 5
        gm = 10.0
        gb = 25.0
        vaMax = 5
        vm = 0.20
        vb = 0.20
        rsMax = 5
        rm = 100.0
        rb = 650.0
    WBT_PSY = np.array
    WBT_ISO = np.array
    WBT_LIL = np.array
    # WBT_CHROMA = np.array
    WBGT_PSY = np.array
    WBGT_ISO = np.array
    WBGT_LIL = np.array
    # WBGT_CHROMA = np.array

    if (GENERATE_NEW):
        WBT_PSY = np.zeros((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
        WBT_ISO = np.zeros((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
        WBT_LIL = np.zeros((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
        # WBT_CHROMA = np.zeros((taMax, rhMax, bpMax, tgMax, vaMax, rsMax, 3))
        WBGT_PSY = np.zeros((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
        WBGT_ISO = np.zeros((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
        WBGT_LIL = np.zeros((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
        # WBGT_CHROMA = np.zeros((taMax, rhMax, bpMax, tgMax, vaMax, rsMax, 3))
        total = taMax*rhMax*bpMax*tgMax*vaMax*rsMax
        count = 0
        startTime = time.perf_counter_ns()
        timeTally = 0
        estTime = 0
        for iA in range (0, taMax):
            a = am*iA+ab
            for iH in range(0,rhMax):
                h = hm*iH+hb
                for iP in range(0, bpMax):
                    p = pm*iP+pb
                    if RUN_PSYCHRO:
                        w_PSY = psychrometric_wet_bulb(a, p, h)
                    for iG in range (0, tgMax):
                        g = gm*iG+gb
                        for iV in range (0, vaMax):
                            v = vm*iV+vb
                            if RUN_ISO:
                                w_ISO = approximate_wet_bulb_ISO(v, a, g, h)
                            for iR in range(0, rsMax):
                                r = rm*iR+rb
                                if RUN_PSYCHRO:
                                    WBT_PSY[iA, iH, iP, iG, iV, iR] = w_PSY
                                    WBGT_PSY[iA, iH, iP, iG, iV, iR] = 0.7*w_PSY + 0.2*g + 0.1*a
                                if RUN_ISO:
                                    WBT_ISO[iA, iH, iP, iG, iV, iR] = w_ISO
                                    WBGT_ISO[iA, iH, iP, iG, iV, iR] = 0.7*w_ISO + 0.2*g + 0.1*a
                                if RUN_LILJEGREN:
                                    #wet_bulb_liljegren(t_aC, rh_percent, p_mBar, windspeed_mps, radiance_Wpm2, zenithAngle_deg):
                                    w_LIL = wet_bulb_liljegren(a, h, p, v, r, Z_ANGLE)
                                    WBT_LIL[iA, iH, iP, iG, iV, iR] = w_LIL
                                    WBGT_LIL[iA, iH, iP, iG, iV, iR] = 0.7*w_LIL + 0.2*g + 0.1*a
                                # if RUN_PSYCHRO and RUN_ISO and RUN_LILJEGREN:
                                #     w_Max = np.nanmax((w_PSY,w_ISO,w_LIL))
                                #     WBT_CHROMA[iA, iH, iP, iG, iV, iR, 0] = w_PSY/w_Max
                                #     WBT_CHROMA[iA, iH, iP, iG, iV, iR, 1] = w_ISO/w_Max
                                #     WBT_CHROMA[iA, iH, iP, iG, iV, iR, 2] = w_LIL/w_Max
                                #     wBGT_Max = np.nanmax((WBGT_PSY[iA, iH, iP, iG, iV, iR],WBGT_ISO[iA, iH, iP, iG, iV, iR],WBGT_LIL[iA, iH, iP, iG, iV, iR]))
                                #     WBGT_CHROMA[iA, iH, iP, iG, iV, iR, 0] = WBGT_PSY[iA, iH, iP, iG, iV, iR]/wBGT_Max
                                #     WBGT_CHROMA[iA, iH, iP, iG, iV, iR, 1] = WBGT_ISO[iA, iH, iP, iG, iV, iR]/wBGT_Max
                                #     WBGT_CHROMA[iA, iH, iP, iG, iV, iR, 2] = WBGT_LIL[iA, iH, iP, iG, iV, iR]/wBGT_Max
                                count = count + 1
                                # os.system("cls")
                                # print(f"Building matrices...\nProgress: {count}/{total} ({(100*count/total):.3f}%)")
                            # os.system("cls")
                            # print(f"Building matrices...\nProgress: {count}/{total} ({(100*count/total):.3f}%)")
                        # os.system("cls")
                        # print(f"Building matrices...\nProgress: {count}/{total} ({(100*count/total):.3f}%)")
                        # timeTally = time.perf_counter_ns() - startTime
                        # print(f"elapsed: {timeTally/(10.0**9.0):0.3f}s, estimated: {(timeTally*total/count/(10.0**9.0)):0.3f}s")
                    os.system("cls")
                    print(f"Building matrices...\nProgress: {count}/{total} ({(100*count/total):.3f}%)")
                    timeTally = time.perf_counter_ns() - startTime
                    print(f"elapsed: {timeTally/(10.0**9.0):0.3f}s, estimated: {(timeTally*total/count/(10.0**9.0)):0.3f}s")
                # os.system("cls")
                # print(f"Building matrices...\nProgress: {count}/{total} ({(100*count/total):.3f}%)")
                # timeTally = time.perf_counter_ns() - startTime
                # print(f"elapsed: {timeTally/(10.0**9.0):0.3f}s, estimated: {(timeTally*total/count/(10.0**9.0)):0.3f}s")
            # os.system("cls")
            # print(f"Building matrices...\nProgress: {count}/{total} ({(100*count/total):.3f}%)")
            # timeTally = time.perf_counter_ns() - startTime
            # print(f"elapsed: {timeTally/(10.0**9.0):0.3f}s, estimated: {(timeTally*total/count/(10.0**9.0)):0.3f}s")

        if ERROR_COUNT > 0:
            print(f"\nWARNING: {ERROR_COUNT} matrix cells contain NAN values as a result of inputs failing to produce a root within specified bounds.\n")

        # taMax = 26
        # am = 1
        # ab = 25.0
        # rhMax = 26
        # hm = 4
        # hb = 0.0
        # bpMax = 26
        # pm = 4
        # pb = 950
        # tgMax = 26
        # gm = 1
        # gb = 35.0
        # vaMax = 26
        # vm = 0.1
        # vb = 0.1
        # rsMax = 26
        # rm = 50
        # rb = 0
        if SINGLE_MATRIX:
            if RUN_PSYCHRO:
                np.save("WBT_PSYdb1.npy", WBT_PSY)
                np.save("WBT_PSYdb1_key.npy", np.array([["Ambient Temp", "°C", am, ab],
                                                        ["Relative Humidity", "%", hm, hb],
                                                        ["Barometric Pressure", "mbar", pm, pb],
                                                        ["Globe Temp", "°C", gm, gb],
                                                        ["Wind Velocity", "m/s", vm, vb],
                                                        ["Solar Irradiance", "W/m^2", rm, rb]]))
            if RUN_ISO:
                np.save("WBT_ISOdb1.npy", WBT_ISO)
                np.save("WBT_ISOdb1_key.npy", np.array([["Ambient Temp", "°C", am, ab],
                                                        ["Relative Humidity", "%", hm, hb],
                                                        ["Barometric Pressure", "mbar", pm, pb],
                                                        ["Globe Temp", "°C", gm, gb],
                                                        ["Wind Velocity", "m/s", vm, vb],
                                                        ["Solar Irradiance", "W/m^2", rm, rb]]))
            if RUN_LILJEGREN:
                np.save("WBT_LILdb1.npy", WBT_LIL)
                np.save("WBT_LILdb1_key.npy", np.array([["Ambient Temp", "°C", am, ab],
                                                        ["Relative Humidity", "%", hm, hb],
                                                        ["Barometric Pressure", "mbar", pm, pb],
                                                        ["Globe Temp", "°C", gm, gb],
                                                        ["Wind Velocity", "m/s", vm, vb],
                                                        ["Solar Irradiance", "W/m^2", rm, rb]]))
            if RUN_PSYCHRO:
                np.save("WBGT_PSYdb1.npy", WBGT_PSY)
                np.save("WBGT_PSYdb1_key.npy", np.array([["Ambient Temp", "°C", am, ab],
                                                        ["Relative Humidity", "%", hm, hb],
                                                        ["Barometric Pressure", "mbar", pm, pb],
                                                        ["Globe Temp", "°C", gm, gb],
                                                        ["Wind Velocity", "m/s", vm, vb],
                                                        ["Solar Irradiance", "W/m^2", rm, rb]]))
            if RUN_ISO:
                np.save("WBGT_ISOdb1.npy", WBGT_ISO)
                np.save("WBGT_ISOdb1_key.npy", np.array([["Ambient Temp", "°C", am, ab],
                                                        ["Relative Humidity", "%", hm, hb],
                                                        ["Barometric Pressure", "mbar", pm, pb],
                                                        ["Globe Temp", "°C", gm, gb],
                                                        ["Wind Velocity", "m/s", vm, vb],
                                                        ["Solar Irradiance", "W/m^2", rm, rb]]))
            if RUN_LILJEGREN:
                np.save("WBGT_LILdb1.npy", WBGT_LIL)
                np.save("WBGT_LILdb1_key.npy", np.array([["Ambient Temp", "°C", am, ab],
                                                        ["Relative Humidity", "%", hm, hb],
                                                        ["Barometric Pressure", "mbar", pm, pb],
                                                        ["Globe Temp", "°C", gm, gb],
                                                        ["Wind Velocity", "m/s", vm, vb],
                                                        ["Solar Irradiance", "W/m^2", rm, rb]]))
        elif RUN_3D:
            np.save("WBT_PSYdb3d.npy", WBT_PSY)
            np.save("WBT_ISOdb3d.npy", WBT_ISO)
            np.save("WBT_LILdb3d.npy", WBT_LIL)
            np.save("WBGT_PSYdb3d.npy", WBGT_PSY)
            np.save("WBGT_ISOdb3d.npy", WBGT_ISO)
            np.save("WBGT_LILdb3d.npy", WBGT_LIL)
        else:
            np.save("WBT_PSYdb.npy", WBT_PSY)
            np.save("WBT_PSYdb_key.npy", np.array([["Ambient Temp", "°C", am, ab],
                                                     ["Relative Humidity", "%", hm, hb],
                                                     ["Barometric Pressure", "mbar", pm, pb],
                                                     ["Globe Temp", "°C", gm, gb],
                                                     ["Wind Velocity", "m/s", vm, vb],
                                                     ["Solar Irradiance", "W/m^2", rm, rb]]))
            np.save("WBT_ISOdb.npy", WBT_ISO)
            np.save("WBT_ISOdb_key.npy", np.array([["Ambient Temp", "°C", am, ab],
                                                     ["Relative Humidity", "%", hm, hb],
                                                     ["Barometric Pressure", "mbar", pm, pb],
                                                     ["Globe Temp", "°C", gm, gb],
                                                     ["Wind Velocity", "m/s", vm, vb],
                                                     ["Solar Irradiance", "W/m^2", rm, rb]]))
            np.save("WBT_LILdb.npy", WBT_LIL)
            np.save("WBT_LILdb_key.npy", np.array([["Ambient Temp", "°C", am, ab],
                                                     ["Relative Humidity", "%", hm, hb],
                                                     ["Barometric Pressure", "mbar", pm, pb],
                                                     ["Globe Temp", "°C", gm, gb],
                                                     ["Wind Velocity", "m/s", vm, vb],
                                                     ["Solar Irradiance", "W/m^2", rm, rb]]))
            np.save("WBGT_PSYdb.npy", WBGT_PSY)
            np.save("WBGT_PSYdb_key.npy", np.array([["Ambient Temp", "°C", am, ab],
                                                     ["Relative Humidity", "%", hm, hb],
                                                     ["Barometric Pressure", "mbar", pm, pb],
                                                     ["Globe Temp", "°C", gm, gb],
                                                     ["Wind Velocity", "m/s", vm, vb],
                                                     ["Solar Irradiance", "W/m^2", rm, rb]]))
            np.save("WBGT_ISOdb.npy", WBGT_ISO)
            np.save("WBGT_ISOdb_key.npy", np.array([["Ambient Temp", "°C", am, ab],
                                                     ["Relative Humidity", "%", hm, hb],
                                                     ["Barometric Pressure", "mbar", pm, pb],
                                                     ["Globe Temp", "°C", gm, gb],
                                                     ["Wind Velocity", "m/s", vm, vb],
                                                     ["Solar Irradiance", "W/m^2", rm, rb]]))
            np.save("WBGT_LILdb.npy", WBGT_LIL)
            np.save("WBGT_LILdb_key.npy", np.array([["Ambient Temp", "°C", am, ab],
                                                     ["Relative Humidity", "%", hm, hb],
                                                     ["Barometric Pressure", "mbar", pm, pb],
                                                     ["Globe Temp", "°C", gm, gb],
                                                     ["Wind Velocity", "m/s", vm, vb],
                                                     ["Solar Irradiance", "W/m^2", rm, rb]]))
    else:
        if SINGLE_MATRIX:
            WBT_PSY = np.load("WBT_PSYdb1.npy")
            WBT_ISO = np.load("WBT_ISOdb1.npy")
            WBT_LIL = np.load("WBT_LILdb1.npy")
            WBGT_PSY = np.load("WBGT_PSYdb1.npy")
            WBGT_ISO = np.load("WBGT_ISOdb1.npy")
            WBGT_LIL = np.load("WBGT_LILdb1.npy")
        elif RUN_3D:
            WBT_PSY = np.load("WBT_PSYdb3d.npy")
            WBT_ISO = np.load("WBT_ISOdb3d.npy")
            WBT_LIL = np.load("WBT_LILdb3d.npy")
            WBGT_PSY = np.load("WBGT_PSYdb3d.npy")
            WBGT_ISO = np.load("WBGT_ISOdb3d.npy")
            WBGT_LIL = np.load("WBGT_LILdb3d.npy")
        else:
            WBT_PSY = np.load("WBT_PSYdb.npy")
            WBT_ISO = np.load("WBT_ISOdb.npy")
            WBT_LIL = np.load("WBT_LILdb.npy")
            WBGT_PSY = np.load("WBGT_PSYdb.npy")
            WBGT_ISO = np.load("WBGT_ISOdb.npy")
            WBGT_LIL = np.load("WBGT_LILdb.npy")

    paramSettings = ((ab, am, taMax), (hb, hm, rhMax), (pb, pm, bpMax),
                     (gb, gm, tgMax), (vb, vm, vaMax), (rb, rm, rsMax))
    np.save("paramSettings.npy", paramSettings)

    if GENERATE_ONLY:
        pass
    elif SINGLE_MATRIX:
        if RUN_PSYCHRO and RUN_ISO and RUN_LILJEGREN:
            weird = True
            if weird:
                fig, axes = plt.subplot_mosaic([['plot1', 'plot2']], layout='constrained')
                # normal1 = mpl.colors.Normalize(vmin=np.nanmin(WBT_CHROMA[0,:,0,0,:,0,:]), vmax=np.nanmax(WBT_CHROMA[0,:,0,0,:,0,:]))
                # normal2 = mpl.colors.Normalize(vmin=np.nanmin(WBGT_CHROMA[0,:,0,0,:,0,:]), vmax=np.nanmax(WBGT_CHROMA[0,:,0,0,:,0,:]))
                # WBT_CHROMA[0,:,0,0,:,0,:] += np.nanmin(WBT_CHROMA[0,:,0,0,:,0,:])
                plt.subplot(121)
                # print(WBT_CHROMA[0,:,0,0,:,0])
                plt.imshow(WBT_CHROMA[0,:,0,0,:,0], origin='lower', norm=mpl.colors.Normalize(vmin=np.nanmin(WBT_CHROMA[0,:,0,0,:,0,:]), vmax=np.nanmax(WBT_CHROMA[0,:,0,0,:,0,:])))
                plt.subplot(122)
                plt.imshow(WBGT_CHROMA[0,:,0,0,:,0,:], origin='lower', norm=mpl.colors.Normalize(vmin=np.nanmin(WBGT_CHROMA[0,:,0,0,:,0,:]), vmax=np.nanmax(WBGT_CHROMA[0,:,0,0,:,0,:])))
            else:
                # ((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
                fig, axes = plt.subplot_mosaic([['plot1', 'plot2', 'plot3'],['plot4', 'plot5', 'plot6']], layout='tight')
                plt.subplot(231)
                plt.imshow(WBGT_PSY[0,0,0,0,:,:].transpose(), cmap=COLORMAP_ALT, origin='lower')
                plt.setp(axes['plot1'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,11), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),11).round(3))
                plt.title(f"Psychrometric Model: Ta x Va; Tg,RH={gm*0+gb:.1f},{hm*0+hb:.1f}")
                plt.xlabel("Dry bulb temp")
                plt.ylabel("Air velocity")
                plt.colorbar()
                plt.subplot(232)
                plt.imshow(WBGT_ISO[0,0,0,0,:,:].transpose(), cmap=COLORMAP_ALT, origin='lower')
                plt.setp(axes['plot2'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,11), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),11).round(3))
                plt.title(f"ISO Model: Ta x Va; Tg,RH={gm*0+gb:.1f},{hm*0+hb:.1f}")
                plt.xlabel("Dry bulb temp")
                plt.ylabel("Air velocity")
                plt.colorbar()
                plt.subplot(233)
                plt.imshow(WBGT_LIL[0,0,0,0,:,:].transpose(), cmap=COLORMAP_ALT, origin='lower')
                plt.setp(axes['plot3'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,11), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),11).round(3))
                plt.title(f"Liljegren Model: Ta x Va; Tg,RH={gm*0+gb:.1f},{hm*0+hb:.1f}")
                plt.xlabel("Dry bulb temp")
                plt.ylabel("Air velocity")
                plt.colorbar()
                plt.subplot(234)
                plt.imshow(WBT_PSY[0,0,0,0,:,:].transpose(), cmap=COLORMAP, origin='lower')
                plt.setp(axes['plot4'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,11), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),11).round(3))
                plt.title(f"Psychrometric Model: Ta x Va; Tg,RH={gm*0+gb:.1f},{hm*0+hb:.1f}")
                plt.xlabel("Dry bulb temp")
                plt.ylabel("Air velocity")
                plt.colorbar()
                plt.subplot(235)
                plt.imshow(WBT_ISO[0,0,0,0,:,:].transpose(), cmap=COLORMAP, origin='lower')
                plt.setp(axes['plot5'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,11), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),11).round(3))
                plt.title(f"ISO Model: Ta x Va; Tg,RH={gm*0+gb:.1f},{hm*0+hb:.1f}")
                plt.xlabel("Dry bulb temp")
                plt.ylabel("Air velocity")
                plt.colorbar()
                plt.subplot(236)
                plt.imshow(WBT_LIL[0,0,0,0,:,:].transpose(), cmap=COLORMAP, origin='lower')
                plt.setp(axes['plot6'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(rb,rb+rm*(rsMax-1),11).round(3), yticks=np.linspace(0,50,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
                plt.title(f"Liljegren Model: Ta x Va; Tg,RH={gm*0+gb:.1f},{hm*0+hb:.1f}")
                plt.xlabel("Dry bulb temp")
                plt.ylabel("Air velocity")
                plt.colorbar()
        elif RUN_LILJEGREN and RUN_3D:
            # ((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
            print(WBT_LIL[:,:,0,0,:,0].shape)
            normal1 = mpl.colors.Normalize(vmin=np.nanmin(WBGT_LIL[:,:,0,0,0,0], ), vmax=np.nanmax(WBGT_LIL[:,:,0,0,0,0]))
            x = y = z = np.linspace(0,10,11,endpoint=True)
            z = np.linspace(1,10,10,endpoint=True)
            X, Y, Z = np.meshgrid(x, y, z)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(X, Y, Z, s=200, c=WBT_LIL[:,:,0,0,:,0], cmap=COLORMAP_ALT)
            plt.setp(ax, xticks=np.linspace(0,(taMax-1),11), yticks=np.linspace(0,(rhMax-1),11), zticks=np.linspace(0,(vaMax-1),10), xticklabels=np.linspace(am,am+am*(taMax-1),11).round(3), yticklabels=np.linspace(hb,hb+hm*(rhMax-1),11).round(3), zticklabels=np.linspace(vb,vb+vm*(vaMax-1),10).round(3))
            fig.colorbar(scatter, ax=ax)
        elif RUN_LILJEGREN and RUN_LILJEGREN:
            # ((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
            normal1 = mpl.colors.Normalize(vmin=np.nanmin(WBGT_LIL[0:2,:,0,0,:,0], ), vmax=np.nanmax(WBGT_LIL[0:2,:,0,0,:,0]))
            normal2 = mpl.colors.Normalize(vmin=np.nanmin(WBT_LIL[0:2,:,0,0,:,0], ), vmax=np.nanmax(WBT_LIL[0:2,:,0,0,:,0]))
            fig, axes = plt.subplot_mosaic([['plot1', 'plot2', 'plot3'],['plot4','plot5','plot6']], layout='constrained', sharey=True)
            plt.subplot(231)
            plt.imshow(WBGT_LIL[0,:,0,0,:,0].transpose(), norm=normal1, cmap=COLORMAP_ALT, origin='lower')
            d = [[hm,hb,rhMax],[vm,vb,vaMax]]
            plt.setp(axes['plot1'], xticks=np.linspace(0,100,11), yticks=np.linspace(0,100,11), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),11).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),11).round(3))
            plt.title(f"Liljegren WBGT: RH x Va; DB={ab:.0f}")
            plt.xlabel("Relative Humidity (%)")
            plt.ylabel("Wind Speed (m/s)")
            plt.subplot(234)
            plt.imshow(WBT_LIL[0,:,0,0,:,0].transpose(), norm=normal2, cmap=COLORMAP, origin='lower')
            d = [[hm,hb,rhMax],[vm,vb,vaMax]]
            plt.setp(axes['plot4'], xticks=np.linspace(0,100,11), yticks=np.linspace(0,100,11), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),11).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),11).round(3))
            plt.title(f"Liljegren WBT: RH x Va; DB={ab:.0f}")
            plt.xlabel("Relative Humidity (%)")
            plt.ylabel("Wind Speed (m/s)")
            plt.subplot(232)
            plt.imshow(WBGT_LIL[1,:,0,0,:,0].transpose(), norm=normal1, cmap=COLORMAP_ALT, origin='lower')
            d = [[hm,hb,rhMax],[vm,vb,vaMax]]
            plt.setp(axes['plot2'], xticks=np.linspace(0,100,11), yticks=np.linspace(0,100,11), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),11).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),11).round(3))
            plt.title(f"Liljegren WBGT: RH x Va; DB={ab+am:.0f}")
            plt.title(f"Liljegren WBGT, WBT: RH x Va; DB={ab+am:.0f}")
            plt.xlabel("Relative Humidity (%)")
            plt.ylabel("Wind Speed (m/s)")
            plt.subplot(235)
            plt.imshow(WBT_LIL[1,:,0,0,:,0].transpose(), norm=normal2, cmap=COLORMAP, origin='lower')
            d = [[hm,hb,rhMax],[vm,vb,vaMax]]
            plt.setp(axes['plot5'], xticks=np.linspace(0,100,11), yticks=np.linspace(0,100,11), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),11).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),11).round(3))
            plt.title(f"Liljegren WBT: RH x Va; DB={ab+am:.0f}")
            plt.xlabel("Relative Humidity (%)")
            plt.ylabel("Wind Speed (m/s)")
            plt.subplot(233)
            plt.imshow(WBGT_LIL[2,:,0,0,:,0].transpose(), norm=normal1, cmap=COLORMAP_ALT, origin='lower')
            d = [[hm,hb,rhMax],[vm,vb,vaMax]]
            plt.setp(axes['plot3'], xticks=np.linspace(0,100,11), yticks=np.linspace(0,100,11), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),11).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),11).round(3))
            plt.title(f"Liljegren WBGT: RH x Va; DB={ab+am*2:.0f}")
            plt.xlabel("Relative Humidity (%)")
            plt.ylabel("Wind Speed (m/s)")
            plt.colorbar(mappable=None, norm=normal1, cmap=COLORMAP_ALT)
            plt.subplot(236)
            plt.imshow(WBT_LIL[2,:,0,0,:,0].transpose(), norm=normal2, cmap=COLORMAP, origin='lower')
            d = [[hm,hb,rhMax],[vm,vb,vaMax]]
            plt.setp(axes['plot6'], xticks=np.linspace(0,100,11), yticks=np.linspace(0,100,11), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),11).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),11).round(3))
            plt.title(f"Liljegren WBT: RH x Va; DB={ab+am*2:.0f}")
            plt.xlabel("Relative Humidity (%)")
            plt.ylabel("Wind Speed (m/s)")
            plt.colorbar(mappable=None, norm=normal2, cmap=COLORMAP)
        elif RUN_ISO and RUN_LILJEGREN:
            # ((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
            normal = mpl.colors.Normalize(vmin=np.min(np.concatenate((WBGT_ISO[0,:,:,0],WBGT_LIL[0,:,:,0]), axis=1)), vmax=np.max(np.concatenate((WBGT_ISO[0,:,:,0],WBGT_LIL[0,:,:,0]), axis=1)))
            fig, axes = plt.subplot_mosaic([['plot1', 'plot2']], layout='tight')
            plt.subplot(121)
            plt.imshow(WBGT_ISO[0,0,0,0,:,:], norm=normal, cmap=COLORMAP, origin='lower')
            # plt.setp(axes['plot1'], xticks=np.linspace(0,(taMax-1),11), xticklabels=np.linspace(10,ab+am*(taMax-1),11).round(3), yticks=np.linspace(0,(vaMax-1),6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
            plt.setp(axes['plot1'], xticks=np.linspace(0,(rsMax-1),6), xticklabels=np.linspace(0,rb+rm*(rsMax-1),6).round(3), yticks=np.linspace(0,(vaMax-1),11), yticklabels=np.linspace(0,vb+vm*(vaMax-1),11).round(3))
            # plt.title(f"ISO Model: Ta x Va; Tg,RH={gm*0+gb:.1f},{hm*0+hb:.1f}")
            plt.title(f"ISO Model: S x v_a; Ta,Tg,RH,P={ab:.0f},{gb:.0f},{hb:.0f},{pb:.0f}")
            plt.xlabel("Dry bulb temp")
            plt.ylabel("Air velocity")
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(WBGT_LIL[0,0,0,0,:,:], norm=normal, cmap=COLORMAP, origin='lower')
            # plt.setp(axes['plot2'], xticks=np.linspace(0,(taMax-1),11), xticklabels=np.linspace(10,ab+am*(taMax-1),11).round(3), yticks=np.linspace(0,(vaMax-1),6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
            plt.setp(axes['plot2'], xticks=np.linspace(0,(rsMax-1),6), xticklabels=np.linspace(0,rb+rm*(rsMax-1),6).round(3), yticks=np.linspace(0,(vaMax-1),11), yticklabels=np.linspace(0,vb+vm*(vaMax-1),11).round(3))
            # plt.title(f"Liljegren Model: Ta x Va; Tg,RH={gm*0+gb:.1f},{hm*0+hb:.1f}")
            plt.title(f"Liljegren Model: S x v_a; Ta,Tg,RH,P={ab:.0f},{gb:.0f},{hb:.0f},{pb:.0f}")
            plt.xlabel("Dry bulb temp")
            plt.ylabel("Air velocity")
            plt.colorbar()
        elif RUN_ISO:
            # ((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
            fig, axes = plt.subplot_mosaic([['plot1']], layout='tight')
            plt.imshow(WBGT_ISO[0,0,0,0,:,:], cmap=COLORMAP, origin='lower')
            plt.setp(axes['plot1'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(10,ab+am*(taMax-1),11).round(3), yticks=np.linspace(0,50,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
            plt.title("ISO Model: Ta x Va (RH=60,tg=50)")
            plt.xlabel("Dry bulb temp")
            plt.ylabel("Air velocity")
            plt.colorbar()
        elif RUN_LILJEGREN:
            # ((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
            fig, axes = plt.subplot_mosaic([['plot1']], layout='tight')
            plt.imshow(WBGT_LIL[:,:,0,0,0,0].transpose(), cmap=COLORMAP, origin='lower')
            plt.setp(axes['plot1'], xticks=np.linspace(0,100,11), yticks=np.linspace(0,100,11), xticklabels=np.linspace(ab,ab+am*(taMax-1),11).round(3), yticklabels=np.linspace(hb,hb+hm*(rhMax-1),11).round(3))
            plt.title("Liljegren Model: Ta x RH; P,G,V,S=1020,50,0.5,650")
            plt.xlabel("Dry bulb temp")
            plt.ylabel("Air velocity")
            plt.colorbar()
        elif RUN_PSYCHRO:
            # ((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
            fig, axes = plt.subplot_mosaic([['plot1']], layout='tight')
            plt.imshow(WBGT_PSY[0,0,0,0,:,:].transpose(), cmap=COLORMAP, origin='lower')
            plt.setp(axes['plot1'], xticks=np.linspace(0,50,11), xticklabels=np.linspace(10,ab+am*(taMax-1),11).round(3), yticks=np.linspace(0,50,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
            plt.title(f"Psychrometric Model: Ta x Va; Tg,RH={gm*0+gb:.1f},{hm*0+hb:.1f}")
            plt.xlabel("Dry bulb temp")
            plt.ylabel("Air velocity")
            plt.colorbar()
    else:
        if RUN_3D:
            if RUN_LILJEGREN:
                G = True
                LILJ_DATA = np.zeros((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
                if G: LILJ_DATA = WBGT_LIL
                else: LILJ_DATA = WBT_LIL
                normal = mpl.colors.Normalize(vmin=np.nanmin(LILJ_DATA), vmax=np.nanmax(LILJ_DATA))
                x = y = z = np.linspace(0,5,5,endpoint=True)
                X, Y, Z = np.meshgrid(x, y, z)
                fig = plt.figure(layout='constrained')
                # ((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
                axs = fig.add_subplot(4,5,1, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[:,:,:,2,2,2], cmap=COLORMAP_ALT, norm=normal)
                d = [[am,ab,taMax],[hm,hb,rhMax],[pm,pb,bpMax]]
                plt.setp(axs, title="DB x RH x P", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                axs = fig.add_subplot(4,5,2, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[:,:,2,:,2,2], cmap=COLORMAP_ALT, norm=normal)
                d = [[am,ab,taMax],[hm,hb,rhMax],[gm,gb,tgMax]]
                plt.setp(axs, title="DB x RH x Tg", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                axs = fig.add_subplot(4,5,3, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[:,:,2,2,:,2], cmap=COLORMAP_ALT, norm=normal)
                d = [[am,ab,taMax],[hm,hb,rhMax],[vm,vb,vaMax]]
                plt.setp(axs, title="DB x RH x Va", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                axs = fig.add_subplot(4,5,4, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[:,:,2,2,2,:], cmap=COLORMAP_ALT, norm=normal)
                d = [[am,ab,taMax],[hm,hb,rhMax],[rm,rb,rsMax]]
                plt.setp(axs, title="DB x RH x S", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                # ((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
                axs = fig.add_subplot(4,5,5, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[:,2,:,:,2,2], cmap=COLORMAP_ALT, norm=normal)
                d = [[am,ab,taMax],[pm,pb,bpMax],[gm,gb,tgMax]]
                plt.setp(axs, title="DB x P x Tg", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                axs = fig.add_subplot(4,5,6, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[:,2,:,2,:,2], cmap=COLORMAP_ALT, norm=normal)
                d = [[am,ab,taMax],[pm,pb,bpMax],[vm,vb,vaMax]]
                plt.setp(axs, title="DB x P x Va", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                axs = fig.add_subplot(4,5,7, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[:,2,:,2,2,:], cmap=COLORMAP_ALT, norm=normal)
                d = [[am,ab,taMax],[pm,pb,bpMax],[rm,rb,rsMax]]
                plt.setp(axs, title="DB x P x S", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                axs = fig.add_subplot(4,5,8, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[:,2,2,:,:,2], cmap=COLORMAP_ALT, norm=normal)
                d = [[am,ab,taMax],[gm,gb,tgMax],[vm,vb,vaMax]]
                plt.setp(axs, title="DB x Tg x Va", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                # ((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
                axs = fig.add_subplot(4,5,9, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[:,2,2,:,2,:], cmap=COLORMAP_ALT, norm=normal)
                d = [[am,ab,taMax],[gm,gb,tgMax],[rm,rb,rsMax]]
                plt.setp(axs, title="DB x Tg x S", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                axs = fig.add_subplot(4,5,10, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[:,2,2,2,:,:], cmap=COLORMAP_ALT, norm=normal)
                d = [[am,ab,taMax],[vm,vb,vaMax],[rm,rb,rsMax]]
                plt.setp(axs, title="DB x Va x S", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                axs = fig.add_subplot(4,5,11, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[2,:,:,:,2,2], cmap=COLORMAP_ALT, norm=normal)
                d = [[hm,hb,rhMax],[pm,pb,bpMax],[gm,gb,tgMax]]
                plt.setp(axs, title="RH x P x Tg", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                axs = fig.add_subplot(4,5,12, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[2,:,:,2,:,2], cmap=COLORMAP_ALT, norm=normal)
                d = [[hm,hb,rhMax],[pm,pb,bpMax],[vm,vb,vaMax]]
                plt.setp(axs, title="RH x P x Va", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                # ((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
                axs = fig.add_subplot(4,5,13, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[2,:,:,2,2,:], cmap=COLORMAP_ALT, norm=normal)
                d = [[hm,hb,rhMax],[pm,pb,bpMax],[rm,rb,rsMax]]
                plt.setp(axs, title="RH x P x S", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                axs = fig.add_subplot(4,5,14, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[2,:,2,:,:,2], cmap=COLORMAP_ALT, norm=normal)
                d = [[hm,hb,rhMax],[gm,gb,tgMax],[vm,vb,vaMax]]
                plt.setp(axs, title="RH x Tg x Va", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                axs = fig.add_subplot(4,5,15, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[2,:,2,:,2,:], cmap=COLORMAP_ALT, norm=normal)
                d = [[hm,hb,rhMax],[gm,gb,tgMax],[rm,rb,rsMax]]
                plt.setp(axs, title="RH x Tg x S", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                axs = fig.add_subplot(4,5,16, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[2,:,2,2,:,:], cmap=COLORMAP_ALT, norm=normal)
                d = [[hm,hb,rhMax],[vm,vb,vaMax],[rm,rb,rsMax]]
                plt.setp(axs, title="RH x Va x S", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                # ((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
                axs = fig.add_subplot(4,5,17, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[2,2,:,:,:,2], cmap=COLORMAP_ALT, norm=normal)
                d = [[pm,pb,bpMax],[gm,gb,tgMax],[vm,vb,vaMax]]
                plt.setp(axs, title="P x Tg x Va", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                axs = fig.add_subplot(4,5,18, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[2,2,:,:,2,:], cmap=COLORMAP_ALT, norm=normal)
                d = [[pm,pb,bpMax],[gm,gb,tgMax],[rm,rb,rsMax]]
                plt.setp(axs, title="P x Tg x S", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                axs = fig.add_subplot(4,5,19, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[2,2,:,2,:,:], cmap=COLORMAP_ALT, norm=normal)
                d = [[pm,pb,bpMax],[vm,vb,vaMax],[rm,rb,rsMax]]
                plt.setp(axs, title="P x Va x S", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                # fig.colorbar(scatter, normal=normal, ax=axs)
                axs = fig.add_subplot(4,5,20, projection='3d')
                scatter = axs.scatter(X, Y, Z, s=50, c=LILJ_DATA[2,2,2,:,:,:], cmap=COLORMAP_ALT, norm=normal)
                d = [[gm,gb,tgMax],[vm,vb,vaMax],[rm,rb,rsMax]]
                plt.setp(axs, title="Tg x Va x S", xticks=np.linspace(0,5,5,endpoint=True), yticks=np.linspace(0,5,5,endpoint=True), zticks=np.linspace(0,5,5,endpoint=True), xticklabels=np.linspace(d[0][1],d[0][1]+d[0][0]*(d[0][2]-1),5).round(3), yticklabels=np.linspace(d[1][1],d[1][1]+d[1][0]*(d[1][2]-1),5).round(3), zticklabels=np.linspace(d[2][1],d[2][1]+d[2][0]*(d[2][2]-1),5).round(3))
                fig.colorbar(scatter, ax=fig.axes, norm=normal)
                print(LILJ_DATA[:,:,:,2,2,2])
                print()
                print(LILJ_DATA[:,:,2,:,2,2])
                print()
                print(LILJ_DATA[:,:,2,2,:,2])
        if RUN_PSYCHRO and RUN_ISO and RUN_LILJEGREN:
            # ((taMax, rhMax, bpMax, tgMax, vaMax, rsMax))
            fig, axes = plt.subplot_mosaic([['plot1', 'plot2', 'plot3'],['plot4', 'plot5', 'plot6']], layout='tight')
            plt.subplot(231)
            plt.imshow(WBGT_PSY[5,:,5,5,:,5].transpose(), cmap=COLORMAP_ALT, origin='lower')
            # plt.setp(axes['plot1'], xticks=np.linspace(0,rhMax-1,rhMax), xticklabels=np.linspace(hb,hb+hm*(rhMax-1),11).round(3), yticks=np.linspace(0,vaMax-1,vaMax), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),11).round(3))
            plt.setp(axes['plot1'], xticks=np.linspace(0,20,11), yticks=np.linspace(0,19,11), xticklabels=np.linspace(0,100,11).round(2), yticklabels=np.linspace(0,1,11).round(2))
            plt.title(f"Psychrometric WBGT: RH x v_a; Tdb,P,Tg,S={ab+am*5:.0f},{pb+pm*5:.0f},{gb+gm*5:.0f},{rb+rm*5:.0f}")
            # plt.xlabel("Relative Humidity (%)")
            plt.ylabel("Wind Speed (m/s)")
            plt.colorbar()
            plt.subplot(232)
            plt.imshow(WBGT_ISO[5,:,5,5,:,5].transpose(), cmap=COLORMAP_ALT, origin='lower')
            # plt.setp(axes['plot2'], xticks=np.linspace(0,rhMax-1,rhMax), xticklabels=np.linspace(hb,hb+hm*(rhMax-1),11).round(3), yticks=np.linspace(0,vaMax-1,vaMax), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),11).round(3))
            plt.title(f"ISO WBGT: RH x v_a; Tdb,P,Tg,S={ab+am*5:.0f},{pb+pm*5:.0f},{gb+gm*5:.0f},{rb+rm*5:.0f}")
            # plt.xlabel("Relative Humidity (%)")
            plt.ylabel("Wind Speed (m/s)")
            plt.colorbar()
            plt.subplot(233)
            plt.imshow(WBGT_LIL[5,:,5,5,:,5].transpose(), cmap=COLORMAP_ALT, origin='lower')
            # plt.setp(axes['plot3'], xticks=np.linspace(0,rhMax-1,rhMax), xticklabels=np.linspace(hb,hb+hm*(rhMax-1),11).round(3), yticks=np.linspace(0,vaMax-1,vaMax), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),11).round(3))
            plt.title(f"Liljegren WBGT: RH x v_a; Tdb,P,Tg,S={ab+am*5:.0f},{pb+pm*5:.0f},{gb+gm*5:.0f},{rb+rm*5:.0f}")
            # plt.xlabel("Relative Humidity (%)")
            plt.ylabel("Wind Speed (m/s)")
            plt.colorbar()
            plt.subplot(234)
            plt.imshow(WBT_PSY[5,:,5,5,:,5].transpose(), cmap=COLORMAP, origin='lower')
            # plt.setp(axes['plot4'], xticks=np.linspace(0,rhMax-1,rhMax), xticklabels=np.linspace(hb,hb+hm*(rhMax-1),11).round(3), yticks=np.linspace(0,vaMax-1,vaMax), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),11).round(3))
            plt.title(f"Psychrometric WBT: RH x v_a; Tdb,P,Tg,S={ab+am*5:.0f},{pb+pm*5:.0f},{gb+gm*5:.0f},{rb+rm*5:.0f}")
            plt.xlabel("Relative Humidity (%)")
            plt.ylabel("Wind Speed (m/s)")
            plt.colorbar()
            plt.subplot(235)
            plt.imshow(WBT_ISO[5,:,5,5,:,5].transpose(), cmap=COLORMAP, origin='lower')
            # plt.setp(axes['plot5'], xticks=np.linspace(0,rhMax-1,rhMax), xticklabels=np.linspace(hb,hb+hm*(rhMax-1),11).round(3), yticks=np.linspace(0,vaMax-1,vaMax), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),11).round(3))
            plt.title(f"ISO WBT: RH x v_a; Tdb,P,Tg,S={ab+am*5:.0f},{pb+pm*5:.0f},{gb+gm*5:.0f},{rb+rm*5:.0f}")
            plt.xlabel("Relative Humidity (%)")
            plt.ylabel("Wind Speed (m/s)")
            plt.colorbar()
            plt.subplot(236)
            plt.imshow(WBT_LIL[5,:,5,5,:,5].transpose(), cmap=COLORMAP, origin='lower')
            # plt.setp(axes['plot6'], xticks=np.linspace(0,rhMax-1,rhMax), xticklabels=np.linspace(hb,hb+hm*(rhMax-1),11).round(3), yticks=np.linspace(0,vaMax-1,vaMax), yticklabels=np.linspace(vb,vb+vm*(vaMax-1),6).round(3))
            plt.title(f"Liljegren WBT: RH x v_a; Tdb,P,Tg,S={ab+am*5:.0f},{pb+pm*5:.0f},{gb+gm*5:.0f},{rb+rm*5:.0f}")
            plt.xlabel("Relative Humidity (%)")
            plt.ylabel("Wind Speed (m/s)")
            plt.colorbar()
    if not GENERATE_ONLY:
        plt.show()