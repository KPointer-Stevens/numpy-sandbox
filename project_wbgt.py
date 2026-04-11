import os
import numpy as np
import matplotlib.pyplot as plt

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

# Returns partial vapor pressure in kPa:
def Buck_equation(DB):
    d = DB
    if d == -257.14:
        d = -257.15
        print("DB ERR! new DB=", d)
    return 0.61121 * np.exp((18.678-(d/234.5))*(d/(257.14+d)))

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

    return w_guess

# Approximates wet bulb temp from v_a, t_a, t_g, RH, and pressure:
def approximate_wet_bulb_ISO(v, a, g, h, p):
    w_psy = 0
    if v == 0:
        return psychrometric_wet_bulb(a, p, h)
    if v < 0.01:    # do not factor wind if near-zero to avoid artifacts inherent to ISO model
        w_psy = psychrometric_wet_bulb(a, p, h)

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
    
    if v < 0.01:
        return w_psy*(1-(v/0.01)) + w_guess*(v/0.01)
        # return w_psy*(1-(v/0.5)) + w_guess*(v/0.5)

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
        Q_net = Q_evap + Q_rad + Q_conv                         # total net heat flux: when Q_net ≈ 0, t_w is correct
        # print(Q_net_prev, Q_net)

        if abs(Q_net) < 0.02:
            return t_wC
        
        if (Q_net_prev > 0 and Q_net < 0) or (Q_net_prev < 0 and Q_net > 0):
            dir *= -1
            incr /= 10
        
        t_wC += incr * dir                                      # new wet bulb guess
    
    return t_wC


GENERATE_MASTER_ARRAY = False

# SCRIPT START:
tgMax = 3
gm = 10.0
gb = 50.0
taMax = 201
am = 0.25
ab = 10.0
vaMax = 201
vm = 0.0075
vb = 0.0
rhMax = 3
hm = 10.0
hb = 60.0
WBGT = np.zeros((tgMax, taMax, vaMax, rhMax))


if (GENERATE_MASTER_ARRAY):
    total = tgMax*taMax*vaMax*rhMax
    count = 0
    for Tg in range (0, tgMax):
        g = gm*Tg+gb
        for Ta in range (0, taMax):
            a = am*Ta+ab
            for Va in range (0, vaMax):
                v = vm*Va+vb
                for Hum in range(0,rhMax):
                    h = hm*Hum+hb
                    # w = approximate_wet_bulb_psychro(v, a, g, h, 1013.5)
                    w = approximate_wet_bulb_ISO(v, a, g, h)
                    WBGT[Tg, Ta, Va, Hum] = 0.7*w + 0.2*g + 0.1*a
                    count = count + 1
            os.system("cls")
            print("Building matrix...")
            print(f"Progress: {count}/{total} ({(100*count/total):.3f}%)")
        os.system("cls")
        print("Building matrix...")
        print(f"Progress: {count}/{total} ({(100*count/total):.3f}%)")
    np.save("WBGTdb.npy", WBGT)
else:
    WBGT = np.load("WBGTdb.npy")


image1 = WBGT[0,:,:,0].transpose()
image2 = WBGT[1,:,:,0].transpose()
image3 = WBGT[0,:,:,1].transpose()
image4 = WBGT[1,:,:,1].transpose()

# fig, axes = plt.subplots(2,4, layout='tight')
# fig, axes = plt.subplot_mosaic([['plot1', 'plot2', 'plot3', 'plot4'], ['plot5', 'plot6', 'plot7', 'plot8']], layout='tight')
fig, axes = plt.subplot_mosaic([['plot1', 'plot2'], ['plot3', 'plot4']], layout='tight')
plt.subplot(221)
plt.imshow(image1, cmap='coolwarm', origin='lower')
plt.setp(axes['plot1'], xticks=np.linspace(0,200,11), xticklabels=np.linspace(10,ab+am*(taMax-1),11).round(3), yticks=np.linspace(0,200,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
plt.title("Ta x Va (RH=60,tg=50)")
plt.xlabel("Dry bulb temp")
plt.ylabel("Air velocity")
plt.colorbar()
plt.subplot(222)
plt.imshow(image2, cmap='coolwarm', origin='lower')
plt.setp(axes['plot2'], xticks=np.linspace(0,200,11), xticklabels=np.linspace(10,ab+am*(taMax-1),11).round(3), yticks=np.linspace(0,200,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
plt.title("Ta x Va (RH=60,tg=60)")
plt.xlabel("Dry bulb temp")
plt.ylabel("Air velocity")
plt.colorbar()
plt.subplot(223)
plt.imshow(image3, cmap='coolwarm', origin='lower')
plt.setp(axes['plot3'], xticks=np.linspace(0,200,11), xticklabels=np.linspace(10,ab+am*(taMax-1),11).round(3), yticks=np.linspace(0,200,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
plt.title("Ta x Va (RH=70,tg=50)")
plt.xlabel("Dry bulb temp")
plt.ylabel("Air velocity")
plt.colorbar()
plt.subplot(224)
plt.imshow(image4, cmap='coolwarm', origin='lower')
plt.setp(axes['plot4'], xticks=np.linspace(0,200,11), xticklabels=np.linspace(10,ab+am*(taMax-1),11).round(3), yticks=np.linspace(0,200,6), yticklabels=np.linspace(0,vb+vm*(vaMax-1),6).round(3))
plt.title("Ta x Va (RH=70,tg=60)")
plt.xlabel("Dry bulb temp")
plt.ylabel("Air velocity")
plt.colorbar()
plt.show()
