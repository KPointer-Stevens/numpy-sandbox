import os
import numpy as np
import sympy as smp
import matplotlib.pyplot as plt


def dew_pt(DB, RH):
    gamma = np.log(RH/100)+(17.625*DB)/(243.04+DB)
    return (243.04*gamma)/(17.625-gamma)

def mean_radiant_from_globe(GT, AV, DB):
    return np.float_power(np.float_power(GT+273,4)+((1.1*np.float_power(10,8)*np.float_power(AV,0.6))/(0.95*np.float_power(0.15,0.4)))*(GT-DB),0.25)-273

# returns vapor pressure in kPa
def Buck_equation(DB):
    d = DB
    if d == -257.14:
        d = -257.15
        print("DB ERR! new DB=", d)
    return 0.61121 * np.exp((18.678-(d/234.5))*(d/(257.14+d)))

# Approximates wet bulb temp from DB, AV, 
def approximate_wet_bulb(v, a, g, h):
    # Arbitrary starting values:
    w = dew_pt(a, h)
    p_w = Buck_equation(w)
    p_a = Buck_equation(a)
    r = mean_radiant_from_globe(g, v, a)
    print("w=", w, ", p_w=", p_w, ", p_a=", p_a, ", r=", r)
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
            return w
        
        # if (w < -100):
        #     dir = 1
        
        w = w + incr * dir
        p_w = Buck_equation(w)
        
        term1 = 4.18*np.float_power(v,0.444)*(a-w) + np.float_power(10, -8)*(np.float_power(r+273,4)-np.float_power(w+273,4))
        term2 = 77.1*np.float_power(v,0.421)*(p_w-h*p_a)

        result_prev = result
        result = term1-term2
        print(result_prev, "->", result)

        if (abs(result) < 0.05):
                print("SUCCESS!")
                return w
        if (result < 0 and result_prev > 0) or (result_prev < 0 and result > 0):
            dir = dir * -1
            incr = incr/10
            print("dir=", dir, ", incr=", incr)

    return w


# SCRIPT BODY START:
dry_bulb = 25
globe = 35
humidity = 75
air_velocity = 0.5
print("dry bulb=", dry_bulb, "\nglobe temp=", globe, "\nhumidity=", humidity, "\nwind speed=", air_velocity)

wet_bulb = approximate_wet_bulb(air_velocity, dry_bulb, globe, humidity)
print("WBT=", wet_bulb)

wbgt = 0.7*wet_bulb + 0.2*globe + 0.1*dry_bulb
print("WBGT=", wbgt)