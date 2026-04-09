import os
import numpy as np
# import sympy as smp
import matplotlib.pyplot as plt

# Calculates rough dew point, for starting guess:
def dew_pt(DB, RH):
    gamma = np.log(RH/100)+(17.625*DB)/(243.04+DB)
    return (243.04*gamma)/(17.625-gamma)

# Returns median radiant temperature:
def mean_radiant_from_globe(GT, AV, DB):
    # return np.float_power(np.float_power(GT+273,4)+((1.1*np.float_power(10,8)*np.float_power(AV,0.6))/(0.95*np.float_power(0.15,0.4)))*(GT-DB),0.25)-273
    return np.float_power(np.float_power(GT+273,4)+(((110000000*np.float_power(AV,0.6))/(0.95*np.float_power(0.15,0.4)))*(GT-DB)),0.25)-273

# Returns partial vapor pressure in kPa:
def Buck_equation(DB):
    d = DB
    if d == -257.14:
        d = -257.15
        print("DB ERR! new DB=", d)
    return 0.61121 * np.exp((18.678-(d/234.5))*(d/(257.14+d)))

# Approximates wet bulb temp from v_a, t_a, t_g, and RH:
def approximate_wet_bulb(v, a, g, h):
    # Arbitrary starting values:
    w = dew_pt(a, h)
    p_w = Buck_equation(w)
    p_a = Buck_equation(a)
    r = mean_radiant_from_globe(g, v, a)
    print("\nw=", w, ", p_w=", p_w, ", p_a=", p_a, ", r=", r)
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
        
        w = w + incr * dir
        p_w = Buck_equation(w)

        # term1 = 4.18*np.float_power(v,0.444)*(a-w) + np.float_power(10,-8)*(np.float_power(r+273,4) - np.float_power(w+273,4))
        # term2 = 77.1*np.float_power(v,0.421)*(p_w-h*p_a)
        # print("w=", w)
        term1 = 4.18*np.float_power(v,0.444)*(a-w) + np.float_power(10,-8) * ((np.float_power(r+273,4))-(np.float_power(w+273,4)))
        print("r=", r)
        # print("T1=", term1)
        term2 = 77.1*np.float_power(v,0.421)*(p_w-(h*p_a))
        # print("T2=", term2, "\n")

        result_prev = result
        result = term1-term2

        if (abs(result) < 0.02):
                print("final p_w= ", p_w)
                print("final result= ", result)
                return w
        if (result < 0 and result_prev > 0) or (result_prev < 0 and result > 0):
            # print(result_prev, w, dir, incr)
            dir = dir * -1
            incr = incr/10
            # print(result, w, dir, incr, "\n")

    return w


GENERATE_MASTER_ARRAY = False

print("4.18*v(a-w)=", 4.18*np.float_power(0.1,0.444)*(30-145.1798))

print("v=", 0.2, ", a=", 30, ", g=", 40, ", h=", 100)
print("t_nw =", approximate_wet_bulb(0.2,30,40,100), "\n")

# SCRIPT START:
tgMax = 26
gm = 1.0
gb = 20.0
taMax = 26
am = 1.0
ab = 20.0
vaMax = 26
vm = 0.2
vb = 0.0
rhMax = 26
hm = 5.0
hb = 0.0
WBGT = np.zeros((tgMax, taMax, vaMax, rhMax))


if (GENERATE_MASTER_ARRAY):
    total = tgMax*taMax*vaMax*rhMax
    count = 0
    for Tg in range (0, tgMax):
        for Ta in range (0, taMax):
            for Va in range (0, vaMax):
                for Hum in range(0,rhMax):
                    # Tw = approximate_wet_bulb(vaScale[Va], taScale[Ta], tgScale[Tg], rhScale[Hum])
                    v = vm*Va+vb
                    a = am*Ta+ab
                    g = gm*Tg+gb
                    h = hm*Hum+hb
                    Tw = approximate_wet_bulb(v, a, g, h)
                    WBGT[Tg, Ta, Va, Hum] = 0.7*Tw + 0.2*g + 0.1*a
                    count = count + 1
            os.system("cls")
            print("Building table...")
            print(f"Progress: {count}/{total} ({(100*count/total):.3f}%)")
        os.system("cls")
        print("Building table...")
        print(f"Progress: {count}/{total} ({(100*count/total):.3f}%)")
    np.save("WBGTdb.npy", WBGT)
else:
    WBGT = np.load("WBGTdb.npy")


fig, (ax1, ax2) = plt.subplots(2,2, layout="constrained")
# plt.figure(1)
plt.subplot(221)
plt.imshow(WBGT[15,5,:,:], cmap='coolwarm')
plt.title(("Va x RH; Tg,Ta=", gm*15+gb, am*5+ab))
plt.xlabel("Wind Speed")
plt.ylabel("Humidity")
plt.colorbar()
# plt.figure(2)
plt.subplot(222)
plt.imshow(WBGT[15,:,:,20], cmap='coolwarm')
plt.title(("Ta x Va; Tg,RH=", gm*15+gb, hm*20+hb))
plt.xlabel("Dry Bulb Temp")
plt.ylabel("Wind Speed")
plt.colorbar()
# plt.figure(3)
plt.subplot(223)
plt.imshow(WBGT[:,:,1,20], cmap='coolwarm')
plt.title(("Tg x Ta; Va,RH=", vm*1+vb, hm*20+hb))
plt.xlabel("Globe Temp")
plt.ylabel("Dry Bulb Temp")
plt.colorbar()
# plt.figure(4)
plt.subplot(224)
plt.imshow(WBGT[:,5,1,:], cmap='coolwarm')
plt.title(("Tg x RH; Ta,Va=", am*5+ab, vm*1+vb))
plt.xlabel("Globe Temp")
plt.ylabel("Humidity")
plt.colorbar()
# plt.show()
