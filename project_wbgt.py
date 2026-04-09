import os
import numpy as np
# import sympy as smp
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

# Approximates wet bulb temp from v_a, t_a, t_g, and RH:
def approximate_wet_bulb(v, a, g, h):
    # Arbitrary starting values:
    w = dew_pt(a, h)
    p_w = Buck_equation(w)
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
            return w
        
        w = w + incr * dir
        p_w = Buck_equation(w)

        term1 = 4.18*np.float_power(v,0.444)*(a-w) + 0.00000001*((np.float_power(r+273,4))-(np.float_power(w+273,4)))
        term2 = 77.1*np.float_power(v,0.421)*(p_w-((h/100)*p_a))

        result_prev = result
        result = term1-term2

        if (abs(result) < 0.02):
                # print("final p_w= ", p_w)
                # print("final result= ", result)
                return w
        if (result < 0 and result_prev > 0) or (result_prev < 0 and result > 0):
            # print(result_prev, w, dir, incr)
            dir = dir * -1
            incr = incr/10
            # print(result, w, dir, incr, "\n")

    return w


GENERATE_MASTER_ARRAY = False

# SCRIPT START:
tgMax = 3
gm = 10.0
gb = 50.0
taMax = 201
am = 0.25
ab = 10.0
vaMax = 201
vm = 0.025
vb = 0.0
rhMax = 3
hm = 10.0
hb = 60.0
WBGT = np.zeros((tgMax, taMax, vaMax, rhMax))


if (GENERATE_MASTER_ARRAY):
    total = tgMax*taMax*vaMax*rhMax
    count = 0
    for Tg in range (0, tgMax):
        for Ta in range (0, taMax):
            for Va in range (0, vaMax):
                for Hum in range(0,rhMax):
                    v = vm*Va+vb
                    a = am*Ta+ab
                    g = gm*Tg+gb
                    h = hm*Hum+hb
                    Tw = approximate_wet_bulb(v, a, g, h)
                    WBGT[Tg, Ta, Va, Hum] = 0.7*Tw + 0.2*g + 0.1*a
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

fig, axes = plt.subplots(2,2, layout='tight')
plt.subplot(221)
plt.imshow(image1, cmap='coolwarm', origin='lower')
plt.title("Ta x Va (RH=60,tg=50)")
plt.xlabel("Dry bulb temp")
plt.setp(axes, xticks=np.linspace(0,200,11), xticklabels=np.linspace(10,60,11), yticks=np.linspace(0,200,11), yticklabels=np.linspace(0,5,11))
plt.ylabel("Air velocity")
plt.colorbar()
plt.subplot(222)
plt.imshow(image2, cmap='coolwarm', origin='lower')
plt.title("Ta x Va (RH=60,tg=60)")
plt.xlabel("Dry bulb temp")
plt.ylabel("Air velocity")
plt.colorbar()
plt.subplot(223)
plt.imshow(image3, cmap='coolwarm', origin='lower')
plt.title("Ta x Va (RH=70,tg=50)")
plt.xlabel("Dry bulb temp")
plt.ylabel("Air velocity")
plt.colorbar()
plt.subplot(224)
plt.imshow(image4, cmap='coolwarm', origin='lower')
plt.title("Ta x Va (RH=70,tg=60)")
plt.xlabel("Dry bulb temp")
plt.ylabel("Air velocity")
plt.colorbar()
plt.show()
