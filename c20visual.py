#%%
#python3 -m pip install module
import time as tt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
from matplotlib import colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from scipy.optimize import curve_fit
from scipy import special
from scipy.integrate import cumulative_simpson as simpson
import os
import random
import matplotlib.image as mpimg
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
plt.style.use('classic')
mat.rcParams['figure.facecolor'] = 'white'
print(np.random.randint(1,9))

# print(u'\u03B1') # alpha
# print(u'\u03C3') # sigma
# %%
c20 = -1.083*10**(-3)
gamma = 3.986*10**14
re = 6378*1e3
r = re+500*1e3
i = 87.35*np.pi/180
u_list = np.linspace(0, 2, 1000)
fac = 3/2 * c20 * gamma * re**2 / (r**4)
print(fac)

def f2(rsw, u_list):
    print("hal")
    f = np.array([])
    for k in range(0, len(u_list)):
        u = u_list[k] * np.pi
        if (rsw == 1):
            term = 1 - 1.5 * np.sin(i)**2 * (1 - np.cos(2*u))
        elif (rsw == 2):
            term = np.sin(i)**2 * np.sin(2*u)
        else:
            term = np.sin(2*i) * np.sin(u)
        f = np.append(f, term)
    f = fac * f
    return(f)

fr_list = f2(1, u_list)
fs_list = f2(2, u_list)
fw_list = f2(3, u_list)
# %%
fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.plot(u_list, fr_list, 'tab:orange', ls = 'solid',
         lw = 1, label = r'$F_R$')
plt.plot(u_list, fs_list, 'tab:blue', ls = 'solid',
         lw = 1, label = r'$F_S$')
plt.plot(u_list, fw_list, 'tab:green', ls = 'solid',
         lw = 1, label = r'$F_W$')
plt.xlabel(r'$u$ [$\pi$]', fontsize = 20)
plt.ylabel(r'$F$ [$\frac{\text{m}}{\text{s}^2}$]',fontsize = 20)
plt.legend()
plt.show(fig)
plt.close(fig)

# %%

# %%
print(3/2)
#%%
