# %% import stuff
# python3 -m pip install module
# python -m pip install numpy
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat

from VanVid_functions_final import master_lst, master_ele
from VanVid_functions_final import master_integrator

plt.style.use('classic')
plt.rcParams['text.usetex'] = True
mat.rcParams['figure.facecolor'] = 'white'
# %%
print(os.getcwd())
os.chdir('..')
#os.chdir('FurtherResearch/')
print(os.getcwd())
print(os.listdir())
# %% Get data location ('swarm24/' in my case)
path = 'swarm24/'
oslist = os.listdir(path)
# %% create master files if not already done (swarm24 has folders LST and ELE)
# create master LST files
osc_list = [["r", 11], ["e", 21], ["omega_small", 24],["u", 17]]
master_lst('swarm24/LST', 1, osc_list, 'swarm24/')

# create master ELE files
master_ele('swarm24/ELE', 'swarm24/')

# create master ACC files if ACC data is available
# master_acc... # need to write your own function for this (data files may be different from my case)
# %% set time interval for the integration
MJD_interval = [60431, 60445] 
MJD_0, MJD_end = MJD_interval[0], MJD_interval[1]
n_days_tot = MJD_end - MJD_0
# %% Integration
a_int_pca_list = master_integrator(path, MJD_interval, 'PCA')
print("pca integration complete")

if ('ACC.txt' in oslist):
    a_int_acc_list = master_integrator(path, MJD_interval, 'ACC')
    print("acc integration complete")
# %% split into integrated a and a_dot
a_int_pca, a_dot_pca = a_int_pca_list[0], a_int_pca_list[1]

if ('ACC.txt' in oslist):
    a_int_acc, a_dot_acc = a_int_acc_list[0], a_int_acc_list[1]
# %%
a_int_error_scaling_factor = 1e5
a_dot_error_scaling_factor = 1e9

fig, ax = plt.subplots(dpi = 300)
ax.set_title("a_int")
ax.plot(a_int_pca[:, 0], a_int_pca[:, 1], 'r-')
ax.fill_between(a_int_pca[:, 0],
                a_int_pca[:, 1] - a_int_pca[:, 2] * a_int_error_scaling_factor,
                a_int_pca[:, 1] + a_int_pca[:, 2] * a_int_error_scaling_factor,
                color = 'orange', ls = 'solid')
plt.show(fig)
plt.close(fig)

fig, ax = plt.subplots(dpi = 300)
ax.set_title("a_dot")
ax.plot(a_dot_pca[:, 0], a_dot_pca[:, 1], 'b-')
ax.fill_between(a_dot_pca[:, 0],
                a_dot_pca[:, 1] - a_dot_pca[:, 2] * a_dot_error_scaling_factor,
                a_dot_pca[:, 1] + a_dot_pca[:, 2] * a_dot_error_scaling_factor,
                color = 'deepskyblue', ls = 'solid')
plt.show(fig)
plt.close(fig)