# %%
#python3 -m pip install module
import time as tt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
from matplotlib import colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits import mplot3d as d3
from scipy.optimize import curve_fit
from scipy import special
from scipy.integrate import cumulative_simpson as simpson
from astropy.time import Time
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
from functions import spectrum, smoother, fitter
from functions import plot_func_6, fft_logplot_spec_3, decrease_plot_adv
from functions import array_denormalize, array_normalize, array_modifier
from functions import array_columns, decrease_unit_day
from functions import arg_finder, fft, cl_lin
from functions import decrease_plot_hyperadv, step_data_generation
from functions import file_extreme_day_new, ele_gen_txt
from functions import get_n_s, lst_sqrs_adjstmnt_adv, insert
from functions import fit_func_1, fit_func_1_derivs
from functions import fit_func_2, fit_func_2_derivs
from functions import fit_func_3, fit_func_3_derivs
from functions import lst_sqrs_adjstmnt, ravel, fit_func_adv, n_objects
from functions import quotient, encode_list, list_code, N_inv_co, round_list
from functions import C_n_q_tilde, E_n_q_q_prime, F_n
from functions import N_eff, G_n_q_tilde, H_n, AT_P
from functions import x_q_n_vec, I_n, A_x, compute_data
from functions import v, splitter, m0_eff, L_constr
from functions import lsa_cond, lst_sqrs_adjstmnt_eff_adv
from functions import pre_lst_sqrs_adjstmt, fitter, mjd_to_mmdd, xaxis_year
#%%
print(os.getcwd())
os.chdir('/Users/levin/Documents/Uni/Bachelorthesis new')
print(os.getcwd())
#%%
#ele_gen_txt("grace18ele", 1)
# %%
path_ele = "grace18ele/year_normal/year_normal.txt"
path_acc = "ultimatedata/GF-C/all_corr.txt"

save_on = 0
path = os.getcwd() + "/updates/update 10/grace/c/"

step_fac = 10

#58356
MJD_interval = [58360, 58362]
MJD_0, MJD_end = MJD_interval[0], MJD_interval[1]
n_days_tot = MJD_end - MJD_0

vline_list1 = [58354.25, 58355.25, 59958.915, 59961.875, 59967.125,
               59988.71, 59995.416, 59999.5, 60001.78125, 60002.427,
               60017.17, 60018.17, 60023.9375, 60026.375, 60053.389]
vline_list2 = [58356]
vline_list_specs1 = ['k', 0.5, 1, "CME"]
vline_list_specs2 = ['gold', 0.75, 1, "26.08"]
# %%
MJD_0, MJD_end = MJD_interval[0], MJD_interval[1]
n_days_tot = MJD_end - MJD_0

data_ele = np.loadtxt(path_ele)
data_acc = np.loadtxt(path_acc)

data_ele = array_modifier(data_ele, MJD_0, n_days_tot)
data_acc = array_modifier(data_acc, MJD_0, n_days_tot)

data_ele = step_data_generation(data_ele, step_fac)
# %% COMPARE ACC VS ELE DIRECTION WISE
dir_list = ['R', 'S', 'W', 'A']
col_list = [['crimson', 'midnightblue', 'forestgreen', 'indigo'],
            ['tomato', 'deepskyblue', 'lime', 'fuchsia']]
for i in range(0, 4):
    col_ele = col_list[1][i]
    col_acc = col_list[0][i]
    
    mean_ele = np.mean(data_ele[:, i + 1])
    mean_acc = np.mean(data_acc[:, i + 1])
    
    dir = dir_list[i]
    mean_ele_str = r'\overline{D}_{\text{ELE}}'.replace('D', dir)
    mean_acc_str = r'\overline{D}_{\text{ACC}}'.replace('D', dir)
    str1 = r' $\left[ %s, %s \right]$' % (mean_ele_str, mean_acc_str)
    str2 = r' = [%.2e, %.2e] $\frac{m}{s^2}$' % (mean_ele, mean_acc)
    title = dir + '-acceleration:' + str1 + str2
    
    fig, ax = plt.subplots(figsize = (20, 5), dpi = 300)
    ax.plot(data_ele[:, 0], data_ele[:, i + 1],
            ls = 'solid', lw = 1,
            color = col_ele, alpha = 1,
            label = 'ELE')
    #ax.plot(data_acc[:, 0], data_acc[:, i + 1],
    #        ls = 'solid', lw = 1,
    #        color = col_acc, alpha = 0.75,
    #        label = 'ACC')
    ax.xaxis.set_major_formatter(mjd_to_mmdd)
    ax.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    ax.set_ylabel(r'$[\frac{m}{s^2}]$', fontsize = 15)
    fig.suptitle(title, fontsize = 17.5)
    plt.figlegend(fontsize = 15, markerscale = 5, loc = 1,
                  bbox_to_anchor = (1, 1), bbox_transform = ax.transAxes,
                  ncols = 1, labelspacing = 0)
    ax.grid()
    plt.show(fig)
    plt.close(fig)
# %% COMPARE ACC VS ELE ALL TOGETHER
fig, ax = plt.subplots(figsize = (15, 5), dpi = 300)
for i in range(0, 4):
    col_ele = col_list[1][i]
    col_acc = col_list[0][i]
    ax.plot(data_ele[:, 0], data_ele[:, i + 1],
             ls = 'solid', lw = 1,
             color = col_ele, alpha = 0.875,
             label = dir_list[i] + ' ELE')
    ax.plot(data_acc[:, 0], data_acc[:, i + 1],
             ls = 'solid', lw = 1,
             color = col_acc, alpha = 0.75,
             label = dir_list[i] + ' ACC')
ax.xaxis.set_major_formatter(mjd_to_mmdd)
ax.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
ax.set_ylabel(r'$[\frac{m}{s^2}]$', fontsize = 15)
fig.suptitle('All directions', fontsize = 17.5)
plt.figlegend(fontsize = 12, markerscale = 5, loc = 1,
              bbox_to_anchor = (1, 1), bbox_transform = ax.transAxes,
              ncols = 4, labelspacing = 0)
ax.grid()
plt.show(fig)
plt.close(fig)
# %% KRAUSS COMPARISON ########################
fig, ax = plt.subplots(figsize = (20, 5), dpi = 300)
col_ele = col_list[1][3]
col_acc = col_list[0][3]
ax.semilogy(data_ele[:, 0], data_ele[:, 4] - 2e-8,
            ls = 'solid', lw = 1,
            color = col_ele, alpha = 1,
            label = 'ELE')
#ax.plot(data_acc[:, 0], data_acc[:, 4],
#         ls = 'solid', lw = 1,
#         color = col_acc, alpha = 0.75,
#         label = 'ACC')
ax.xaxis.set_major_formatter(mjd_to_mmdd)
ax.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
ax.set_ylabel(r'$[\frac{m}{s^2}]$', fontsize = 15)
fig.suptitle(dir_list[3] + ' shifted by -2e-8 m/s^2', fontsize = 17.5)
plt.figlegend(fontsize = 15, markerscale = 5, loc = 1,
              bbox_to_anchor = (1, 1), bbox_transform = ax.transAxes,
              ncols = 1, labelspacing = 0)
ax.set_ylim(10**(-9), 10**(-7))
ax.grid(which = 'minor')
ax.grid()
plt.show(fig)
plt.close(fig)
# %%
