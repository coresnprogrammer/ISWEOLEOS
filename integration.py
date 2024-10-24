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
from functions import arg_finder, fft, step_data_generation
from functions import decrease_plot_hyperadv
from functions import file_extreme_day_new
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
from functions import pre_lst_sqrs_adjstmt, fitter, mjd_to_mmdd
# %%
string_list = ["u_sat", "beta_sun", "u_sun",
               "a", "e", "i", "Omega_upp", "omega_low", "T0",
               "rho", "r", "h_ell"]
name_list = [["satellite anomaly", r"$u_{sat}$", r"[$^{\circ}$]",
              r"$\bar{u}_{sat}$", r"$\widehat{u}_{sat}$", r"$\tilde{u}_{sat}$"],
             ["beta angle", r"$\beta_{\odot}$", r"[$^{\circ}$]",
              r"$\bar{\beta}_{\odot}$", r"$\widehat{\beta}_{\odot}$", r"$\tilde{\beta}_{\odot}$"],
             ["sun anomaly", r"$u_{\odot}$", r"[$^{\circ}$]",
              r"$\bar{u}_{\odot}$", r"$\widehat{u}_{\odot}$", r"$\tilde{u}_{\odot}$"],
             ["semi-major axis", r"$a$", r"[$m$]", r"$\bar{a}$", r"$\widehat{a}$", r"$\tilde{a}$"],
             ["eccentricity", r"$e$", r"[$-$]", r"$\bar{e}$", r"$\widehat{e}$", r"$\tilde{e}$"],
             ["inclination", r"$i$", r"[$^{\circ}$]", r"$\bar{i}$", r"$\widehat{i}$", r"$\tilde{i}$"],
             ["longitude of ascending node", r"$\Omega$", r"[$^{\circ}$]",
              r"$\bar{\Omega}$", r"$\widehat{\Omega}$", r"$\tilde{\Omega}$"],
             ["argument of periapsis", r"$\omega$", r"[$^{\circ}$]",
              r"$\bar{\omega}$", r"$\widehat{\omega}$", r"$\tilde{\omega}$"],
             ["time of periapsis passage", r"$T_0$", r"[$s$]",
              r"$\bar{T}_0$", r"$\widehat{T}_0$", r"$\tilde{T}_0$"],
             ["air density", r"$\varrho$", r"[$kg m^{-3}$]",
              r"$\bar{\varrho}$", r"$\widehat{\varrho}$", r"$\tilde{\varrho}$"],
             ["radius", r"$r$", r"[$m$]", r"$\bar{r}$", r"$\widehat{r}$", r"$\tilde{r}$"],
             ["ell. height", r"$h_{ell}$", r"[$m$]",
              r"$\bar{h}_{ell}$", r"$\widehat{h}_{ell}$", r"$\tilde{h}_{ell}$"]]
#%%
print(os.getcwd())
os.chdir('/Users/levin/Documents/Uni/Bachelorthesis new')
print(os.getcwd())
#%%
#file_extreme_day_new('grace18newlst', 2)
#%%
# general constants
sec_to_day = 1 / (24 * 60 * 60)
G  = 6.674e-11 / (sec_to_day**2)
Me = 5.972e24
μ = G * Me

save_on = 0
path = os.getcwd() + "/updates/update 10/grace/c/"

foldername_lst = "grace18newlst"
foldername_acc = "ultimatedata/GF-C"
path_ele = "grace18ele/year_normal/year_normal.txt"
file_name = "nongra"
#58356
MJD_interval = [58352, 58362]
MJD_0, MJD_end = MJD_interval[0], MJD_interval[1]
n_days_tot = MJD_end - MJD_0

str_list = ["a", "e", "omega_low", "u_sat"]

a_symb, a_unit, a_dot = r"$a$", r"[$m$]", r"$\dot{a}$"

# prepare el, el0, acc
el_data_list = []
for i in range(0, len(str_list)):
    el_str = str_list[i]
    name_el = foldername_lst + 2 * ("/year_" + file_name) + "_" + el_str + ".txt"
    el_data = np.loadtxt(name_el, skiprows=1)  # load data
    el_data = array_denormalize(el_data)  # denormalize data
    el_data = array_modifier(el_data, MJD_0, n_days_tot)  # trim data
    #el_data = array_normalize(el_data, 0)  # normalize data
    #el_data = el_data[1:]  # cut MJD_0
    el_data_list.append(el_data)
a_data, e_data = el_data_list[0], el_data_list[1]
ω_data, u_data = el_data_list[2], el_data_list[3]

name_acc = foldername_acc + "/all_corr.txt"
acc_data = np.loadtxt(name_acc) # unnormalized
# first plott acc then modify
acc_R_data_0 = array_columns(acc_data, [0, 1])
acc_S_data_0 = array_columns(acc_data, [0, 2])
acc_W_data_0 = array_columns(acc_data, [0, 3])
acc_A_data_0 = array_columns(acc_data, [0, 4])

mean_data = np.loadtxt(foldername_acc + "/mean.txt")
scale_data = np.loadtxt(foldername_acc + "/scale.txt")
bias_data = np.loadtxt(foldername_acc + "/bias.txt")


ele_data = np.loadtxt(path_ele)

ele_R_data_0 = array_columns(ele_data, [0, 1])
ele_S_data_0 = array_columns(ele_data, [0, 2])
ele_W_data_0 = array_columns(ele_data, [0, 3])
ele_A_data_0 = array_columns(ele_data, [0, 4])


# specifically for VXYZ_EF
#t_list = V_data[:, 0]
#V_list = np.sqrt(np.sum((V_data[:, 1:])**2, 1)) # |V|
#V_data = np.vstack((t_list, V_list)).T

vline_list1 = [58354.25, 58355.25, 59958.915, 59961.875, 59967.125,
               59988.71, 59995.416, 59999.5, 60001.78125, 60002.427,
               60017.17, 60018.17, 60023.9375, 60026.375, 60053.389]
vline_list2 = [58356]
vline_list_specs1 = ['k', 0.5, 1, "CME"]
vline_list_specs2 = ['gold', 0.75, 1, "26.08"]
#%%
from astropy.time import Time
def mjd_to_mmdd(t, pos):
    t_obj = Time(str(t), format = 'mjd')
    t_iso = t_obj.iso
    string = t_iso[5 : 10].replace('-', '.')
    return(string)

def mjd_to_ddmm(t, pos):
    t_obj = Time(str(t), format = 'mjd')
    t_iso = t_obj.iso
    mm = t_iso[5 : 7]
    dd = t_iso[8 : 10]
    string = dd + "." + mm
    return(string)

def xaxis_year(t_list, time_format):
    t0 = t_list[0]
    t_obj = Time(str(t0), format = 'mjd')
    t_iso = t_obj.iso
    string = t_iso[0 : 4]
    string = string + " + " + time_format
    return(string)

def plot_func_6_un(data_array_list, data_spec_array, y_label,
                   ref_array_list, ref_spec_array, ref_y_spec_list,
                   tit, fosi, n_cols,
                   MJD_0, MJD_end, xlimits, ylimits,
                   vline_specs, z_order, nongra_fac, grid,
                   vline_list1, vline_list_specs1,
                   vline_list2, vline_list_specs2,
                   save_specs):
    # data_array_list = [data_1, data_2, ...]
    # data_spec_array = [alpha_i, col_i, lab_i, lw_i]
    # y_label: label for y axis
    # ref_array_list: [ref_1, ref_2, ...]
    # ref_spec_array: [alpha_i, col_i, lab_i, lw_i]
    # ref_y_spec_list: [y_label, y_axis_col]
    # tit: title, string like
    # MJD_0: data start
    # MJD_end: data end
    # xlimits = [x_start, x_end] (to deactivate: x_start = x_end)
    # ylimits = [y_low, y_upp]: (to deactivate: y_low = y_upp)
    # vline_specs = [col, alpha, lw]
    # z_order: if 1 --> data in front of ref
    # nongra_fac: for scaling nongra data
    # grid: if 1 --> grid to data
    # vline_list = [vline_i]
    # vline_list_specs = [col_i, alpha_i, lw_i, label]
    # save_specs = [1, path] # 0,1 for saving
    xstart = xlimits[0] + MJD_0
    xend = MJD_end - xlimits[1]
    n_days = xend - xstart
    new_data_array_list = 0
    new_ref_array_list = 0
    if (xlimits[0] != xlimits[1]):
        new_data_array_list = []
        new_ref_array_list = []
        for i in range(0, len(data_array_list)):
            data_i = data_array_list[i]
            data_i = array_modifier(data_i, xstart, n_days)
            new_data_array_list.append(data_i)
        for i in range(0, len(ref_array_list)):
            ref_i = ref_array_list[i]
            ref_i = array_modifier(ref_i, xstart, n_days)
            new_ref_array_list.append(ref_i)
    else:
        new_data_array_list = data_array_list
        new_ref_array_list = ref_array_list
    
    fig, ax1 = plt.subplots(figsize = (10, 5), dpi = 300)
    fig.suptitle(tit, fontsize = 17.5)
    
    for i in range(0, len(new_data_array_list)):
        x_data = new_data_array_list[i][:, 0]
        y_data = new_data_array_list[i][:, 1]
        α = data_spec_array[i][0]
        col = data_spec_array[i][1]
        lab = data_spec_array[i][2]
        width = data_spec_array[i][3]
        if (i == 1):
            if (nongra_fac != 1):
                lab = lab + ' (scaled by ' + str(nongra_fac) + ')'
                y_data *= nongra_fac
        ax1.plot(x_data, y_data, color = col, ls = '-', lw = width,
                 alpha = α, zorder = 5, label = lab)
    ax1.xaxis.set_major_formatter(mjd_to_mmdd)
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    ax1.set_ylabel(y_label, fontsize = 15)
    y_low, y_upp = ylimits[0], ylimits[1]
    if (xstart != xend):
        ax1.set_xlim(xstart, xend)
    if (y_low != y_upp):
        ax1.set_ylim(y_low, y_upp)
    if (grid == 1):
        ax1.grid()
    
    if (len(new_ref_array_list) != 0):
        ax2 = ax1.twinx()
        for i in range(0, len(new_ref_array_list)):
            x_ref = new_ref_array_list[i][:, 0]
            y_ref = new_ref_array_list[i][:, 1]
            α = ref_spec_array[i][0]
            col = ref_spec_array[i][1]
            lab = ref_spec_array[i][2]
            width = ref_spec_array[i][3]
            ax2.plot(x_ref, y_ref,
                     ls = '-', lw = width, color = col, alpha = α,
                     label = lab)
        ax2.set_ylabel(ref_y_spec_list[0], color = ref_y_spec_list[1], fontsize = 15)
        ax2.tick_params(axis = 'y', labelcolor = ref_y_spec_list[1])
        
        if (z_order == 1):
            ax1.set_zorder(ax2.get_zorder()+1)
            ax1.set_frame_on(False)
    
    c = 0 # for giving only one label to vlines from vline_list1
    for i in range(0, len(vline_list1)):
        vline = vline_list1[i]
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs1[3]
                c += 1
            plt.axvline(vline, color = vline_list_specs1[0],
                        alpha = vline_list_specs1[1], lw = vline_list_specs1[2],
                        ls = (0, (4, 4)), zorder = 3,
                        label = c_label)
    
    c = 0 # for giving only one label to vlines from vline_list2
    for i in range(0, len(vline_list2)):
        vline = vline_list2[i]
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs2[3]
                c += 1
            plt.axvline(vline, color = vline_list_specs2[0],
                        alpha = vline_list_specs2[1], lw = vline_list_specs2[2],
                        ls = (0, (4, 4)), zorder = 3,
                        label = c_label)
    
    plt.figlegend(fontsize = fosi, markerscale = 5, loc = 1,
                  bbox_to_anchor = (1, 1), bbox_transform = ax1.transAxes,
                  labelspacing = 0, ncols = n_cols)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    
    plt.show(fig)
    plt.close(fig)

data_array_list = [acc_R_data_0, ele_R_data_0,
                   acc_S_data_0, ele_S_data_0,
                   acc_W_data_0, ele_W_data_0,
                   acc_A_data_0, ele_A_data_0]
data_spec_array = [[0.75, 'crimson', r'$R$ (ACC)', 1], [0.75, 'tomato', r'$R$ (ELE)', 1],
                   [0.75, 'midnightblue', r'$S$ (ACC)', 1], [0.75, 'deepskyblue', r'$S$ (ELE)', 1],
                   [0.75, 'forestgreen', r'$W$ (ACC)', 1], [0.75, 'lime', r'$W$ (ELE)', 1],
                   [0.75, 'indigo', r'$A$ (ACC)', 1], [0.75, 'fuchsia', r'$A$ (ELE)', 1]]

y_label = "m/s^2"
ref_array_list, ref_spec_array, ref_y_spec_list = [], [], []
tit = "ACC and ELE: R, S, W, A"
y_mean_list = []
y_std_list = []
for i in range(0, len(data_array_list)):
    y_data = data_array_list[i][:, 1]
    y_mean = np.mean(y_data)
    y_std = np.std(y_data)
    
    y_mean_list.append(y_mean)
    y_std_list.append(y_std)

factor = 0.02
y_mean = min(np.abs(y_mean_list))
y_std = max(y_std_list) * factor
xlimits = [0, 0]

ylimits = [y_mean - y_std, y_mean + y_std]
vline_specs = ["gold", 0, 1]
z_order = 1
nongra_fac = 1
grid = 1
fosi = 10
n_cols = 5
save_specs = [0]
plot_func_6_un(data_array_list, data_spec_array, y_label,
               ref_array_list, ref_spec_array, ref_y_spec_list,
               tit, fosi, n_cols,
               acc_R_data_0[0, 0], acc_R_data_0[-1, 0], xlimits, ylimits,
               vline_specs, z_order, nongra_fac, grid,
               vline_list1, vline_list_specs1,
               vline_list2, vline_list_specs2,
               save_specs)

#%%
acc_R_data = array_modifier(acc_R_data_0, MJD_0, n_days_tot)
acc_S_data = array_modifier(acc_S_data_0, MJD_0, n_days_tot)
acc_W_data = array_modifier(acc_W_data_0, MJD_0, n_days_tot)
acc_A_data = array_modifier(acc_A_data_0, MJD_0, n_days_tot)

ele_data = array_modifier(ele_data, MJD_0, n_days_tot)
#%%
mean_acc_R = np.mean(acc_R_data[:, 1])
mean_acc_S = np.mean(acc_S_data[:, 1])
mean_acc_W = np.mean(acc_W_data[:, 1])

mean_ele_R = np.mean(ele_data[:, 1])
mean_ele_S = np.mean(ele_data[:, 2])
mean_ele_W = np.mean(ele_data[:, 3])

Δ_mean_R = mean_ele_R - mean_acc_R
Δ_mean_S = mean_ele_S - mean_acc_S
Δ_mean_W = mean_ele_W - mean_acc_W

acc_R_data_shifted = acc_R_data + 0 # avoid pointer problems
acc_S_data_shifted = acc_S_data + 0 # avoid pointer problems
acc_W_data_shifted = acc_W_data + 0 # avoid pointer problems

acc_R_data_shifted[:, 1] += Δ_mean_R
acc_S_data_shifted[:, 1] += Δ_mean_S
acc_W_data_shifted[:, 1] += Δ_mean_W

print(np.mean(acc_R_data_shifted[:, 1]), mean_ele_R)
print(np.mean(acc_S_data_shifted[:, 1]), mean_ele_S)
print(np.mean(acc_W_data_shifted[:, 1]), mean_ele_W)
#%%
step_fac = 72 # 72 * 5s = 6min
ele_data = step_data_generation(ele_data, step_fac)

ele_R_data = array_columns(ele_data, [0, 1])
ele_S_data = array_columns(ele_data, [0, 2])
ele_W_data = array_columns(ele_data, [0, 3])
ele_A_data = array_columns(ele_data, [0, 4])
#%%
data_array_list = [acc_R_data, ele_R_data,
                   acc_S_data, ele_S_data,
                   acc_W_data, ele_W_data,
                   acc_A_data, ele_A_data]

data_spec_array = [[0.75, 'crimson', r'$R$ (ACC)', 1], [0.75, 'tomato', r'$R$ (ELE)', 1],
                   [0.75, 'midnightblue', r'$S$ (ACC)', 1], [0.75, 'deepskyblue', r'$S$ (ELE)', 1],
                   [0.75, 'forestgreen', r'$W$ (ACC)', 1], [0.75, 'lime', r'$W$ (ELE)', 1],
                   [0.75, 'indigo', r'$A$ (ACC)', 1], [0.75, 'fuchsia', r'$A$ (ELE)', 1]]

y_mean_list = []
y_std_list = []
for i in range(0, len(data_array_list)):
    y_data = data_array_list[i][:, 1]
    y_mean = np.mean(y_data)
    y_std = np.std(y_data)
    
    y_mean_list.append(y_mean)
    y_std_list.append(y_std)

factor = 8
y_mean = min(np.abs(y_mean_list))
y_std = max(y_std_list) * factor
ylimits = [y_mean - y_std, y_mean + y_std]
fosi = 10
n_cols = 5
plot_func_6_un(data_array_list, data_spec_array, y_label,
               ref_array_list, ref_spec_array, ref_y_spec_list,
               tit, fosi, n_cols,
               acc_R_data[0, 0], acc_R_data[-1, 0], xlimits, ylimits,
               vline_specs, z_order, nongra_fac, grid,
               vline_list1, vline_list_specs1,
               vline_list2, vline_list_specs2,
               save_specs)
#%%
def a_dot_RSW(a, e, ω, u, R, S, σ_a, σ_e, σ_ω):
    ν = (u - ω) * np.pi / 180
    fac = 2 * np.sqrt(a**3 / (μ * (1 - e**2)))
    term1 = e * np.sin(ν) * R
    term2 = (1 + e * np.cos(ν)) * S
    
    a_dot = fac * (term1 + term2)
    
    term_σ_a = 3 / 2 * (term1 + term2) / a
    term_σ_e = (np.sin(ν) * R + (e + np.cos(ν)) * S) / (1 - e**2)
    term_σ_ω = e * (np.cos(ν) * R - np.sin(ν) * S)
    
    term_σ_a_2 = (term_σ_a * σ_a)**2
    term_σ_e_2 = (term_σ_e * σ_e)**2
    term_σ_ω_2 = (term_σ_ω * σ_ω)**2
    
    a_dot_error = fac * np.sqrt(term_σ_a_2 + term_σ_e_2 + term_σ_ω_2)
    
    return(a_dot, a_dot_error)

def rk45_2dt(dt, a_n, u_list, R_list, S_list, e, ω, σ_a, σ_e, σ_ω): # step = 2 * dt
    u_n, u_np1, u_np2 = u_list[0], u_list[1], u_list[2] # u(t, t + dt, t + 2dt)
    R_n, R_np1, R_np2 = R_list[0], R_list[1], R_list[2] # R(t, t + dt, t + 2dt)
    S_n, S_np1, S_np2 = S_list[0], S_list[1], S_list[2] # S(t, t + dt, t + 2dt)
    
    k1, σ_k1 = 2 * dt * a_dot_RSW(a_n, e, ω, u_n, R_n, S_n, σ_a, σ_e, σ_ω)
    k2, σ_k2 = 2 * dt * a_dot_RSW(a_n + k1 / 2, e, ω, u_np1, R_np1, S_np1, σ_k1 / 2, σ_e, σ_ω)
    k3, σ_k3 = 2 * dt * a_dot_RSW(a_n + k2 / 2, e, ω, u_np1, R_np1, S_np1, σ_k2 / 2, σ_e, σ_ω)
    k4, σ_k4 = 2 * dt * a_dot_RSW(a_n + k3, e, ω, u_np2, R_np2, S_np2, σ_k3, σ_e, σ_ω)
    
    a_np2 = a_n + (k1 + k4) / 6 + (k2 + k3) / 3
    a_np2_error = np.sqrt(σ_a**2 + (σ_k1**2 + σ_k4**2) / 36 + (σ_k2**2 + σ_k3**2) / 9)
    return(a_np2, a_np2_error)

def rk45_discrete(a_data, e_data, ω_data, u_data, acc_R_data, acc_S_data, σ_a_0):
    e = e_data[0, 1]
    ω = ω_data[0, 1]
    
    σ_e = np.std(e_data[:, 1])
    σ_ω = np.std(ω_data[:, 1])
    
    u_list_list = u_data[:, 1]
    R_list_list = acc_R_data[:, 1] / (sec_to_day**2)
    S_list_list = acc_S_data[:, 1] / (sec_to_day**2)
    
    t_0 = a_data[0, 0] 
    a_0 = a_data[0, 1]
    a_dot_0, σ_a_dot_0 = a_dot_RSW(a_0, e, ω, u_list_list[0], R_list_list[0],
                                   S_list_list[0], σ_a_0, σ_e, σ_ω)
    
    a_int_data = np.array([[t_0, a_0, σ_a_0]])
    a_dot_data = np.array([[t_0, a_dot_0, a_dot_0]])
    
    dt = a_data[1, 0] - a_data[0, 0] # 5s
    for i in range(0, int(len(a_data) / 2) - 1):
        # _np2 for_(n+2)
        n = 2 * i
        t_np2 = a_data[n + 2, 0]
        a_n = a_int_data[-1, 1]
        σ_a = a_int_data[-1, 2]
        
        u_list = u_list_list[n : n + 2 + 1]
        R_list = R_list_list[n : n + 2 + 1]
        S_list = S_list_list[n : n + 2 + 1]
        
        a_np2, σ_a_np2 = rk45_2dt(dt, a_n, u_list, R_list, S_list,
                                  e, ω, σ_a, σ_e, σ_ω)
        
        a_dot_np2, σ_a_dot_np2 = a_dot_RSW(a_np2, e, ω, u_list[-1],
                                           R_list[-1], S_list[-1],
                                           σ_a_np2, σ_e, σ_ω)
        
        a_int_row = np.array([t_np2, a_np2, σ_a_np2])
        a_dot_row = np.array([t_np2, a_dot_np2, σ_a_dot_np2])
        
        a_int_data = np.vstack((a_int_data, a_int_row))
        a_dot_data = np.vstack((a_dot_data, a_dot_row))
    
    return([a_int_data, a_dot_data])
#%%
a_data_int_acc_list = rk45_discrete(a_data, e_data, ω_data, u_data, acc_R_data, acc_S_data, 0)
a_data_int_ele_list = rk45_discrete(a_data, e_data, ω_data, u_data, ele_R_data, ele_S_data, 0)
a_data_int_acc_shifted_list = rk45_discrete(a_data, e_data, ω_data, u_data,
                                       acc_R_data_shifted, acc_S_data_shifted, 0)

a_data_int_acc, 



#%% SAVE INTEGRATED ACC DATA #################################
#np.savetxt(foldername_acc + '/' + 'a_data_int_acc.txt', a_data_int_acc)
#%%
a_data_norm = array_normalize(a_data, 0)
a_data_norm = a_data_norm[1:]

a_data_int_acc_norm = array_normalize(a_data_int_acc, 0)
a_data_int_acc_norm = a_data_int_acc_norm[1:]

a_data_int_ele_norm = array_normalize(a_data_int_ele, 0)
a_data_int_ele_norm = a_data_int_ele_norm[1:]

a_data_int_acc_shifted_norm = array_normalize(a_data_int_acc_shifted, 0)
a_data_int_acc_shifted_norm = a_data_int_acc_shifted_norm[1:]

data_array_list = [a_data_norm, a_data_int_acc_norm,
                   a_data_int_ele_norm, a_data_int_acc_shifted_norm]
data_spec_array = [[0.75, 'y', r'$a_{\text{LST}}$', 0.1],
                   [0.75, 'r', r'$a_{\text{ACC}}$', 1],
                   [0.75, 'b', r'$a_{\text{ELE}}$', 1],
                   [0.75, 'g', r'$a_{\text{ACC, shifted}}$', 1]]
y_label = a_symb + " " + a_unit
ref_array_list, ref_spec_array, ref_y_spec_list = [], [], []
tit = r'semimajor axis from LST-data and numerical integration'
xlimits = [0, 0]
mean = np.mean(a_data_int_acc_norm[:, 1])
std = np.std(a_data_int_acc_norm[:, 1])
ylimits = [mean - 1.5 * std, mean + 1.5 * std]
vline_specs = ["gold", 0, 1]
z_order = 1
nongra_fac = 1
grid = 1
save_specs = [0]
vline_list2 = []
vline_list_specs2 = ['gold', 0.75, 1, "26.08"]
plot_func_6(data_array_list, data_spec_array, y_label,
            ref_array_list, ref_spec_array, ref_y_spec_list,
            tit, MJD_0, MJD_end, xlimits, ylimits,
            vline_specs, z_order, nongra_fac, grid,
            vline_list1, vline_list_specs1,
            vline_list2, vline_list_specs2,
            save_specs)
#%%
t_list = a_data_int_acc_norm[:, 0]

Δ_acc_ele = a_data_int_acc_norm[:, 1] - a_data_int_ele_norm[:, 1]
Δ_acc_acc_shifted = a_data_int_acc_norm[:, 1] - a_data_int_acc_shifted_norm[:, 1]
Δ_ele_acc_shifted = a_data_int_ele_norm[:, 1] - a_data_int_acc_shifted_norm[:, 1]

Δ_acc_ele = np.vstack((t_list, Δ_acc_ele)).T
Δ_acc_acc_shifted = np.vstack((t_list, Δ_acc_acc_shifted)).T
Δ_ele_acc_shifted = np.vstack((t_list, Δ_ele_acc_shifted)).T


data_array_list = [Δ_acc_ele, Δ_acc_acc_shifted, Δ_ele_acc_shifted]
data_spec_array = [[0.75, 'violet', r'$a_{\text{ACC}} - a_{\text{ELE}}$', 1],
                   [0.75, 'brown', r'$a_{\text{ACC}} - a_{\text{ACC, shifted}}$', 1],
                   [0.75, 'turquoise', r'$a_{\text{ELE}} - a_{\text{ACC, shifted}}$', 1]]
y_label = 'Δa' + " " + a_unit
ref_array_list, ref_spec_array, ref_y_spec_list = [], [], []
tit = r'differences'
xlimits = [0, 0]
ylimits = [0, 0]
vline_specs = ["gold", 0, 1]
z_order = 1
nongra_fac = 1
grid = 1
save_specs = [0]
vline_list2 = []
vline_list_specs2 = ['gold', 0.75, 1, "26.08"]
plot_func_6(data_array_list, data_spec_array, y_label,
            ref_array_list, ref_spec_array, ref_y_spec_list,
            tit, MJD_0, MJD_end, xlimits, ylimits,
            vline_specs, z_order, nongra_fac, grid,
            vline_list1, vline_list_specs1,
            vline_list2, vline_list_specs2,
            save_specs)

ylimits = [-12e-4, 1e-4]
#ylimits = [-1e-4, 1e-4]
#ylimits = [-5e-6, 5e-6]
plot_func_6(data_array_list, data_spec_array, y_label,
            ref_array_list, ref_spec_array, ref_y_spec_list,
            tit, MJD_0, MJD_end, xlimits, ylimits,
            vline_specs, z_order, nongra_fac, grid,
            vline_list1, vline_list_specs1,
            vline_list2, vline_list_specs2,
            save_specs)
#%%
fft_list_acc = fft(a_data_int_acc)
p_o_list_acc = fft_list_acc[0]
a_o_list_acc = fft_list_acc[1]

fft_list_ele = fft(a_data_int_ele)
p_o_list_ele = fft_list_ele[0]
a_o_list_ele = fft_list_ele[1]

fig = plt.figure(figsize = (11, 5), dpi = 300)
plt.title("spectrum of integrated a", fontsize = 17.5)
plt.loglog(p_o_list_acc, a_o_list_acc,
           ls = 'solid', lw = 1, color = 'b',
           alpha = 1, label = r'$a$ (ACC)')
plt.loglog(p_o_list_ele, a_o_list_ele * 0.5,
           ls = 'solid', lw = 1, color = 'g',
           alpha = 1, label = r'$a$ (ELE)')
plt.xlabel("Period [d]", fontsize = 15)
plt.ylabel("Amplitude", fontsize = 15)
plt.legend(fontsize = 12.5, labelspacing = 0.1)
plt.axvline(3.28e-2, color = 'gold', alpha = 0.25,
            ls = (0, (5, 10)), lw = 1.5)
plt.axvline(1.64e-2, color = 'gold', alpha = 0.25,
            ls = (0, (5, 10)), lw = 1.5)
plt.axvline(6.56e-2, color = 'gold', alpha = 0.25,
            ls = (0, (5, 10)), lw = 1.5)
plt.xlim(0.01, 0.1)
plt.ylim(0.002, 0.1)
plt.grid()
plt.show(fig)
plt.close(fig)
# %%
lcm_list_acc = [0, 50, 1, 0.066, 0.95, 10]
lcm_list_ele = [0, 50, 1, 0.066, 0.95, 10]
lcm_list_list = [lcm_list_acc, lcm_list_ele]

interval_list_acc = [[3.27e-2, 3.29e-2]]#[6.55e-2, 6.57e-2], [1.63e-2, 1.65e-2]
interval_list_ele = [[3.27e-2, 3.29e-2]]


acc_spectrum_list = spectrum(a_data_int_acc, 3, lcm_list_acc, interval_list_acc)
acc_p_o, acc_a_o = acc_spectrum_list[0], acc_spectrum_list[1]
acc_per, acc_amp = acc_spectrum_list[2], acc_spectrum_list[3]

ele_spectrum_list = spectrum(a_data_int_ele, 3, lcm_list_ele, interval_list_ele)
ele_p_o, ele_a_o = ele_spectrum_list[0], ele_spectrum_list[1]
ele_per, ele_amp = ele_spectrum_list[2], ele_spectrum_list[3]

p_o_list = [acc_p_o, ele_p_o]
a_o_list = [acc_a_o, ele_a_o]
per_list = [acc_per, ele_per]
amp_list = [acc_amp, ele_amp]

spectrum_p = [acc_p_o, ele_p_o]
spectrum_a = [acc_a_o, ele_a_o]
spectrum_data_specs = [[0.75, 'b', r'$a_{\text{ACC}}$'],
                       [0.75, 'g', r'$a_{\text{ELE}}$']]
spectrum_per = [acc_per, ele_per]
spectrum_amp = [acc_amp, ele_amp]
marker_spec_array = [[(7.5, -7.5), "o", "dodgerblue",
                      "periods used for " + r'$\overline{a}_{\text{ACC}}$', r'$\text{ACC}$'],
                     [(7.5, -7.5), "o", "forestgreen",
                      "periods used for " + r'$\overline{a}_{\text{ELE}}$', r'$\text{ELE}$']]
Δt_list = []
Δt_spec_array = [[(0, (4, 4)), "r", 'a']]
spectrum_tit = "spectrum"
v_line_specs = ['b', 0.5, 1]
xlimits = [0, 0]
#xlimits = [0.01, 0.1]
ylimits = [0, 0]
#ylimits = [0.0001, 1]
#[0.0145, 0.0185], [0.019, 0.026], [0.045, 0.1]
v_line_list = []
log_fac_list = [5]
location = 2
fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                   spectrum_per, spectrum_amp, marker_spec_array,
                   Δt_list, Δt_spec_array, spectrum_tit, 1,
                   log_fac_list, location,
                   v_line_list, v_line_specs,
                   xlimits, ylimits, [0])
# %%

t_fit_a = tt.time()
configuration_list = [[10, 0, 0, 1]]
print("len of data: ", len(a_data_int_acc), len(a_data_int_ele))

#configuration0a = "n_" + str(n_fac) + "/"
#configuration1 = "-n_" + str(n_fac) + "-R"
#configuration3 = "-w_%.0e.png" % ω
el_data_list = [a_data_int_acc, a_data_int_ele]

el_fit_list_list = []
el_para_fit_list = []
el_para_e_fit_list = []
el_m0_list = []
τ_fit_list_list = []
slope_gen_list_list = []
for i in range(0, len(el_data_list)):
    configurations = configuration_list[0]
    n_fac = configurations[0]
    R = configurations[1]
    ω = configurations[2]
    constr = configurations[3]
    
    #print("periods: ", el_per[:R])
    
    n_partition = n_fac * n_days_tot
    
    el_data = el_data_list[i]
    el_per = per_list[i]
    el_amp = amp_list[i]
    
    # fit
    el_fit_list = fitter(el_data, el_per[:R], el_amp[:R], n_partition,
                         5, ω, constr)
    el_fit = el_fit_list[0]
    el_para_fit, el_para_e_fit = el_fit_list[1], el_fit_list[2]
    el_m0, τ_fit_list = el_fit_list[3], el_fit_list[4]
    slope_gen_list = el_fit_list[5]
    
    el_fit_list_list.append(el_fit)
    el_para_fit_list.append(el_para_fit)
    el_para_e_fit_list.append(el_para_e_fit)
    el_m0_list.append(el_m0)
    τ_fit_list_list.append(τ_fit_list)
    slope_gen_list_list.append(slope_gen_list)
t_fit_b = tt.time()
print("Time: ", t_fit_b - t_fit_a)
print("DONE!!!")
# %%
el_fit_p_o_list = []
el_fit_a_o_list = []
el_fit_per_list = []
el_fit_amp_list = []
for i in range(0, len(el_data_list)):#len(data_both)
    el_fit = el_fit_list_list[i]
    # data_spectrum
    el_fit_spectrum_list = spectrum(el_fit, 0, lcm_list_list[i], [[0, 0]])
    el_fit_p_o, el_fit_a_o = el_fit_spectrum_list[0], el_fit_spectrum_list[1]
    el_fit_per, el_fit_amp = el_fit_spectrum_list[2], el_fit_spectrum_list[3]
    
    el_fit_p_o_list.append(el_fit_p_o)
    el_fit_a_o_list.append(el_fit_a_o)
    el_fit_per_list.append(el_fit_per)
    el_fit_amp_list.append(el_fit_amp)
#%%
def dollar_remove(string):
    if (string[0] == '$'):
        return(string[1 : -1])
    else:
        return(string)

def fft_logplot_spec_3(p_o_list, a_o_list, data_spec_array,
                       per_list_list, amp_list_list, marker_spec_array,
                       Δt_list, Δt_spec_array, tit, log,
                       log_fac_list, location,
                       vlines_list, vlines_specs, xlimits, ylimits,
                       save_specs):
    # p_o_list = [p_o_1, p_o_2, ...]
    # a_o_list = [a_o_1, a_o_2, ...]
    # data_spec_array = [alpha_i, col_i, lab_i]; alpha, color, label
    # per_list_list = [per_list_1, per_list_2, ...] for markers
    # amp_list_list = [amp_list_1, amp_list_2, ...] for markers
    # marker_spec_array = [pos_set, symb_i, col_i, leg_i, text_i];
    #                      position, symbol, color, label for legend, text label
    # Δt_list: for vlines
    # Δt_spec_array = [ls_i, col_i, lab_i]; linestyle, color, label
    # tit: title
    # log: if 0 --> semilogx, if 1 --> loglog
    # log_fac_list -> to shift spectrum of p_o_2, p_o_3, etc
    # location: location of legend
    # vlines_list: helpful to determine interval_list in peak_finder
    # vlines_specs = [col, alpha, lw]
    # xlimits: for zooming
    # ylimits: for zooming
    # save_specs = [1, path] # 0,1 for saving
    print("fft_logplot_spec_3 version 4")
    n_data = len(p_o_list)
    n_marker = len(per_list_list)
    n_Δt = len(Δt_list)
    
    new_a_o_list = [a_o_list[0]]
    for i in range(0, n_data - 1):
        new_a_o = a_o_list[i + 1] / log_fac_list[i]
        new_a_o_list.append(new_a_o)
    
    if (len(amp_list_list) != 0):
        new_amp_list_list = [amp_list_list[0]]
        for i in range(0, n_marker - 1):
            new_amp_list = amp_list_list[i + 1] / log_fac_list[i]
            new_amp_list_list.append(new_amp_list)
    else:
        new_amp_list_list = []
    
    fig = plt.figure(figsize = (10, 5), dpi = 300)
    plt.title(tit, fontsize = 17.5)
    
    if (log == 0):
        for i in  range(0, n_data):
            col = data_spec_array[i][1]
            α = data_spec_array[i][0]
            lab = data_spec_array[i][2]
            if (i >= 1):
                if (log_fac_list[i - 1] != 1):
                    lab = lab + ' / ' + str(log_fac_list[i - 1])
            plt.semilogx(p_o_list[i], new_a_o_list[i],
                         color = col, alpha = α,
                         ls = 'solid', lw = 1, label = lab)
    else:
        for i in range(0, n_data):
            col = data_spec_array[i][1]
            α = data_spec_array[i][0]
            lab = data_spec_array[i][2]
            if (i >= 1):
                if (log_fac_list[i - 1] != 1):
                    lab = lab + ' / ' + str(log_fac_list[i - 1])
            plt.loglog(p_o_list[i], new_a_o_list[i],
                       color = col, alpha = α,
                       ls = 'solid', lw = 1, label = lab)
    
    k = 0
    for i in range(0, n_marker):
        plt.scatter(per_list_list[i], new_amp_list_list[i],
                    color = marker_spec_array[i][2], s = 25,
                    marker = marker_spec_array[i][1], fc = 'None', lw = 0.5,
                    alpha = 1, label = marker_spec_array[i][3])
        for j in range(0, len(per_list_list[i])):
            str_lab = dollar_remove(marker_spec_array[i][4])
            string = r'$t_{xyz}^{(0)}$'.replace('0', str(j + 1)).replace('xyz', str_lab)
            plt.annotate(string, (per_list_list[i][j], new_amp_list_list[i][j]),
                         marker_spec_array[i][0], textcoords = 'offset points',
                         fontsize = 10)
            string_add = ' = %.2e d' % per_list_list[i][j]
            plt.figtext(0.92, 0.875 - k, string + string_add,
                        fontsize = 10, ha = 'left')
            k += 0.05
        k += 0.03
    
    for i in range(0, n_Δt):
        lab1 = r'$Δt_{xyz}$ = '.replace('xyz', dollar_remove(Δt_spec_array[i][2]))
        lab2 = str(np.round(Δt_list[i], 3)) + ' d'
        plt.axvline(Δt_list[i], color = Δt_spec_array[i][1], alpha = 0.5,
                    ls = Δt_spec_array[i][0], lw = 5,
                    label = lab1 + lab2)
    
    for i in range(0, len(vlines_list)):
        vline = vlines_list[i]
        plt.axvline(vline, color = vlines_specs[0],
                    alpha = vlines_specs[1], lw = vlines_specs[2],
                    ls = 'solid')
    
    if (xlimits[0] != xlimits[1]):
        plt.xlim(xlimits[0], xlimits[1])
    
    if (ylimits[0] != ylimits[1]):
        plt.ylim(ylimits[0], ylimits[1])
    
    plt.xlabel(r'period [d]', fontsize = 15)
    plt.ylabel(r'amplitude', fontsize = 15)
    plt.legend(fontsize = 12.5, loc = location)
    plt.grid()
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)


#N_n_n = "n_" + str(lcm_list_el[2]) + "_" + str(lcm_list_el_bar[2]) + "_" + str(lcm_list_el_hat[2])
#N_n_title = "N = (" + str(lcm_list_el[2]) + "," + str(lcm_list_el_bar[2]) + "," + str(lcm_list_el_hat[2]) + ")"
def cl_lin(liste, cmap): # lineare Skalierung
    norm_list = []
    bottom = min(liste)
    top = max(liste)
    for i in range(len(liste)):
        norm_list.append((liste[i] - bottom) / (top - bottom))
    colist = []
    for i in range(len(liste)):
        el = norm_list[i]
        tup = cmap(el)
        colist.append(tup)
    return(colist)

def lab_gen(x, configuration):
    #configuration_list = [[n_fac, R, ω, constr]]
    n_fac = configuration[0]
    R = configuration[1]
    ω = configuration[2]
    constr = configuration[3]
    lab = r'$_{n_fac}^{(R)}\tilde{x}_{ω}^{(constr)}$'
    lab = lab.replace("x", x)
    lab = lab.replace("n_fac", str(n_fac))
    lab = lab.replace("R", str(R))
    lab = lab.replace('ω', '%.0e' % ω)
    lab = lab.replace('constr', str(constr))
    return(lab)

path_add = "_" + str(0) + "_" + str(ω)
#if (os.path.exists(path) == False):
#    os.mkdir(os.getcwd() + "/" + file_name[:-1])
xlimit_list = [[0, 0]]
log_fac_list = [10, 2, 20]
# for plots

col_list = cl_lin(np.arange(len(p_o_list) * 2), mat.colormaps['brg'])
# spectrum
spectrum_p = p_o_list + el_fit_p_o_list
spectrum_a = a_o_list + el_fit_a_o_list

spectrum_data_specs = [[0.75, 'b', r'$a_{\text{ACC}}$'],
                       [0.75, 'g', r'$a_{\text{ELE}}$'],
                       [0.75, 'dodgerblue', r'$\overline{a}_{\text{ACC}}$'],
                       [0.75, 'forestgreen', r'$\overline{a}_{\text{ELE}}$']]

spectrum_per = per_list
spectrum_amp = amp_list
marker_spec_array = [[(-5, 7.5), "o", "dodgerblue",
                      r'periods used for $\overline{a}_{\text{ACC}}$', r'$\text{ACC}$'],
                     [(-5, 7.5), "o", "forestgreen",
                      r'periods used for $\overline{a}_{\text{ELE}}$', r'$\text{ELE}$']]
Δt_list = []
Δt_spec_array = []
vlines = []
vline_specs = ['k', 0.5, 0.5]
xlimits = [0, 0]
ylimits = [0, 0]
spectrum_tit_1 = r'spectrum of $a$ and $\overline{a}$'
spectrum_tit_2 = r", $m_0 = $ [" + ", ".join("%.1e" % m0 for m0 in el_m0_list) + "] m"
spectrum_tit = spectrum_tit_1 + spectrum_tit_2
file_name_fft = "spec/spec_" + "_" + 'a' + ".png"
file_name_fft = file_name + file_name_fft
fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                   spectrum_per, spectrum_amp, marker_spec_array,
                   Δt_list, Δt_spec_array, spectrum_tit, 1,
                   log_fac_list, 4,
                   vlines, vline_specs, xlimits, ylimits,
                   [save_on, path + "spec" + path_add])
print("yes")
print("DONE!!!")
# %%
def decrease_unit_day(str_unit):
    # str_unit: for example r'[$kg m^{-3}$]'
    inner_string = str_unit[2:-2]
    inner_string = r'\frac{' + inner_string + r'}{d}'
    new_string = str_unit[:2] + inner_string + str_unit[-2:]
    return(new_string)

def decrease_plot_hyperadv(slope_data_list, slope_specs_list,
                           add_data_list, add_data_specs_list,
                           el_label, el_unit,
                           ref_data_list, ref_specs_list, ref_lab_list,
                           MJD_0, MJD_end, xlimits, ylimits, title, grid,
                           fac, location, anchor,
                           vline_list1, vline_list_specs1,
                           vline_list2, vline_list_specs2,
                           save_specs):
    # slope_data_list = [[τ_i, s_i, e_s_i]]
    # slope_specs_list = [[e_fac, ms, mew, capsize, mfc, mec, ecolor]]
    # add_data_list = [[t_k_i, z_k_i], (k+1), ...]
    # add_data_specs_list = [[α_k, col_k, lab_k, width_k, ms_k], (k+1), ...]
    # el_label = stringlike
    # el_unit = el_unit
    # ref_data_list = reference
    # ref_specs_list = [α, col, lab, width]
    # ref_lab_list = [ref_lab, ref_y_axis_lab]
    # n_partition: how many fits per day
    # el_Δt_n: smoothing period
    # MJD_0: start of data
    # MJD_end: end of data
    # xlimits = [xstart, xend]
    # ylimits = [ylow, yupp]
    # title: title of plot
    # grid: if 1 --> grid for data
    
    xstart = xlimits[0] + MJD_0
    xend = MJD_end - xlimits[1]
    n_days = xend - xstart
    new_slope_data_list = 0
    new_ref_data_list = 0
    new_add_data_list = 0
    if (xlimits[0] != xlimits[1]):
        MJD_start = xstart - MJD_0
        n_days = xend - xstart
        new_slope_data_list = []
        new_add_data_list = []
        new_ref_data_list = []
        for i in range(0, len(slope_data_list)):
            data_i = slope_data_list[i]
            data_i = array_modifier(data_i, xstart - MJD_0, n_days)
            new_slope_data_list.append(data_i)
        for i in range(0, len(add_data_list)):
            data_i = add_data_list[i]
            data_i = array_modifier(data_i, xstart - MJD_0, n_days)
            new_add_data_list.append(data_i)
        for i in range(0, len(ref_data_list)):
            ref_i = ref_data_list[i]
            ref_i = array_modifier(ref_i, xstart - MJD_0, n_days)
            new_ref_data_list.append(ref_i)
    else:
        new_slope_data_list = slope_data_list
        new_add_data_list = add_data_list
        new_ref_data_list = ref_data_list
    
    y_label = 'slope of ' + el_label
    unit_day = decrease_unit_day(el_unit)
    
    fig, ax1 = plt.subplots(figsize = (10, 5), dpi = 500)
    fig.suptitle(title, fontsize = 17.5)
    
    for i in range(0, len(new_slope_data_list)):
        slope_data = new_slope_data_list[i]
        slope_specs = slope_specs_list[i]
        e_fac = slope_specs[0]
        ax1.errorbar(slope_data[:, 0], slope_data[:, 1],
                     yerr = e_fac * slope_data[:, 2],
                     ls = 'none', marker = 'o',
                     ms = slope_specs[1], mew = slope_specs[2],
                     capsize = slope_specs[3], mfc = slope_specs[4],
                     mec = slope_specs[5], ecolor = slope_specs[6],
                     label = 'slope (' + str(e_fac) + r' $\cdot$ error)',
                     zorder = 5)
        ax1.plot(slope_data[:, 0], slope_data[:, 1],
                 'b-', lw = 0.5, alpha = 1, zorder = 4)
    
    for i in range(0, len(new_add_data_list)):
        add_data = new_add_data_list[i]
        add_data_specs = add_data_specs_list[i]
        α = add_data_specs[0]
        col = add_data_specs[1]
        lab = add_data_specs[2]
        width = add_data_specs[3]
        size = add_data_specs[4]
        ax1.plot(add_data[:, 0], fac * add_data[:, 1],
                 color = col, alpha = α, ls = 'solid', lw = width,
                 marker = 'o', ms = size,
                 label = lab, zorder = 6)
    
    ax1.set_xlabel("MJD " + str(MJD_0) + " + t [d]", fontsize = 15)
    ax1.set_ylabel(y_label + ' ' + unit_day, fontsize = 15)
    if (grid == 1):
        ax1.grid()
    if (xstart != xend):
        ax1.set_xlim(xstart - MJD_0, xend - MJD_0)
    y_low, y_upp = ylimits[0], ylimits[1]
    if (y_low != y_upp):
        ax1.set_ylim(y_low, y_upp)
    
    ax2 = ax1.twinx()
    for i in range(0, len(new_ref_data_list)):
        ref_data = new_ref_data_list[i]
        ref_specs = ref_specs_list[i]
        α = ref_specs[0]
        col = ref_specs[1]
        lab = ref_specs[2]
        width = ref_specs[3]
        ax2.plot(ref_data[:, 0], ref_data[:, 1],
                 ls = '-', lw = width, color = col, alpha = α,
                 label = lab)
    
    ax2.set_ylabel(ref_lab_list[0] + " " + ref_lab_list[1],
                   color = col, fontsize = 15)
    ax2.tick_params(axis = 'y', labelcolor = col)

    #if (xstart <= 60002 and xend >= 60002):
    #    plt.axvline(60002 - MJD_0, color = 'k',
    #                ls = (0, (4, 4)), lw = 1.5, zorder = 3)
    #if (xstart <= 60027 and xend >= 60027):
    #    plt.axvline(60027 - MJD_0, color = 'k',
    #                ls = (0, (4, 4)), lw = 1.5, zorder = 3)
    #if (xstart <= 60058 and xend >= 60058):
    #    plt.axvline(60058 - MJD_0, color = 'k',
    #                ls = (0, (4, 4)), lw = 1.5, zorder = 3)
    #if (xstart <= 58356 and xend >= 58356):
    #    plt.axvline(58356 - MJD_0, color = 'k',
    #                ls = (0, (4, 4)), lw = 1.5, zorder = 3)
    c = 0
    for i in range(0, len(vline_list1)):
        vline = vline_list1[i]
        col = vline_list_specs1[0]
        α = vline_list_specs1[1]
        width = vline_list_specs1[2]
        lab = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                lab = vline_list_specs1[3]
                c += 1
            plt.axvline(vline - MJD_0, color = col, alpha = α,
                        ls = (0, (4, 4)), lw = width,
                        label = lab, zorder = 3)
    c = 0
    for i in range(0, len(vline_list2)):
        vline = vline_list2[i]
        col = vline_list_specs2[0]
        α = vline_list_specs2[1]
        width = vline_list_specs2[2]
        lab = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                lab = vline_list_specs2[3]
                c += 1
            plt.axvline(vline - MJD_0, color = col, alpha = α,
                        ls = (0, (4, 4)), lw = width,
                        label = lab, zorder = 3)
    
    plt.figlegend(fontsize = 12.5, markerscale = 2.5, loc = location,
                  bbox_to_anchor = anchor, bbox_transform = ax1.transAxes)
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.set_frame_on(False)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    
    plt.show(fig)
    plt.close(fig)


def decr_bar_new(slope_data_list, slope_specs, efac,
                 add_data_list, add_data_specs_list,
                 el_label, el_unit,
                 flx_data, flx_spec_list,
                 MJD_0, MJD_end, xlimits, ylimits, title, grid,
                 fac, location, anchor,
                 vline_list1, vline_list_specs1,
                 vline_list2, vline_list_specs2,
                 save_specs):
    # slope_data_list = [[τ_i, s_i, e_s_i]]
    # slope_specs_list = [[ms, mew, mfc, lw, α]]
    # add_data_list = [[t_k_i, z_k_i], (k+1), ...]
    # add_data_specs_list = [[α_k, col_k, lab_k, width_k, ms_k], (k+1), ...]
    # el_label = stringlike
    # el_unit = el_unit
    # ref_data_list = reference
    # ref_specs_list = [α, col, lab, width]
    # ref_lab_list = [ref_lab, ref_y_axis_lab]
    # n_partition: how many fits per day
    # el_Δt_n: smoothing period
    # MJD_0: start of data
    # MJD_end: end of data
    # xlimits = [xstart, xend]
    # ylimits = [ylow, yupp]
    # title: title of plot
    # grid: if 1 --> grid for data
    
    xstart = xlimits[0] + MJD_0
    xend = MJD_end - xlimits[1]
    n_days = xend - xstart
    new_slope_data_list = 0
    new_add_data_list = 0
    if (xlimits[0] != xlimits[1]):
        MJD_start = xstart - MJD_0
        n_days = xend - xstart
        new_slope_data_list = []
        new_add_data_list = []
        for i in range(0, len(slope_data_list)):
            data_i = slope_data_list[i]
            data_i = array_modifier(data_i, MJD_start, n_days)
            new_slope_data_list.append(data_i)
        for i in range(0, len(add_data_list)):
            data_i = add_data_list[i]
            data_i = array_modifier(data_i, MJD_start, n_days)
            new_add_data_list.append(data_i)
        if (len(flx_data) != 0):
            flx_data = array_modifier(flx_data, MJD_start, n_days)
    else:
        new_slope_data_list = slope_data_list
        new_add_data_list = add_data_list
    
    y_label = el_label
    unit_day = decrease_unit_day(el_unit)
    
    fig, ax1 = plt.subplots(figsize = (10, 5), dpi = 500)
    fig.suptitle(title, fontsize = 17.5)
    
    for i in range(0, len(new_slope_data_list)):
        slope_data = new_slope_data_list[i]
        slope_specs = slope_specs_list[i]
        x_slope = slope_data[:, 0]
        y_slope = slope_data[:, 1]
        y_e_slope = slope_data[:, 2]
        
        l_w = slope_specs[0]
        lcol = slope_specs[1]
        m_s = slope_specs[2]
        mcol = slope_specs[3]
        ecol = slope_specs[4]
        ew = slope_specs[5]
        α = slope_specs[6]
        lab = slope_specs[7]
        ax1.errorbar(x_slope + MJD_0, y_slope, yerr = efac * y_e_slope,
                     ls = '-', marker = '.',
                     lw = l_w, color = lcol,
                     ms = m_s, mfc = mcol, mew = 0,
                     ecolor = ecol, elinewidth = ew,
                     alpha = α, label = lab)
    
    for i in range(0, len(new_add_data_list)):
        add_data = new_add_data_list[i]
        add_data_specs = add_data_specs_list[i]
        α = add_data_specs[0]
        col = add_data_specs[1]
        lab = add_data_specs[2]
        width = add_data_specs[3]
        size = add_data_specs[4]
        ax1.plot(add_data[:, 0] + MJD_0, fac * add_data[:, 1],
                 color = col, alpha = α, ls = 'solid', lw = width,
                 marker = 'o', ms = size,
                 label = lab, zorder = 6)
    
    ax1.xaxis.set_major_formatter(mjd_to_mmdd)
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    ax1.set_ylabel(y_label + ' ' + unit_day, fontsize = 15)
    if (grid == 1):
        ax1.grid()
    if (xstart != xend):
        ax1.set_xlim(xstart, xend)
    y_low, y_upp = ylimits[0], ylimits[1]
    if (y_low != y_upp):
        ax1.set_ylim(y_low, y_upp)
    
    if (len(flx_data) != 0):
        len_flx_data = len(flx_data.T) - 1
        col_map = flx_spec_list[2]
        α = flx_spec_list[0]
        lab_stat = flx_spec_list[1]
        ax2 = ax1.twinx()
        bar_width = 1 / (len_flx_data)
        
        colmap = mat.colormaps[col_map]
        col_list = cl_lin(np.linspace(0, 1, len(flx_data.T) - 1), colmap)
        norming = colors.BoundaryNorm(np.arange(1 / 2, len(flx_data.T) - 1, 1), colmap.N) # discrete bins
        
        α = flx_spec_list[0]
        lab_stat = flx_spec_list[1]
        if (lab_stat == 1):
            lab = r'$\text{ap}_i$'.replace('i', str(i))
        else:
            lab = None
        for i in range(1, len_flx_data):
            t_ap = flx_data[:, 0]
            ap = flx_data[:, i]
            ax2.bar(t_ap + (i + 1 / 2) * bar_width + MJD_0, ap,
                    width = bar_width, alpha = α, color = col_list[i - 1],
                    label = lab)
        
        sm = ScalarMappable(cmap = colmap, norm = norming)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, cax = ax2.inset_axes([1.1, 0, 0.02, 1]),
                            ticks = np.arange(1, len(flx_data.T) - 1, 1),
                            alpha = α)
        cbar.set_label(r'$i$', rotation = 0, labelpad = 10)
        
        ax2.set_ylabel(r'$\text{ap}_i$', color = 'k', fontsize = 15, rotation = 0)
        ax2.tick_params(axis = 'y', labelcolor = 'k')
    
    c = 0
    for i in range(0, len(vline_list1)):
        vline = vline_list1[i]
        col = vline_list_specs1[0]
        α = vline_list_specs1[1]
        width = vline_list_specs1[2]
        lab = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                lab = vline_list_specs1[3]
                c += 1
            plt.axvline(vline, color = col, alpha = α,
                        ls = (0, (4, 4)), lw = width,
                        label = lab, zorder = 3)
    c = 0
    for i in range(0, len(vline_list2)):
        vline = vline_list2[i]
        col = vline_list_specs2[0]
        α = vline_list_specs2[1]
        width = vline_list_specs2[2]
        lab = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                lab = vline_list_specs2[3]
                c += 1
            plt.axvline(vline, color = col, alpha = α,
                        ls = (0, (4, 4)), lw = width,
                        label = lab, zorder = 3)
    
    plt.figlegend(fontsize = 10, markerscale = 1, loc = location,
                  bbox_to_anchor = anchor, bbox_transform = ax1.transAxes,
                  labelspacing = 0.1)
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.set_frame_on(False)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)


def decr_bar_adv(slope_data_list, slope_specs,
                 e_fac, ave, el_label, el_unit,
                 flx_data, flx_spec_list,
                 MJD_0, MJD_end, xlimits, ylimits, title, grid,
                 fac, location, anchor, n_cols,
                 vline_list1, vline_list_specs1,
                 vline_list2, vline_list_specs2,
                 save_specs):
    # slope_data_list = [[τ_i, s_i, e_s_i]]
    # slope_specs_list = [[ms, mew, mfc, lw, α]]
    # add_data_list = [[t_k_i, z_k_i], (k+1), ...]
    # add_data_specs_list = [[α_k, col_k, lab_k, width_k, ms_k], (k+1), ...]
    # el_label = stringlike
    # el_unit = el_unit
    # ref_data_list = reference
    # ref_specs_list = [α, col, lab, width]
    # ref_lab_list = [ref_lab, ref_y_axis_lab]
    # n_partition: how many fits per day
    # el_Δt_n: smoothing period
    # MJD_0: start of data
    # MJD_end: end of data
    # xlimits = [xstart, xend]
    # ylimits = [ylow, yupp]
    # title: title of plot
    # grid: if 1 --> grid for data
    
    xstart = xlimits[0] + MJD_0
    xend = MJD_end - xlimits[1]
    n_days = xend - xstart
    new_slope_data_list = 0
    if (xlimits[0] != xlimits[1]):
        MJD_start = xstart - MJD_0
        n_days = xend - xstart
        new_slope_data_list = []
        new_add_data_list = []
        for i in range(0, len(slope_data_list)):
            data_i = slope_data_list[i]
            data_i = array_modifier(data_i, MJD_start, n_days)
            new_slope_data_list.append(data_i)
        if (len(flx_data) != 0):
            flx_data = array_modifier(flx_data, MJD_start, n_days)
    else:
        new_slope_data_list = slope_data_list
    
    fig, ax1 = plt.subplots(figsize = (11, 5), dpi = 500)
    mean_list = []
    for i in range(0, len(new_slope_data_list)):
        slope_data = new_slope_data_list[i]
        slope_specs = slope_specs_list[i]
        x_slope = slope_data[:, 0]
        y_slope = slope_data[:, 1]
        y_e_slope = slope_data[:, 2]
        y_low_slope = y_slope - y_e_slope * e_fac
        y_upp_slope = y_slope + y_e_slope * e_fac
        
        l_w = slope_specs[0]
        lcol = slope_specs[1]
        ecol = slope_specs[2]
        α = slope_specs[3]
        e_α = slope_specs[4]
        lab = slope_specs[5]
        e_lab = slope_specs[6]
        if (e_fac != 1):
            e_lab = e_lab + r' $\cdot$ ' + str(e_fac) 
        
        if (ave == 1):
            mean = np.mean(y_slope)
            y_slope = y_slope / mean
            y_low_slope = y_low_slope / mean
            y_upp_slope = y_upp_slope / mean
            mean_list.append(mean)
        
        ax1.plot(x_slope + MJD_0, y_slope,
                 ls = '-', lw = l_w, color = lcol,
                 alpha =  α, label = lab, zorder = 10)
        ax1.fill_between(x_slope + MJD_0, y_low_slope, y_upp_slope,
                         color = ecol, alpha = e_α, zorder = 9,
                         label = e_lab)
    
    ax1.xaxis.set_major_formatter(mjd_to_mmdd)
    ax1.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
    ax1.tick_params(axis = 'x', which = 'major',
                    width = 2, length = 8,
                    direction = 'out', top = False)
    ax1.tick_params(axis = 'x', which = 'minor',
                    width = 1, length = 4,
                    direction = 'out', top = False)
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    if (ave == 0):
        y_lab = r'$\dot{\overline{el}}$'
        y_lab = y_lab.replace('el', el_label)
        y_lab = y_lab + ' [' + el_unit + ']'
        ax1.set_ylabel(y_lab, fontsize = 15)
    else:
        y_lab = r'$\frac{\dot{\overline{el}}}{\langle\dot{\overline{el}}\rangle}$'
        y_lab = y_lab.replace('el', el_label)
        
        if (len(new_slope_data_list) > 1):
            title = title + r', $\langle\dot{\overline{el}}\rangle$ = $ ['
            title = title.replace('el', el_label)
            title = title + ', '.join('%.1e' % mean for mean in mean_list) + '] ' + el_unit
        else:
            title = title + r', $\langle\dot{\overline{el}}\rangle$ = %.1e ' % mean_list[0] + el_unit
            title = title.replace('el', el_label)
        ax1.set_ylabel(y_lab, rotation = 0, labelpad = 10, fontsize = 15)
    fig.suptitle(title, fontsize = 17.5)
    if (grid == 1):
        ax1.grid()
    if (xstart != xend):
        ax1.set_xlim(xstart, xend)
    y_low, y_upp = ylimits[0], ylimits[1]
    if (y_low != y_upp):
        ax1.set_ylim(y_low, y_upp)
    
    if (len(flx_data) != 0):
        len_flx_data = len(flx_data.T) - 1
        α = flx_spec_list[0]
        col_map_stat = flx_spec_list[1]
        
        ax2 = ax1.twinx()
        bar_width = 1 / (len_flx_data)
        
        if (col_map_stat == 1):
            col_map = flx_spec_list[2]
            colmap = mat.colormaps[col_map]
            col_list = cl_lin(np.linspace(0, 1, len(flx_data.T) - 1), colmap)
            norming = colors.BoundaryNorm(np.arange(1 / 2, len(flx_data.T) - 1, 1), colmap.N) # discrete bins
            
            for i in range(1, len_flx_data + 1):
                t_ap = flx_data[:, 0]
                ap = flx_data[:, i]
                ax2.bar(t_ap + (i - 1 / 2) * bar_width + MJD_0, ap,
                        width = bar_width, alpha = α,
                        color = col_list[i - 1], lw = 0)
            
            sm = ScalarMappable(cmap = colmap, norm = norming)
            sm.set_array([])
            
            cbar = plt.colorbar(sm, cax = ax2.inset_axes([1.1, 0, 0.02, 1]),
                                ticks = np.arange(1, len(flx_data.T) - 1, 1),
                                alpha = α)
            cbar.set_label(r'$i$', rotation = 0, labelpad = 10)
            flx_ylabel = r'$\text{ap}_i$'
            ax2.set_ylabel(flx_ylabel, color = 'k', fontsize = 15, rotation = 0)
            ax2.tick_params(axis = 'y', labelcolor = 'k')
        
        else:
            col = flx_spec_list[2]
            for i in range(1, len_flx_data + 1):
                t_ap = flx_data[:, 0]
                ap = flx_data[:, i]
                
                ax2.bar(t_ap + (i - 1 / 2) * bar_width + MJD_0, ap,
                        width = bar_width, alpha = α,
                        color = col, lw = 0)
            flx_ylabel =  r'$\text{ap}$'
            ax2.set_ylabel(flx_ylabel, color = col, fontsize = 15, rotation = 0)
            ax2.tick_params(axis = 'y', labelcolor = col, color = col)
    
    c = 0
    for i in range(0, len(vline_list1)):
        vline = vline_list1[i]
        col = vline_list_specs1[0]
        α = vline_list_specs1[1]
        width = vline_list_specs1[2]
        lab = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                lab = vline_list_specs1[3]
                c += 1
            plt.axvline(vline, color = col, alpha = α,
                        ls = (0, (4, 4)), lw = width,
                        label = lab, zorder = 3)
    c = 0
    for i in range(0, len(vline_list2)):
        vline = vline_list2[i]
        col = vline_list_specs2[0]
        α = vline_list_specs2[1]
        width = vline_list_specs2[2]
        lab = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                lab = vline_list_specs2[3]
                c += 1
            plt.axvline(vline, color = col, alpha = α,
                        ls = (0, (4, 4)), lw = width,
                        label = lab, zorder = 3)
    
    if (len(flx_data) != 0):
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.set_frame_on(False)
    
    plt.figlegend(fontsize = 12.5, markerscale = 1, loc = location,
                  bbox_to_anchor = anchor, bbox_transform = ax1.transAxes,
                  labelspacing = 0.1, ncols = n_cols, columnspacing = 1)
    
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)

def gen_a_dot_array(n_t, slope_gen_list):
    # n_t: number of t points
    τ_list = slope_gen_list[0]
    Δτ = slope_gen_list[1]
    a_bar_list = slope_gen_list[2]
    σ2_a_bar_list = slope_gen_list[3]
    # t_list should be array
    # warning min(t_list) >= min(τ_list)
    #     and max(t_list) < max(τ_list)
    # τ_lis, a_bar_list and σ2_a_bar_list should all
    # be sorted from lowest to highest chronologically
    τ_0 = τ_list[0]
    τ_n_s = τ_list[-1]
    t_list = np.linspace(τ_0, τ_n_s, n_t + 1)[: -1]
    
    a_dot_list = np.array([])
    s_a_dot_list = np.array([])
    
    for j in range(1, len(τ_list)):
        a_bar_n_1 = a_bar_list[j - 1]
        a_bar_n = a_bar_list[j]
        σ2_a_bar_n_1 = σ2_a_bar_list[j - 1]
        σ2_a_bar_n = σ2_a_bar_list[j]
        
        a_dot_j = (a_bar_n - a_bar_n_1) / Δτ
        s_a_dot_j = np.sqrt((σ2_a_bar_n + σ2_a_bar_n_1) / (Δτ**2))
        
        a_dot_list = np.append(a_dot_list, a_dot_j)
        s_a_dot_list = np.append(s_a_dot_list, s_a_dot_j)
    
    array = np.zeros(3)
    for i in range(0, len(t_list)):
        t_i = t_list[i]
        for j in range(1, len(τ_list)):
            if (j * Δτ + τ_0 > t_i):
                a_dot_i = a_dot_list[j - 1]
                s_a_dot_i = s_a_dot_list[j - 1]
                
                row_i = np.array([t_i, a_dot_i, s_a_dot_i])
                array = np.vstack((array, row_i))
                break
    array = array[1:]
    return(array)

def lab_gen_add(x, configuration, add):
    #configuration_list = [[n_fac, R, ω, constr]]
    n_fac = configuration[0]
    R = configuration[1]
    ω = configuration[2]
    constr = configuration[3]
    lab = r'$_{n_fac}^{(R)}\dot{\bar{x}}_{\text{add}}^{(ω)}$'
    lab = lab.replace("x", x, 1)
    lab = lab.replace("n_fac", str(n_fac))
    lab = lab.replace("R", str(R))
    lab = lab.replace('ω', '%.0e' % ω)
    lab = lab.replace('add', add)
    return(lab)

vline_list1 = [58354.25, 58355.25, 59958.915, 59961.875, 59967.125,
               59988.71, 59995.416, 59999.5, 60001.78125, 60002.427,
               60017.17, 60018.17, 60023.9375, 60026.375, 60053.389]

xlimit_list = [[-0.1, -0.1]]
vline_specs = ["gold", 0.75, 1]
e_fac = 100
fac = 1

str_0 = ""
str_1 = r'$\bar{a}$'
if (len(el_m0_list) > 1):
    str_2 = r", $m_0 = $ [" + ", ".join("%.1e" % m0 for m0 in el_m0_list) + "] m"
else:
    str_2 = r", $m_0 = $ %.1e m" % el_m0_list[0]

n_slope = n_days_tot * 1
slope_data_list = []
slope_specs_list = []
#color_group_list = [['magenta', 'deeppink', 'mediumvioletred'],
#                    ['darkturquoise', 'royalblue', 'darkblue']]
#color_group_list = [['r', 'r'],
#                    ['b', 'b']]
#color_group_list = [['g', 'g'],
#                    ['b', 'b']]
#color_group_list = [['yellow', 'yellow'],
#                    ['red', 'red']]
color_group_list = [['forestgreen', 'forestgreen'],
                    ['royalblue', 'royalblue']]
add_list = [r'ACC', r'ELE']
for i in range(0, len(el_fit_list_list)):
    configuration = configuration_list[0]
    n_fac = configuration[0]
    R = configurations[1]
    ω = configuration[2]
    constr = configuration[3]
    
    slope_gen_list = slope_gen_list_list[i]
    
    m0 = el_m0_list[i]
    
    n_slope = 10 * n_days_tot * n_fac
    slope_data = gen_a_dot_array(n_slope, slope_gen_list)
    #slope_specs = [l_w, lcol, m_s, mcol, ecol, ew, α, lab]
    # [l_w, lcol, ecol, α, e_α, lab, e_lab]
    add = add_list[i]
    lab = lab_gen_add('a', configuration, add)
    e_lab = lab_gen_add('σ', configuration, add)
    color_index = i
    lcol = color_group_list[color_index][0]
    #mcol = color_group_list[color_index][1]
    #ecol = color_group_list[color_index][2]
    ecol = color_group_list[color_index][1]
    #slope_specs = [2, lcol, ecol, 1 - i / 2.5, 0.25, lab, e_lab]
    slope_specs = [2, lcol, ecol, 1, 0.5 - i * 0.15, lab, e_lab]
    slope_data_list.append(slope_data)
    slope_specs_list.append(slope_specs)
    
    #name = '%d %d %f %d' % (n_fac, R, ω, constr)
    #name = name.replace(' ', '_') + '.txt'
    #head = str(m0)
    
    #np.savetxt(foldername_acc + '/' + name, slope_data,
    #           header = head, comments = '')

from functions import flx_get_data
path_flx = 'All Data new/FLXAP_P_MJD.FLX'
data_flx_apday, data_ap = flx_get_data(path_flx, MJD_0, n_days_tot)

#slope_specs = [5.,  0, "deeppink", 1.5, 1]
add_data_list = []
add_data_specs_list = []
el_label = 'a'
el_unit = r'$\frac{m}{d}$'
flx_data = data_ap
#flx_spec_list = [0.75, 1, 'plasma']
flx_spec_list = [1, 0, 'crimson']
title = str_0 + "decrease of " + str_1 + str_2

ylimits = [-8, 0] #[-110, -40]
z_order = 1
nongra_fac = 1
configuration0b = "decr"
path_add = "0"
ave = 0
n_cols = 1
location = 4
position = (1, 0.125)
decr_bar_adv(slope_data_list, slope_specs_list,
             e_fac, ave, el_label, el_unit,
             flx_data, flx_spec_list,
             MJD_0, MJD_end, xlimit_list[0], ylimits,
             title, 1, fac, location, position, n_cols,
             vline_list1, vline_list_specs1,
             vline_list2, vline_list_specs2,
             [save_on, path + path_add])
# %%
def decr_bar_adv(slope_data_list, slope_specs,
                 e_fac_list, ave, el_label, el_unit,
                 flx_data, flx_spec_list,
                 MJD_0, MJD_end, xlimits, ylimits, title, grid,
                 fac, location, anchor, n_cols,
                 vline_list1, vline_list_specs1,
                 vline_list2, vline_list_specs2,
                 save_specs):
    # slope_data_list = [[τ_i, s_i, e_s_i]]
    # slope_specs_list = [[ms, mew, mfc, lw, α]]
    # add_data_list = [[t_k_i, z_k_i], (k+1), ...]
    # add_data_specs_list = [[α_k, col_k, lab_k, width_k, ms_k], (k+1), ...]
    # el_label = stringlike
    # el_unit = el_unit
    # ref_data_list = reference
    # ref_specs_list = [α, col, lab, width]
    # ref_lab_list = [ref_lab, ref_y_axis_lab]
    # n_partition: how many fits per day
    # el_Δt_n: smoothing period
    # MJD_0: start of data
    # MJD_end: end of data
    # xlimits = [xstart, xend]
    # ylimits = [ylow, yupp]
    # title: title of plot
    # grid: if 1 --> grid for data
    print(len(slope_data_list))
    xstart = xlimits[0] + MJD_0
    xend = MJD_end - xlimits[1]
    n_days = xend - xstart
    new_slope_data_list = 0
    if (xlimits[0] != xlimits[1]):
        MJD_start = xstart - MJD_0
        n_days = xend - xstart
        new_slope_data_list = []
        new_add_data_list = []
        for i in range(0, len(slope_data_list)):
            data_i = slope_data_list[i]
            data_i = array_modifier(data_i, MJD_start, n_days)
            new_slope_data_list.append(data_i)
        if (len(flx_data) != 0):
            flx_data = array_modifier(flx_data, MJD_start, n_days)
    else:
        new_slope_data_list = slope_data_list
    
    fig, ax1 = plt.subplots(figsize = (11, 5), dpi = 500)
    mean_list = []
    for i in range(0, len(new_slope_data_list)):
        print("e_fac_list", e_fac_list)
        print(e_fac_list[i])
        e_fac = e_fac_list[i]
        slope_data = new_slope_data_list[i]
        slope_specs = slope_specs_list[i]
        x_slope = slope_data[:, 0]
        y_slope = slope_data[:, 1]
        y_e_slope = slope_data[:, 2]
        y_low_slope = y_slope - y_e_slope * e_fac
        y_upp_slope = y_slope + y_e_slope * e_fac
        
        l_w = slope_specs[0]
        lcol = slope_specs[1]
        ecol = slope_specs[2]
        α = slope_specs[3]
        e_α = slope_specs[4]
        lab = slope_specs[5]
        e_lab = slope_specs[6]
        if (e_fac != 1):
            e_lab = str(e_fac) + r' $\cdot$ ' + e_lab
        
        if (ave == 1):
            mean = np.mean(y_slope)
            y_slope = y_slope / mean
            y_low_slope = y_low_slope / mean
            y_upp_slope = y_upp_slope / mean
            mean_list.append(mean)
        
        ax1.plot(x_slope + MJD_0, y_slope,
                 ls = '-', lw = l_w, color = lcol,
                 alpha =  α, label = lab, zorder = 10)
        ax1.fill_between(x_slope + MJD_0, y_low_slope, y_upp_slope,
                         color = ecol, alpha = e_α, zorder = 9,
                         label = e_lab)
    
    ax1.xaxis.set_major_formatter(mjd_to_mmdd)
    ax1.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
    ax1.tick_params(axis = 'x', which = 'major',
                    width = 2, length = 8,
                    direction = 'out', top = False)
    ax1.tick_params(axis = 'x', which = 'minor',
                    width = 1, length = 4,
                    direction = 'out', top = False)
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    if (ave == 0):
        y_lab = r'$\dot{\overline{el}}$'
        y_lab = y_lab.replace('el', el_label)
        y_lab = y_lab + ' [' + el_unit + ']'
        ax1.set_ylabel(y_lab, fontsize = 15)
    else:
        y_lab = r'$\frac{\dot{\overline{el}}}{\langle\dot{\overline{el}}\rangle}$'
        y_lab = y_lab.replace('el', el_label)
        
        if (len(new_slope_data_list) > 1):
            title = title + r', $\langle\dot{\overline{el}}\rangle$ = $ ['
            title = title.replace('el', el_label)
            title = title + ', '.join('%.1e' % mean for mean in mean_list) + '] ' + el_unit
        else:
            title = title + r', $\langle\dot{\overline{el}}\rangle$ = %.1e ' % mean_list[0] + el_unit
            title = title.replace('el', el_label)
        ax1.set_ylabel(y_lab, rotation = 0, labelpad = 10, fontsize = 15)
    fig.suptitle(title, fontsize = 17.5)
    if (grid == 1):
        ax1.grid()
    if (xstart != xend):
        ax1.set_xlim(xstart, xend)
    y_low, y_upp = ylimits[0], ylimits[1]
    if (y_low != y_upp):
        ax1.set_ylim(y_low, y_upp)
    
    if (len(flx_data) != 0):
        len_flx_data = len(flx_data.T) - 1
        α = flx_spec_list[0]
        col_map_stat = flx_spec_list[1]
        
        ax2 = ax1.twinx()
        bar_width = 1 / (len_flx_data)
        
        if (col_map_stat == 1):
            col_map = flx_spec_list[2]
            colmap = mat.colormaps[col_map]
            col_list = cl_lin(np.linspace(0, 1, len(flx_data.T) - 1), colmap)
            norming = colors.BoundaryNorm(np.arange(1 / 2, len(flx_data.T) - 1, 1), colmap.N) # discrete bins
            
            for i in range(1, len_flx_data):
                t_ap = flx_data[:, 0]
                ap = flx_data[:, i]
                ax2.bar(t_ap + (i + 1 / 2) * bar_width + MJD_0, ap,
                        width = bar_width, alpha = α,
                        color = col_list[i - 1], lw = 0)
            
            sm = ScalarMappable(cmap = colmap, norm = norming)
            sm.set_array([])
            
            cbar = plt.colorbar(sm, cax = ax2.inset_axes([1.1, 0, 0.02, 1]),
                                ticks = np.arange(1, len(flx_data.T) - 1, 1),
                                alpha = α)
            cbar.set_label(r'$i$', rotation = 0, labelpad = 10)
            flx_ylabel = r'$\text{ap}_i$'
            ax2.set_ylabel(flx_ylabel, color = 'k', fontsize = 15, rotation = 0)
            ax2.tick_params(axis = 'y', labelcolor = 'k')
        
        else:
            col = flx_spec_list[2]
            for i in range(1, len_flx_data):
                t_ap = flx_data[:, 0]
                ap = flx_data[:, i]
                
                ax2.bar(t_ap + (i + 1 / 2) * bar_width + MJD_0, ap,
                        width = bar_width, alpha = α,
                        color = col, lw = 0)
            flx_ylabel =  r'$\text{ap}$'
            ax2.set_ylabel(flx_ylabel, color = col, fontsize = 15, rotation = 0)
            ax2.tick_params(axis = 'y', labelcolor = col, color = col)
    
    c = 0
    for i in range(0, len(vline_list1)):
        vline = vline_list1[i]
        col = vline_list_specs1[0]
        α = vline_list_specs1[1]
        width = vline_list_specs1[2]
        lab = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                lab = vline_list_specs1[3]
                c += 1
            plt.axvline(vline, color = col, alpha = α,
                        ls = (0, (4, 4)), lw = width,
                        label = lab, zorder = 3)
    c = 0
    for i in range(0, len(vline_list2)):
        vline = vline_list2[i]
        col = vline_list_specs2[0]
        α = vline_list_specs2[1]
        width = vline_list_specs2[2]
        lab = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                lab = vline_list_specs2[3]
                c += 1
            plt.axvline(vline, color = col, alpha = α,
                        ls = (0, (4, 4)), lw = width,
                        label = lab, zorder = 3)
    
    if (len(flx_data) != 0):
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.set_frame_on(False)
    
    plt.figlegend(fontsize = 12.5, markerscale = 1, loc = location,
                  bbox_to_anchor = anchor, bbox_transform = ax1.transAxes,
                  labelspacing = 0.1, ncols = n_cols, columnspacing = 1)
    
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)

def erase_slash_txt(string):
    index = 0
    counter = 0
    while (counter < 10):
        index = string.find('/')
        if (index == -1):
            break
        string = string[index + 1 :]
        counter += 1
    return(string[: -4])

def get_configuration(path, class_list):
    string = erase_slash_txt(path)
    configuration = []
    index = 0
    counter = 0
    while (counter <= 10):
        index = string.find('_')
        if (index == -1):
            break
        parameter = class_list[counter](string[:index])
        string = string[index + 1 :]
        configuration.append(parameter)
        counter += 1
    configuration.append(class_list[-1](string[-1]))
    return(configuration)

from astropy.time import Time
def mjd_to_mmdd(t, pos):
    t_obj = Time(str(t), format = 'mjd')
    t_iso = t_obj.iso
    string = t_iso[5 : 10].replace('-', '.')
    return(string)

def mjd_to_ddmm(t, pos):
    t_obj = Time(str(t), format = 'mjd')
    t_iso = t_obj.iso
    mm = t_iso[5 : 7]
    dd = t_iso[8 : 10]
    string = dd + "." + mm
    return(string)

def xaxis_year(t_list, time_format):
    t0 = t_list[0]
    t_obj = Time(str(t0), format = 'mjd')
    t_iso = t_obj.iso
    string = t_iso[0 : 4]
    string = string + " + " + time_format
    return(string)

def lab_gen(x, configuration):
    #configuration_list = [[n_fac, R, ω, constr]]
    n_fac = configuration[0]
    R = configuration[1]
    ω = configuration[2]
    constr = configuration[3]
    lab = r'$_{n_fac}^{(R)}\tilde{x}_{ω}^{(constr)}$'
    lab = lab.replace("x", x)
    lab = lab.replace("n_fac", str(n_fac))
    lab = lab.replace("R", str(R))
    lab = lab.replace('ω', '%.0e' % ω)
    lab = lab.replace('constr', str(constr))
    return(lab)

class_list = [int, int, float, int] # [class(n_fac), class(R), class(ω), class(constr)]
path_list = ['newoscele/GRACE_LST/10_3_10000.000000_1.txt',
             'ultimatedata/GF-C/10_3_10000.000000_1.txt']
lab_add_list = [' (LST-Data)', ' (ACC-Data)']

color_group_list = [['lime', 'darkblue'],
                    ['darkgreen', 'cyan']]


new_configuration_list = []
new_slope_data_list = []
new_el_m0_list = []
slope_specs_list = []
for i in range(0, len(path_list)):
    path = path_list[i]
    configuration = get_configuration(path, class_list)
    
    file = open(path, 'r')
    lines = file.readlines()
    file.close()
    m0 = float(lines[0])
    
    new_slope_data = np.loadtxt(path, skiprows = 1)
    
    new_configuration_list.append(configuration)
    new_slope_data_list.append(new_slope_data)
    new_el_m0_list.append(m0)
    
    lab_add = lab_add_list[i]
    lab = lab_gen('a', configuration) + lab_add
    e_lab = lab_gen('σ', configuration) + lab_add
    color_index = i
    lcol = color_group_list[color_index][0]
    #mcol = color_group_list[color_index][1]
    #ecol = color_group_list[color_index][2]
    ecol = color_group_list[color_index][1]
    #slope_specs = [2, lcol, ecol, 1 - i / 2.5, 0.25, lab, e_lab]
    slope_specs = [2, lcol, ecol, 1 - i / 2.5, 0.75, lab, e_lab]
    slope_specs_list.append(slope_specs)

from functions import flx_get_data
path_flx = 'All Data new/FLXAP_P_MJD.FLX'
data_flx_apday, data_ap = flx_get_data(path_flx, MJD_0, n_days_tot)

print(len(new_slope_data_list))

sat_name = 'GFOC'



fac = 1
str_0 = sat_name + ' - '
str_1 = r'$\tilde{a}$'
if (len(new_el_m0_list) > 1):
    str_2 = r", $m_0 = $ [" + ", ".join("%.1e" % m0 for m0 in new_el_m0_list) + "] m"
else:
    str_2 = r", $m_0 = $ %.1e m" % new_el_m0_list[0]



e_fac_list = [1, 100]
#slope_specs = [5.,  0, "deeppink", 1.5, 1]
add_data_list = []
add_data_specs_list = []
el_label = 'a'
el_unit = r'$\frac{m}{d}$'
flx_data = data_ap
#flx_spec_list = [0.75, 1, 'plasma']
flx_spec_list = [1, 0, 'crimson']
title = str_0 + "decrease of " + str_1 + str_2

xlimits = [0, 0]
ylimits = [-20, 20] #[-110, -40]
z_order = 1
nongra_fac = 1
configuration0b = "decr"
path_add = "0"
ave = 0
n_cols = 3

decr_bar_adv(new_slope_data_list, slope_specs_list,
             e_fac_list, ave, el_label, el_unit,
             flx_data, flx_spec_list,
             MJD_0, MJD_end, xlimits, ylimits,
             title, 1, fac, 1, (1., 1.), n_cols,
             vline_list1, vline_list_specs1,
             vline_list2, vline_list_specs2,
             [save_on, path + path_add])
# %%
