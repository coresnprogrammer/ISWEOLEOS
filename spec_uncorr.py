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
save_on = 0
path = os.getcwd() + "/updates/update 10/grace/c/"

foldername_acc = "ultimatedata/GF-C"
file_name = "nongra"
#58356
# 61 days before data hole
#MJD_interval = [58372, 58382] # [58352, 58413]
#MJD_interval = [58352, 58413]
#MJD_interval = [58370, 58410]
MJD_interval = [58400, 58420]

MJD_0, MJD_end = MJD_interval[0], MJD_interval[1]
n_days_tot = MJD_end - MJD_0

str_list = ["a", "e", "omega_low", "u_sat"]

a_symb, a_unit, a_dot = r"$a$", r"[$m$]", r"$\dot{a}$"

# prepare el, el0, acc
name_acc_corr = foldername_acc + "/all_corr.txt"
name_acc_unco = foldername_acc + "/all_uncorr.txt"

acc_data_corr = np.loadtxt(name_acc_corr) # unnormalized
acc_data_unco = np.loadtxt(name_acc_unco) # unnormalized
# first plott acc then modify
acc_R_data_0_corr = array_columns(acc_data_corr, [0, 1])
acc_S_data_0_corr = array_columns(acc_data_corr, [0, 2])
acc_W_data_0_corr = array_columns(acc_data_corr, [0, 3])
acc_A_data_0_corr = array_columns(acc_data_corr, [0, 4])

acc_R_data_0_unco = array_columns(acc_data_unco, [0, 1])
acc_S_data_0_unco = array_columns(acc_data_unco, [0, 2])
acc_W_data_0_unco = array_columns(acc_data_unco, [0, 3])
acc_A_data_0_unco = array_columns(acc_data_unco, [0, 4])

mean_data_0 = np.loadtxt(foldername_acc + "/mean.txt")
scale_data_0 = np.loadtxt(foldername_acc + "/scale.txt")
bias_data_0 = np.loadtxt(foldername_acc + "/bias.txt")


vline_list1 = [58354.25, 58355.25, 59958.915, 59961.875, 59967.125,
               59988.71, 59995.416, 59999.5, 60001.78125, 60002.427,
               60017.17, 60018.17, 60023.9375, 60026.375, 60053.389]
vline_list2 = [58414]
vline_list_specs1 = ['k', 1, 1, "CME"]
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

data_array_list = [acc_R_data_0_corr, acc_R_data_0_unco,
                   acc_S_data_0_corr, acc_S_data_0_unco,
                   acc_W_data_0_corr, acc_W_data_0_unco,
                   acc_A_data_0_corr, acc_A_data_0_unco]
data_spec_array = [[0.75, 'crimson', r'$R$ (corr)', 1], [0.75, 'tomato', r'$R$ (unco)', 1],
                   [0.75, 'midnightblue', r'$S$ (corr)', 1], [0.75, 'deepskyblue', r'$S$ (unco)', 1],
                   [0.75, 'forestgreen', r'$W$ (corr)', 1], [0.75, 'lime', r'$W$ (unco)', 1],
                   [0.75, 'indigo', r'$A$ (corr)', 1], [0.75, 'fuchsia', r'$A$ (unco)', 1]]

y_label = "m/s^2"
ref_array_list, ref_spec_array, ref_y_spec_list = [], [], []
tit = "ACC: R, S, W, A"
y_mean_list = []
y_std_list = []
for i in range(0, len(data_array_list)):
    y_data = data_array_list[i][:, 1]
    y_mean = np.mean(y_data)
    y_std = np.std(y_data)
    
    y_mean_list.append(y_mean)
    y_std_list.append(y_std)

factor = 0.05
y_mean = min(np.abs(y_mean_list))
y_std = max(y_std_list) * factor
xlimits = [130, 60]

ylimits = [y_mean - y_std, y_mean + y_std]
vline_specs = ["gold", 0, 1]
z_order = 0
nongra_fac = 1
grid = 1
fosi = 10
n_cols = 5
save_specs = [0]
vline_list2 = [58413]
vline_list_specs2 = ['gold', 0.75, 5, "Interval"]
plot_func_6_un(data_array_list, data_spec_array, y_label,
               ref_array_list, ref_spec_array, ref_y_spec_list,
               tit, fosi, n_cols,
               acc_R_data_0_corr[0, 0], acc_R_data_0_corr[-1, 0], xlimits, ylimits,
               vline_specs, z_order, nongra_fac, grid,
               vline_list1, vline_list_specs1,
               vline_list2, vline_list_specs2,
               save_specs)
#%%
acc_R_data_corr = array_modifier(acc_R_data_0_corr, MJD_0, n_days_tot)
acc_S_data_corr = array_modifier(acc_S_data_0_corr, MJD_0, n_days_tot)
acc_W_data_corr = array_modifier(acc_W_data_0_corr, MJD_0, n_days_tot)
acc_A_data_corr = array_modifier(acc_A_data_0_corr, MJD_0, n_days_tot)

acc_R_data_unco = array_modifier(acc_R_data_0_unco, MJD_0, n_days_tot)
acc_S_data_unco = array_modifier(acc_S_data_0_unco, MJD_0, n_days_tot)
acc_W_data_unco = array_modifier(acc_W_data_0_unco, MJD_0, n_days_tot)
acc_A_data_unco = array_modifier(acc_A_data_0_unco, MJD_0, n_days_tot)
#%%
data_array_list = [acc_R_data_corr, acc_R_data_unco,
                   acc_S_data_corr, acc_S_data_unco,
                   acc_W_data_corr, acc_W_data_unco,
                   acc_A_data_corr, acc_A_data_unco]
data_spec_array = [[0.75, 'crimson', r'$R$ (corr)', 1], [0.75, 'tomato', r'$R$ (unco)', 1],
                   [0.75, 'midnightblue', r'$S$ (corr)', 1], [0.75, 'deepskyblue', r'$S$ (unco)', 1],
                   [0.75, 'forestgreen', r'$W$ (corr)', 1], [0.75, 'lime', r'$W$ (unco)', 1],
                   [0.75, 'indigo', r'$A$ (corr)', 1], [0.75, 'fuchsia', r'$A$ (unco)', 1]]

y_label = "m/s^2"
ref_array_list, ref_spec_array, ref_y_spec_list = [], [], []
tit = "ACC: R, S, W, A"
y_mean_list = []
y_std_list = []
for i in range(0, len(data_array_list)):
    y_data = data_array_list[i][:, 1]
    y_mean = np.mean(y_data)
    y_std = np.std(y_data)
    
    y_mean_list.append(y_mean)
    y_std_list.append(y_std)

factor = 10
y_mean = min(np.abs(y_mean_list))
y_std = max(y_std_list) * factor
xlimits = [-0.25, -0.25]

ylimits = [y_mean - y_std, y_mean + y_std]
vline_specs = ["gold", 0, 1]
z_order = 0
nongra_fac = 1
grid = 1
fosi = 10
n_cols = 5
save_specs = [0]
plot_func_6_un(data_array_list, data_spec_array, y_label,
               ref_array_list, ref_spec_array, ref_y_spec_list,
               tit, fosi, n_cols,
               acc_R_data_corr[0, 0], acc_R_data_corr[-1, 0], xlimits, ylimits,
               vline_specs, z_order, nongra_fac, grid,
               vline_list1, vline_list_specs1,
               vline_list2, vline_list_specs2,
               save_specs)
for i in range(0, 4):
    data_list = data_array_list[2 * i : 2 * (i + 1)]
    data_specs = data_spec_array[2 * i : 2 * (i + 1)]
    plot_func_6_un(data_list, data_specs, y_label,
                   ref_array_list, ref_spec_array, ref_y_spec_list,
                   tit, fosi, n_cols,
                   acc_R_data_corr[0, 0], acc_R_data_corr[-1, 0], xlimits, ylimits,
                   vline_specs, z_order, nongra_fac, grid,
                   vline_list1, vline_list_specs1,
                   vline_list2, vline_list_specs2,
                   save_specs)
#%%
fft_list_acc_R_corr = fft(acc_R_data_corr)
fft_list_acc_S_corr = fft(acc_S_data_corr)
fft_list_acc_W_corr = fft(acc_W_data_corr)
fft_list_acc_A_corr = fft(acc_A_data_corr)

fft_list_acc_R_unco = fft(acc_R_data_unco)
fft_list_acc_S_unco = fft(acc_S_data_unco)
fft_list_acc_W_unco = fft(acc_W_data_unco)
fft_list_acc_A_unco = fft(acc_A_data_unco)


p_o_list_acc_R_corr, a_o_list_acc_R_corr = fft_list_acc_R_corr[0], fft_list_acc_R_corr[1]
p_o_list_acc_S_corr, a_o_list_acc_S_corr = fft_list_acc_S_corr[0], fft_list_acc_S_corr[1]
p_o_list_acc_W_corr, a_o_list_acc_W_corr = fft_list_acc_W_corr[0], fft_list_acc_W_corr[1]
p_o_list_acc_A_corr, a_o_list_acc_A_corr = fft_list_acc_A_corr[0], fft_list_acc_A_corr[1]

p_o_list_acc_R_unco, a_o_list_acc_R_unco = fft_list_acc_R_unco[0], fft_list_acc_R_unco[1]
p_o_list_acc_S_unco, a_o_list_acc_S_unco = fft_list_acc_S_unco[0], fft_list_acc_S_unco[1]
p_o_list_acc_W_unco, a_o_list_acc_W_unco = fft_list_acc_W_unco[0], fft_list_acc_W_unco[1]
p_o_list_acc_A_unco, a_o_list_acc_A_unco = fft_list_acc_A_unco[0], fft_list_acc_A_unco[1]


p_o_list_acc_list_corr = [p_o_list_acc_R_corr, p_o_list_acc_S_corr,
                          p_o_list_acc_W_corr, p_o_list_acc_A_corr]
a_o_list_acc_list_corr = [a_o_list_acc_R_corr, a_o_list_acc_S_corr,
                          a_o_list_acc_W_corr, a_o_list_acc_A_corr]

p_o_list_acc_list_unco = [p_o_list_acc_R_unco, p_o_list_acc_S_unco,
                          p_o_list_acc_W_unco, p_o_list_acc_A_unco]
a_o_list_acc_list_unco = [a_o_list_acc_R_unco, a_o_list_acc_S_unco,
                          a_o_list_acc_W_unco, a_o_list_acc_A_unco]
#%%
tit_list = ['R', 'S', 'W', 'A']
label_list = [[r'$R_{\text{corr}}$', r'$R_{\text{unco}}$'],
              [r'$S_{\text{corr}}$', r'$S_{\text{unco}}$'],
              [r'$W_{\text{corr}}$', r'$W_{\text{unco}}$'],
              [r'$A_{\text{corr}}$', r'$A_{\text{unco}}$']]
col_list = [['crimson', 'tomato'],
            ['midnightblue', 'deepskyblue'],
            ['forestgreen', 'lime'],
            ['indigo', 'fuchsia']]

fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.title("spectrum of all accelerations", fontsize = 17.5)

for i in range(0, 4):
    lab_acc_corr, lab_acc_unco = label_list[i][0], label_list[i][1]
    p_o_acc_corr, a_o_acc_corr = p_o_list_acc_list_corr[i], a_o_list_acc_list_corr[i]
    p_o_acc_unco, a_o_acc_unco = p_o_list_acc_list_unco[i], a_o_list_acc_list_unco[i]
    fac = 1000**i
    plt.loglog(p_o_acc_corr, a_o_acc_corr / fac, col_list[i][0],
               ls = 'solid', lw = 1, alpha = 0.75,
               label = lab_acc_corr + '/%.1e' % fac, zorder = 2)
    plt.loglog(p_o_acc_unco, a_o_acc_unco / fac, col_list[i][1],
                   ls = 'solid', lw = 1, alpha = 0.75,
               label = lab_acc_unco + '/%.1e' % fac, zorder = 2)

plt.axvline(5.55, color = 'k', lw = 2.5, ls = (0, (5, 10)), alpha = 0.5,
            label = '5.55 d', zorder = 1)
plt.xlabel("Period [d]", fontsize = 15)
plt.ylabel("Amplitude", fontsize = 15)
plt.legend(fontsize = 5, labelspacing = 0.1,
           loc = 4, ncols = 5, prop = 'monospace',
           columnspacing = 1)
plt.grid()
plt.show(fig)
plt.close(fig)
#%%
log_fac = 1000
for i in range(0, 4):
    lab = tit_list[i]
    lab_acc_corr, lab_acc_unco = label_list[i][0], label_list[i][1]
    p_o_acc_corr, a_o_acc_corr = p_o_list_acc_list_corr[i], a_o_list_acc_list_corr[i]
    p_o_acc_unco, a_o_acc_unco = p_o_list_acc_list_unco[i], a_o_list_acc_list_unco[i]
    
    fig = plt.figure(figsize = (11, 5), dpi = 300)
    plt.title("spectrum of acceleration: " + lab, fontsize = 17.5)
    
    plt.loglog(p_o_acc_corr, a_o_acc_corr, col_list[i][0],
               ls = 'solid', lw = 1, alpha = 0.75,
               label = lab_acc_corr, zorder = 2)
    plt.loglog(p_o_acc_unco, a_o_acc_unco / log_fac, col_list[i][1],
               ls = 'solid', lw = 1, alpha = 0.75,
               label = lab_acc_unco + ' / %.1e' % log_fac, zorder = 2)
    
    plt.axvline(5.55, color = 'k', lw = 2.5, ls = (0, (5, 10)), alpha = 0.5,
                label = '5.55 d', zorder = 1)
    
    plt.xlabel("Period [d]", fontsize = 15)
    plt.ylabel("Amplitude", fontsize = 15)
    plt.legend(fontsize = 12.5, labelspacing = 0.1,
               loc = 4)
    plt.grid()
    plt.show(fig)
    plt.close(fig)
# %%
#########################################################
################### MEAN, SCALE, BIAS ###################
#########################################################
data_array_list = [mean_data_0, scale_data_0, bias_data_0]

data_spec_array = [[0.75, 'r', r'R', 1],
                   [0.75, 'b', r'S', 1],
                   [0.75, 'g', r'W', 1]]

y_label = "m/s^2"
ref_array_list, ref_spec_array, ref_y_spec_list = [], [], []
tit_0 = "Entire time span - ACC: "
tit_add_list = ['mean', 'scale', 'bias']
vline_specs = ["gold", 0, 1]
z_order = 0
nongra_fac = 1
grid = 1
fosi = 10
n_cols = 5
save_specs = [0]

factor_list = [2, 0.25, 2.5]
xlimits = [0, 0]

for i in range(0, 3):
    tit = tit_0 + tit_add_list[i]
    factor = factor_list[i]
    
    y_data_list = []
    for j in range(0, 3):
        data = array_columns(data_array_list[i], [0, j + 1])
        y_data_list.append(data)
    
    y_mean_list = []
    y_std_list = []
    for j in range(0, 3):
        y_data = y_data_list[j][:, 1]
        y_mean = np.mean(y_data)
        y_std = np.std(y_data)

        y_mean_list.append(y_mean)
        y_std_list.append(y_std)

    y_mean = np.mean(np.array(y_mean_list))
    #y_mean = min(np.abs(np.array(y_mean_list)))
    y_std = max(y_std_list) * factor

    ylimits = [y_mean - y_std, y_mean + y_std]
    
    plot_func_6_un(y_data_list, data_spec_array, y_label,
                   ref_array_list, ref_spec_array, ref_y_spec_list,
                   tit, fosi, n_cols,
                   acc_R_data_0_corr[0, 0], acc_R_data_0_corr[-1, 0],
                   xlimits, ylimits,
                   vline_specs, z_order, nongra_fac, grid,
                   vline_list1, vline_list_specs1,
                   vline_list2, vline_list_specs2,
                   save_specs)
# %%
######################################################
################# MINUS ITS OWN MEAN #################
######################################################

data_array_list = [mean_data_0, scale_data_0, bias_data_0]

data_spec_array = [[0.75, 'r', r'R', 1],
                   [0.75, 'b', r'S', 1],
                   [0.75, 'g', r'W', 1]]

y_label = "m/s^2"
ref_array_list, ref_spec_array, ref_y_spec_list = [], [], []
tit_0 = "Entire time span, mean value subtracted - ACC: "
tit_add_list = ['mean', 'scale', 'bias']
vline_specs = ["gold", 0, 1]
z_order = 0
nongra_fac = 1
grid = 1
fosi = 10
n_cols = 5
save_specs = [0]

factor_list = [0.15, 0.25, 1]

xlimits = [-0.5, -0]

for i in range(0, 3):
    tit = tit_0 + tit_add_list[i]
    factor = factor_list[i]
    fac1, fac2 = 1, 1
    if (i == 0):
        fac1 = 0.1
        fac2 = 0.7
    
    y_data_list = []
    for j in range(0, 3):
        data = array_columns(data_array_list[i], [0, j + 1])
        data_time = data[:, 0]
        data_yyyy = data[:, 1]
        data_mean = np.mean(data_yyyy)
        
        data_0 = np.vstack((data_time, data_yyyy - data_mean)).T
        y_data_list.append(data_0)
    
    y_std_list = []
    for j in range(0, 3):
        y_data = y_data_list[j][:, 1]
        y_std = np.std(y_data)
        y_std_list.append(y_std)
    
    y_std = max(y_std_list) * factor

    ylimits = [0 - y_std * fac1, 0 + y_std * fac2]
    
    plot_func_6_un(y_data_list, data_spec_array, y_label,
                   ref_array_list, ref_spec_array, ref_y_spec_list,
                   tit, fosi, n_cols,
                   acc_R_data_0_corr[0, 0], acc_R_data_0_corr[-1, 0],
                   xlimits, ylimits,
                   vline_specs, z_order, nongra_fac, grid,
                   vline_list1, vline_list_specs1,
                   vline_list2, vline_list_specs2,
                   save_specs)
#%%
#################################################
#################### TRIMMED ####################
#################################################

mean_data = array_modifier(mean_data_0, MJD_0, n_days_tot)
scale_data = array_modifier(scale_data_0, MJD_0, n_days_tot)
bias_data = array_modifier(bias_data_0, MJD_0, n_days_tot)

data_array_list = [mean_data, scale_data, bias_data]

data_spec_array = [[0.75, 'r', r'R', 1],
                   [0.75, 'b', r'S', 1],
                   [0.75, 'g', r'W', 1]]

y_label = "m/s^2"
ref_array_list, ref_spec_array, ref_y_spec_list = [], [], []
tit_0 = "Trimmed time span - ACC: "
tit_add_list = ['mean', 'scale', 'bias']
vline_specs = ["gold", 0, 1]
z_order = 0
nongra_fac = 1
grid = 1
fosi = 10
n_cols = 5
save_specs = [0]

factor_list = [500, 2.5, 10] # [58352, 58413]
#factor_list = [3000, 2.5, 25] # [58372, 58382]
#factor_list = [800, 2.5, 15] # [58370, 58410]
xlimits = [-0.25, -0.25]

for i in range(0, 3):
    tit = tit_0 + tit_add_list[i]
    factor = factor_list[i]
    
    y_data_list = []
    for j in range(0, 3):
        data = array_columns(data_array_list[i], [0, j + 1])
        y_data_list.append(data)
    
    y_mean_list = []
    y_std_list = []
    for j in range(0, 3):
        y_data = y_data_list[j][:, 1]
        y_mean = np.mean(y_data)
        y_std = np.std(y_data)

        y_mean_list.append(y_mean)
        y_std_list.append(y_std)

    y_mean = np.mean(np.array(y_mean_list))
    #y_mean = min(np.abs(np.array(y_mean_list)))
    y_std = max(y_std_list) * factor

    ylimits = [y_mean - y_std, y_mean + y_std]
    
    plot_func_6_un(y_data_list, data_spec_array, y_label,
                   ref_array_list, ref_spec_array, ref_y_spec_list,
                   tit, fosi, n_cols,
                   acc_R_data_corr[0, 0], acc_R_data_corr[-1, 0],
                   xlimits, ylimits,
                   vline_specs, z_order, nongra_fac, grid,
                   vline_list1, vline_list_specs1,
                   vline_list2, vline_list_specs2,
                   save_specs)
# %%
######################################################
################# MINUS ITS OWN MEAN #################
######################################################

data_array_list = [mean_data, scale_data, bias_data]

data_spec_array = [[0.75, 'r', r'R', 1],
                   [0.75, 'b', r'S', 1],
                   [0.75, 'g', r'W', 1]]

y_label = "m/s^2"
ref_array_list, ref_spec_array, ref_y_spec_list = [], [], []
tit_0 = "Trimmed time span, mean value subtracted - ACC: "
tit_add_list = ['mean', 'scale', 'bias']
vline_specs = ["gold", 0, 1]
z_order = 0
nongra_fac = 1
grid = 1
fosi = 10
n_cols = 5
save_specs = [0]

factor_list = [2.5, 2.5, 2.5]
xlimits = [-0.5, -0]

for i in range(0, 3):
    tit = tit_0 + tit_add_list[i]
    factor = factor_list[i]
    
    y_data_list = []
    for j in range(0, 3):
        data = array_columns(data_array_list[i], [0, j + 1])
        data_time = data[:, 0]
        data_yyyy = data[:, 1]
        data_mean = np.mean(data_yyyy)
        
        data_0 = np.vstack((data_time, data_yyyy - data_mean)).T
        y_data_list.append(data_0)
    
    y_std_list = []
    for j in range(0, 3):
        y_data = y_data_list[j][:, 1]
        y_std = np.std(y_data)
        y_std_list.append(y_std)
    
    y_std = max(y_std_list) * factor

    ylimits = [0 - y_std, 0 + y_std]
    
    plot_func_6_un(y_data_list, data_spec_array, y_label,
                   ref_array_list, ref_spec_array, ref_y_spec_list,
                   tit, fosi, n_cols,
                   acc_R_data_corr[0, 0], acc_R_data_corr[-1, 0],
                   xlimits, ylimits,
                   vline_specs, z_order, nongra_fac, grid,
                   vline_list1, vline_list_specs1,
                   vline_list2, vline_list_specs2,
                   save_specs)
# %%
