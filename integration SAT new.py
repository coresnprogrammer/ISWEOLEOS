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
from functions import pre_lst_sqrs_adjstmt, fitter, mjd_to_mmdd, man_gen_file
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
os.chdir('/Users/levin/Documents/Uni/Bachelor/Bachelorthesis new')
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

satname = 'SENTINEL-1A'
year = '2023'
satshort = 'SE1A'
yearshort = '23'

image_folder = 'Thesis/LATEX/Images/Results/' + satname + '_' + year + '/'

# foldername_lst = "All Data new/oscelegrace"
# path_ele = "All Data new/elegrace/year_normal/year_normal.txt"
# path_ele_error = "All Data new/elegrace/year_normal/year_normal_errors.txt"
# file_name = "nongra"

# foldername_lst = "All Data new/osceleswarm"
# path_ele = "All Data new/eleswarm/year_normal/year_normal.txt"
# path_ele_error = "All Data new/eleswarm/year_normal/year_normal_errors.txt"
# file_name = "nongra"

foldername_lst = "All Data new/oscelesentinel"
path_ele = "All Data new/elesentinel/year_normal/year_normal.txt"
path_ele_error = "All Data new/elesentinel/year_normal/year_normal_errors.txt"
file_name = "nongra"

# foldername_lst = "oscele24/swma"
# path_ele = "final24/swma/year_normal/year_normal.txt"
# path_ele_error = "final24/swma/year_normal/year_normal_errors.txt"
# file_name = "normal"

# foldername_lst = "sentinel_collection/se2a"
# path_ele = "final24/se2a/year_normal/year_normal.txt"
# path_ele_error = "final24/se2a/year_normal/year_normal_errors.txt"
# file_name = "normal"

# foldername_lst = "sentinel_collection/se3a"
# path_ele = "final24/se3a/year_normal/year_normal.txt"
# path_ele_error = "final24/se3a/year_normal/year_normal_errors.txt"
# file_name = "normal"


#MJD_interval = [58340, 58413] # year 2018 (2018.08.10 - 2018.10.22)
MJD_interval = [59990, 60040] # year 2023 (2023.02.15 - 2023.04.06)
#MJD_interval = [60431, 60445] # year 2024 (2024.05.01 - 2024.05.14)
MJD_interval_lst = MJD_interval
MJD_interval_int = MJD_interval


MJD_0_lst, MJD_end_lst = MJD_interval_lst[0], MJD_interval_lst[1]
MJD_0_int, MJD_end_int = MJD_interval_int[0], MJD_interval_int[1]

n_days_tot_lst = MJD_end_lst - MJD_0_lst
n_days_tot_int = MJD_end_int - MJD_0_int
n_days_tot_max = max([MJD_end_lst, MJD_end_int]) - min([MJD_0_lst, MJD_0_int])

MJD_0_list = [MJD_0_lst, MJD_0_int]
MJD_end_list = [MJD_end_lst, MJD_end_int]
MJD_0 = min(MJD_0_list)
MJD_end = max(MJD_end_list)


str_list = ["a", "e", "omega_low", "u_sat"]

a_symb, a_unit, a_dot = r"$a$", r"[$m$]", r"$\dot{a}$"

# prepare el, el0, acc
el_data_list = []
for i in range(0, len(str_list)):
    el_str = str_list[i]
    name_el = foldername_lst + 2 * ("/year_" + file_name) + "_" + el_str + ".txt"
    el_data = np.loadtxt(name_el, skiprows=1)  # load data
    el_data = array_denormalize(el_data)  # denormalize data
    el_data = array_modifier(el_data, MJD_0_lst, n_days_tot_lst)  # trim data
    #el_data = array_normalize(el_data, 0)  # normalize data
    #el_data = el_data[1:]  # cut MJD_0
    el_data_list.append(el_data)
a_data, e_data = el_data_list[0], el_data_list[1]
ω_data, u_data = el_data_list[2], el_data_list[3]

print(a_data[-1,0] - a_data[0,0])

ele_data = np.loadtxt(path_ele)

ele_R_data_0 = array_columns(ele_data, [0, 1])
ele_S_data_0 = array_columns(ele_data, [0, 2])
ele_W_data_0 = array_columns(ele_data, [0, 3])
ele_A_data_0 = array_columns(ele_data, [0, 4])


# specifically for VXYZ_EF
#t_list = V_data[:, 0]
#V_list = np.sqrt(np.sum((V_data[:, 1:])**2, 1)) # |V|
#V_data = np.vstack((t_list, V_list)).T

vline_list1 = []

from astropy.time import Time
def yyyymmddhhmm_to_mjd(string):
    t_obj = Time(string, format = 'isot', scale = 'utc')
    t_mjd = t_obj.mjd
    mjd_float = float(t_mjd)
    return(mjd_float)

addcmelist = ['2018-07-10T11:25Z', '2018-08-24T05:50Z',
              '2018-08-25T06:00Z', '2023-02-13T17:02Z',
              '2023-02-20T09:52Z', '2023-02-24T12:08Z',
              '2023-02-26T18:43Z', '2023-02-27T10:15Z',
              '2023-03-14T03:58Z', '2023-03-15T03:48Z',
              '2023-03-15T03:48Z', '2023-03-20T22:37Z',
              '2023-03-23T09:10Z', '2023-03-23T09:10Z',
              '2023-04-19T09:22Z', '2023-04-23T17:00Z',
              '2024-04-26T00:17Z', '2024-05-05T11:30Z',
              '2024-05-10T16:36Z', '2024-05-10T16:36Z',
              '2024-05-10T16:36Z', '2024-05-10T16:36Z',
              '2024-05-10T16:36Z', '2024-05-11T09:30Z',
              '2024-05-11T09:30Z', '2024-05-11T20:30Z',
              '2024-05-12T08:55Z', '2024-05-15T18:13Z',
              '2024-05-17T12:40Z']
for i in range(0, len(addcmelist)):
    vline_list1.append(yyyymmddhhmm_to_mjd(addcmelist[i]))

vline_list_specs1 = ['goldenrod', 1, 2.5, "CME", (0, (4, 4)), 2]


vline_list2 = []
vline_list_specs2 = ['saddlebrown', 1, 1.5, "MAN", (0, (4, 4)), 2]
# put as comments if it is not sentinel-1a!!!
"""
man_data = np.loadtxt("hyperdata/hyperdata2/year/all.txt")
man_data = array_denormalize(man_data)
man_data = array_modifier(man_data, MJD_0, n_days_tot_max)[:,0]
print(man_data)
vline_list2 = man_data

go = []
go_all = []
for i in range(0, len(man_data) - 1):
    diff = man_data[i + 1] - man_data[i]
    if (diff > 5.5 and diff < 7.5):
        go.append(diff)
    go_all.append(diff)
print(go)
fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.plot(np.arange(len(go)), go, 'r.')
plt.plot(np.arange(len(go_all)),go_all,'b.')
plt.show(fig)
plt.close(fig)
print(np.mean(go), np.std(go))
print(np.mean(go_all), np.std(go_all))
vline_list_specs1 = ['goldenrod', 1, 1, "CME", (0, (4, 12)), 2]
vline_list_specs2 = ['saddlebrown', 1, 1, "MAN", (-6, (2, 4,2,8)), 2]
"""

#"""
def yyddd_to_mjd(npfloat):
    string = '%.f' % npfloat
    newstring = '20%s:%s' % (string[:2], string[2:])
    t_obj = Time(newstring, format = 'yday', scale = 'utc')
    #t_mjd = t_obj.mjd
    #mjd_float = float(t_mjd)
    return(t_obj.iso)

#man_gen_file("finalmans/s3a")
man_data = np.loadtxt("finalmans/s2a/year/all.txt")
man_data = array_denormalize(man_data)
man_data = array_modifier(man_data, MJD_0, n_days_tot_max)[:,0]
print(man_data)
vline_list2 = man_data


# for se3a
# vline_list2_data = np.loadtxt("man_se2a_se3a/S2A_Maneuvers_2024.txt")
# print(vline_list2_data)
# vline_list_specs1 = ['gold', 1, 1, "CME", (0, (4, 12)), 2]
# vline_list_specs2 = ['saddlebrown', 1, 1, "MAN", (-6, (2, 4,2,8)), 2]
# for i in range(0, len(vline_list2_data)):
#     vline_list2.append(yyddd_to_mjd(vline_list2_data[i]))
# print(vline_list2)

#%%
el_num = arg_finder(string_list, 'a')
el_word, el_symb = name_list[el_num][0], name_list[el_num][1]
el_unit, el_bar = name_list[el_num][2], name_list[el_num][3]
el_hat, el_tilde = name_list[el_num][4], name_list[el_num][5]
ele = string_list[el_num]
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
    ax1.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
    ax1.tick_params(axis = 'x', which = 'major',
                    width = 2, length = 8,
                    direction = 'out', top = False)
    ax1.tick_params(axis = 'x', which = 'minor',
                    width = 1, length = 4,
                    direction = 'out', top = False)
    ax1.get_yaxis().get_major_formatter().set_useMathText(True)
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    ax1.set_ylabel(y_label,rotation='horizontal', fontsize = 20)
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
                        ls = vline_list_specs1[4], zorder = 3,
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
                        ls = vline_list_specs2[4], zorder = 3,
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

data_array_list = [ele_R_data_0, ele_S_data_0, ele_W_data_0, ele_A_data_0]
data_spec_array = [[0.75, 'tomato', r'$R$ (PCA)', 1], [0.75, 'deepskyblue', r'$S$ (PCA)', 1],
                   [0.75, 'lime', r'$W$ (PCA)', 1], [0.75, 'fuchsia', r'$A$ (PCA)', 1]]

y_label = "m/s^2"
ref_array_list, ref_spec_array, ref_y_spec_list = [], [], []
tit = "PCA: R, S, W, A"
y_mean_list = []
y_std_list = []
for i in range(0, len(data_array_list)):
    y_data = data_array_list[i][:, 1]
    y_mean = np.mean(y_data)
    y_std = np.std(y_data)
    
    y_mean_list.append(y_mean)
    y_std_list.append(y_std)

factor = 1
y_mean = min(np.abs(y_mean_list))
y_std = max(y_std_list) * factor
xlimits = [0, 0]

ylimits = [y_mean - y_std, y_mean + y_std]
vline_specs = ["gold", 0, 1]
z_order = 0
nongra_fac = 1
grid = 1
fosi = 10
n_cols = 3
save_specs = [0]
plot_func_6_un(data_array_list, data_spec_array, y_label,
               ref_array_list, ref_spec_array, ref_y_spec_list,
               tit, fosi, n_cols,
               MJD_0, MJD_end, xlimits, ylimits,
               vline_specs, z_order, nongra_fac, grid,
               vline_list1, vline_list_specs1,
               vline_list2, vline_list_specs2,
               save_specs)
#%%
#acc_R_data = array_modifier(acc_R_data_0, MJD_0_int, n_days_tot_int)
#acc_S_data = array_modifier(acc_S_data_0, MJD_0_int, n_days_tot_int)
#acc_W_data = array_modifier(acc_W_data_0, MJD_0_int, n_days_tot_int)
#acc_A_data = array_modifier(acc_A_data_0, MJD_0_int, n_days_tot_int)

ele_data = array_modifier(ele_data, MJD_0_int, n_days_tot_int)
#%%
data_array_list = [array_columns(ele_data, [0, 1]),
                   array_columns(ele_data, [0, 2]),
                   array_columns(ele_data, [0, 3]),
                   array_columns(ele_data, [0, 4])]
data_spec_array = [[0.75, 'tomato', r'$R$ (PCA)', 1],
                   [0.75, 'deepskyblue', r'$S$ (PCA)', 1],
                   [0.75, 'lime', r'$W$ (PCA)', 1],
                   [0.75, 'fuchsia', r'$A$ (PCA)', 1]]

y_label = "m/s^2"
ref_array_list, ref_spec_array, ref_y_spec_list = [], [], []
tit = "PCA: R, S, W, A"
y_mean_list = []
y_std_list = []
for i in range(0, len(data_array_list)):
    y_data = data_array_list[i][:, 1]
    y_mean = np.mean(y_data)
    y_std = np.std(y_data)
    
    y_mean_list.append(y_mean)
    y_std_list.append(y_std)

factor = 1
y_mean = min(np.abs(y_mean_list))
y_std = max(y_std_list) * factor
xlimits = [0, 0]

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
               MJD_0_int, MJD_end_int, xlimits, ylimits,
               vline_specs, z_order, nongra_fac, grid,
               vline_list1, vline_list_specs1,
               vline_list2, vline_list_specs2,
               save_specs)
#%%
fft_list_ele_R = fft(array_columns(ele_data, [0, 1]))
fft_list_ele_S = fft(array_columns(ele_data, [0, 2]))
fft_list_ele_W = fft(array_columns(ele_data, [0, 3]))
fft_list_ele_A = fft(array_columns(ele_data, [0, 4]))

p_o_list_ele_R, a_o_list_ele_R = fft_list_ele_R[0], fft_list_ele_R[1]
p_o_list_ele_S, a_o_list_ele_S = fft_list_ele_S[0], fft_list_ele_S[1]
p_o_list_ele_W, a_o_list_ele_W = fft_list_ele_W[0], fft_list_ele_W[1]
p_o_list_ele_A, a_o_list_ele_A = fft_list_ele_A[0], fft_list_ele_A[1]

p_o_list_ele_list = [p_o_list_ele_R, p_o_list_ele_S, p_o_list_ele_W, p_o_list_ele_A]
a_o_list_ele_list = [a_o_list_ele_R, a_o_list_ele_S, a_o_list_ele_W, a_o_list_ele_A]
#%%
tit_list = ['R', 'S', 'W', 'A']
label_list = [r'$R_{\text{PCA}}$',
              r'$S_{\text{PCA}}$',
              r'$W_{\text{PCA}}$',
              r'$A_{\text{PCA}}$']
col_list = ['tomato', 'deepskyblue', 'lime', 'fuchsia']

fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.title("spectrum of all accelerations", fontsize = 17.5)

for i in range(0, 4):
    lab_ele = label_list[i]
    p_o_ele, a_o_ele = p_o_list_ele_list[i], a_o_list_ele_list[i]
    fac = 1000**i
    plt.loglog(p_o_ele, a_o_ele / fac, col_list[i],
                   ls = 'solid', lw = 1, alpha = 0.75,
               label = lab_ele + '/%.1e' % fac, zorder = 2)

plt.axvline(5.55, color = 'k', lw = 2.5, ls = (0, (5, 10)), alpha = 0.5,
            label = '5.55 d', zorder = 1)
plt.xlabel("Period [d]", fontsize = 15)
plt.ylabel(r"Amplitude [$\frac{\text{m}}{\text{s}^2}$]", fontsize = 15)
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
    lab_ele = label_list[i]
    p_o_ele, a_o_ele = p_o_list_ele_list[i], a_o_list_ele_list[i]
    
    fig = plt.figure(figsize = (10, 5), dpi = 300)
    plt.title("spectrum of acceleration: " + lab, fontsize = 17.5)
    
    plt.loglog(p_o_ele, a_o_ele / log_fac, col_list[i],
               ls = 'solid', lw = 1, alpha = 0.75,
               label = lab_ele + ' / %.1e' % log_fac, zorder = 2)
    
    plt.axvline(5.55, color = 'k', lw = 2.5, ls = (0, (5, 10)), alpha = 0.5,
                label = '5.55 d', zorder = 1)
    
    plt.xlabel("Period [d]", fontsize = 15)
    plt.ylabel(r"Amplitude [$\frac{\text{m}}{\text{s}^2}$]", fontsize = 15)
    plt.legend(fontsize = 12.5, labelspacing = 0.1,
               loc = 4)
    plt.grid()
    plt.show(fig)
    plt.close(fig)
#%%
"""
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

print(mean_acc_R, mean_ele_R, np.mean(acc_R_data_shifted[:, 1]))
print(mean_acc_S, mean_ele_S, np.mean(acc_S_data_shifted[:, 1]))
print(mean_acc_W, mean_ele_W, np.mean(acc_W_data_shifted[:, 1]))
"""
#%%
step_fac = 12 # 12 * 30s = 6min
t1 = tt.time()
ele_data_step = step_data_generation(ele_data, step_fac)
t2 = tt.time()
print("time: %.3f" % (t2 - t1))

ele_R_data = array_columns(ele_data_step, [0, 1])
ele_S_data = array_columns(ele_data_step, [0, 2])
ele_W_data = array_columns(ele_data_step, [0, 3])
ele_A_data = array_columns(ele_data_step, [0, 4])
#%%
data_array_list = [ele_R_data, ele_S_data,ele_W_data, ele_A_data]

data_spec_array = [[0.75, 'tomato', r'$R$ (PCA)', 1],
                   [0.75, 'deepskyblue', r'$S$ (PCA)', 1],
                   [0.75, 'lime', r'$W$ (PCA)', 1],
                   [0.75, 'fuchsia', r'$A$ (PCA)', 1]]

y_mean_list = []
y_std_list = []
for i in range(0, len(data_array_list)):
    y_data = data_array_list[i][:, 1]
    y_mean = np.mean(y_data)
    y_std = np.std(y_data)
    
    y_mean_list.append(y_mean)
    y_std_list.append(y_std)

xlimits = [0, 0]
factor = 8
y_mean = min(np.abs(y_mean_list))
y_std = max(y_std_list) * factor
ylimits = [y_mean - y_std, y_mean + y_std]
fosi = 10
n_cols = 5
plot_func_6_un(data_array_list, data_spec_array, y_label,
               ref_array_list, ref_spec_array, ref_y_spec_list,
               tit, fosi, n_cols,
               MJD_0_int, MJD_end_int, xlimits, ylimits,
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
    
    k1, σ_k1 = a_dot_RSW(a_n, e, ω, u_n, R_n, S_n, σ_a, σ_e, σ_ω)
    k1, σ_k1 = 2 * dt * k1, 2 * dt * σ_k1
    k2, σ_k2 = a_dot_RSW(a_n + k1 / 2, e, ω, u_np1, R_np1, S_np1, np.sqrt(σ_a**2 +  σ_k1**2 / 4), σ_e, σ_ω)
    k2, σ_k2 = 2 * dt * k2, 2 * dt * σ_k2
    k3, σ_k3 = a_dot_RSW(a_n + k2 / 2, e, ω, u_np1, R_np1, S_np1, np.sqrt(σ_a**2 +  σ_k2**2 / 4), σ_e, σ_ω)
    k3, σ_k3 = 2 * dt * k3, 2 * dt * σ_k3
    k4, σ_k4 = a_dot_RSW(a_n + k3, e, ω, u_np2, R_np2, S_np2, np.sqrt(σ_a**2 +  σ_k3**2), σ_e, σ_ω)
    k4, σ_k4 = 2 * dt * k4, 2 * dt * σ_k4
    
    a_np2 = a_n + (k1 + k4) / 6 + (k2 + k3) / 3
    a_np2_error = np.sqrt(σ_a**2 + (σ_k1**2 + σ_k4**2) / 36 + (σ_k2**2 + σ_k3**2) / 9)
    return(a_np2, a_np2_error)

def ele_σ_for_rk45(path_ele_σ, mjd_interval):
    data = np.loadtxt(path_ele_σ)
    mjd_start = mjd_interval[0]
    n_days = mjd_interval[1] - mjd_start
    data_trimmed = array_modifier(data, mjd_start, n_days)
    σ_a = np.mean(data_trimmed[:, 1])
    σ_e = np.mean(data_trimmed[:, 2])
    σ_ω = np.mean(data_trimmed[:, 5])
    return(σ_a, σ_e, σ_ω)

def rk45_discrete(a_data, e_data, ω_data, u_data, acc_R_data, acc_S_data,
                  path_ele_σ, mjd_interval):
    e = e_data[0, 1]
    ω = ω_data[0, 1]
    
    σ_a_0, σ_e, σ_ω = ele_σ_for_rk45(path_ele_σ, mjd_interval)
    
    u_list_list = u_data[:, 1]
    R_list_list = acc_R_data[:, 1] / (sec_to_day**2)
    S_list_list = acc_S_data[:, 1] / (sec_to_day**2)
    
    t_0 = a_data[0, 0] 
    a_0 = a_data[0, 1]
    a_dot_0, σ_a_dot_0 = a_dot_RSW(a_0, e, ω, u_list_list[0], R_list_list[0],
                                   S_list_list[0], σ_a_0, σ_e, σ_ω)
    
    a_int_data = np.array([[t_0, a_0, σ_a_0]])
    a_dot_data = np.array([[t_0, a_dot_0, σ_a_dot_0]])
    
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
a_data_for_int = array_modifier(a_data, MJD_0_int, n_days_tot_int)
e_data_for_int = array_modifier(e_data, MJD_0_int, n_days_tot_int)
ω_data_for_int = array_modifier(ω_data, MJD_0_int, n_days_tot_int)
u_data_for_int = array_modifier(u_data, MJD_0_int, n_days_tot_int)

print(len(ele_R_data), len(a_data_for_int))
print((ele_R_data[1, 0] - ele_R_data[0, 0])*24*60*60)
print((a_data_for_int[1, 0] - a_data_for_int[0, 0])*24*60*60)
#%%
t_1 = tt.time()
a_int_ele_list = rk45_discrete(a_data_for_int, e_data_for_int,
                               ω_data_for_int, u_data_for_int,
                               ele_R_data, ele_S_data,
                               path_ele_error, MJD_interval_int)
t_2 = tt.time()
print("a_int_ele_list done - time: %.3f s" % (t_2 - t_1))
#%%
a_int_ele, a_dot_ele = a_int_ele_list[0], a_int_ele_list[1]
#%%
##################################################
##################################################
#################### FAILSAFE ####################
##################################################
##################################################
folder_collection = 'big_data_collection/FINAL/' + satshort + yearshort + '/'
run = 1
add = folder_collection + 'run_' + str(run) + '-'

name_a_int_ele = add + 'a_int_ele' + '.txt'
name_a_dot_ele = add + 'a_dot_ele' + '.txt'

name_list = [name_a_int_ele, name_a_dot_ele]

""" # set "#" before
array_list = [a_int_ele, a_dot_ele]

for i in range(0, len(name_list)):
    name_i = name_list[i]
    array_i = array_list[i]
    np.savetxt(name_i, array_i)
""" # set "#" before
#""" # set "#" before
a_int_ele = np.loadtxt(name_a_int_ele)
a_dot_ele = np.loadtxt(name_a_dot_ele)
#""" # set "#" before
##################################################
##################################################
##################################################
##################################################
##################################################
#%% PLOT LST, ACC AND ELE
def t_min_t_max(data_list):
    t_min_list = []
    t_max_list = []
    for i in range(0, len(data_list)):
        t_min_list.append(min(data_list[i][:, 0]))
        t_max_list.append(max(data_list[i][:, 0]))
    return(min(t_min_list), max(t_max_list))

def scientific_to_exponent(number, coma_digits):
    # number = 13000
    # coma_digits: # of digits after ","
    if (coma_digits > 10): # error
        return(1)
    scientific_str = r'%.10E' % number
    fac_str = scientific_str[: 2 + coma_digits]
    if (fac_str[-1] == '0' and coma_digits == 1):
        fac_str = fac_str[:-2]
    exp_str = scientific_str[-3:]
    if (exp_str[-2] == '0'):
        exp_str = exp_str[0] + exp_str[-1]
    if (exp_str[0] == '+'):
        exp_str = exp_str[1:]
    if (fac_str == '1'):
        if (exp_str == '0'):
            new_str = r'$1$'
        elif (exp_str == '1'):
            new_str = r'$10$'
        else:
            new_str = r'$10^{%s}$' % (exp_str)
    else:
        if (exp_str == '0'):
            new_str = r'$%s$' % fac_str
        else:
            new_str = r'$%s\cdot10^{%s}$' % (fac_str, exp_str)
    return(new_str)

def plot_data(data_list, data_spec_array, y_label,
              ref_list, ref_spec_array, ref_y_spec_list,
              tit, xlimits, ylimits, grid,
              location, anchor, n_cols, fosi, labspa,
              vline_list1, vline_list_specs1,
              vline_list2, vline_list_specs2,
              save_specs):
    """ description of input parameters
    data_list        : [data_1, data_2, ...]
    data_spec_array  : [alpha_i, col_i, lab_i, lw_i]
    y_label          : label for y-axis
    ref_list         : [ref_1, ref_2, ...]
    ref_spec_array   : [alpha_i, col_i, lab_i, lw_i]
    ref_y_spec_list  : [y_label, y_axis_col]
    tit              : title, string like
    xlimits          : zoom values
    ylimits          : limit values
    grid             : if 1 --> grid to data
    vline_list1      : [vline_i]
    vline_list_specs1: [col_i, alpha_i, lw_i, label]
    vline_list2      : [vline_i]
    vline_list_specs2: [col_i, alpha_i, lw_i, label]
    save_specs       : [1, path] # 0,1 for saving
    """
    t_min, t_max = t_min_t_max(data_list)
    xstart = t_min + xlimits[0]
    xend = t_max - xlimits[1]
    n_days = xend - xstart
    new_data_list = 0
    new_ref_list = 0
    if (xlimits[0] != xlimits[1]):
        new_data_list = []
        new_ref_list = []
        for i in range(0, len(data_list)):
            data_i = data_list[i]
            data_i = array_modifier(data_i, xstart, n_days)
            new_data_list.append(data_i)
        for i in range(0, len(ref_list)):
            ref_i = ref_list[i]
            ref_i = array_modifier(ref_i, xstart, n_days)
            new_ref_list.append(ref_i)
    else:
        new_data_list = data_list
        new_ref_list = ref_list
    
    fig, ax1 = plt.subplots(figsize = (10, 5), dpi = 300)
    fig.suptitle(tit, fontsize = 17.5)
    
    for i in range(0, len(new_data_list)):
        x_data = new_data_list[i][:, 0]
        y_data = new_data_list[i][:, 1]
        α = data_spec_array[i][0]
        col = data_spec_array[i][1]
        lab = data_spec_array[i][2]
        width = data_spec_array[i][3]
        if (len(new_data_list[i].T) == 3):
            # plot with errors
            s_fac = data_spec_array[i][5]
            s_y = new_data_list[i][:, 2] * s_fac
            e_α = data_spec_array[i][4]
            e_lab = data_spec_array[i][6]
            lab_fac = scientific_to_exponent(s_fac, 1)
            lab = lab + r' (' + e_lab + r' $\times$ ' + lab_fac + ')'
            ax1.fill_between(x_data, y_data - s_y, y_data + s_y,
                             color = col, alpha = e_α, zorder = 4)
        ax1.plot(x_data, y_data, color = col, ls = '-', lw = width,
                 alpha = α, zorder = 5, label = lab)
    ax1.xaxis.set_major_formatter(mjd_to_mmdd)
    ax1.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
    ax1.tick_params(axis = 'x', which = 'major',
                    width = 2, length = 8,
                    direction = 'out', top = False)
    ax1.tick_params(axis = 'x', which = 'minor',
                    width = 1, length = 4,
                    direction = 'out', top = False)
    ax1.get_yaxis().get_major_formatter().set_useMathText(True)
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    ax1.set_ylabel(y_label, fontsize = 15)
    y_low, y_upp = ylimits[0], ylimits[1]
    if (xstart != xend):
        ax1.set_xlim(xstart, xend)
    if (y_low != y_upp):
        ax1.set_ylim(y_low, y_upp)
    if (grid == 1):
        ax1.grid()
    
    if (len(ref_array_list) != 0):
        ax2 = ax1.twinx()
        for i in range(0, len(new_ref_list)):
            x_ref = new_ref_list[i][:, 0]
            y_ref = new_ref_list[i][:, 1]
            α = ref_spec_array[i][0]
            col = ref_spec_array[i][1]
            lab = ref_spec_array[i][2]
            width = ref_spec_array[i][3]
            ax2.plot(x_ref, y_ref,
                     ls = '-', lw = width, color = col, alpha = α,
                     label = lab)
        ax2.set_ylabel(ref_y_spec_list[0], color = ref_y_spec_list[1], fontsize = 15)
        ax2.tick_params(axis = 'y', labelcolor = ref_y_spec_list[1])
        
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.set_frame_on(False)
    
    c = 0 # for giving only one label to vlines from vline_list1
    for i in range(0, len(vline_list1)):
        vline = vline_list1[i]
        col = vline_list_specs1[0]
        alph = vline_list_specs1[1]
        liwi = vline_list_specs1[2]
        linst = vline_list_specs1[4]
        zo = 5 + vline_list_specs1[5]
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs1[3]
                c += 1
            plt.axvline(vline, color = col,
                        alpha = alph, lw = liwi,
                        ls = linst, zorder = zo,
                        label = c_label)
    
    c = 0 # for giving only one label to vlines from vline_list2
    for i in range(0, len(vline_list2)):
        vline = vline_list2[i]
        col = vline_list_specs2[0]
        alph = vline_list_specs2[1]
        liwi = vline_list_specs2[2]
        linst = vline_list_specs2[4]
        zo = 5 + vline_list_specs2[5]
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs2[3]
                c += 1
            plt.axvline(vline, color = col,
                        alpha = alph, lw = liwi,
                        ls = linst, zorder = zo,
                        label = c_label)
    
    plt.figlegend(fontsize = fosi, markerscale = 5, loc = location,
                  bbox_to_anchor = anchor, bbox_transform = ax1.transAxes,
                  labelspacing = labspa, ncols = n_cols, columnspacing = 2.5)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    
    plt.show(fig)
    plt.close(fig)

data_list = [a_int_ele] # a_data
data_spec_array = [#[0.5, 'y', r'$a_{\text{LST}}$', 0.1, 0.25, 1],
                   [0.75, 'b', r'$a_{\text{PCA}}$', 1, 0.25, 1e4, r'$\sigma$']]
a_unit =r'[$\text{m}$]'
y_label = a_symb + " " + a_unit
ref_list, ref_spec_array, ref_y_spec_list = [], [], []
tit = r'semi-major axis from LST-data and numerical integration'
xlimits = [0, 0]
mean = np.mean(a_int_ele[:, 1])
std = np.std(a_int_ele[:, 1])
ylimits = [mean - 2 * std, mean + 2 * std]
grid = 1
save_specs = [0]
location, anchor = 1, (1, 1)
n_cols = 1
fosi = 12.5
labspa = 0.5
plot_data(data_list, data_spec_array, y_label,
          ref_list, ref_spec_array, ref_y_spec_list,
          tit, xlimits, ylimits, grid,
          location, anchor, n_cols, fosi, labspa,
          vline_list1, vline_list_specs1,
          vline_list2, vline_list_specs2,
          [0])
#%%
"""
t_list = a_int_acc[:, 0]

Δ_acc_ele = a_int_acc[:, 1] - a_int_ele[:, 1]
Δ_acc_shifted_ele = a_int_acc_shifted[:, 1] - a_int_ele[:, 1]
Δ_acc_acc_shifted = a_int_acc[:, 1] - a_int_acc_shifted[:, 1]

Δ_acc_ele = np.vstack((t_list, Δ_acc_ele)).T
Δ_acc_shifted_ele = np.vstack((t_list, Δ_acc_shifted_ele)).T
Δ_acc_acc_shifted = np.vstack((t_list, Δ_acc_acc_shifted)).T


data_list = [Δ_acc_ele, Δ_acc_shifted_ele, Δ_acc_acc_shifted]
data_spec_array = [[0.875, 'violet', r'$a_{\text{ACC}} - a_{\text{PCA}}$', 1],
                   [0.5, 'lightseagreen', r'$a_{\text{ACC, shifted}} - a_{\text{PCA}}$', 1],
                   [0.75, 'brown', r'$a_{\text{ACC}} - a_{\text{ACC, shifted}}$', 1]]
y_label = 'Δa' + " " + a_unit
ref_list, ref_spec_array, ref_y_spec_list = [], [], []
tit = None #r'differences'
xlimits = [0, 0]
ylimits = [-0.15, 0.25]
grid = 1
save_specs = [0]
vline_list2 = []
vline_list_specs2 = ['gold', 0.75, 1, "26.08"]
location, anchor = 2, (0, 1)
fosi = 15
labspa = 0.5
n_cols = 2
plot_data(data_list, data_spec_array, y_label,
          ref_list, ref_spec_array, ref_y_spec_list,
          tit, xlimits, ylimits, grid,
          location, anchor, n_cols, fosi, labspa,
          vline_list1, vline_list_specs1,
          vline_list2, vline_list_specs2,
          save_specs)
xlimits = [30,30]
ylimits = [-15e-3, 5e-4]
location, anchor = 1, (1, 1)
n_cols = 1
plot_data(data_list, data_spec_array, y_label,
          ref_list, ref_spec_array, ref_y_spec_list,
          tit, xlimits, ylimits, grid,
          location, anchor, n_cols, fosi, labspa,
          vline_list1, vline_list_specs1,
          vline_list2, vline_list_specs2,
          save_specs)
"""
#%%
fft_list_ele = fft(a_int_ele)
p_o_list_ele = fft_list_ele[0]
a_o_list_ele = fft_list_ele[1]

fig = plt.figure(figsize = (11, 5), dpi = 300)
plt.title("spectrum of integrated a", fontsize = 17.5)
plt.loglog(p_o_list_ele, a_o_list_ele,
           ls = 'solid', lw = 1, color = 'g',
           alpha = 1, label = r'$a_{\text{PCA}}$')
plt.xlabel("Period [d]", fontsize = 15)
plt.ylabel("Amplitude [m]", fontsize = 15)
plt.legend(fontsize = 12.5, labelspacing = 0.1,
           loc = 4)
"""
plt.axvline(3.28e-2, color = 'gold', alpha = 0.25,
            ls = (0, (5, 10)), lw = 1.5)
plt.axvline(1.64e-2, color = 'gold', alpha = 0.25,
            ls = (0, (5, 10)), lw = 1.5)
plt.axvline(6.56e-2, color = 'gold', alpha = 0.25,
            ls = (0, (5, 10)), lw = 1.5)
plt.xlim(0.01, 0.1)
plt.ylim(0.002, 0.1)
"""
plt.grid()
plt.show(fig)
plt.close(fig)
# %%
lcm_list_ele = [0, 50, 1, 0.066, 0.95, 10]
lcm_list_list = [lcm_list_ele]

interval_list_ele = [[0,0]]#[[0.02175,0.022]]

ele_spectrum_list = spectrum(a_int_ele, 3, lcm_list_ele, interval_list_ele)
ele_p_o, ele_a_o = ele_spectrum_list[0], ele_spectrum_list[1]
ele_per, ele_amp = ele_spectrum_list[2], ele_spectrum_list[3]

p_o_list = [ele_p_o]
a_o_list = [ele_a_o]
per_list = [ele_per]
amp_list = [ele_amp]

spectrum_p = p_o_list
spectrum_a = a_o_list
spectrum_data_specs = [[0.75, 'b', r'$a_{\text{PCA}}$']]
spectrum_per = per_list
spectrum_amp = amp_list
marker_spec_array = [[(5, -7.5), 50, "o", "dodgerblue",
                      "periods used for " + r'$\overline{a}_{\text{PCA}}$', r'$\text{PCA}$']]
Δt_list = []
Δt_spec_array = [[(0, (4, 4)), "r", 'a']]
spectrum_tit = "spectrum"
v_line_specs = ['b', 0.5, 1]
unit = r'[m]'
xlimits = [0, 0]
#xlimits = [0.01, 0.1]
ylimits = [0, 0]
#ylimits = [0.0001, 1]
#[0.0145, 0.0185], [0.019, 0.026], [0.045, 0.1]
v_line_list = []
log_fac_list = [5, 25]
location = 2
fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                   spectrum_per, spectrum_amp, marker_spec_array,
                   Δt_list, Δt_spec_array, spectrum_tit, unit, 1,
                   log_fac_list, location,
                   v_line_list, v_line_specs,
                   xlimits, ylimits, [0])
#%%
def dollar_remove(string):
    if (string[0] == '$'):
        return(string[1 : -1])
    else:
        return(string)

def fft_logplot_spec_3(p_o_list, a_o_list, data_spec_array,
                       per_list_list, amp_list_list, marker_spec_array,
                       Δt_list, Δt_spec_array, tit, unit, log,
                       log_fac_list, location, fosi, labspa, n_cols,
                       vlines_list, vlines_specs, xlimits, ylimits,
                       modus, save_specs):
    # p_o_list = [p_o_1, p_o_2, ...]
    # a_o_list = [a_o_1, a_o_2, ...]
    # data_spec_array = [alpha_i, col_i, lab_i]; alpha, color, label
    # per_list_list = [per_list_1, per_list_2, ...] for markers
    # amp_list_list = [amp_list_1, amp_list_2, ...] for markers
    # marker_spec_array = [pos_set, size, symb_i, col_i, leg_i, text_i];
    #                      position, size, symbol, color, label for legend, text label
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
    if (modus == 'd'): # day
        γ = 1
        p_lab = '[d]'
    elif (modus == 'h'): # hour
        γ = 24
        p_lab = '[h]'
    elif (modus == 'm'): # minute
        γ = 24 * 60
        p_lab = '[min]'
    elif (modus == 's'): # second
        γ = 24 * 60 * 60
        p_lab = '[s]'
    p = p_lab[1:-1]
    print("fft_logplot_spec_3 version 8")
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
            plt.semilogx(p_o_list[i]*γ, new_a_o_list[i],
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
            plt.loglog(p_o_list[i]*γ, new_a_o_list[i],
                       color = col, alpha = α,
                       ls = 'solid', lw = 1, label = lab)
    
    k = 0
    for i in range(0, n_marker):
        coords = marker_spec_array[i][0]
        size = marker_spec_array[i][1]
        mark = marker_spec_array[i][2]
        col = marker_spec_array[i][3]
        lab = marker_spec_array[i][4]
        text = marker_spec_array[i][5]
        plt.scatter(per_list_list[i]*γ, new_amp_list_list[i],
                    color = col, s = size,
                    marker = mark,
                    fc = 'None', lw = 1,
                    alpha = 1, label = lab)
        for j in range(0, len(per_list_list[i])):
            str_lab = dollar_remove(text)
            string = r'$p_{0}$'.replace('0', str(j + 1))
            plt.annotate(string, (per_list_list[i][j]*γ, new_amp_list_list[i][j]),
                         coords, textcoords = 'offset points',
                         fontsize = 15)
            string_add = ' = %.6e ' % (per_list_list[i][j]*γ)
            #plt.figtext(0.92, 0.875 - k, string + string_add,
            #            fontsize = 10, ha = 'left')
            print(string + string_add + p)
            k += 0.05
        k += 0.03
    
    for i in range(0, n_Δt):
        lab1 = r'$Δt_{xyz}$ = '.replace('xyz', dollar_remove(Δt_spec_array[i][2]*γ))
        lab2 = str(np.round(Δt_list[i], 3)) + ' ' + p
        plt.axvline(Δt_list[i]*γ, color = Δt_spec_array[i][1], alpha = 0.5,
                    ls = Δt_spec_array[i][0], lw = 5,
                    label = lab1 + lab2)
    
    for i in range(0, len(vlines_list)):
        vline = vlines_list[i]
        plt.axvline(vline*γ, color = vlines_specs[0],
                    alpha = vlines_specs[1], lw = vlines_specs[2],
                    ls = 'solid')
    
    if (xlimits[0] != xlimits[1]):
        plt.xlim(xlimits[0], xlimits[1])
    
    if (ylimits[0] != ylimits[1]):
        plt.ylim(ylimits[0], ylimits[1])
    
    plt.xlabel(r'period ' + p_lab, fontsize = 15)
    plt.ylabel(r'amplitude ' + unit, fontsize = 15)
    plt.legend(fontsize = fosi, loc = location,
               labelspacing = labspa, ncols = n_cols)
    plt.grid()
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)

lcm_list_el = [0, 50, 1, 0.09, 0.95, 10]
interval_list_el = [[0,0]]#[[1.75,2]]#for se1a

#n_days_tot_lst = 61
el_data = array_modifier(a_data, MJD_0_lst, n_days_tot_lst)

el_p_o_list = []
el_a_o_list = []
el_per_list = []
el_amp_list = []

# data_spectrum
el_spectrum_list = spectrum(el_data, -1, lcm_list_el, interval_list_el)
el_p_o, el_a_o = el_spectrum_list[0], el_spectrum_list[1]
el_per, el_amp = el_spectrum_list[2], el_spectrum_list[3]

el_p_o_list.append(el_p_o)
el_a_o_list.append(el_a_o)
el_per_list.append(el_per)
el_amp_list.append(el_amp)


spectrum_p = [el_p_o]
spectrum_a = [el_a_o]
spectrum_data_specs = [[1, "chocolate", el_symb]]
spectrum_per = [el_per]
spectrum_amp = [el_amp]
marker_spec_array = [[(5, -5), 50, "o", "k",
                      "peaks used for " + r'$a_{\text{Fit}}$', el_symb]]
#for se1a
#marker_spec_array = [[(-15, 0), 10, "o", "k",
#                      "peaks used for " + r'$a_{\text{Fit}}$', el_symb]]
Δt_list = []
Δt_spec_array = [[(0, (4, 4)), "r", el_symb]]
spectrum_tit = None
v_line_specs = ['b', 0.5, 1]
xlimits = [1, 1e5]
#xlimits = [0.01, 0.1]
ylimits = [0, 0]
ylimits = [1e-1, 1e4]
v_line_list = []#[0.019, 0.026, 0.05, 0.1, 5.55]
log_fac_list = []
location = 4
fosi = 17.5
labspa = 0.5
n_cols = 1
modus = 'm'
fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                   spectrum_per, spectrum_amp, marker_spec_array,
                   Δt_list, Δt_spec_array, spectrum_tit, "[m]", 1,
                   log_fac_list, location, fosi, labspa, n_cols,
                   v_line_list, v_line_specs,
                   xlimits, ylimits, modus,
                   [0, image_folder + 'Spectrum_a.png'])
#%%
print("len a_data: ", len(el_data))
print("timestep: ", (el_data[1,0] - el_data[0,0])*24*60*60)
print("n_days: ", (el_data[-1,0] - el_data[0,0]))
#%% downsample a_data
def down_sample(array, fac):
    new_array = np.zeros((1, len(array.T)))
    c = 0
    l = len(array)
    while (c < l):
        row = array[c]
        new_array = np.vstack((new_array, row))
        c += fac
    new_array = new_array[1:]
    return(new_array)

a_data_s = el_data #down_sample(el_data, 6)
print("len a_data_s: ", len(a_data_s))
print("timestep: ", (a_data_s[1,0] - a_data_s[0,0])*24*60*60)
print("n_days: ", (a_data_s[-1,0] - a_data_s[0,0]))
# %%
#6.912629386800108
configuration_list = [[10, 1, 1e3, 1],
                      [10, 1, 1e4, 1],
                      [10, 1, 1e5, 1]]
#configuration0a = "n_" + str(n_fac) + "/"
#configuration1 = "-n_" + str(n_fac) + "-R"
#configuration3 = "-w_%.0e.png" % ω
el_data_list = [a_data_s]
#n_days_tot_lst = 61

el_fit_list_list = []
el_para_fit_list = []
el_para_e_fit_list = []
el_m0_list = []
τ_fit_list_list = []
slope_gen_list_list = []
bar_list_list = []
x_list_list = []
Kxx_list = []
#%%
t_fit_a = tt.time()
for i in range(0, len(configuration_list)):
    configurations = configuration_list[i]
    n_fac = configurations[0]
    R = configurations[1]
    ω = configurations[2]
    constr = configurations[3]
    
    #print("periods: ", el_per[:R])
    
    n_partition = n_fac * n_days_tot_lst
    
    el_data = el_data_list[0]
    el_per = el_per_list[0]
    el_amp = el_amp_list[0]
    #el_per = [el_per_list[i][0], 4.75, 5.55]
    #el_amp = [el_amp_list[i][0], 5, 5]
    
    # fit
    t1 = tt.time()
    el_fit_list = fitter(el_data, el_per[:R], el_amp[:R],
                         n_partition, 5, ω, constr)
    t2 = tt.time()
    print("time: %.3f" % (t2 - t1))
    el_fit = el_fit_list[0]
    el_para_fit, el_para_e_fit = el_fit_list[1], el_fit_list[2]
    el_m0, τ_fit_list = el_fit_list[3], el_fit_list[4]
    slope_gen_list = el_fit_list[5]
    bar_list = el_fit_list[6]
    x_list = el_fit_list[7]
    Kxx = el_fit_list[8]
    
    el_fit_list_list.append(el_fit)
    el_para_fit_list.append(el_para_fit)
    el_para_e_fit_list.append(el_para_e_fit)
    el_m0_list.append(el_m0)
    τ_fit_list_list.append(τ_fit_list)
    slope_gen_list_list.append(slope_gen_list)
    bar_list_list.append(bar_list)
    x_list_list.append(x_list)
    Kxx_list.append(Kxx)
t_fit_b = tt.time()
print("Time: ", t_fit_b - t_fit_a)
print("DONE!!!")
#%%
##################################################
##################################################
#################### FAILSAFE ####################
##################################################
##################################################
folder_collection = 'big_data_collection/FINAL/' + satshort + yearshort + '/'
run = 1
add = folder_collection + 'run_' + str(run) + '-'
""" # set "#" before
# TO SAVE DATA
for i in range(0, len(configuration_list)):
    configurations = configuration_list[i]
    n_fac = configurations[0]
    R = configurations[1]
    ω = configurations[2]
    constr = configurations[3]
    config = 'n_%d-R_%d-w_%.1E' % (n_fac, R, ω)
    
    fit_name = add + 'fit-' + config + '.txt'
    para_name = add + 'para-' + config + '.txt'
    para_e_name = add + 'para_e-' + config + '.txt'
    m0_name = add + 'm0-' + config + '.txt'
    τ_fit_name = add + 'tau_fit-' + config + '.txt'
    
    slope_tau_name = add + 'slope_tau-' + config + '.txt'
    slope_deltatau_name = add + 'slope_deltatau-' + config + '.txt'
    slope_a_bar_name = add + 'slope_a_bar-' + config + '.txt'
    slope_sigma2_a_bar_name = add + 'slope_sigma2_a_bar-' + config + '.txt'
    
    bar_name = add + 'bar-' + config + '.txt'
    
    x_list_name = add + 'x_list-' + config + '.txt'
    Kxx_name = add + 'kxx-' + config + '.txt'
    
    
    fit = el_fit_list_list[i]
    para = el_para_fit_list[i]
    para_e = el_para_e_fit_list[i]
    m0 = el_m0_list[i]
    τ_fit = τ_fit_list_list[i]
    
    slope = slope_gen_list_list[i]
    slope_tau = slope[0]
    slope_deltatau = slope[1]
    slope_a_bar = slope[2]
    slope_sigma2_a_bar = slope[3]
    
    bar = bar_list_list[i]    
    
    x_list = x_list_list[i]
    Kxx = Kxx_list[i]
    
    np.savetxt(fit_name, fit)
    np.savetxt(para_name, para)
    np.savetxt(para_e_name, para_e)
    np.savetxt(m0_name, np.array([m0]))
    np.savetxt(τ_fit_name, τ_fit)
    
    np.savetxt(slope_tau_name, slope_tau)
    np.savetxt(slope_deltatau_name, np.array([slope_deltatau]))
    np.savetxt(slope_a_bar_name, slope_a_bar)
    np.savetxt(slope_sigma2_a_bar_name, slope_sigma2_a_bar)
    
    np.savetxt(bar_name, bar)
    
    np.savetxt(x_list_name, x_list)
    np.savetxt(Kxx_name, Kxx)
""" # set "#" before
#""" # set "#" before
# TO LOAD DATA
for i in range(0, len(configuration_list)):
    configurations = configuration_list[i]
    n_fac = configurations[0]
    R = configurations[1]
    ω = configurations[2]
    constr = configurations[3]
    config = 'n_%d-R_%d-w_%.1E' % (n_fac, R, ω)
    
    
    fit_name = add + 'fit-' + config + '.txt'
    para_name = add + 'para-' + config + '.txt'
    para_e_name = add + 'para_e-' + config + '.txt'
    m0_name = add + 'm0-' + config + '.txt'
    τ_fit_name = add + 'tau_fit-' + config + '.txt'
    
    slope_tau_name = add + 'slope_tau-' + config + '.txt'
    slope_deltatau_name = add + 'slope_deltatau-' + config + '.txt'
    slope_a_bar_name = add + 'slope_a_bar-' + config + '.txt'
    slope_sigma2_a_bar_name = add + 'slope_sigma2_a_bar-' + config + '.txt'
    
    bar_name = add + 'bar-' + config + '.txt'
    
    x_list_name = add + 'x_list-' + config + '.txt'
    Kxx_name = add + 'kxx-' + config + '.txt'
    
    
    fit = np.loadtxt(fit_name)
    para = np.loadtxt(para_name)
    para_e = np.loadtxt(para_e_name)
    m0 = np.loadtxt(m0_name)
    τ_fit = np.loadtxt(τ_fit_name)
    
    slope_tau = np.loadtxt(slope_tau_name)
    slope_deltatau = np.loadtxt(slope_deltatau_name)
    slope_a_bar = np.loadtxt(slope_a_bar_name)
    slope_sigma2_a_bar = np.loadtxt(slope_sigma2_a_bar_name)
    slope = [slope_tau, slope_deltatau, slope_a_bar, slope_sigma2_a_bar]
    
    bar = np.loadtxt(bar_name)
    bar_error = np.sqrt(bar[:, 2])
    bar_wo_error = bar[:, : 2]
    bar = np.vstack((bar_wo_error.T, bar_error)).T
    
    #x_list = np.loadtxt(x_list_name)
    #Kxx = np.loadtxt(Kxx_name)
    
    
    el_fit_list_list.append(fit)
    el_para_fit_list.append(para)
    el_para_e_fit_list.append(para_e)
    el_m0_list.append(m0)
    τ_fit_list_list.append(τ_fit)
    slope_gen_list_list.append(slope)
    bar_list_list.append(bar)
    #x_list_list.append(x_list)
    #Kxx_list.append(Kxx)
#""" # set "#" before
##################################################
##################################################
##################################################
##################################################
##################################################
#%% PLOT LST, ELE AND ACC
int_list = [a_int_ele]

y_0 = 0
for i in range(0, len(bar_list_list)):
    bar_list = bar_list_list[i]
    bar_list = array_modifier(bar_list, MJD_0_int, 1)
    y_0 += bar_list[0, 1] / len(bar_list_list)

new_int_list = []
for i in range(0, len(int_list)):
    int_i = int_list[i]
    y_i = int_i[0, 1]
    Δy = y_0 - y_i
    
    int_1 = int_i[:, 0]
    int_2 = int_i[:, 1]
    int_3 = int_i[:, 2]
    new_int = np.vstack((int_1, int_2 + Δy, int_3)).T
    
    new_int_list.append(new_int)

data_list = bar_list_list + new_int_list

from functions import cl_lin
red1 = np.linspace(1,0,128)
green1 = np.zeros(128)
blue1 = np.linspace(0,1,128)
red2 = np.zeros(128)
green2 = np.linspace(0,1,128)
blue2 = np.linspace(1,0,128)

clist1 = np.vstack((red1, green1, blue1)).T
clist2 = np.vstack((red2, green2, blue2)).T
clist3 = np.vstack((clist1, clist2))

cols = []
for i in clist3:
    cols.append(tuple(i))
my_cmap = ListedColormap(cols)

col_list = ['r', 'b', 'g']

data_spec_array = []

color_list = cl_lin(np.arange(3), my_cmap)

for i in range(0, len(configuration_list)):
    len_con = len(configuration_list)
    configuration = configuration_list[i]
    n_fac = configuration[0]
    R = configuration[1]
    ω = configuration[2]
    print("ω = ",ω)
    constr = configuration[3]
    if (len_con > 1):
        col = col_list[i]
    else:
        col = 'y'
    lab = r'$\bar{a}_{\psi_{%d}}$' % (i + 1)
    e_lab = r'$\sigma_{\psi_{%d}}$' % (i + 1)
    e_fac_list = [5e0, 5e0, 5e0]
    e_alpha_list = [0.4, 0.4, 0.4]
    lw_list = [7.5, 5, 2.5]
    specs = [1, col, lab, 1.5, e_alpha_list[i], e_fac_list[i], e_lab]
    data_spec_array.append(specs)

data_spec_array.append([1, 'violet', r'$a_{\text{PCA}}$', 1.5, 0.5, 1e2, r'$\sigma_{\text{PCA}}$'])


y_label = a_symb + " " + a_unit
ref_list, ref_spec_array, ref_y_spec_list = [], [], []
tit = None

mean = np.mean(a_int_ele[:, 1])
std = np.std(a_int_ele[:, 1])
#ylimits = [mean - 2 * std, mean + 2 * std]
#ylimits = [mean - 89.5 * std, mean - 84.5 * std]

xlimits = [0,0]
#xlimits = [32.5, 8.5]
#xlimits = [8, 33]
#xlimits = [8, 1]
ylimits = [7177920,7177960]
#ylimits = [6864800,6865100]
grid = 1
save_specs = [0]
w_period = 4.75
#w_period = 5.55
#vline_list2 = np.arange(13) * np.ones(13) * w_period + 58345.25
#vline_list2 = np.arange(11) * np.ones(11) * w_period + 58342.5
location, anchor = 3, (0, 0.)
n_cols = 1
fosi = 17.5
labspa = 0.25
vline_list_specs1 = ['goldenrod', 1, 2.5, "CME", (0, (4, 12)), 2]
vline_list_specs2 = ['saddlebrown', 1, 1.5, "MAN", (-6, (2, 4,2,8)), 2]

def plot_data(data_list, data_spec_array, y_label,
              ref_list, ref_spec_array, ref_y_spec_list,
              tit, xlimits, ylimits, grid,
              location, anchor, n_cols, fosi, labspa,
              vline_list1, vline_list_specs1,
              vline_list2, vline_list_specs2,
              save_specs):
    """ description of input parameters
    data_list        : [data_1, data_2, ...]
    data_spec_array  : [alpha_i, col_i, lab_i, lw_i]
    y_label          : label for y-axis
    ref_list         : [ref_1, ref_2, ...]
    ref_spec_array   : [alpha_i, col_i, lab_i, lw_i]
    ref_y_spec_list  : [y_label, y_axis_col]
    tit              : title, string like
    xlimits          : zoom values
    ylimits          : limit values
    grid             : if 1 --> grid to data
    vline_list1      : [vline_i]
    vline_list_specs1: [col_i, alpha_i, lw_i, label]
    vline_list2      : [vline_i]
    vline_list_specs2: [col_i, alpha_i, lw_i, label]
    save_specs       : [1, path] # 0,1 for saving
    """
    t_min, t_max = t_min_t_max(data_list)
    xstart = t_min + xlimits[0]
    xend = t_max - xlimits[1]
    n_days = xend - xstart
    new_data_list = 0
    new_ref_list = 0
    if (xlimits[0] != xlimits[1]):
        new_data_list = []
        new_ref_list = []
        for i in range(0, len(data_list)):
            data_i = data_list[i]
            data_i = array_modifier(data_i, xstart, n_days)
            new_data_list.append(data_i)
        for i in range(0, len(ref_list)):
            ref_i = ref_list[i]
            ref_i = array_modifier(ref_i, xstart, n_days)
            new_ref_list.append(ref_i)
    else:
        new_data_list = data_list
        new_ref_list = ref_list
    
    fig, ax1 = plt.subplots(figsize = (10, 5), dpi = 300)
    fig.suptitle(tit, fontsize = 17.5)
    
    for i in range(0, len(new_data_list)):
        x_data = new_data_list[i][:, 0]
        y_data = new_data_list[i][:, 1]
        α = data_spec_array[i][0]
        col = data_spec_array[i][1]
        lab = data_spec_array[i][2]
        width = data_spec_array[i][3]
        if (len(new_data_list[i].T) == 3):
            # plot with errors
            s_fac = data_spec_array[i][5]
            s_y = new_data_list[i][:, 2] * s_fac
            e_α = data_spec_array[i][4]
            e_lab = data_spec_array[i][6]
            lab_fac = scientific_to_exponent(s_fac, 1)
            lab = lab + r' (' + e_lab + r' $\times$ ' + lab_fac + ')'
            ax1.fill_between(x_data, y_data - s_y, y_data + s_y,
                             color = col, alpha = e_α, zorder = 4)
        ax1.plot(x_data, y_data, color = col, ls = '-', lw = width,
                 alpha = α, zorder = 5, label = lab)
    ax1.xaxis.set_major_formatter(mjd_to_mmdd)
    ax1.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
    ax1.tick_params(axis = 'x', which = 'major',
                    width = 2, length = 8,
                    direction = 'out', top = False)
    ax1.tick_params(axis = 'x', which = 'minor',
                    width = 1, length = 4,
                    direction = 'out', top = False)
    ax1.get_yaxis().get_major_formatter().set_useMathText(True)
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    ax1.set_ylabel(y_label, fontsize = 15)
    y_low, y_upp = ylimits[0], ylimits[1]
    if (xstart != xend):
        ax1.set_xlim(xstart, xend)
    if (y_low != y_upp):
        ax1.set_ylim(y_low, y_upp)
    if (grid == 1):
        ax1.grid()
    
    if (len(ref_array_list) != 0):
        ax2 = ax1.twinx()
        for i in range(0, len(new_ref_list)):
            x_ref = new_ref_list[i][:, 0]
            y_ref = new_ref_list[i][:, 1]
            α = ref_spec_array[i][0]
            col = ref_spec_array[i][1]
            lab = ref_spec_array[i][2]
            width = ref_spec_array[i][3]
            ax2.plot(x_ref, y_ref,
                     ls = '-', lw = width, color = col, alpha = α,
                     label = lab)
        ax2.set_ylabel(ref_y_spec_list[0], color = ref_y_spec_list[1], fontsize = 15)
        ax2.tick_params(axis = 'y', labelcolor = ref_y_spec_list[1])
        
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.set_frame_on(False)
    
    c = 0 # for giving only one label to vlines from vline_list1
    for i in range(0, len(vline_list1)):
        vline = vline_list1[i]
        col = vline_list_specs1[0]
        alph = vline_list_specs1[1]
        liwi = vline_list_specs1[2]
        linst = vline_list_specs1[4]
        zo = 5 + vline_list_specs1[5]
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs1[3]
                c += 1
            plt.axvline(vline, color = col,
                        alpha = alph, lw = liwi,
                        ls = linst, zorder = zo,
                        label = c_label)
    
    c = 0 # for giving only one label to vlines from vline_list2
    for i in range(0, len(vline_list2)):
        vline = vline_list2[i]
        col = vline_list_specs2[0]
        alph = vline_list_specs2[1]
        liwi = vline_list_specs2[2]
        linst = vline_list_specs2[4]
        zo = 5 + vline_list_specs2[5]
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs2[3]
                c += 1
            plt.axvline(vline, color = col,
                        alpha = alph, lw = liwi,
                        ls = linst, zorder = zo,
                        label = c_label)
    
    plt.figlegend(fontsize = fosi, markerscale = 5, loc = location,
                  bbox_to_anchor = anchor, bbox_transform = ax1.transAxes,
                  labelspacing = labspa, ncols = n_cols, columnspacing = 0)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    
    plt.show(fig)
    plt.close(fig)

def plot_data_new(data_list, data_spec_array, y_label,
                  ref_list, ref_spec_array, ref_y_spec_list,
                  tit, xlimits, ylimits, grid,
                  location, anchor, n_cols, fosi, labspa,
                  vline_list1, vline_list_specs1,
                  vline_list2, vline_list_specs2,
                  rectangle, save_specs):
    """ description of input parameters
    data_list        : [data_1, data_2, ...]
    data_spec_array  : [alpha_i, col_i, lab_i, lw_i]
    y_label          : label for y-axis
    ref_list         : [ref_1, ref_2, ...]
    ref_spec_array   : [alpha_i, col_i, lab_i, lw_i]
    ref_y_spec_list  : [y_label, y_axis_col]
    tit              : title, string like
    xlimits          : zoom values
    ylimits          : limit values
    grid             : if 1 --> grid to data
    vline_list1      : [vline_i]
    vline_list_specs1: [col_i, alpha_i, lw_i, label]
    vline_list2      : [vline_i]
    vline_list_specs2: [col_i, alpha_i, lw_i, label]
    save_specs       : [1, path] # 0,1 for saving
    """
    t_min, t_max = t_min_t_max(data_list)
    xstart = t_min + xlimits[0]
    xend = t_max - xlimits[1]
    n_days = xend - xstart
    new_data_list = 0
    new_ref_list = 0
    if (xlimits[0] != xlimits[1]):
        new_data_list = []
        new_ref_list = []
        for i in range(0, len(data_list)):
            data_i = data_list[i]
            data_i = array_modifier(data_i, xstart, n_days)
            new_data_list.append(data_i)
        for i in range(0, len(ref_list)):
            ref_i = ref_list[i]
            ref_i = array_modifier(ref_i, xstart, n_days)
            new_ref_list.append(ref_i)
    else:
        new_data_list = data_list
        new_ref_list = ref_list
    
    fig, ax1 = plt.subplots(figsize = (15, 5), dpi = 300)
    fig.suptitle(tit, fontsize = 17.5)
    
    for i in range(0, len(new_data_list)):
        x_data = new_data_list[i][:, 0]
        y_data = new_data_list[i][:, 1]
        α = data_spec_array[i][0]
        col = data_spec_array[i][1]
        lab = data_spec_array[i][2]
        width = data_spec_array[i][3]
        if (len(new_data_list[i].T) == 3):
            # plot with errors
            s_fac = data_spec_array[i][5]
            s_y = new_data_list[i][:, 2] * s_fac
            e_α = data_spec_array[i][4]
            e_lab = data_spec_array[i][6]
            lab_fac = scientific_to_exponent(s_fac, 1)
            lab = lab + r' (' + e_lab + r'$\times$' + lab_fac + ')'
            ax1.fill_between(x_data, y_data - s_y, y_data + s_y,
                             color = col, alpha = e_α, zorder = 4)
        ax1.plot(x_data, y_data, color = col, ls = '-', lw = width,
                 alpha = α, zorder = 5, label = lab)
    

    rec_x_a, rec_x_b = rectangle[0][0], rectangle[1][0]
    rec_y_a, rec_y_b = rectangle[0][1], rectangle[1][1]
    rectangle_x = np.linspace(rec_x_a, rec_x_b, 100)
    rectangle_y = np.linspace(rec_y_a, rec_y_b, 100)
    rec_list = np.ones(100)
    rec_col = rectangle[2]
    rec_width = rectangle[3]
    rec_alpha = rectangle[4]

    ax1.plot(rectangle_x, rec_y_a * rec_list,
             ls = 'solid', lw = rec_width,
             color = rec_col, alpha = rec_alpha,
             zorder = 10)
    ax1.plot(rectangle_x, rec_y_b * rec_list,
             ls = 'solid', lw = rec_width,
             color = rec_col, alpha = rec_alpha,
             zorder = 10)
    ax1.plot(rec_x_a * rec_list, rectangle_y,
             ls = 'solid', lw = rec_width,
             color = rec_col, alpha = rec_alpha,
             zorder = 10)
    ax1.plot(rec_x_b * rec_list, rectangle_y,
             ls = 'solid', lw = rec_width,
             color = rec_col, alpha = rec_alpha,
             zorder = 10)

    ax1.xaxis.set_major_formatter(mjd_to_mmdd)
    ax1.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
    ax1.tick_params(axis = 'x', which = 'major',
                    width = 2, length = 8,
                    direction = 'out', top = False)
    ax1.tick_params(axis = 'x', which = 'minor',
                    width = 1, length = 4,
                    direction = 'out', top = False)
    ax1.get_yaxis().get_major_formatter().set_useMathText(True)
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    ax1.set_ylabel(y_label, fontsize = 15)
    y_low, y_upp = ylimits[0], ylimits[1]
    if (xstart != xend):
        ax1.set_xlim(xstart, xend)
    if (y_low != y_upp):
        ax1.set_ylim(y_low, y_upp)
    if (grid == 1):
        ax1.grid()
    
    if (len(ref_array_list) != 0):
        ax2 = ax1.twinx()
        for i in range(0, len(new_ref_list)):
            x_ref = new_ref_list[i][:, 0]
            y_ref = new_ref_list[i][:, 1]
            α = ref_spec_array[i][0]
            col = ref_spec_array[i][1]
            lab = ref_spec_array[i][2]
            width = ref_spec_array[i][3]
            ax2.plot(x_ref, y_ref,
                     ls = '-', lw = width, color = col, alpha = α,
                     label = lab)
        ax2.set_ylabel(ref_y_spec_list[0], color = ref_y_spec_list[1], fontsize = 15)
        ax2.tick_params(axis = 'y', labelcolor = ref_y_spec_list[1])
        
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.set_frame_on(False)
    
    c = 0 # for giving only one label to vlines from vline_list1
    for i in range(0, len(vline_list1)):
        vline = vline_list1[i]
        col = vline_list_specs1[0]
        alph = vline_list_specs1[1]
        liwi = vline_list_specs1[2]
        linst = vline_list_specs1[4]
        zo = 5 + vline_list_specs1[5]
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs1[3]
                c += 1
            plt.axvline(vline, color = col,
                        alpha = alph, lw = liwi,
                        ls = linst, zorder = zo,
                        label = c_label)
    
    c = 0 # for giving only one label to vlines from vline_list2
    for i in range(0, len(vline_list2)):
        vline = vline_list2[i]
        col = vline_list_specs2[0]
        alph = vline_list_specs2[1]
        liwi = vline_list_specs2[2]
        linst = vline_list_specs2[4]
        zo = 5 + vline_list_specs2[5]
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs2[3]
                c += 1
            plt.axvline(vline, color = col,
                        alpha = alph, lw = liwi,
                        ls = linst, zorder = zo,
                        label = c_label)
    
    plt.figlegend(fontsize = fosi, markerscale = 5, loc = location,
                  bbox_to_anchor = anchor, bbox_transform = ax1.transAxes,
                  labelspacing = labspa, ncols = n_cols, columnspacing = 1)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    
    plt.show(fig)
    plt.close(fig)

def plot_data_rec(data_list, data_spec_array, y_label,
                  ref_list, ref_spec_array, ref_y_spec_list,
                  tit, xlimits, ylimits, grid,
                  location, anchor, n_cols, fosi, labspa,
                  vline_list1, vline_list_specs1,
                  vline_list2, vline_list_specs2,
                  rectangle, save_specs):
    """ description of input parameters
    data_list        : [data_1, data_2, ...]
    data_spec_array  : [alpha_i, col_i, lab_i, lw_i]
    y_label          : label for y-axis
    ref_list         : [ref_1, ref_2, ...]
    ref_spec_array   : [alpha_i, col_i, lab_i, lw_i]
    ref_y_spec_list  : [y_label, y_axis_col]
    tit              : title, string like
    xlimits          : zoom values
    ylimits          : limit values
    grid             : if 1 --> grid to data
    vline_list1      : [vline_i]
    vline_list_specs1: [col_i, alpha_i, lw_i, label]
    vline_list2      : [vline_i]
    vline_list_specs2: [col_i, alpha_i, lw_i, label]
    save_specs       : [1, path] # 0,1 for saving
    """
    t_min, t_max = t_min_t_max(data_list)
    xstart = t_min + xlimits[0]
    xend = t_max - xlimits[1]
    n_days = xend - xstart
    new_data_list = 0
    new_ref_list = 0
    if (xlimits[0] != xlimits[1]):
        new_data_list = []
        new_ref_list = []
        for i in range(0, len(data_list)):
            data_i = data_list[i]
            data_i = array_modifier(data_i, xstart, n_days)
            new_data_list.append(data_i)
        for i in range(0, len(ref_list)):
            ref_i = ref_list[i]
            ref_i = array_modifier(ref_i, xstart, n_days)
            new_ref_list.append(ref_i)
    else:
        new_data_list = data_list
        new_ref_list = ref_list
    
    fig, ax1 = plt.subplots(figsize = (15, 5), dpi = 300)
    fig.suptitle(tit, fontsize = 17.5)
    
    for i in range(0, len(new_data_list)):
        x_data = new_data_list[i][:, 0]
        y_data = new_data_list[i][:, 1]
        α = data_spec_array[i][0]
        col = data_spec_array[i][1]
        lab = data_spec_array[i][2]
        width = data_spec_array[i][3]
        if (len(new_data_list[i].T) == 3):
            # plot with errors
            s_fac = data_spec_array[i][5]
            s_y = new_data_list[i][:, 2] * s_fac
            e_α = data_spec_array[i][4]
            e_lab = data_spec_array[i][6]
            lab_fac = scientific_to_exponent(s_fac, 1)
            lab = lab + r' (' + e_lab + r'$\times$' + lab_fac + ')'
            ax1.fill_between(x_data, y_data - s_y, y_data + s_y,
                             color = col, alpha = e_α, zorder = 4)
        ax1.plot(x_data, y_data, color = col, ls = '-', lw = width,
                 alpha = α, zorder = 5, label = lab)
    

    rec_x_a, rec_x_b = rectangle[0][0], rectangle[1][0]
    rec_y_a, rec_y_b = rectangle[0][1], rectangle[1][1]
    rectangle_x = np.linspace(rec_x_a, rec_x_b, 100)
    rectangle_y = np.linspace(rec_y_a, rec_y_b, 100)
    rec_list = np.ones(100)
    rec_col = rectangle[2]
    rec_width = rectangle[3]
    rec_alpha = rectangle[4]

    ax1.plot(rectangle_x, rec_y_a * rec_list,
             ls = 'solid', lw = rec_width,
             color = rec_col, alpha = rec_alpha,
             zorder = 10)
    ax1.plot(rectangle_x, rec_y_b * rec_list,
             ls = 'solid', lw = rec_width,
             color = rec_col, alpha = rec_alpha,
             zorder = 10)
    ax1.plot(rec_x_a * rec_list, rectangle_y,
             ls = 'solid', lw = rec_width,
             color = rec_col, alpha = rec_alpha,
             zorder = 10)
    ax1.plot(rec_x_b * rec_list, rectangle_y,
             ls = 'solid', lw = rec_width,
             color = rec_col, alpha = rec_alpha,
             zorder = 10)

    ax1.xaxis.set_major_formatter(mjd_to_mmdd)
    ax1.xaxis.set_major_locator(mat.ticker.MultipleLocator(1))
    ax1.xaxis.set_minor_locator(mat.ticker.MultipleLocator(0.5))
    ax1.tick_params(axis = 'x', which = 'major',
                    width = 2, length = 8,
                    direction = 'out', top = False)
    ax1.tick_params(axis = 'x', which = 'minor',
                    width = 1, length = 4,
                    direction = 'out', top = False)
    ax1.get_yaxis().get_major_formatter().set_useMathText(True)
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    ax1.set_ylabel(y_label, fontsize = 15)
    y_low, y_upp = ylimits[0], ylimits[1]
    if (xstart != xend):
        ax1.set_xlim(xstart, xend)
    if (y_low != y_upp):
        ax1.set_ylim(y_low, y_upp)
    if (grid == 1):
        ax1.grid()
    ax1.set_xlim(rectangle[0][0], rectangle[1][0])
    ax1.set_ylim(rectangle[0][1], rectangle[1][1])
    
    if (len(ref_array_list) != 0):
        ax2 = ax1.twinx()
        for i in range(0, len(new_ref_list)):
            x_ref = new_ref_list[i][:, 0]
            y_ref = new_ref_list[i][:, 1]
            α = ref_spec_array[i][0]
            col = ref_spec_array[i][1]
            lab = ref_spec_array[i][2]
            width = ref_spec_array[i][3]
            ax2.plot(x_ref, y_ref,
                     ls = '-', lw = width, color = col, alpha = α,
                     label = lab)
        ax2.set_ylabel(ref_y_spec_list[0], color = ref_y_spec_list[1], fontsize = 15)
        ax2.tick_params(axis = 'y', labelcolor = ref_y_spec_list[1])
        
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.set_frame_on(False)
    
    c = 0 # for giving only one label to vlines from vline_list1
    for i in range(0, len(vline_list1)):
        vline = vline_list1[i]
        col = vline_list_specs1[0]
        alph = vline_list_specs1[1]
        liwi = vline_list_specs1[2]
        linst = vline_list_specs1[4]
        zo = 5 + vline_list_specs1[5]
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs1[3]
                c += 1
            plt.axvline(vline, color = col,
                        alpha = alph, lw = liwi,
                        ls = linst, zorder = zo,
                        label = c_label)
    
    c = 0 # for giving only one label to vlines from vline_list2
    for i in range(0, len(vline_list2)):
        vline = vline_list2[i]
        col = vline_list_specs2[0]
        alph = vline_list_specs2[1]
        liwi = vline_list_specs2[2]
        linst = vline_list_specs2[4]
        zo = 5 + vline_list_specs2[5]
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs2[3]
                c += 1
            plt.axvline(vline, color = col,
                        alpha = alph, lw = liwi,
                        ls = linst, zorder = zo,
                        label = c_label)
    
    plt.figlegend(fontsize = fosi, markerscale = 5, loc = location,
                  bbox_to_anchor = anchor, bbox_transform = ax1.transAxes,
                  labelspacing = labspa, ncols = n_cols, columnspacing = 1)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    
    plt.show(fig)
    plt.close(fig)

#rectangle = [[MJD_0_lst+33,7070995],[MJD_0_lst+37,7071010],'k',1,1]
#rectangle = [[MJD_0_lst+17.5,7070955],[MJD_0_lst+20,7070970],'k',2,1]#this one
#rectangle = [[MJD_0_lst+19.5,7070990],[MJD_0_lst+22.5,7071005],'k',1,1]

plot_data(data_list, data_spec_array, y_label,
          ref_list, ref_spec_array, ref_y_spec_list,
          tit, xlimits, ylimits, grid,
          location, anchor, n_cols, fosi, labspa,
          vline_list1, vline_list_specs1,
          vline_list2, vline_list_specs2,
          [0, image_folder + 'Fit_Int.jpg'])
"""
plot_data_new(data_list, data_spec_array, y_label,
              ref_list, ref_spec_array, ref_y_spec_list,
              tit, xlimits, ylimits, grid,
              location, anchor, n_cols, fosi, labspa,
              vline_list1, vline_list_specs1,
              vline_list2, vline_list_specs2,
              rectangle, [0, image_folder + 'Fit_Int.jpg'])
plot_data_rec(data_list, data_spec_array, y_label,
              ref_list, ref_spec_array, ref_y_spec_list,
              tit, xlimits, ylimits, grid,
              1, (1,1), 3, fosi, labspa,
              vline_list1, vline_list_specs1,
              vline_list2, vline_list_specs2,
              rectangle, [0, image_folder + 'Fit_Int_z.jpg'])
"""
#%%
n_s = n_fac * n_days_tot_lst
τ_fit_list = τ_fit_list_list[0]

μ_r_list_1 = x_list_list[0][1 * (n_s + 1) : (1 + 1) * (n_s + 1)]
η_r_list_1 = x_list_list[0][(1 + 1) * (n_s + 1) : (1 + 1 + 1) * (n_s + 1)]

μ_r_list_2 = x_list_list[1][1 * (n_s + 1) : (1 + 1) * (n_s + 1)]
η_r_list_2 = x_list_list[1][(1 + 1) * (n_s + 1) : (1 + 1 + 1) * (n_s + 1)]

μ_r_list_3 = x_list_list[2][1 * (n_s + 1) : (1 + 1) * (n_s + 1)]
η_r_list_3 = x_list_list[2][(1 + 1) * (n_s + 1) : (1 + 1 + 1) * (n_s + 1)]

fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.plot(τ_fit_list, μ_r_list_3 - μ_r_list_1,
         'r-', lw = 1, alpha = 0.25,
         label = None)
plt.plot(τ_fit_list, μ_r_list_3 - μ_r_list_2,
         'b-', lw = 1, alpha = 0.25,
         label = None)
plt.plot(τ_fit_list, μ_r_list_3 - μ_r_list_1,
         'r.', ms = 1.5,
         label = 'μ_1(ψ_3) - μ_1(ψ_1)')
plt.plot(τ_fit_list, μ_r_list_3 - μ_r_list_2,
         'b.', ms = 1.5,
         label = 'μ_1(ψ_3) - μ_1(ψ_2)')
#plt.ylim(-5,5)

plt.grid()
plt.legend()
plt.show(fig)
plt.close(fig)

fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.plot(τ_fit_list, η_r_list_3 - η_r_list_1,
         'r-', lw = 1, alpha = 0.25,
         label = None)
plt.plot(τ_fit_list, η_r_list_3 - η_r_list_2,
         'b-', lw = 1, alpha = 0.25,
         label = None)
plt.plot(τ_fit_list, η_r_list_3 - η_r_list_1,
         'r.', ms = 1.5,
         label = 'η_1(ψ_3) - η_1(ψ_1)')
plt.plot(τ_fit_list, η_r_list_3 - η_r_list_2,
         'b.', ms = 1.5,
         label = 'η_1(ψ_3) - η_1(ψ_2)')
#plt.ylim(-5,5)

plt.grid()
plt.legend()
plt.show(fig)
plt.close(fig)

for k in range(0, len(configuration_list)):
    configurations = configuration_list[k]
    n_fac = configurations[0]
    R = configurations[1]
    ω = configurations[2]
    constr = configurations[3]
    
    Kxx = Kxx_list[k]
    
    def format_coord(x, y):
        col = round(x)
        row = round(y)
        nrows, ncols = Kxx.shape
        if 0 <= col < ncols and 0 <= row < nrows:
            z = Kxx[row, col]
            return f'x={x:1.4f}, y={y:1.4f}, z={z:1.4f}'
        else:
            return f'x={x:1.4f}, y={y:1.4f}'
    
    fig, ax = plt.subplots(dpi = 300)
    ax.imshow(np.log(Kxx), cmap = mat.colormaps['gist_rainbow'])
    ax.format_coord = format_coord
    plt.show(fig)
    plt.close(fig)
    print(np.max(Kxx))
    print(np.min(Kxx))
# %%
el_fit_p_o_list = []
el_fit_a_o_list = []
el_fit_per_list = []
el_fit_amp_list = []
lcm_list_fit = [0, 50, 1, 0.5, 0.95, 10]
for i in range(0, len(bar_list_list)):#len(data_both)
    el_fit = bar_list_list[i]
    # data_spectrum
    el_fit_spectrum_list = spectrum(el_fit, 0, lcm_list_fit, [[0, 0]])
    el_fit_p_o, el_fit_a_o = el_fit_spectrum_list[0], el_fit_spectrum_list[1]
    el_fit_per, el_fit_amp = el_fit_spectrum_list[2], el_fit_spectrum_list[3]
    
    el_fit_p_o_list.append(el_fit_p_o)
    el_fit_a_o_list.append(el_fit_a_o)
    el_fit_per_list.append(el_fit_per)
    el_fit_amp_list.append(el_fit_amp)
# manually delete 
#%%
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

col_list = ['r', 'b', 'g', 'violet'] #cl_lin(np.arange(3), my_cmap)
# spectrum
#spectrum_p = [el_fit_p_o_list[0]] + p_o_list
#spectrum_a = [el_fit_a_o_list[0]] + a_o_list

spectrum_p = el_fit_p_o_list + p_o_list
spectrum_a = el_fit_a_o_list + a_o_list

spectrum_data_specs = []
marker_spec_array = []

for i in range(0, len(configuration_list)):
    conf = configuration_list[i]
    specs = [0.75, col_list[i],r'$\bar{a}_{\psi_{%d}}$' % (i+1)]
    m_specs = [(2.5, -10), 50, "o", col_list[i],
               r'periods of $\bar{a}_{\text{Fit}}$', r'$\text{Fit}$']
    spectrum_data_specs.append(specs)
    marker_spec_array.append(m_specs)

spectrum_data_specs.append([1, col_list[3], r'$a_{\text{PCA}}$'])
marker_spec_array.append([(2.5, -10), 50, "o", col_list[3], r'periods of $a_{\text{PCA}}$', r'$\text{PCA}$'])


spectrum_per = []#el_fit_per_list + per_list
spectrum_amp = el_fit_amp_list + amp_list

def fft_logplot_spec_x(p_o_list, a_o_list, data_spec_array,
                       per_list_list, amp_list_list, marker_spec_array,
                       Δt_list, Δt_spec_array, tit, unit, log,
                       log_fac_list, location, fosi, labspa, n_cols,
                       vlines_list, vlines_specs, xlimits, ylimits,
                       modus, save_specs):
    # p_o_list = [p_o_1, p_o_2, ...]
    # a_o_list = [a_o_1, a_o_2, ...]
    # data_spec_array = [alpha_i, col_i, lab_i]; alpha, color, label
    # per_list_list = [per_list_1, per_list_2, ...] for markers
    # amp_list_list = [amp_list_1, amp_list_2, ...] for markers
    # marker_spec_array = [pos_set, size, symb_i, col_i, leg_i, text_i];
    #                      position, size, symbol, color, label for legend, text label
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
    if (modus == 'd'): # day
        γ = 1
        p_lab = '[d]'
    elif (modus == 'h'): # hour
        γ = 24
        p_lab = '[h]'
    elif (modus == 'm'): # minute
        γ = 24 * 60
        p_lab = '[min]'
    elif (modus == 's'): # second
        γ = 24 * 60 * 60
        p_lab = '[s]'
    p = p_lab[1:-1]
    print("fft_logplot_spec_3 version 8")
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
            plt.semilogx(p_o_list[i]*γ, new_a_o_list[i],
                         color = col, alpha = α,
                         ls = 'solid', lw = 1, label = lab)
    else:
        for i in range(0, n_data):
            col = data_spec_array[i][1]
            α = data_spec_array[i][0]
            lab = data_spec_array[i][2]
            if (i >= 1):
                if (log_fac_list[i - 1] != 1):
                    lab = lab + r'$\cdot$' + scientific_to_exponent(1/log_fac_list[i - 1], 1)
                    #scientific_to_exponent(s_fac, 1)
                    #lab = lab + ' / ' + str(log_fac_list[i - 1])
            plt.loglog(p_o_list[i]*γ, new_a_o_list[i],
                       color = col, alpha = α,
                       ls = 'solid', lw = 1, label = lab)
    
    k = 0
    for i in range(0, n_marker):
        coords = marker_spec_array[i][0]
        size = marker_spec_array[i][1]
        mark = marker_spec_array[i][2]
        col = marker_spec_array[i][3]
        lab = marker_spec_array[i][4]
        text = marker_spec_array[i][5]
        plt.scatter(per_list_list[i]*γ, new_amp_list_list[i],
                    color = col, s = size,
                    marker = mark,
                    fc = 'None', lw = 1,
                    alpha = 1, label = lab)
        for j in range(0, len(per_list_list[i])):
            str_lab = dollar_remove(text)
            string = r'$p_{0}$'.replace('0', str(j + 1))
            plt.annotate(string, (per_list_list[i][j]*γ, new_amp_list_list[i][j]),
                         coords, textcoords = 'offset points',
                         fontsize = 15)
            string_add = ' = %.6e ' % (per_list_list[i][j]*γ)
            #plt.figtext(0.92, 0.875 - k, string + string_add,
            #            fontsize = 10, ha = 'left')
            print(string + string_add + p)
            k += 0.05
        k += 0.03
    
    for i in range(0, n_Δt):
        lab1 = r'$Δt_{xyz}$ = '.replace('xyz', dollar_remove(Δt_spec_array[i][2]*γ))
        lab2 = str(np.round(Δt_list[i], 3)) + ' ' + p
        plt.axvline(Δt_list[i]*γ, color = Δt_spec_array[i][1], alpha = 0.5,
                    ls = Δt_spec_array[i][0], lw = 5,
                    label = lab1 + lab2)
    
    for i in range(0, len(vlines_list)):
        vline = vlines_list[i]
        plt.axvline(vline*γ, color = vlines_specs[0],
                    alpha = vlines_specs[1], lw = vlines_specs[2],
                    ls = 'solid')
    
    if (xlimits[0] != xlimits[1]):
        plt.xlim(xlimits[0], xlimits[1])
    
    if (ylimits[0] != ylimits[1]):
        plt.ylim(ylimits[0], ylimits[1])
    
    plt.xlabel(r'period ' + p_lab, fontsize = 15)
    plt.ylabel(r'amplitude ' + unit, fontsize = 15)
    plt.legend(fontsize = fosi, loc = location,
               labelspacing = labspa, ncols = n_cols)
    plt.grid()
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)

Δt_list = []
Δt_spec_array = []
vlines = []
#print(0.03*24*60)
vline_specs = ['k', 0.5, 0.5]
xlimits = [1, 1e5]
ylimits = [0, 0]
spectrum_tit_1 = r'spectrum of $a$ and $\overline{a}$'
spectrum_tit_2 = r", $m_0 = $ [" + ", ".join("%.1e" % m0 for m0 in el_m0_list) + "] m"
spectrum_tit = None #spectrum_tit_1 + spectrum_tit_2
file_name_fft = "spec/spec_" + "_" + 'a' + ".png"
file_name_fft = file_name + file_name_fft
log_fac_list = 8**np.arange(1, 6)
#log_fac_list = [10,100,10000]
location = 4
fosi = 17.5
labspa = 0.25
n_cols = 1
fft_logplot_spec_3(spectrum_p[:], spectrum_a[:], spectrum_data_specs[:],
                   spectrum_per, spectrum_amp, marker_spec_array,
                   Δt_list, Δt_spec_array, spectrum_tit, "[m]", 1,
                   log_fac_list, location, fosi, labspa, n_cols,
                   vlines, vline_specs, xlimits, ylimits, 'm',
                   [0, image_folder + 'Spectrum_Fit_Int.png']) # jpg is not worth it
print("yes")
print("DONE!!!")
#%%
from functions import flx_get_data_ymd
path_flx = 'FLXAP_P.FLX'
data_flx_apday, data_ap = flx_get_data_ymd(path_flx)
data_ap = array_modifier(data_ap, MJD_0, n_days_tot_max)
#%%
from functions import gen_a_dot_array

def decrease_unit_day(str_unit):
    # str_unit: for example r'[$kg m^{-3}$]'
    inner_string = str_unit[2:-2]
    inner_string = r'\frac{' + inner_string + r'}{d}'
    new_string = str_unit[:2] + inner_string + str_unit[-2:]
    return(new_string)

def lab_gen_add(x, configuration, add):
    #configuration_list = [[n_fac, R, ω, constr]]
    n_fac = configuration[0]
    R = configuration[1]
    ω = configuration[2]
    constr = configuration[3]
    lab = r'$_{n_fac}^{(R)}\dot{x}_{\text{add}}^{(ω)}$'
    lab = lab.replace("x", x, 1)
    lab = lab.replace("n_fac", str(n_fac))
    lab = lab.replace("R", str(R))
    lab = lab.replace('ω', '%.0e' % ω)
    lab = lab.replace('add', add)
    return(lab)

def decr_bar_ult(slope_data_list, slope_specs_list,
                 e_fac_list, average, el_label, el_unit,
                 flx_data, flx_spec_list,
                 xlimits, ylimits, title, grid,
                 location, anchor, n_cols, fosi, labspa,
                 vline_list1, vline_list_specs1,
                 vline_list2, vline_list_specs2,
                 save_specs):
    # slope_data_list = [[τ_i, s_i, e_s_i]]
    # slope_specs = [[l_w, lcol, ecol, α, e_α, lab, e_lab]]
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
    t_min, t_max = t_min_t_max(slope_data_list)
    xstart = t_min + xlimits[0]
    xend = t_max - xlimits[1]
    n_days = xend - xstart
    new_slope_data_list = 0
    if (xlimits[0] != xlimits[1]):
        new_slope_data_list = []
        for i in range(0, len(slope_data_list)):
            data_i = slope_data_list[i]
            data_i = array_modifier(data_i, xstart, n_days)
            new_slope_data_list.append(data_i)
    else:
        new_slope_data_list = slope_data_list
    if (len(flx_data) != 0):
        flx_data = array_modifier(flx_data, xstart, n_days)
    
    
    fig, ax1 = plt.subplots(figsize = (10, 5), dpi = 300)
    mean_list = []
    for i in range(0, len(new_slope_data_list)):
        slope_data = new_slope_data_list[i]
        slope_specs = slope_specs_list[i]
        x_slope = slope_data[:, 0]
        y_slope = slope_data[:, 1]
        
        s_fac = e_fac_list[i]
        s_y_slope = slope_data[:, 2] * s_fac
        lab_fac = scientific_to_exponent(s_fac, 1)
        # slope_specs = [[l_w, lcol, ecol, α, e_α, lab, e_lab]]
        l_w = slope_specs[0]
        lcol = slope_specs[1]
        ecol = slope_specs[2]
        α = slope_specs[3]
        e_α = slope_specs[4]
        lab = slope_specs[5]
        e_lab = slope_specs[6]
        if (s_fac != 1):
            e_lab = e_lab + r' $\times$ ' + lab_fac
        
        if (average == 1):
            mean = np.mean(y_slope)
            y_slope = y_slope / mean
            s_y_slope = s_y_slope / mean
            mean_list.append(mean)
        
        ax1.plot(x_slope, y_slope,
                 ls = '-', lw = l_w, color = lcol,
                 alpha = α, label = lab, zorder = 10)
        ax1.fill_between(x_slope, y_slope - s_y_slope, y_slope + s_y_slope,
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
    ax1.set_xlabel(xaxis_year([xstart], 'mm.dd'), fontsize = 15)
    if (average == 0):
        y_lab = r'$\dot{el}$'
        y_lab = y_lab.replace('el', el_label)
        y_lab = y_lab + ' [' + el_unit + ']'
        ax1.set_ylabel(y_lab, fontsize = 15)
    else:
        y_lab = r'$\frac{\dot{el}}{\langle\dot{el}\rangle}$'
        y_lab = y_lab.replace('el', el_label)
        
        if (len(new_slope_data_list) > 1):
            message = r'$\langle\dot{\overline{el}}\rangle$ = ['
            message = message.replace('el', el_label)
            message = message + ', '.join('%.3f' % mean for mean in mean_list) + '] ' + el_unit
            
            print(message)
        else:
            message = r'$\langle\dot{\overline{el}}\rangle$ = %.3f ' % mean_list[0] + el_unit
            message = message.replace('el', el_label)
            print(message)
        ax1.set_ylabel(y_lab, rotation = 0, labelpad = 10, fontsize = 15)
    #fig.suptitle(title, fontsize = 15)
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
                ax2.bar(t_ap + (i - 1 / 2) * bar_width, ap,
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
                
                ax2.bar(t_ap + (i - 1 / 2) * bar_width, ap,
                        width = bar_width, alpha = α,
                        color = col, lw = 0)
            flx_ylabel =  r'$ap$'
            ax2.set_ylabel(flx_ylabel, color = col, fontsize = 15, rotation = 0)
            ax2.tick_params(axis = 'y', labelcolor = col, color = col)
    
    c = 0
    for i in range(0, len(vline_list1)):
        vline = vline_list1[i]
        col = vline_list_specs1[0]
        α = vline_list_specs1[1]
        width = vline_list_specs1[2]
        style = vline_list_specs1[4]
        lab = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                lab = vline_list_specs1[3]
                c += 1
            ax1.axvline(vline, color = col, alpha = α,
                        ls = style, lw = width,
                        label = lab, zorder = 200)
    c = 0
    for i in range(0, len(vline_list2)):
        vline = vline_list2[i]
        col = vline_list_specs2[0]
        α = vline_list_specs2[1]
        width = vline_list_specs2[2]
        style = vline_list_specs2[4]
        lab = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                lab = vline_list_specs2[3]
                c += 1
            ax1.axvline(vline, color = col, alpha = α,
                        ls = style, lw = width,
                        label = lab, zorder = 20)
    
    if (len(flx_data) != 0):
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.set_frame_on(False)
    
    plt.figlegend(fontsize = fosi, markerscale = 5, loc = location,
                  bbox_to_anchor = anchor, bbox_transform = ax1.transAxes,
                  labelspacing = labspa, ncols = n_cols, columnspacing = 1)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)


n_slope = n_days_tot_int * 1
slope_data_list = []
slope_specs_list = []

color_group_list = [['forestgreen', 'forestgreen'],
                    ['royalblue', 'royalblue']]

color_list = cl_lin(np.arange(3), my_cmap)

add_list_1 = len(configuration_list) * [r'Fit']
for i in range(0, len(configuration_list)):
    configuration = configuration_list[i]
    n_fac = configuration[0]
    R = configurations[1]
    ω = configuration[2]
    constr = configuration[3]
    
    slope_gen_list = slope_gen_list_list[i]
    
    m0 = el_m0_list[i]
    
    n_slope = 10 * n_days_tot_lst * n_fac
    slope_data = gen_a_dot_array(n_slope, slope_gen_list)
    add = add_list_1[i]
    lab = r'$\dot{\bar{a}}_{\psi_{%d}}$' % (i + 1)
    e_lab = r'$\sigma_{\psi_{%d}}$' % (i + 1)
    color_index = i
    lcol = color_list[i]
    #mcol = color_group_list[color_index][1]
    #ecol = color_group_list[color_index][2]
    ecol = color_list[i]
    #slope_specs = [lw, lcol, ecol, α, e_α, lab, e_lab]
    slope_specs = [1.5, lcol, ecol, 1, 0.333, lab, e_lab]
    slope_data_list.append(slope_data)
    slope_specs_list.append(slope_specs)


a_dot_list = [a_dot_ele]
add_list_2 = [r'PCA']
for i in range(len(configuration_list)*0, 0*len(configuration_list) + 1):
    add = add_list_2[i - len(configuration_list)*0]
    lab = r'$\dot{a}_{\text{%s}}$' % add
    e_lab = r'$\sigma_{\text{%s}}$' % add
    #col = ['tomato','deepskyblue','lime'][i]
    #col = ['crimson','midnightblue','forestgreen'][i]
    col = 'violet'
    lcol = col
    ecol = col
    #slope_specs = [lw, lcol, ecol, α, e_α, lab, e_lab]
    slope_specs = [1.5, lcol, ecol, 0.75, 0.25, lab, e_lab]
    slope_data = a_dot_list[i - len(configuration_list)*0]
    
    slope_data_list.append(slope_data)
    slope_specs_list.append(slope_specs)


#slope_specs = [5.,  0, "deeppink", 1.5, 1]
add_data_list = []
add_data_specs_list = []
el_label = 'a'
el_unit = r'$\frac{\text{m}}{\text{d}}$'
flx_data = data_ap
#flx_spec_list = [0.75, 1, 'plasma']
flx_spec_list = [1, 0, 'k']

str_0 = r'GFOC - '
str_1 = r'$\bar{a}$'
if (len(el_m0_list) > 1):
    str_2 = r', $m_0 = $ [' + ', '.join('%.1e' % m0 for m0 in el_m0_list) + '] m'
else:
    str_2 = r', $m_0 = $ %.1e m' % el_m0_list[0]
title = None #str_0 + r'decrease of ' + str_1 + str_2

xlimits = [0, 0]
ylimits = [-70,40]

average = 0
n_cols = 4
location = 2 #10
anchor = (0., 1) #(0.5, -0.225)
fosi = 17.5
labspa = 0.5
e_fac_list = len(configuration_list) * [10e-1] + 3 * [1e7]
#vline_list_specs1 = ['gold', 1, 1, "CME", (4, (4, 4)), 2]
#vline_list_specs2 = ['saddlebrown', 1, 1, "MAN", (0, (4, 12)), 2]
decr_bar_ult(slope_data_list[3:], slope_specs_list[3:],
             e_fac_list[3:], average, el_label, el_unit,
             flx_data, flx_spec_list,
             xlimits, ylimits, title, grid,
             location, anchor, n_cols, fosi, labspa,
             vline_list1, vline_list_specs1,
             vline_list2, vline_list_specs2,
             [0, image_folder + 'Slope_Int_z.jpg']) # save only int
#Slope_Int
#Slope_Int_Average
#%%
fit_p_list = []
fit_a_list = []
for i in range(0, len(configuration_list)):
    slope = slope_data_list[i]
    fft_list = fft(slope)
    p_fft = fft_list[0]
    a_fft = fft_list[1]
    fit_p_list.append(p_fft)
    fit_a_list.append(a_fft)
col_list = ['r','b','g']
fig = plt.figure(figsize=(10,5),dpi=300)
for i in range(0, 3):
    plt.loglog(fit_p_list[i],fit_a_list[i]/(10**(2*i)),color = col_list[i])
plt.ylim(1e-9,1e1)
plt.grid()
plt.show(fig)
plt.close(fig)
#%%
#60020 - 60024

n_p = 1
a_dot_ele_lcm_list = [0, 50, n_p, 0.1, 0.9, 10]
a_dot_lcm_list = [a_dot_ele_lcm_list]

a_dot_ele_interval_list = [[0, 0]]
a_dot_interval_list = [a_dot_ele_interval_list]

a_dot_list = [a_dot_ele]
#a_dot_list = [a_dot_ele, array_modifier(a_dot_ele,60020,4)]

a_dot_p_o_list = []
a_dot_a_o_list = []
a_dot_per_list = []
a_dot_amp_list = []

for i in range(0, len(a_dot_list)):
    a_dot = a_dot_list[i]
    lcm_list = a_dot_lcm_list[0]#
    interval_list = a_dot_interval_list[0]#
    
    #"""#for se1a comment start
    spectrum_list = spectrum(a_dot, -1, lcm_list, interval_list)
    p_o_list, a_o_list = spectrum_list[0], spectrum_list[1]
    per_list, amp_list = spectrum_list[2], spectrum_list[3]
    
    a_dot_p_o_list.append(p_o_list)
    a_dot_a_o_list.append(a_o_list)
    a_dot_per_list.append(per_list)
    a_dot_amp_list.append(amp_list)
    #"""#for se1a comment end
    
    # not for se1a comment start
    """
    if (i == 0):
        spectrum_list = fft(a_dot)
        p_o_list, a_o_list = spectrum_list[0], spectrum_list[1]
        a_dot_p_o_list.append(p_o_list)
        a_dot_a_o_list.append(a_o_list)
    else:
        spectrum_list = spectrum(a_dot, -1, lcm_list, interval_list)
        p_o_list, a_o_list = spectrum_list[0], spectrum_list[1]
        per_list, amp_list = spectrum_list[2], spectrum_list[3]

        a_dot_p_o_list.append(p_o_list)
        a_dot_a_o_list.append(a_o_list)
        a_dot_per_list.append(per_list)
        a_dot_amp_list.append(amp_list)
    """
    #not for se1a comment end


###################################################
#################### SMOOTHING ####################
###################################################
from functions import smoother

smoothed_list = []
per0_list_list = []
amp0_list_list = []
Δt_n_list = []
q = 0
for i in range(0, len(a_dot_list)):
    print(len(a_dot_list))
    full_a_dot_data = a_dot_list[i]
    per_list = a_dot_per_list[i]#
    amp_list = a_dot_amp_list[i]#
    lcm_list = a_dot_lcm_list[0]#
    
    a_dot_data = array_columns(full_a_dot_data, [0, 1])
    σ_a_dot_data = array_columns(full_a_dot_data, [0, 2])
    
    a_dot_smooth_list = smoother(a_dot_data, per_list, amp_list,
                                 lcm_list, -1, q)
    σ_a_dot_smooth_list = smoother(σ_a_dot_data, per_list, amp_list,
                                   lcm_list, -1, q)
    
    a_dot_smoothed = a_dot_smooth_list[0]
    σ_a_dot_smoothed = σ_a_dot_smooth_list[0]
    a_dot_data_smoothed = np.vstack((a_dot_smoothed.T,
                                     σ_a_dot_smoothed[:, 1])).T
    
    per0_list = a_dot_smooth_list[1]
    amp0_list = a_dot_smooth_list[2]
    Δt_n = a_dot_smooth_list[4]
    
    smoothed_list.append(a_dot_data_smoothed)
    per0_list_list.append(per0_list)
    amp0_list_list.append(amp0_list)
    Δt_n_list.append(Δt_n)

smoothed_a_dot_ele_lcm_list = [0, 50, 1, 0.1, 0.95, 10]
smoothed_lcm_list = [smoothed_a_dot_ele_lcm_list]

smoothed_a_dot_ele_interval_list = [[0, 0]]
smoothed_interval_list = [smoothed_a_dot_ele_interval_list]

smoothed_a_dot_p_o_list = []
smoothed_a_dot_a_o_list = []
smoothed_a_dot_per_list = []
smoothed_a_dot_amp_list = []

for i in range(0, len(a_dot_list)):
    smoothed_a_dot = smoothed_list[i]
    lcm_list = smoothed_lcm_list[0]#
    interval_list = smoothed_interval_list[0]#
    
    spectrum_list = spectrum(smoothed_a_dot, -1, lcm_list, interval_list)
    p_o_list, a_o_list = spectrum_list[0], spectrum_list[1]
    per_list, amp_list = spectrum_list[2], spectrum_list[3]
    
    smoothed_a_dot_p_o_list.append(p_o_list)
    smoothed_a_dot_a_o_list.append(a_o_list)
    smoothed_a_dot_per_list.append(per_list)
    smoothed_a_dot_amp_list.append(amp_list)
#%%
def riffle(list1, list2):
    newlist = []
    for i in range(0, len(list1)):
        newlist.append(list1[i])
        newlist.append(list2[i])
    return(newlist)

p_o_list = a_dot_p_o_list + smoothed_a_dot_p_o_list
a_o_list = a_dot_a_o_list + smoothed_a_dot_a_o_list
per_list_list = per0_list_list + smoothed_a_dot_per_list
amp_list_list = amp0_list_list + smoothed_a_dot_amp_list

#"""#not for se1a
data_specs = [[1, 'violet', r'$\dot{a}_{\text{PCA}}$', 1],
              [1, 'fuchsia', r'$\tilde{\dot{a}}_{\text{PCA}}$', 1]]

marker_specs = [[(5, -7.5), 50, "o", "violet",
                 "periods used for " + r'$\tilde{\dot{a}}_{\text{PCA}}$', r'$\text{PCA}$']]
#"""#not for se1a

"""# for se1a
data_specs = [[1, 'violet', r'$\dot{a}_{\text{PCA}}$', 1],
              [1, 'purple', r'$\overset{\frown}{\dot{a}}_{\text{PCA}}$', 1],
              [1, 'fuchsia', r'$\tilde{\dot{a}}_{\text{PCA}}$', 1],
              [1, 'indigo', r'$\overset{\frown}{\tilde{\dot{a}}}_{\text{PCA}}$', 1]]

marker_specs = [[(5, -7.5), 50, "o", "purple",
                 "periods used for " + r'$\tilde{\dot{a}}_{\text{PCA}}$', r'$\text{PCA}$']]
"""# for se1a

Δt_list = Δt_n_list
Δt_spec_array = [[(0, (2, 5)), "k", 0.5, 'PCA']]

tit_add_1 = r'$[N_{\text{PCA}}]$'
tit_add_2 = r' = $[%d]$' % (a_dot_lcm_list[0][2])
tit = None #r'spectrum of $\dot{a}$ and $\bar{\dot{a}}$, ' + tit_add_1 + tit_add_2



v_line_specs = ['b', 0.5, 1]
xlimits = [0,0]
ylimits = [0, 0]
v_line_list = []


def fft_logplot_spec_8(p_o_list, a_o_list, data_specs,
                       per_list_list, amp_list_list, marker_specs,
                       Δt_list, Δt_spec_array, tit, log,
                       log_fac_list, location, anchor,
                       fosi, n_cols, labspa,
                       vlines_list, vlines_specs, xlimits, ylimits,
                       modus,
                       save_specs):
    # p_o_list = [p_o_1, p_o_2, ...]
    # a_o_list = [a_o_1, a_o_2, ...]
    # data_spec_array = [alpha_i, col_i, lab_i]; alpha, color, label
    # per_list_list = [per_list_1, per_list_2, ...] for markers
    # amp_list_list = [amp_list_1, amp_list_2, ...] for markers
    # marker_spec_array = [pos_set, size, symb_i, col_i, leg_i, text_i];
    #                      position, size, symbol, color, label for legend, text label
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
    print("fft_logplot_spec_8 version 9")
    if (modus == 'd'): # day
        γ = 1
        p_lab = '[d]'
    elif (modus == 'h'): # hour
        γ = 24
        p_lab = '[h]'
    elif (modus == 'm'): # minute
        γ = 24 * 60
        p_lab = '[min]'
    elif (modus == 's'): # second
        γ = 24 * 60 * 60
        p_lab = '[s]'
    p = p_lab[1:-1]
    n_data = len(p_o_list)
    n_marker = len(per_list_list)
    n_Δt = len(Δt_list)
    
    new_a_o_list = [a_o_list[0]]
    for i in range(0, n_data - 1):
        new_a_o = a_o_list[i + 1] * log_fac_list[i]
        new_a_o_list.append(new_a_o)
    
    if (len(amp_list_list) != 0):
        new_amp_list_list = [amp_list_list[0]]
        for i in range(0, n_marker - 1):
            new_amp_list = amp_list_list[i + 1] * log_fac_list[2*i+1]
            new_amp_list_list.append(new_amp_list)
    else:
        new_amp_list_list = []
    
    fig = plt.figure(figsize = (12, 5), dpi = 300)
    plt.title(tit, fontsize = 17.5)
    
    if (log == 0):
        for i in  range(0, n_data):
            col = data_specs[i][1]
            α = data_specs[i][0]
            lab = data_specs[i][2]
            
            fac = log_fac_list[i - 1]
            if (i != 0 and fac != 1):
                lab_fac = scientific_to_exponent(fac, 0)
                lab = lab + r' $\times$ ' + lab_fac
            plt.semilogx(p_o_list[i]*γ, new_a_o_list[i],
                         color = col, alpha = α,
                         ls = 'solid', lw = 1, label = lab)
    else:
        for i in range(0, n_data):
            col = data_specs[i][1]
            α = data_specs[i][0]
            lab = data_specs[i][2]
            
            fac = log_fac_list[i - 1]
            if (i != 0 and fac != 1):
                lab_fac = scientific_to_exponent(fac, 1)
                lab = lab + r' $\times$ ' + lab_fac
            plt.loglog(p_o_list[i]*γ, new_a_o_list[i],
                       color = col, alpha = α,
                       ls = 'solid', lw = 1, label = lab)
    
    k = 0
    for i in range(0, n_marker):
        coords = marker_specs[i][0]
        size = marker_specs[i][1]
        mark = marker_specs[i][2]
        col = marker_specs[i][3]
        lab = marker_specs[i][4]
        text = marker_specs[i][5]
        plt.scatter(per_list_list[i]*γ, new_amp_list_list[i],
                    color = col, s = size,
                    marker = mark,
                    fc = 'None', lw = 1.5,
                    alpha = 1, label = lab)
        for j in range(0, len(per_list_list[i])):
            str_lab = dollar_remove(text)
            string = r'$p_{xyz}^{(0)}$'.replace('0', str(j + 1)).replace('xyz', str_lab)
            plt.annotate(string, (per_list_list[i][j]*γ, new_amp_list_list[i][j]),
                         coords, textcoords = 'offset points',
                         fontsize = 15)
            string_add = ' = %.6e d' % (per_list_list[i][j]*γ)
            #plt.figtext(0.92, 0.875 - k, string + string_add,
            #            fontsize = 10, ha = 'left')
            k += 0.05
        k += 0.03
    
    for i in range(0, n_Δt):
        lisi = Δt_spec_array[i][0]
        col = Δt_spec_array[i][1]
        α = Δt_spec_array[i][2]
        lab = Δt_spec_array[i][3]
        lab1 = r'$Δt_{\text{xyz}}$ = '.replace('xyz', dollar_remove(lab))
        lab2 = str(np.round(Δt_list[i]*γ, 2)) + ' ' + p
        plt.axvline(Δt_list[i]*γ, color = col, alpha = α,
                    ls = lisi, lw = 5,
                    label = lab1 + lab2)
    
    for i in range(0, len(vlines_list)):
        vline = vlines_list[i]*γ
        plt.axvline(vline, color = vlines_specs[0],
                    alpha = vlines_specs[1], lw = vlines_specs[2],
                    ls = 'solid')
    
    if (xlimits[0] != xlimits[1]):
        plt.xlim(xlimits[0], xlimits[1])
    
    if (ylimits[0] != ylimits[1]):
        plt.ylim(ylimits[0], ylimits[1])
    
    plt.xlabel(r'period ' + p_lab, fontsize = 15)
    plt.ylabel(r'amplitude [$\frac{\text{m}}{\text{d}}$]', fontsize = 15)
    plt.legend(fontsize = fosi, loc = location,
               bbox_to_anchor = anchor,
               ncols = n_cols, labelspacing = labspa)
    plt.grid()
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)



#location = 10#for se1a
#anchor = (0.5, 1.15)#for se1a
location = 4
anchor = (1, 0)
fosi = 17.5
n_cols = 1#for se1a
#n_cols = 1
labspa = 0.5
#log_fac_list = [1e-3,1e-6,1e-9]#for se1a
log_fac_list = [1e-1]

"""#for se1a
amp_list_new = []
for i in range(0, len(amp_list_list)):
    amp_list_new.append(amp_list_list[i]/1e3)
"""#for se1a
xlimits = [1,1e5]
fft_logplot_spec_8(p_o_list, a_o_list, data_specs,
                   per_list_list[0:1], amp_list_list, marker_specs, #amp_list_new # per_list_list[1:2]
                   Δt_list[:], Δt_spec_array, tit, 1,#Δt_list[1:2]
                   log_fac_list, location, anchor,
                   fosi, n_cols, labspa,
                   v_line_list, v_line_specs,
                   xlimits, ylimits, 'm',
                   [0, image_folder + 'Slope_Int_Spectrum.png'])
print("per_list_list: ", per_list_list)
print("Δt_list: ", Δt_list)
#%%
slope_data_list = []
slope_specs_list = []

color_list = cl_lin(np.arange(3), my_cmap)
add_list_1 = len(configuration_list) * [r'Fit']
for i in range(0, len(configuration_list)):
    configuration = configuration_list[i]
    n_fac = configuration[0]
    R = configurations[1]
    ω = configuration[2]
    constr = configuration[3]
    
    slope_gen_list = slope_gen_list_list[i]
    
    m0 = el_m0_list[i]
    
    n_slope = 10 * n_days_tot_lst * n_fac
    slope_data = gen_a_dot_array(n_slope, slope_gen_list)
    #slope_specs = [l_w, lcol, m_s, mcol, ecol, ew, α, lab]
    # [l_w, lcol, ecol, α, e_α, lab, e_lab]
    add = add_list_1[i]
    lab = r'$\dot{\bar{a}}_{\psi_{%d}}$' % (i + 1)
    e_lab = r'$\sigma_{\psi_{%d}}$' % (i + 1)
    color_index = i
    lcol = color_list[i]
    #mcol = color_group_list[color_index][1]
    #ecol = color_group_list[color_index][2]
    ecol = color_list[i]
    #slope_specs = [lw, lcol, ecol, α, e_α, lab, e_lab]
    slope_specs = [1.25, lcol, ecol, 1, 0.333, lab, e_lab]
    slope_data_list.append(slope_data)
    slope_specs_list.append(slope_specs)
    
    
    
    #short_start = 60001 # 2023 first storm
    short_start = 60026 # 2023 second storm
    #short_start = 60440 # 2024
    
    short_slope = array_modifier(slope_data,short_start,3)
    ind_min = np.argmin(short_slope[:,1])
    date = short_slope[ind_min,0]
    decay = short_slope[ind_min,1]
    t_obj = Time(str(date), format = 'mjd')
    t_iso = t_obj.iso
    string = t_iso[5 : 16].replace('-', '.')
    print(ω, string, decay)

a_dot_smoothed_list = smoothed_list
add_list_2 = [r'PCA']
for i in range(0, 1):
    add = add_list_2[i]
    lab = r'$\tilde{\dot{a}}_{\text{%s}}$' % add
    e_lab = r'$\tilde{\sigma}_{\text{%s}}$' % add
    #col = ['crimson', 'midnightblue', 'forestgreen'][i]
    #col = ['tomato', 'deepskyblue', 'lime'][i]
    col = 'fuchsia'
    lcol = col
    ecol = col
    α_list = [1,1,1]
    slope_specs = [1, lcol, ecol, 1, 0.25, lab, e_lab]
    slope_data = a_dot_smoothed_list[i]
    
    slope_data_list.append(slope_data)
    slope_specs_list.append(slope_specs)
    
    
    short_start = 60001 # 2023 first storm
    #short_start = 60026 # 2023 second storm
    #short_start = 60440 # 2024
    
    short_slope = array_modifier(slope_data,short_start,3)
    ind_min = np.argmin(short_slope[:,1])
    date = short_slope[ind_min,0]
    decay = short_slope[ind_min,1]
    t_obj = Time(str(date), format = 'mjd')
    t_iso = t_obj.iso
    string = t_iso[5 : 16].replace('-', '.')
    print(add, string, decay)

#slope_specs = [5.,  0, "deeppink", 1.5, 1]
add_data_list = []
add_data_specs_list = []
el_label = 'a'
el_unit = r'$\frac{\text{m}}{\text{d}}$'
flx_data = data_ap
#flx_spec_list = [0.75, 1, 'plasma']
flx_spec_list = [1, 0, 'k']

str_0 = 'GFOC - '
str_1 = r'$\bar{a}$'
if (len(el_m0_list) > 1):
    str_2 = r", $m_0 = $ [" + ", ".join("%.1e" % m0 for m0 in el_m0_list) + "] m"
else:
    str_2 = r", $m_0 = $ %.1e m" % el_m0_list[0]
title = None #str_0 + "decrease of " + str_1 + str_2

grid = 1
xlimits = [0, 0]
#ylimits = [0.5, 2]#average
ylimits = [-35, 20]
average = 0
n_cols = 2
fosi = 17.5
labspa = 0.25
location = 2
anchor = (0., 1.0)
#anchor = (0.5, 1.275)
#anchor = (0.5, -0.275) # for fit
#anchor = (0.5, -0.225) # for int # DO NOT FORGET !!!!!!!!!!!!!!!!!!!!
e_fac_list = [1, 1, 1] + [1e7]
#vline_list_specs1 = ['gold', 1, 1, "CME", (2, (4, 4)), 2]

decr_bar_ult(slope_data_list[3:], slope_specs_list[3:],
             e_fac_list[3:], average, el_label, el_unit,
             flx_data, flx_spec_list,
             xlimits, ylimits, title, grid,
             location, anchor, n_cols, fosi, labspa,
             vline_list1, vline_list_specs1,
             vline_list2, vline_list_specs2,
             [0, image_folder + 'Slope_Int_Smooth_z.jpg'])
#Slope_Fit
#Slope_Fit_Average
#Slope_Int_Smooth
#Slope_Int_Smooth_Average
#%%
print((slope_data_list[3:][0][1,0]-slope_data_list[3:][0][0,0])*24*60*60)
# %%
def mjd_to_mmdd_test(array):
    new_array = []
    for t in array:
        t_obj = Time(str(t), format = 'mjd')
        t_iso = t_obj.iso
        string = t_iso[5 : 10].replace('-', '.')
        new_array.append(string)
    return(new_array)

def mjd_to_mmdd_one(t):
    t_obj = Time(str(t), format = 'mjd')
    t_iso = t_obj.iso
    print(t_iso)
    string = t_iso[5 : 10].replace('-', '.')
    return(string)
print(mjd_to_mmdd_one(vline_list2[0]))
# %%