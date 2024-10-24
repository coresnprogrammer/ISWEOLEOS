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
from functions import arg_finder, fft, cl_lin
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

sat_name = 'GFOC'
save_on = 0
path = os.getcwd() + "/updates/update 10/grace/c/"

#lst_path = "grace18newlst/year_nongra/year_nongra_a.txt"
lst_path = "newoscele/GRACE_LST/year_nongra/year_nongra_a.txt"
integration_path = "ultimatedata/GF-C/a_data_int_acc.txt"
#58356
MJD_interval = [58331, 58381]
#MJD_interval = [58346, 58366]
MJD_interval = [58351, 58361]
#MJD_interval = [58281, 58431] # 58119 - 58483
MJD_0, MJD_end = MJD_interval[0], MJD_interval[1]
n_days_tot = MJD_end - MJD_0

a_symb, a_unit, a_dot = r"$a$", r"[$m$]", r"$\dot{a}$"

lcm_list_el = [0, 50, 1, 0.09, 0.95, 10]
interval_list_el = [[0.0218, 0.022], [0.065, 0.066]] # [5.5, 5.6] [0.065, 0.066],[0.0725, 0.0775], [0.08, 0.085]

vline_list1 = [58354.25, 58355.25, 59958.915, 59961.875, 59967.125,
               59988.71, 59995.416, 59999.5, 60001.78125, 60002.427,
               60017.17, 60018.17, 60023.9375, 60026.375, 60053.389]
vline_list2 = []
vline_list_specs1 = ['k', 0.5, 1, "CME"]
vline_list_specs2 = ['gold', 0.75, 1, "26.08"]
#%%
a_data_int_acc = np.loadtxt(integration_path)
a_lst = np.loadtxt(lst_path, skiprows = 1)
a_lst = array_denormalize(a_lst)
a_lst = array_modifier(a_lst, MJD_0, n_days_tot)
# %%
el_p_o_list = []
el_a_o_list = []
el_per_list_0 = []
el_amp_list_0 = []
# data_spectrum
el_spectrum_list = spectrum(a_lst, 3, lcm_list_el, interval_list_el)
el_p_o, el_a_o = el_spectrum_list[0], el_spectrum_list[1]
el_per, el_amp = el_spectrum_list[2], el_spectrum_list[3]

el_per = np.append(el_per, 5.55) # 5.55
el_amp = np.append(el_amp, 10)

el_p_o_list.append(el_p_o)
el_a_o_list.append(el_a_o)
el_per_list_0.append(el_per)
el_amp_list_0.append(el_amp)

spectrum_p = [el_p_o]
spectrum_a = [el_a_o]
spectrum_data_specs = [[0.75, "r", a_symb]]
spectrum_per = [el_per]
spectrum_amp = [el_amp]
print("########################")
print("periods")
print(spectrum_per)
print("########################")
print(el_p_o[20:40])
print(a_lst[-1, 0] - a_lst[0, 0])
print("--------------")
marker_spec_array = [[(7.5, -7.5), "o", "k",
                      "peaks used for " + a_symb, a_symb]]
Δt_list = []
Δt_spec_array = [[(0, (4, 4)), "r", a_symb]]
spectrum_tit = "test"
v_line_specs = ['b', 0.5, 1]
xlimits = [0, 0]
xlimits = [0.01, 10]
ylimits = [0, 0]
ylimits = [0.1, 100]
#[0.0145, 0.0185], [0.019, 0.026], [0.045, 0.125]
#v_line_list = [0.019, 0.026, 0.05, 0.075, 0.095, 0.1, 0.25, 0.5]
v_line_list = [0.0218, 0.022, 0.065, 0.066,
               0.0725, 0.0775, 0.08, 0.085, 5.5, 5.6]
#[array([0.03282784, 0.02188044, 0.0656126 , 0.08182782, 0.09803294,
#       0.45868624])]
fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                   spectrum_per, spectrum_amp, marker_spec_array,
                   Δt_list, Δt_spec_array, spectrum_tit, 1, [], 0,
                   v_line_list, v_line_specs,
                   xlimits, ylimits, [0])
#%%
el_per_list = [[el_per_list_0[0][0], el_per_list_0[0][3],
                el_per_list_0[0][2], el_per_list_0[0][1]]]
el_amp_list = [[el_amp_list_0[0][0], el_amp_list_0[0][3],
                el_amp_list_0[0][2], el_amp_list_0[0][1]]]

el_per_list = el_per_list_0
el_amp_list = el_amp_list_0
#%%
t_fit_a = tt.time()
configuration_list = []
print(len(a_lst), len(a_data_int_acc))

configuration_list = [[20, 4, 0, 1],
                      [20, 4, 1e-3, 1],
                      [20, 4, 1e0, 1],
                      [20, 4, 1e1, 1],
                      [20, 4, 1e2, 1],
                      [20, 4, 1e3, 1],
                      [20, 4, 1e4, 1],
                      [20, 4, 1e5, 1],
                      [20, 4, 1e6, 1],
                      [20, 4, 1e9, 1]]

# Fit
a_int_ta = a_data_int_acc[0, 0]
a_int_tb = a_data_int_acc[-1, 0]
n_fac = int(np.ceil(a_int_tb - a_int_ta))
n_partition = 10 * n_fac

configuration_list = [[n_fac, 0, 0, 1]] + configuration_list

el_fit_list_list = []
el_para_fit_list = []
el_para_e_fit_list = []
el_m0_list = []
τ_fit_list_list = []
slope_gen_list_list = []
bar_data_list = []

for i in range(0, len(configuration_list)):
    configurations = configuration_list[i]
    n_fac = configurations[0]
    R = configurations[1]
    ω = configurations[2]
    constr = configurations[3]
    
    n_partition = n_fac * n_days_tot
    
    if (i == 0):
        el_data = a_data_int_acc
    else:
        el_data = a_lst
    
    el_per = el_per_list[0]
    el_amp = el_amp_list[0]
    
    # fit
    el_fit_list = fitter(el_data, el_per[:R], el_amp[:R],
                         n_partition, 5, ω, constr)
    el_fit = el_fit_list[0]
    el_para_fit, el_para_e_fit = el_fit_list[1], el_fit_list[2]
    el_m0, τ_fit_list = el_fit_list[3], el_fit_list[4]
    slope_gen_list, bar_data = el_fit_list[5], el_fit_list[6]
    
    el_fit_list_list.append(el_fit)
    el_para_fit_list.append(el_para_fit)
    el_para_e_fit_list.append(el_para_e_fit)
    el_m0_list.append(el_m0)
    τ_fit_list_list.append(τ_fit_list)
    slope_gen_list_list.append(slope_gen_list)
    bar_data_list.append(bar_data)
t_fit_b = tt.time()
print("Time: ", t_fit_b - t_fit_a)
print("DONE!!!")
#%%
el_fit_p_o_list = []
el_fit_a_o_list = []
el_fit_per_list = []
el_fit_amp_list = []
for i in range(0, len(configuration_list)):#len(data_both)
    el_fit = el_fit_list_list[i]
    # data_spectrum
    el_fit_spectrum_list = spectrum(el_fit, 3, lcm_list_el, [[0, 0]])
    el_fit_p_o, el_fit_a_o = el_fit_spectrum_list[0], el_fit_spectrum_list[1]
    el_fit_per, el_fit_amp = el_fit_spectrum_list[2], el_fit_spectrum_list[3]
    
    el_fit_p_o_list.append(el_fit_p_o)
    el_fit_a_o_list.append(el_fit_a_o)
    el_fit_per_list.append(el_fit_per)
    el_fit_amp_list.append(el_fit_amp)
#%%
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

vline_list_specs1 = ['k', 0.75, 1, "CME"]
vline_list_specs2 = ['r', 0.25, 1, "MAN"]
path_add = "_" + str(0) + "_" + str(ω)
#if (os.path.exists(path) == False):
#    os.mkdir(os.getcwd() + "/" + file_name[:-1])
xlimit_list = [[0, 0]]
log_fac_list = 2**np.arange(1, len(configuration_list) + 1)
# for plots
el_per, el_amp = el_per_list[0], el_amp_list[0]
col_add = 0
if (len(configuration_list) == 1):
    col_add = 1
col_list = cl_lin(np.arange(len(configuration_list) + col_add), mat.colormaps['gist_rainbow'])
# spectrum
spectrum_p = el_p_o_list + el_fit_p_o_list
spectrum_a = el_a_o_list + el_fit_a_o_list

spectrum_data_specs = [[0.75, "k", a_symb]]
for i in range(0, len(configuration_list)):
    spec_list = [0.75, col_list[i],
                 lab_gen('a', configuration_list[i])]
    spectrum_data_specs.append(spec_list)

spectrum_per = [el_per]
spectrum_amp = [el_amp]
#spectrum_per = [[5]]
#spectrum_amp = [[2.3]]
marker_spec_array = [[(7.5, -7.5), "o", "k", "periods used", a_symb]]
Δt_list = []
Δt_spec_array = []
vlines = []
vline_specs = ['k', 0.5, 0.5]
xlimits = [0, 0]
ylimits = [0, 0]
spectrum_tit_1 = sat_name + " data: spectrum of " + a_symb + " and " + a_dot
spectrum_tit_2 = ""#r", $m_0 = $ [" + ", ".join("%.1e" % m0 for m0 in el_m0_list) + "] m"
spectrum_tit = spectrum_tit_1 + spectrum_tit_2
file_name_fft = "spec/spec_" + "_" + a_symb + ".png"
file_name_fft = file_name_fft
fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                   spectrum_per, spectrum_amp, marker_spec_array,
                   Δt_list, Δt_spec_array, spectrum_tit, 1,
                   log_fac_list, 2,
                   vlines, vline_specs, xlimits, ylimits,
                   [save_on, path + "spec" + path_add])
print("yes")
print("DONE!!!")
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

def plot_func_6_no(data_array_list, data_spec_array, y_label,
                   ref_array_list, ref_spec_array, ref_y_spec_list,
                   tit, MJD_0, MJD_end, xlimits, ylimits,
                   location, anchor, fosi, n_cols,
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
    xstart = xlimits[0]
    xend = MJD_end - MJD_0 - xlimits[1]
    n_days = xend - xstart
    new_data_array_list = 0
    new_ref_array_list = 0
    if (max(np.abs(np.array([xlimits[0], xlimits[1]]))) != 0):
        new_data_array_list = []
        new_ref_array_list = []
        for i in range(0, len(data_array_list)):
            data_i = data_array_list[i]
            data_i = array_modifier(data_i, xstart + MJD_0, n_days)
            new_data_array_list.append(data_i)
        for i in range(0, len(ref_array_list)):
            ref_i = ref_array_list[i]
            ref_i = array_modifier(ref_i, xstart + MJD_0, n_days)
            new_ref_array_list.append(ref_i)
    else:
        new_data_array_list = data_array_list
        new_ref_array_list = ref_array_list
    
    fig, ax1 = plt.subplots(figsize = (8, 8), dpi = 300)
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
        ax1.set_xlim(xstart + MJD_0, xend + MJD_0)
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
        vline = vline_list1[i] - MJD_0
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs1[3]
                c += 1
            plt.axvline(vline + MJD_0, color = vline_list_specs1[0],
                        alpha = vline_list_specs1[1], lw = vline_list_specs1[2],
                        ls = (0, (4, 4)), zorder = 3,
                        label = c_label)
    
    c = 0 # for giving only one label to vlines from vline_list2
    for i in range(0, len(vline_list2)):
        vline = vline_list2[i] - MJD_0
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs2[3]
                c += 1
            plt.axvline(vline + MJD_0, color = vline_list_specs2[0],
                        alpha = vline_list_specs2[1], lw = vline_list_specs2[2],
                        ls = (0, (4, 4)), zorder = 3,
                        label = c_label)
    
    plt.figlegend(fontsize = fosi, markerscale = 5, loc = location,
                  bbox_to_anchor = anchor, bbox_transform = ax1.transAxes,
                  ncols = n_cols, labelspacing = 0, prop = 'monospace')
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    
    plt.show(fig)
    plt.close(fig)

MJD_interval_new = [58351, 58361]
MJD_interval_new = MJD_interval
MJD_0_new, MJD_end_new = MJD_interval_new[0], MJD_interval_new[1]
n_days_tot_new = MJD_end_new - MJD_0_new

a_data_int_acc_trim = array_modifier(a_data_int_acc, MJD_0_new, n_days_tot_new)

max_mjd = max([MJD_0_new, a_data_int_acc_trim[0, 0]])

bar_start = 0
bar_data_list_trim = []
for i in range(1, len(bar_data_list)):
    bar_data = bar_data_list[i]
    bar_data_trim = array_modifier(bar_data, MJD_0_new, n_days_tot_new)
    bar_data_list_trim.append(bar_data_trim)
    bar_sync = array_modifier(bar_data, max_mjd, 2)
    if (i >= 2):
        bar_start += bar_sync[0, 1]
bar_start /= (len(bar_data_list)-2)

shift = bar_start - a_data_int_acc_trim[0, 1]
a_data_int_acc_shifted = np.vstack((a_data_int_acc_trim[:, 0],
                                    a_data_int_acc_trim[:, 1] + shift)).T

# integration 3 times for label reasons
data_list = 3 * [a_data_int_acc_shifted] + bar_data_list_trim + [a_data_int_acc_shifted]


data_array_list = data_list
cmap = mat.colormaps['gist_rainbow']
col_list = cl_lin(np.arange(len(bar_data_list)-1), cmap)


header = r'( n, R,    ω   ,    $m_0$   )'
data_spec_array = [[0.25, 'k', 'ACC integration', 10],
                   [0, 'w', ' ', 0],
                   [0, 'w', header, 0]]
for i in range(1, len(configuration_list)):
    configuration = configuration_list[i]
    n_fac = configuration[0]
    R = configuration[1]
    ω = configuration[2]
    constr = configuration[3]
    m0 = el_m0_list[i]
    
    lab = '(%2d, %d, %.1e, %.2e)' % (n_fac, R, ω, m0)
    
    col = col_list[i-1]
    
    specs = [1, col, lab, 1]
    data_spec_array.append(specs)
data_spec_array.append([0, 'w', ' ', 0])

y_label = a_symb + " " + a_unit
ref_array_list, ref_spec_array, ref_y_spec_list = [], [], []
tit = r'$\bar{a}$ of ACC compared with LST'

#ylimits = [mean - 1.7 * std, mean + 1.9 * std]
#ylimits = [mean - 250 * std, mean + 10 * std]
ydown = 6.86945e6
ylimits = [ydown, ydown + 2e2]
vline_specs = ["gold", 0, 1]
z_order = 1
nongra_fac = 1
grid = 1
save_specs = [0]
xlimits = [-0.1, -0.1]
vline_list2 = np.arange(0) * 4.5 + 4. + MJD_0
vline_list_specs2 = ['gold', 0.75, 1, "5.55 d"]
location = 2
anchor = (0., 1.)
fosi = 10
n_cols = 1
plot_func_6_no(data_array_list, data_spec_array, y_label,
               ref_array_list, ref_spec_array, ref_y_spec_list,
               tit, MJD_0_new, MJD_end_new, xlimits, ylimits,
               location, anchor, fosi, n_cols,
               vline_specs, z_order, nongra_fac, grid,
               vline_list1, vline_list_specs1,
               vline_list2, vline_list_specs2,
               save_specs)
#%%
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
                 fac, location, anchor, fosi, n_cols,
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
        
        ax1.plot(x_slope, y_slope,
                 ls = '-', lw = l_w, color = lcol,
                 alpha =  α, label = lab, zorder = 10)
        ax1.fill_between(x_slope, y_low_slope, y_upp_slope,
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
    
    plt.figlegend(fontsize = fosi, markerscale = 1, loc = location,
                  bbox_to_anchor = anchor, bbox_transform = ax1.transAxes,
                  labelspacing = 0.1, ncols = n_cols, columnspacing = 1)
    
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)


def decr_bar_eff(slope_data_list, slope_specs,
                 e_fac_list, ave, el_label, el_unit,
                 flx_data, flx_spec_list,
                 MJD_0, MJD_end, xlimits, ylimits, title, grid,
                 fac, location, anchor, fosi, n_cols,
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
            e_lab = e_lab + r' $\cdot$ ' + str(e_fac) 
        
        if (ave == 1):
            mean = np.mean(y_slope)
            y_slope = y_slope / mean
            y_low_slope = y_low_slope / mean
            y_upp_slope = y_upp_slope / mean
            mean_list.append(mean)
        
        ax1.plot(x_slope, y_slope,
                 ls = '-', lw = l_w, color = lcol,
                 alpha =  α, label = lab, zorder = 10)
    
    for i in range(0, len(new_slope_data_list)):
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
            e_lab = e_lab + r' $\cdot$ ' + str(e_fac) 
        ax1.fill_between(x_slope, y_low_slope, y_upp_slope,
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
    
    plt.figlegend(fontsize = fosi, markerscale = 1, loc = location,
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
fac = 1

str_0 = ""
str_1 = r'$\bar{a}$'
if (len(el_m0_list) > 1):
    str_2 = r", $m_0 = $ [" + ", ".join("%.1e" % m0 for m0 in el_m0_list) + "] m"
else:
    str_2 = r", $m_0 = $ %.1e m" % el_m0_list[0]
str_2 = ''
n_slope = n_days_tot * 1
slope_data_list = []
slope_specs_list = []

cmap = mat.colormaps['rainbow']
col_list = cl_lin(np.arange(len(configuration_list) - 1), cmap)

for i in range(0, len(configuration_list)):
    configuration = configuration_list[i]
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
    l_α = 0.75
    if (i == 0):
        add = 'ACC'
        lcol = 'k'
        ecol = 'k'
    else:
        #if (i == 1):
        #    l_α = 1
        add = 'LST'
        lcol = col_list[i-1]
        ecol = col_list[i-1]
    lab = lab_gen_add('a', configuration, add)
    e_lab = lab_gen_add('σ', configuration, add)
    #slope_specs = [2, lcol, ecol, 1 - i / 2.5, 0.25, lab, e_lab]
    slope_specs = [2, lcol, ecol, l_α, 0.125, lab, e_lab]
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
flx_spec_list = [0.25, 0, 'chocolate']
title = str_0 + "decrease of " + str_1 + str_2
vline_list2 = []
ylimits = [0, 0]
ylimits = [-8, 1]
#ylimits = [-3, 0]
xlimits = [0, 0]
z_order = 1
nongra_fac = 1
configuration0b = "decr"
path_add = "0"
ave = 0
fosi = 12
n_cols = 2
location = 1
position = (1.5, 1.)
vline_list_specs1 = ['k', 0.5, 1, ""]
e_fac_list = [100] + [1e-3] * (len(slope_data_list) - 1)
decr_bar_eff(slope_data_list, slope_specs_list,
             e_fac_list, ave, el_label, el_unit,
             flx_data, flx_spec_list,
             MJD_0, MJD_end, xlimits, ylimits,
             title, 1, fac, location, position,
             fosi, n_cols,
             vline_list1, vline_list_specs1,
             vline_list2, vline_list_specs2,
             [save_on, path + path_add])
# %%

