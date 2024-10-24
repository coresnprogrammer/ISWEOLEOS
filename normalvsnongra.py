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

satname = 'SENTINEL-3A'
year = '2024'
satshort = 'SE3A'
yearshort = '24'

image_folder = 'Thesis/LATEX/Images/Results/' + satname + '_' + year + '/'
foldername_ele = "All Data new/eleswarm"
foldername_lst = "All Data new/osceleswarm"
file_name_list = ['normal', 'nongra']
#58356 is storm
# 61 days before data hole
#MJD_interval_lst = [58347, 58408]
#MJD_interval_int = [58352, 58400] # [58352, 58413]
MJD_interval = [59990, 60040]
MJD_interval_lst = MJD_interval


MJD_0_lst, MJD_end_lst = MJD_interval_lst[0], MJD_interval_lst[1]

n_days_tot_lst = MJD_end_lst - MJD_0_lst
n_days_tot_max = n_days_tot_lst

MJD_0_list = [MJD_0_lst]
MJD_end_list = [MJD_end_lst]
MJD_0 = min(MJD_0_list)
MJD_end = max(MJD_end_list)


str_list = ["a"]

a_symb, a_unit, a_dot = r"$a$", r"[$m$]", r"$\dot{a}$"

# prepare el, el0, acc
el_data_list = []
pca_data_list = []
for i in range(0, len(file_name_list)):
    el_str = str_list[0]
    file_name = file_name_list[i]
    name_el = foldername_lst + 2 * ("/year_" + file_name) + "_" + el_str + ".txt"
    el_data = np.loadtxt(name_el, skiprows=1)  # load data
    el_data = array_denormalize(el_data)  # denormalize data
    el_data = array_modifier(el_data, MJD_0_lst, n_days_tot_lst)  # trim data
    #el_data = array_normalize(el_data, 0)  # normalize data
    #el_data = el_data[1:]  # cut MJD_0
    el_data_list.append(el_data)
    
    #if (file_name == 'normal'):
    #    name_pca = foldername_ele + 2 * ("/year_" + file_name) + "_withoutoffset.txt"
    #else:
    #    name_pca = foldername_ele + 2 * ("/year_" + file_name) + ".txt"
    name_pca = foldername_ele + 2 * ("/year_" + file_name) + ".txt"
    pca_data = np.loadtxt(name_pca)
    pca_data = array_modifier(pca_data, MJD_0_lst, n_days_tot_lst)
    pca_data_list.append(pca_data)
    
a_normal, a_nongra = el_data_list[0], el_data_list[1]
pca_normal, pca_nongra = pca_data_list[0], pca_data_list[1]


vline_list1 = [58354.25, 58355.25, 59958.915, 59961.875, 59967.125,
               59988.71, 59995.416, 59999.5, 60001.78125, 60002.427,
               60017.17, 60018.17, 60023.9375, 60026.375, 60053.389]

from astropy.time import Time
def yyyymmddhhmm_to_mjd(string):
    t_obj = Time(string, format = 'isot', scale = 'utc')
    t_mjd = t_obj.mjd
    mjd_float = float(t_mjd)
    return(mjd_float)

addcmelist = ['2024-05-05T11:30', '2024-05-10T16:36', '2024-05-10T16:36',
 '2024-05-10T16:36', '2024-05-10T16:36', '2024-05-10T16:36',
 '2024-05-11T09:30', '2024-05-11T09:30', '2024-05-11T20:30',
 '2024-05-12T08:55', '2024-05-15T18:13']
for i in range(0, len(addcmelist)):
    vline_list1.append(yyyymmddhhmm_to_mjd(addcmelist[i]))

vline_list2 = [MJD_0_lst, MJD_end_lst]
vline_list_specs1 = ['gold', 1, 1, "CME"]
vline_list_specs2 = ['gold', 1, 5, "Interval"]
#%%
el_num = arg_finder(string_list, 'a')
el_word, el_symb = name_list[el_num][0], name_list[el_num][1]
el_unit, el_bar = name_list[el_num][2], name_list[el_num][3]
el_hat, el_tilde = name_list[el_num][4], name_list[el_num][5]
ele = string_list[el_num]
# %%
fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.plot(a_normal[:, 0], a_normal[:, 1], 'r-', alpha = 0.5)
plt.plot(a_nongra[:, 0], a_nongra[:, 1], 'b-', alpha = 0.5)
plt.show(fig)
plt.close(fig)

fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.plot(a_normal[:, 0], a_normal[:, 1] - a_nongra[:, 1], 'm-')
plt.show(fig)
plt.close(fig)
# %%
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
    exp_str = scientific_str[-3:]
    new_str = r'$%s \cdot 10^{%s}$' % (fac_str, exp_str)
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
            lab_fac = scientific_to_exponent(s_fac, 1)
            lab = lab + r' ($\sigma \times$ ' + lab_fac + ')'
            ax1.fill_between(x_data, y_data - s_y, y_data + s_y,
                             color = col, alpha = e_α, zorder = 4)
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
    
    if (len(ref_list) != 0):
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
    
    plt.figlegend(fontsize = fosi, markerscale = 5, loc = location,
                  bbox_to_anchor = anchor, bbox_transform = ax1.transAxes,
                  labelspacing = labspa, ncols = n_cols)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    
    plt.show(fig)
    plt.close(fig)


a_diff = np.vstack((a_normal[:, 0], a_normal[:, 1] - a_nongra[:, 1])).T
data_list = [a_normal, a_nongra, a_diff] # a_data
#[α, col, lab, width]
data_spec_array = [[1, 'r', r'$a$', 1],
                   [7/12, 'b', r'$a_{\text{ng}}$', 1,],
                   [1, 'orchid', r'$a_{\text{NL}} - a_{\text{NG}}$', 1]]
a_unit =r'[$\text{m}$]'
y_label = a_symb + " " + a_unit
ref_list, ref_spec_array, ref_y_spec_list = [], [], []
tit = None #r'semi-major axis from LST-data and numerical integration'
xlimits = [0, 0]
ylimits = [0, 0]
grid = 1
save_specs = [0]
vline_list2 = []
vline_list_specs2 = ['gold', 0.75, 1, "26.08"]
location, anchor = 1, (1, 1)
n_cols = 1
fosi = 15
labspa = 0.5
vline_list_specs1 = ['goldenrod', 1, 2.5, "CME"]
plot_data(data_list[2:], data_spec_array[2:], y_label,
          ref_list, ref_spec_array, ref_y_spec_list,
          tit, xlimits, ylimits, grid,
          location, anchor, n_cols, fosi, labspa,
          vline_list1, vline_list_specs1,
          vline_list2, vline_list_specs2,
          [0, 'Thesis/LATEX/Images/SWARM-A_normal_minus_nongra.jpg'])
print("mean: ", np.mean(a_diff[:, 1]))
print("std: ", np.std(a_diff[:, 1]))
# %%
y_label = r'$\Gamma$ [$\frac{\text{?}}{\text{s}^2}$]'
ref_list, ref_spec_array, ref_y_spec_list = [], [], []
tit = None #r'semi-major axis from LST-data and numerical integration'
xlimits = [0, 0]
ylimits = [0, 0]
grid = 1
save_specs = [0]
vline_list2 = []
vline_list_specs2 = ['gold', 0.75, 1, "26.08"]
location, anchor = 1, (1, 1)
n_cols = 1
fosi = 12.5
labspa = 0.5
vline_list_specs1 = ['k', 1, 1, "CME"]

lablist = ['R', 'S', 'W', 'A']
t_list = pca_normal[:, 0]
for i in range(0, 4):
    I_normal, I_nongra = pca_normal[:, i + 1], pca_nongra[:, i + 1]
    I_diff = I_normal - I_nongra
    
    I_normal_data = np.vstack((t_list, I_normal)).T
    I_nongra_data = np.vstack((t_list, I_nongra)).T
    I_diff_data = np.vstack((t_list, I_diff)).T
    
    data_list = [I_normal_data, I_nongra_data, I_diff]
    #[α, col, lab, width]
    data_spec_array = [[1, 'r', lablist[i] + ' normal', 1],
                       [7/12, 'b', lablist[i] + ' nongra', 1,],
                       [1, 'orchid', 'difference', 1]]
    
    plot_data(data_list[:2], data_spec_array[:2], y_label,
              ref_list, ref_spec_array, ref_y_spec_list,
              tit, xlimits, ylimits, grid,
              location, anchor, n_cols, fosi, labspa,
              vline_list1, vline_list_specs1,
              vline_list2, vline_list_specs2,
              [0, 'Thesis/LATEX/Images/SWARM-A_normal_minus_nongra.jpg'])
    print("mean: ", np.mean(I_diff))
    print("std: ", np.std(I_diff))
# %%
