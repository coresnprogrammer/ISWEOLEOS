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
# general constants
sec_to_day = 1 / (24 * 60 * 60)
G  = 6.674e-11 / (sec_to_day**2)
Me = 5.972e24
μ = G * Me

save_on = 0
path = os.getcwd() + "/updates/update 10/grace/c/"

year = '2018'
image_path = 'Thesis/LATEX/Images/Results/Indices/Ind' + year

foldername = 'dst and ae'

MJD_interval = [58340, 58413] # year 2018 (2018.08.10 - 2018.10.22)
#MJD_interval = [59990, 60040] # year 2023 (2023.02.15 - 2023.04.06)
#MJD_interval = [60431, 60444] # year 2024 (2024.05.01 - 2024.05.14)

MJD_0 = MJD_interval[0]
MJD_end = MJD_interval[1]
n_days_tot = MJD_end - MJD_0



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
#%%
def str_to_arr(string, delimiter):
    # '32.1, 4.2, -7.3' -> [32.1, 4.2, -7.3]
    # with ',' as delimiter
    values = []
    pos = 0
    c = 0 # counter
    while (c < 20):
        # .find(str, start) returns -1 if str not found
        newpos = string.find(delimiter, pos)
        substr = string[pos : newpos]
        values.append(float(substr))
        pos = newpos + 1
        c += 1
        if (pos == 0):
            break
    return(values)

def master_index_files(foldername):
    dst_name = 'kyoto_dst_index_service' + '.txt'
    ae_name = 'kyoto_ae_indices' + '.txt'
    dst_arrays = []
    ae_arrays = []
    entries = os.listdir(foldername)
    for entry in entries:
        if (entry != '.DS_Store'):
            entry_path = foldername + '/' + entry
            files = os.listdir(entry_path)
            for file in files:
                file_path = entry_path + '/' + file
                size = os.path.getsize(file_path) # avoid empty files
                file_type = 0
                if (file == 'MANIFEST'):
                    file_type = 0
                elif (file == dst_name):
                    file_type = 1
                elif (file == ae_name):
                    file_type = 2
                if (size > 1 and file_type != 0):
                    f = open(file_path, 'r')
                    lines = f.readlines()
                    f.close()
                    rows = []
                    for i in range(0, len(lines) - 1):
                        # -1 because last value is of new day
                        line = lines[i].strip()
                        t = str(yyyymmddhhmm_to_mjd(line[:19]))
                        ind = line[19:]
                        row = str_to_arr(t + ind, ',')
                        rows.append(row)
                    if (file_type == 1):
                        dst_arrays.append(np.array(rows))
                    else:
                        ae_arrays.append(np.array(rows))
    
    t0_list = np.array([])
    for array in dst_arrays:
        t0_list = np.append(t0_list, array[0][0])
    sort_list = np.argsort(t0_list)
    dst_sorted = []
    for i in range(0, len(dst_arrays)):
        dst_sorted.append(dst_arrays[sort_list[i]])
    
    t0_list = np.array([])
    for array in ae_arrays:
        t0_list = np.append(t0_list, array[0][0])
    sort_list = np.argsort(t0_list)
    ae_sorted = []
    for i in range(0, len(ae_arrays)):
        ae_sorted.append(ae_arrays[sort_list[i]])
    
    dst_array = np.vstack(tuple(dst_sorted))
    ae_array = np.vstack(tuple(ae_sorted))
    
    path_dst = foldername + '/dst_master.txt'
    path_ae = foldername + '/ae_master.txt'
    
    np.savetxt(path_dst, dst_array)
    np.savetxt(path_ae, ae_array)
    print("done")

#master_index_files(foldername)

dst_data = np.loadtxt(foldername + '/dst_master.txt')
ae_data = np.loadtxt(foldername + '/ae_master.txt')

dst_data = array_modifier(dst_data, MJD_0, n_days_tot)
ae_data = array_modifier(ae_data, MJD_0, n_days_tot)
#%%
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

def plot_func_6_ind(data1, specs1, y_label_specs1,
                    data2, specs2, y_label_specs2,
                    tit, fosi, n_cols,
                    MJD_0, MJD_end, xlimits, ylimits,
                    z_order, grid,
                    vline_list1, vline_list_specs1,
                    save_specs):
    # data1: index 1
    # specs1: [alpha_i, col_i, lab_i, lw_i]
    # y_label: label for y axis
    # tit: title, string like
    # MJD_0: data start
    # MJD_end: data end
    # xlimits = [x_start, x_end] (to deactivate: x_start = x_end)
    # ylimits = [y_low, y_upp]: (to deactivate: y_low = y_upp)
    # z_order: if 1 --> data in front of ref
    # grid: if 1 --> grid to data
    # vline_list = [vline_i]
    # vline_list_specs = [col_i, alpha_i, lw_i, label]
    # save_specs = [1, path] # 0,1 for saving
    xstart = xlimits[0] + MJD_0
    xend = MJD_end - xlimits[1]
    n_days = xend - xstart
    
    fig, ax1 = plt.subplots(figsize = (10, 5), dpi = 300)
    fig.suptitle(tit, fontsize = 17.5)
    
    if (len(data1) != 0):
        data = data1
        if (xlimits[0] != xlimits[1]):
            data = array_modifier(data, xstart, n_days)
        x_data = data[:, 0]
        y_data = data[:, 1]
        α = specs1[0]
        col = specs1[1]
        lab = specs1[2]
        width = specs1[3]
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
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    ax1.set_ylabel(y_label_specs1[0], color = y_label_specs1[1], fontsize = 15)
    ax1.tick_params(axis = 'y', labelcolor = y_label_specs1[1])
    y_low1, y_upp1 = ylimits[0][0], ylimits[0][1]
    if (xstart != xend):
        ax1.set_xlim(xstart, xend)
    if (y_low1 != y_upp1):
        ax1.set_ylim(y_low1, y_upp1)
    if (grid == 1):
        ax1.grid()
    
    if (len(data2) != 0):
        ax2 = ax1.twinx()
        data = data2
        if (xlimits[0] != xlimits[1]):
            data = array_modifier(data, xstart, n_days)
        x_data = data[:, 0]
        y_data = data[:, 1]
        α = specs2[0]
        col = specs2[1]
        lab = specs2[2]
        width = specs2[3]
        ax2.plot(x_data, y_data, color = col, ls = '-', lw = width,
                 alpha = α, zorder = 5, label = lab)
        ax2.set_ylabel(y_label_specs2[0], color = y_label_specs2[1], fontsize = 15)
        ax2.tick_params(axis = 'y', labelcolor = y_label_specs2[1])
        y_low2, y_upp2 = ylimits[1][0], ylimits[1][1]
        if (y_low2 != y_upp2):
            ax1.set_ylim(y_low2, y_upp2)
        
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




#[alpha_i, col_i, lab_i, lw_i]
specs1 = [0.75, 'r', 'Dst', 1]
specs2 = [0.75, 'b', 'AE', 1]

tit = None
y_label_specs1 = ["Dst [nT]", 'r']
y_label_specs2 = ["AE [nT]", 'b']

xlimits = [0, 0]
ylimits = [[0, 0],[0,0]]
z_order = 0
grid = 1
fosi = 15
n_cols = 3
plot_func_6_ind(dst_data, specs1, y_label_specs1,
                ae_data, specs2, y_label_specs2,
                tit, fosi, n_cols,
                MJD_0, MJD_end, xlimits, ylimits,
                z_order, grid,
                vline_list1, vline_list_specs1,
                [0, image_path + ".png"])
# %%
