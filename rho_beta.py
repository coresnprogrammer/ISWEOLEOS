# %%
# python3 -m pip install module
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
import os
from astropy.time import Time
import matplotlib.image as mpimg
from scipy.integrate import solve_ivp as ode_solve
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
from functions import arg_finder, fft
from functions import decrease_plot_hyperadv
from functions import flx_get_data
from functions import cl_lin
from functions import file_extreme_day_new
from functions import get_n_s, lst_sqrs_adjstmnt_adv, insert
from functions import fit_func_1, fit_func_1_derivs
from functions import fit_func_2, fit_func_2_derivs
from functions import fit_func_3, fit_func_3_derivs
from functions import lst_sqrs_adjstmnt, ravel, fit_func_adv, n_objects
from functions import quotient, encode_list, list_code, N_inv_co, round_list
from functions import mjd_to_mmdd, mjd_to_ddmm, xaxis_year
from functions import lab_gen, plot_bar, plot_func_6
#Δa_day(n_rev_day, fac, a, ρ)
# %%
string_list = ["u_sat", "beta_sun", "u_sun",
               "a", "e", "i", "Omega_upp", "omega_low", "T0",
               "rho", "r", "h_ell"]
name_list = [["satellite anomaly", r"$u_{sat}$", r"[$^{\circ}$]",
              r"$\bar{u}_{sat}$", r"$\widehat{u}_{sat}$", r"$\tilde{u}_{sat}$"],
             ["beta angle", r"$\beta$", r"[$^{\circ}$]",
              r"$\bar{\beta}$", r"$\widehat{\beta}$", r"$\tilde{\beta}$"],
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
#file_extreme_day_new('sat24/sentinel/lst', 1)
#%%
satname = 'GRACE-FO-C'
year = '2018'

image_folder = 'Thesis/LATEX/Images/Results/' + satname + '_' + year + '/'

#foldername = "All Data new/oscelegrace"
foldername = "newoscele/GRACE_LST"
#foldername = "All Data new/osceleswarm"
#foldername = "All Data new/oscelesentinel"

path_flx = 'All Data new/FLXAP_P_MJD.FLX'
file_name_list = ["nongra"]
el_str = "rho"
ref_str = "beta_sun"
MJD_interval = [58340, 58413] # year 2018 (2018.08.10 - 2018.10.22)
#MJD_interval = [59990, 60040] # year 2023 (2023.02.15 - 2023.04.06)
#MJD_interval = [60431, 60444] # year 2024 (2024.05.01 - 2024.05.14)
MJD_0 = MJD_interval[0]
n_days_tot = MJD_interval[1] - MJD_0
save_on = 0
path = os.getcwd() + "/updates/update 10/swarm with flx/"
q = 0
n_partition = 1
lsa_method = 5
ε_I = 10**(-3)
# lcm_list = [p_spec, prec, N, p_max, thresh_quot, limit]
lcm_list_el = [0, 50, 1, 0.09, 0.1, 10]
lcm_list_ref = [0, 50, 1, 0.09, 0.95, 10]

interval_list_el = [[0, 0]]
interval_list_ref = [[0, 0]]# [0.032, 0.033]

vline_list2 = []
"""# put as comments if it is not sentinel-1a!!!
man_data = np.loadtxt("hyperdata/hyperdata2/year/all.txt")
man_data = array_denormalize(man_data)
man_data = array_modifier(man_data, MJD_0, n_days_tot)[:,0]
print(man_data)
vline_list2 = man_data
vline_list_specs2 = ['chocolate', 1, 1, "Maneuver", (0, (4, 4))]
"""
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
man_data = array_modifier(man_data, MJD_0, n_days_tot)[:,0]
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
vline_list_specs1 = ['goldenrod', 1, 2.5, "CME", (0, (4, 12)), 2]
vline_list_specs2 = ['saddlebrown', 1, 1.5, "MAN", (-6, (2, 4,2,8)), 2]
"""

"""
def yyddd_to_mjd(npfloat):
    string = '%.f' % npfloat
    newstring = '20%s:%s' % (string[:2], string[2:])
    t_obj = Time(newstring, format = 'yday', scale = 'utc')
    #t_mjd = t_obj.mjd
    #mjd_float = float(t_mjd)
    return(t_obj.iso)

# for se3a
vline_list2_data = np.loadtxt("man_se2a_se3a/S2A_Maneuvers_2024.txt")
print(vline_list2_data)
vline_list_specs1 = ['gold', 1, 1, "CME", (0, (4, 12)), 2]
vline_list_specs2 = ['saddlebrown', 1, 1, "MAN", (-6, (2, 4,2,8)), 2]
for i in range(0, len(vline_list2_data)):
    vline_list2.append(yyddd_to_mjd(vline_list2_data[i]))
print(vline_list2)
"""
#%%
MJD_0, MJD_end = MJD_interval[0], MJD_interval[1]
n_days_tot = MJD_end - MJD_0

el_num = arg_finder(string_list, el_str)
el_word, el_symb = name_list[el_num][0], name_list[el_num][1]
el_unit, el_bar = name_list[el_num][2], name_list[el_num][3]
el_hat, el_tilde = name_list[el_num][4], name_list[el_num][5]

ref_num = arg_finder(string_list, ref_str)
ref_word, ref_symb = name_list[ref_num][0], name_list[ref_num][1]
ref_unit, ref_bar = name_list[ref_num][2], name_list[ref_num][3]
ref_tild = name_list[ref_num][5]

# prepare data
str_normal_nongra = ["normal", "nongra"]
file_name = "nongra"

el_name = foldername + 2 * ("/year_" + file_name) + "_" + el_str + ".txt"
el_data = np.loadtxt(el_name, skiprows = 1)  # load data
el_data = array_denormalize(el_data)  # denormalize data
el_data = array_modifier(el_data, MJD_0, n_days_tot)  # trim data
#el_data = array_normalize(el_data, 0)  # normalize data
#el_data = el_data[1:]  # cut MJD_0

ref_name = foldername + 2 * ("/year_" + file_name) + "_" + ref_str + ".txt"
ref_data = np.loadtxt(ref_name, skiprows = 1)  # load data
ref_data = array_denormalize(ref_data)  # denormalize data
ref_data = array_modifier(ref_data, MJD_0, n_days_tot)  # trim data
#ref_data = array_normalize(ref_data, 0)  # normalize data
#ref_data = ref_data[1:]  # cut MJD_0
print(el_data)
print(ref_data)
# %%
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
    print("fft_logplot_spec_3 version 7")
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
        coords = marker_spec_array[i][0]
        size = marker_spec_array[i][1]
        mark = marker_spec_array[i][2]
        col = marker_spec_array[i][3]
        lab = marker_spec_array[i][4]
        text = marker_spec_array[i][5]
        plt.scatter(per_list_list[i], new_amp_list_list[i],
                    color = col, s = size,
                    marker = mark,
                    fc = 'None', lw = 0.5,
                    alpha = 1, label = lab)
        for j in range(0, len(per_list_list[i])):
            str_lab = dollar_remove(text)
            string = r'$p_{0}$'.replace('0', str(j + 1))
            plt.annotate(string, (per_list_list[i][j], new_amp_list_list[i][j]),
                         coords, textcoords = 'offset points',
                         fontsize = 10)
            string_add = ' = %.6e d' % per_list_list[i][j]
            #plt.figtext(0.92, 0.875 - k, string + string_add,
            #            fontsize = 10, ha = 'left')
            print(string + string_add)
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

el_spectrum_list = spectrum(el_data, el_num, lcm_list_el, interval_list_el)
el_p_o, el_a_o = el_spectrum_list[0], el_spectrum_list[1]
el_per, el_amp = el_spectrum_list[2], el_spectrum_list[3]
# el smoothing
el_smooth_list = smoother(el_data, el_per, el_amp, lcm_list_el, el_num, q)
el_smooth = el_smooth_list[0]
el_per0, el_amp0 = el_smooth_list[1], el_smooth_list[2]
el_Δt_n = el_smooth_list[4]
# el smoothed spectrum
el_smooth_spectrum_list = spectrum(el_smooth, el_num, lcm_list_el, [[0, 0]])
el_smooth_p_o, el_smooth_a_o = el_smooth_spectrum_list[0], el_smooth_spectrum_list[1]
el_smooth_per, el_smooth_amp = el_smooth_spectrum_list[2], el_smooth_spectrum_list[3]
# el spectrum
spectrum_p = [el_p_o, el_smooth_p_o]
spectrum_a = [el_a_o, el_smooth_a_o]
spectrum_data_specs = [[1, 'r', el_symb],
                       [0.666, 'b', el_tilde]]
spectrum_per = [el_per0]
spectrum_amp = [el_amp0]
marker_spec_array = [[(7.5, 0), 50, 's', 'r', 'peaks used for ' + el_tilde, el_symb]]

Δt_list = [el_Δt_n]
Δt_spec_array = [[(0, (4, 4)), 'g', el_symb]]
spectrum_tit = None

v_line_specs = ['b', 0.5, 1]
v_line_list = []

xlimits = [0, 0]
ylimits = [0, 0]
unit = r'[$\frac{\text{kg}}{\text{m}^3}$]'
fosi = 15
labspa = 0.5
n_cols = 1
fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                   spectrum_per, spectrum_amp, marker_spec_array,
                   Δt_list, Δt_spec_array, spectrum_tit, unit, 1, [1],
                   0, fosi, labspa, n_cols,
                   v_line_list, v_line_specs,
                   xlimits, ylimits,
                   [0, image_folder + 'rho_Spectrum.png'])
#%%
ref_spectrum_list = spectrum(ref_data, ref_num, lcm_list_ref, interval_list_ref)
ref_p_o, ref_a_o = ref_spectrum_list[0], ref_spectrum_list[1]
ref_per, ref_amp = ref_spectrum_list[2], ref_spectrum_list[3]
# ref smoothing
ref_smooth_list = smoother(ref_data, ref_per, ref_amp, lcm_list_ref, ref_num, q)
ref_smooth = ref_smooth_list[0]
ref_per0, ref_amp0 = ref_smooth_list[1], ref_smooth_list[2]
ref_Δt_n = ref_smooth_list[4]
# ref smoothed spectrum
ref_smooth_spectrum_list = spectrum(ref_smooth, ref_num, lcm_list_ref, [[0, 0]])
ref_smooth_p_o, ref_smooth_a_o = ref_smooth_spectrum_list[0], ref_smooth_spectrum_list[1]
ref_smooth_per, ref_smooth_amp = ref_smooth_spectrum_list[2], ref_smooth_spectrum_list[3]
# ref spectrum
spectrum_p = [ref_p_o, ref_smooth_p_o]
spectrum_a = [ref_a_o, ref_smooth_a_o]
spectrum_data_specs = [[0.75, 'r', ref_symb],
                       [0.75, 'b', ref_tild]]
spectrum_per = [ref_per0]
spectrum_amp = [ref_amp0]
marker_spec_array = [[(7.5, 0), 50, 's', 'r', 'peaks used for ' + ref_tild, ref_symb]]
Δt_list = [ref_Δt_n]
Δt_spec_array = [[(0, (4, 4)), 'tab:green', ref_symb]]
spectrum_tit = None #'a'

v_line_specs = ['b', 0.5, 1]
v_line_list = []

xlimits = [0.01, 0.1]
ylimits = [3e-3, 3e-2]
xlimits = [0, 0]
ylimits = [0, 0]
unit = r'[$^\circ$]'
fosi = 15
labspa = 0.5
n_cols = 1
fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                   spectrum_per, spectrum_amp, marker_spec_array,
                   Δt_list, Δt_spec_array, spectrum_tit, unit, 1, [5],
                   0, fosi, labspa, n_cols,
                   v_line_list, v_line_specs,
                   xlimits, ylimits,
                   [0, image_folder + 'beta_Spectrum.png'])
# %%
fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.plot(el_data[:, 0], el_data[:, 1], 'r-', lw = 1)
plt.plot(el_smooth[:, 0], el_smooth[:, 1], 'g-', lw = 1)
plt.show(fig)
plt.close(fig)

fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.plot(ref_data[:, 0], ref_data[:, 1], 'b-', lw = 1)
plt.plot(ref_smooth[:, 0], ref_smooth[:, 1], 'g-', lw = 1)
#plt.xlim(22,24)
#plt.ylim(85,86)
plt.show(fig)
plt.close(fig)
# %%
def plot_func_6(data_array_list, data_spec_array, y_label,
                ref_array_list, ref_spec_array, ref_y_spec_list,
                tit, MJD_0, MJD_end, xlimits, ylimits,
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
            data_i = array_modifier(data_i, xstart - MJD_0, n_days)
            new_data_array_list.append(data_i)
        for i in range(0, len(ref_array_list)):
            ref_i = ref_array_list[i]
            ref_i = array_modifier(ref_i, xstart - MJD_0, n_days)
            new_ref_array_list.append(ref_i)
    else:
        new_data_array_list = data_array_list
        new_ref_array_list = ref_array_list
    
    fig, ax1 = plt.subplots(figsize = (11, 5), dpi = 300)
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
        ax1.plot(x_data + MJD_0, y_data, color = col, ls = '-', lw = width,
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
    
    if (len(ref_array_list) != 0):
        ax2 = ax1.twinx()
        for i in range(0, len(new_ref_array_list)):
            x_ref = new_ref_array_list[i][:, 0]
            y_ref = new_ref_array_list[i][:, 1]
            α = ref_spec_array[i][0]
            col = ref_spec_array[i][1]
            lab = ref_spec_array[i][2]
            width = ref_spec_array[i][3]
            ax2.plot(x_ref + MJD_0, y_ref,
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
    
    plt.figlegend(fontsize = 15, markerscale = 5, loc = 1,
                  bbox_to_anchor = (1, 1), bbox_transform = ax1.transAxes,
                  labelspacing = 0.5)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    
    plt.show(fig)
    plt.close(fig)

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
        new_str = r'$10^{%s}$' % (exp_str)
    else:
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
    data_spec_array  : [alpha_i, col_i, lab_i, lw_i, efac_i, ealpha_i]
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
    print(t_min, t_max)
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
    
    fig, ax1 = plt.subplots(figsize = (12, 5), dpi = 300)
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
    ax1.set_ylabel(y_label, color = data_spec_array[0][1], fontsize = 15)
    ax1.tick_params(axis = 'y', labelcolor = data_spec_array[0][1])
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
        
        ax1.set_zorder(ax2.get_zorder())#+1
        ax1.set_frame_on(False)
    
    #ax1.grid(color = data_spec_array[0][1])
    #ax2.grid(color = ref_y_spec_list[1])
    
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
                  labelspacing = labspa, ncols = n_cols)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    
    plt.show(fig)
    plt.close(fig)

name = file_name_list[0]
tit = None #name + " data: " + el_symb + " and " + ref_symb
# all
data_array_list = [el_data]
data_spec_array = [[1, "red", el_symb, 1],
                   #[0.666, 'red', el_tilde, 1]
                   ]
y_label = el_symb + " " + r'[$\frac{\text{kg}}{\text{m}^3}$]'

ref_array_list = [ref_data]

ref_spec_array = [[1, 'blue', ref_symb, 1],
                  #[0.666, 'green', ref_tild, 1]
                  ]
ref_y_spec_list = [ref_symb + ' ' + ref_unit, 'b']

vline_specs = []
z_order = 0
nongra_fac = 1
# only mean
ylim_offset = np.mean(el_data[:, 1])
ylimits = [0, 0]
xlimit_zoom = [0, 0]
xlimits = [0,0]

configuration0b = "all"
path_add = '0'
#plot_func_6(data_array_list, data_spec_array, y_label,
#            ref_array_list, ref_spec_array, ref_y_spec_list,
#            tit, MJD_0, MJD_end, xlimits, ylimits,
#            vline_specs, z_order, nongra_fac, 1,
#            vline_list1, vline_list_specs1,
#            vline_list2, vline_list_specs2,
#            [0, image_folder + 'rho_beta.png'])


grid = 0
location = 1
anchor = (1,1)
n_cols = 1
fosi = 17.5
labspa = 0.5
plot_data(data_array_list, data_spec_array, y_label,
          ref_array_list, ref_spec_array, ref_y_spec_list,
          tit, xlimits, ylimits, grid,
          location, anchor, n_cols, fosi, labspa,
          vline_list1, vline_list_specs1,
          vline_list2, vline_list_specs2,
          [0, image_folder + 'rho_beta.png'])
# %%
