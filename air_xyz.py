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
from functions import arg_finder, column_abs_averager
from functions import Δa_day, Δa_day_spec, data_mean
from functions import decrease_plot_hyperadv
from functions import file_extreme_day_new
#Δa_day(n_rev_day, fac, a, ρ)
"""
spectrum(data, el_num, lcm_list, interval_list):
    # INPUT
    #    data: [t_i, el_i]
    #    el_num: # of element that gets smoothed (Omega_upp is problematic)
    #    lcm_list: contains parameters that specify the lcm
    #    interval_list: list of intervals where additional peaks should be detected
    # OUTPUT
    #    spectrum_list = [p_o_list, a_o_list, per_list, amp_list]
    #        p_o_list = periods of fft
    #        a_o_list = amplitudes of fft
    #        per_list = periods that have a peak amplitude
    #        amp_list = peak amplitudes sorted from highest to lowest

smoother(data, per_list, amp_list, lcm_list, el_num, q):
    # INPUT
    #    data: [t_i, el_i]
    #    per_list: periods sorted from lowest to highest
    #    amp_list: corresponding amplitudes
    #    lcm_list: contains parameters that specify the lcm
    #    el_num: # of element that gets smoothed (Omega_upp is problematic)
    #    q: polynomial order for filtering
    # OUTPUT
    #    smooth_list = [data_smoothed, per0_list, amp0_list, n_Δt, Δt_n, Δt_dt]
    #        data_smoothed: data smoothed with Savizky-Golay filter
    #        per0_list: periods regarded when smoothed
    #        amp0_list: corresponding amplitudes
    #        n_Δt: width of smoothing window (uneven)
    #        Δt_n: smoothing period
    #        Δt_dt: Δt % dt ("error")

fitter(data, per0_list, amp0_list, n_partition, lsa_method, ε):
    # INPUT
    #    data: [t_i, el_i] to be fitted
    #    per0_list: periods that will be used in fit (fixed or as guesses)
    #    amp0_list: corresponding amplitudes (as guesses)
    #    n_partition: duration of intervals for creating fits
    #    lsa_method: [1 -> t0, per0 fixed, 2 -> t0 fixed]
    #    ε: accuracy of fit
    # OUTPUT
    #    fit_list = [data_fitted, para_fit_array, para_e_fit_array, m0_list]
    #        data_fitted: fitted data (stacked)
    #        para_fit_array: fitted parameters stacked
    #        para_e_fit_array: error of fitted parameters stacked
    #        m0_list: a posteriori errors for each fit

plot_func_6(data_array_list, data_spec_array, y_label,
                ref_array_list, ref_spec_array, ref_y_spec_list,
                tit, MJD_0, MJD_end, xlimits, ylimits,
                vline_col, z_order, nongra_fac, grid):
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
    # vline_col: color for vlines
    # z_order: if 1 --> data in front of ref
    # nongra_fac: for scaling nongra data
    # grid: if 1 --> grid to data

fft_logplot_spec_3(p_o_list, a_o_list, data_spec_array,
                       per_list_list, amp_list_list, marker_spec_array,
                       Δt_list, Δt_spec_array, tit, log):
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

decrease_plot_adv(para_fit, para_e_fit, el_y_label_list, para_specs,
                      ref_data, ref_specs, ref_lab_list,
                      n_partition, el_Δt_n, MJD_0, MJD_end, title, grid)
    # para_fit = [h_i, s_i, a_i, ϕ_i]
    # para_e_fit = [σ_h_i, σ_s_i, σ_a_i, σ_ϕ_i]
    # el_y_label_list = [el, unit]
    # para_specs = [e_fac, ms, mew, capsize, mfc, mec, ecolor]
    # ref_data = reference
    # ref_specs = [α, col, lab, width]
    # ref_lab_list = [ref_lab, ref_y_axis_lab]
    # n_partition: how many fits per day
    # el_Δt_n: smoothing period
    # MJD_0: start of data
    # MJD_end: end of data
    # title: title of plot
    # grid: if 1 --> grid for data
"""
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
#file_extreme_day_new('All Data new/oscelesentinel', 2)
#%%
foldername = "All Data new/oscelesentinel"
file_name_list = ["nongra"]
el_str_list = ["air_x", "air_y", "air_z"]
ref_str = "rho"
MJD_interval = [59950, 60050]
q = 1
n_partition = 0.5
lsa_method = 3
ε = 10 ** (-3)
# lcm_list = [p_spec, prec, N, p_max, thresh_quot, limit]
lcm_list_el = [0, 50, 15, 0.09, 0.95, 10]
lcm_list_el_bar = [0, 50, 15, 0.09, 0.95, 15]
lcm_list_el_hat = [0, 50, 1, 1, 0.95, 15]
lcm_list_ref = [0, 0.1, 1, 0.09, 0.25, 10]
interval_list_el = [[0, 0]] # [[0.05, 0.06], [0.07, 0.09], [0.015, 0.025]]
interval_list_el_bar = [[0, 0]]
interval_list_el_hat = [[0, 0]]
interval_list_ref = [[0.01, 0.0125], [0.0125, 0.015], [0.015, 0.02], [0.02, 0.025]]
xlimit_list = [[0, 0]]
vline_list2 = []
if (foldername[-8:] == "sentinel"):
    man_data = np.loadtxt("hyperdata/hyperdata2/year/all.txt")
    man_data = array_denormalize(man_data)
    man_data = np.ravel(man_data)
    vline_list2 = man_data
# %%
MJD_0, MJD_end = MJD_interval[0], MJD_interval[1]
n_days_tot = MJD_end - MJD_0

el_num = 0
el_word_list = ['air_x', 'air_y', 'air_z']
el_symb_list = [r'$\text{air}_x$', r'$\text{air}_y$', r'$\text{air}_z$']
el_unit = r'[$\frac{\text{nm}}{s^2}$]'
el_bar_list = [r'$\overline{\text{air}}_x$', r'$\overline{\text{air}}_y$', r'$\overline{\text{air}}_z$']
el_hat_list = [r'$\widehat{\text{air}}_x$', r'$\widehat{\text{air}}_y$', r'$\widehat{\text{air}}_z$']
el_tilde_list = [r'$\tilde{\text{air}}_x$', r'$\tilde{\text{air}}_y$', r'$\tilde{\text{air}}_z$']

ref_num = arg_finder(string_list, ref_str)
ref_word, ref_symb = name_list[ref_num][0], name_list[ref_num][1]
ref_unit, ref_bar = name_list[ref_num][2], name_list[ref_num][3]
ref_hat, ref_tilde = name_list[ref_num][4], name_list[ref_num][5]

# prepare data
str_normal_nongra = ["normal", "nongra"]
data_air = []
file_name = file_name_list[0]

for i in range(0, len(el_str_list)):
    name_i = foldername + 2 * ("/year_" + file_name) + "_" + el_str_list[i] + ".txt"
    data_i = np.loadtxt(name_i, skiprows=1)  # load data
    data_i = array_denormalize(data_i)  # denormalize data
    data_i = array_modifier(data_i, MJD_0, n_days_tot)  # trim data
    data_i = array_normalize(data_i, 0)  # normalize data
    data_i = data_i[1:]  # cut MJD_0
    data_air.append(data_i)


ref_name = foldername + 2 * ("/year_" + str_normal_nongra[1]) + "_" + ref_str + ".txt"
data_ref = np.loadtxt(ref_name, skiprows=1)
data_ref = array_denormalize(data_ref)
data_ref = array_modifier(data_ref, MJD_0, n_days_tot)
data_ref = array_normalize(data_ref, 0)
data_ref = data_ref[1:]
# %%
########################################################################################
# PROCESSING REF #######################################################################
########################################################################################
ref_data = data_ref
# ref spectrum
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
ref_smooth_p_o, ref_smooth_a_o = (ref_smooth_spectrum_list[0],
                                  ref_smooth_spectrum_list[1])
ref_smooth_per, ref_smooth_amp = (ref_smooth_spectrum_list[2],
                                  ref_smooth_spectrum_list[3])
# ref spectrum
spectrum_p = [ref_p_o, ref_smooth_p_o]
spectrum_a = [ref_a_o, ref_smooth_a_o]
spectrum_data_specs = [[0.75, "k", ref_symb], [0.75, "chocolate", ref_bar]]
spectrum_per = [ref_per0]
spectrum_amp = [ref_amp0]
marker_spec_array = [[(7.5, 0), "s", "k", "peaks used for " + ref_bar, ref_symb]]
Δt_list = [ref_Δt_n]
Δt_spec_array = [[(0, (4, 4)), "k", ref_symb]]
spectrum_tit = "amplitude - period spectrum of non gravitational \ndata of " + ref_word
"""
fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                   spectrum_per, spectrum_amp, marker_spec_array,
                   Δt_list, Δt_spec_array, spectrum_tit, 1, [1], 4,
                   [0, 0])
"""
print("DONE!!!")
#%%
########################################################################################
# PROCESSING EL ########################################################################
########################################################################################
el_p_o_list = []
el_a_o_list = []
el_per_list = []
el_amp_list = []
for i in range(0, len(data_air)):
    name = file_name_list[0]
    el_data = data_air[i]
    # data_spectrum
    el_spectrum_list = spectrum(el_data, el_num, lcm_list_el, interval_list_el)
    el_p_o, el_a_o = el_spectrum_list[0], el_spectrum_list[1]
    el_per, el_amp = el_spectrum_list[2], el_spectrum_list[3]
    
    el_p_o_list.append(el_p_o)
    el_a_o_list.append(el_a_o)
    el_per_list.append(el_per)
    el_amp_list.append(el_amp)
    
    spectrum_p = [el_p_o]
    spectrum_a = [el_a_o]
    spectrum_data_specs = [[0.75, "r", el_symb_list[i]]]
    spectrum_per = [el_per]
    spectrum_amp = [el_amp]
    marker_spec_array = [[(7.5, -7.5), "s", "r", "peaks used for " + el_bar_list[i], el_symb_list[i]]]
    Δt_list = []
    Δt_spec_array = [[(0, (4, 4)), "r", el_symb_list[i]]]
    spectrum_tit = "test"
    """
    fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                       spectrum_per, spectrum_amp, marker_spec_array,
                       Δt_list, Δt_spec_array, spectrum_tit, 1, [], 0,
                       [0, 0])
    """
print("DONE!!!")
#%%
data_spec_array = [[0.5, "r", el_symb_list[0], 1],
                   [0.5, "b", el_symb_list[1], 1],
                   [0.5, "g", el_symb_list[2], 1]]
y_label = el_symb_list[0] + " " + el_unit
ref_spec_array = [[0.8, "chocolate", ref_bar, 1]]
ref_y_spec_list = [ref_symb + " " + ref_unit, "chocolate"]
tit = "test"
xlimits = [51, 46]
ylimits = [0, 0]
vline_specs = ["gold", 0.75, 1]
z_order = 1
nongra_fac = 1
"""
plot_func_6(data_air, data_spec_array, y_label,
            [ref_smooth], ref_spec_array, ref_y_spec_list,
            tit, MJD_0, MJD_end, xlimits, ylimits,
            vline_specs, z_order, nongra_fac, 0,
            [], [], [], [], [0, 0])
"""
#%%
el_bar_data_list_list = []
el_bar_per0_list = []
el_bar_amp0_list = []
el_bar_Δt_n_list = []
el_bar_p_o_list = []
el_bar_a_o_list = []
el_bar_per_list = []
el_bar_amp_list = []
for i in range(0, len(data_air)):
    el_data = data_air[i]
    el_per = el_per_list[i]
    el_amp = el_amp_list[i]
    # for plots
    el_p_o, el_a_o = el_p_o_list[i], el_a_o_list[i]
    # data smoothing
    el_bar_data_list = smoother(el_data, el_per, el_amp, lcm_list_el, el_num, q)
    el_bar_data = el_bar_data_list[0]
    el_bar_per0, el_bar_amp0 = el_bar_data_list[1], el_bar_data_list[2]
    el_bar_Δt_n = el_bar_data_list[4]
    print("el_bar_per0", el_bar_per0)
    print("el_bar_amp0", el_bar_amp0)
    # smoothed spectrum
    print("000000000000000000000000 MAKE SPECTRUM !!! 000000000000000000000000")
    el_bar_spectrum_list = spectrum(el_bar_data, el_num, lcm_list_el_bar, interval_list_el_bar)
    el_bar_p_o, el_bar_a_o = (el_bar_spectrum_list[0], el_bar_spectrum_list[1])
    el_bar_per, el_bar_amp = (el_bar_spectrum_list[2], el_bar_spectrum_list[3])
    print("lcm_list_el_bar", lcm_list_el_bar)
    print("interval_list_el_bar", interval_list_el_bar)
    print("######################")
    print("el_bar_per", el_bar_per)
    print("el_bar_amp", el_bar_amp)
    print("######################")
    spectrum_p = [el_p_o, el_bar_p_o]
    spectrum_a = [el_a_o, el_bar_a_o]
    spectrum_data_specs = [[0.75, "r", el_symb_list[i]], [0.75, "b", el_bar_list[i]]]
    spectrum_per = [el_bar_per0, el_bar_per]
    spectrum_amp = [el_bar_amp0, el_bar_amp]
    marker_spec_array = [[(7.5, -7.5), "s", "r", "peaks used for " + el_bar_list[i], el_symb_list[i]],
                         [(-7.5, 20), "s", "b", "peaks used for " + el_tilde_list[i], el_bar_list[i]]]
    Δt_list = [el_bar_Δt_n]
    Δt_spec_array = [[(0, (4, 4)), "r", el_symb_list[i]]]
    spectrum_tit = "test"
    """
    fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                       spectrum_per, spectrum_amp, marker_spec_array,
                       Δt_list, Δt_spec_array, spectrum_tit, 1, [5], 0,
                       [0, 0])
    """
    data_spec_array = [[0.8, "r", el_symb_list[i], 1],
                       [0.8, "b", el_bar_list[i], 1]]
    y_label = el_symb_list[i] + " " + el_unit
    ref_spec_array = [[0.8, "k", ref_symb, 1], [0.8, "chocolate", ref_bar, 1]]
    ref_y_spec_list = [ref_symb + " " + ref_unit, "chocolate"]
    tit = "test"
    for j in range(0, len(xlimit_list)):
        ylimits = [0, 0, 0]
        vline_specs = ["gold", 0.75, 1]
        z_order = 1
        nongra_fac = 1
        """
        plot_func_6([data_air[i], el_bar_data], data_spec_array, y_label,
                    [ref_data, ref_smooth], ref_spec_array, ref_y_spec_list,
                    tit, MJD_0, MJD_end, xlimit_list[j], ylimits,
                    vline_specs, z_order, nongra_fac, 0,
                    [], [], [], [], [0, 0])
        """
    
    # return(1)
    el_bar_data_list_list.append(el_bar_data)
    el_bar_per0_list.append(el_bar_per0)
    el_bar_amp0_list.append(el_bar_amp0)
    el_bar_Δt_n_list.append(el_bar_Δt_n)
    el_bar_p_o_list.append(el_bar_p_o)
    el_bar_a_o_list.append(el_bar_a_o)
    el_bar_per_list.append(el_bar_per)
    el_bar_amp_list.append(el_bar_amp)
print("")
print("DONE!!!")
# %%
el_hat_data_list_list = []
el_hat_per0_list = []
el_hat_amp0_list = []
el_hat_Δt_n_list = []
el_hat_p_o_list = []
el_hat_a_o_list = []
el_hat_per_list = []
el_hat_amp_list = []
for i in range(0, len(data_air)):
    el_bar_data = el_bar_data_list_list[i]
    el_bar_per = el_bar_per_list[i]
    el_bar_amp = el_bar_amp_list[i]
    # for plots
    el_p_o, el_a_o = el_p_o_list[i], el_a_o_list[i]
    el_bar_p_o, el_bar_a_o = el_bar_p_o_list[i], el_bar_a_o_list[i]
    el_bar_per0, el_bar_amp0 = el_bar_per0_list[i], el_bar_amp0_list[i]
    el_bar_Δt_n = el_bar_Δt_n_list[i]
    # data smoothing
    el_hat_data_list = smoother(el_bar_data, el_bar_per, el_bar_amp, lcm_list_el_bar, el_num, q)
    el_hat_data = el_hat_data_list[0]
    el_hat_per0, el_hat_amp0 = el_hat_data_list[1], el_hat_data_list[2]
    el_hat_Δt_n = el_hat_data_list[4]
    # smoothed spectrum
    el_hat_spectrum_list = spectrum(el_hat_data, el_num, lcm_list_el_hat, interval_list_el_hat)
    el_hat_p_o, el_hat_a_o = (el_hat_spectrum_list[0], el_hat_spectrum_list[1])
    el_hat_per, el_hat_amp = (el_hat_spectrum_list[2], el_hat_spectrum_list[3])
    
    spectrum_p = [el_p_o, el_bar_p_o, el_hat_p_o]
    spectrum_a = [el_a_o, el_bar_a_o, el_hat_a_o]
    spectrum_data_specs = [[0.75, "r", el_symb_list[i]],
                           [0.75, "b", el_bar_list[i]],
                           [0.75, "g", el_hat_list[i]]]
    spectrum_per = [el_bar_per0, el_hat_per0, el_hat_per]
    spectrum_amp = [el_bar_amp0, el_hat_amp0, np.array(el_hat_amp)]
    marker_spec_array = [[(7.5, -7.5), "s", "r", "peaks used for " + el_bar_list[i], el_symb_list[i]],
                         [(-7.5, -20), "s", "b", "peaks used for " + el_hat_list[i], el_bar_list[i]],
                         [(-7.5, -20), "s", "g", "peaks used for " + el_tilde_list[i], el_hat_list[i]]]
    Δt_list = [el_bar_Δt_n, el_hat_Δt_n]
    Δt_spec_array = [[(0, (4, 4)), "r", el_symb_list[i]], [(0, (4, 4)), "b", el_bar_list[i]]]
    spectrum_tit = "test"
    """
    fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                       spectrum_per, spectrum_amp, marker_spec_array,
                       Δt_list, Δt_spec_array, spectrum_tit, 1, [4, 16], 0,
                       [0, 0])
    """
    data_spec_array = [[0.8, "b", el_bar_list[i], 1], [0.8, "g", el_hat_list[i], 1]]
    y_label = el_symb_list[i] + " " + el_unit
    ref_spec_array = [[0.8, "chocolate", ref_bar, 1]]
    ref_y_spec_list = [ref_symb + " " + ref_unit, "chocolate"]
    tit = "test"
    for j in range(0, len(xlimit_list)):
        yoffset = 6.8695 * 10**6
        ylimits = [j * (yoffset+40), j * (yoffset + 65)]
        vline_specs = ["gold", 0.75, 1]
        z_order = 1
        nongra_fac = 1
        """
        plot_func_6([el_bar_data, el_hat_data], data_spec_array, y_label,
                    [ref_smooth], ref_spec_array, ref_y_spec_list,
                    tit, MJD_0, MJD_end, xlimit_list[j], ylimits,
                    vline_specs, z_order, nongra_fac, 0,
                    [], [], [], [], [0, 0])
        """
    print("TEST SUCCESFUL!!!")
    # return(1)
    el_hat_data_list_list.append(el_hat_data)
    el_hat_per0_list.append(el_hat_per0)
    el_hat_amp0_list.append(el_hat_amp0)
    el_hat_Δt_n_list.append(el_hat_Δt_n)
    el_hat_p_o_list.append(el_hat_p_o)
    el_hat_a_o_list.append(el_hat_a_o)
    el_hat_per_list.append(el_hat_per)
    el_hat_amp_list.append(el_hat_amp)
print("DONE!!!")
#%%
N_n = "n_" + str(lcm_list_el[2]) + "_" + str(lcm_list_el_bar[2])
N_n_title = "N = (" + str(lcm_list_el[2]) + "," + str(lcm_list_el_bar[2]) + ")"
vline_list1 = [59958.915, 59961.875, 59967.125, 59988.71,
              59995.416, 59999.5, 60001.78125, 60002.427,
              60017.17, 60018.17, 60023.9375, 60026.375, 60053.389]
vline_list_specs1 = ['k', 0.25, 1, "CME"]
vline_list_specs2 = ['r', 0.25, 1, "MAN"]
file_name = "updates/update 8/air_data/sentinel23/" 
path = os.getcwd() + "/" + file_name[:-1]
if (os.path.exists(path) == False):
    os.mkdir(os.getcwd() + "/" + file_name[:-1])
for i in range(0, len(data_air)):
    name = file_name_list[0]
    el_data = data_air[i]
    el_p_o, el_a_o = el_p_o_list[i], el_a_o_list[i]
    
    el_bar_data = el_bar_data_list_list[i]
    el_bar_p_o, el_bar_a_o = el_bar_p_o_list[i], el_bar_a_o_list[i]
    el_bar_per0, el_bar_amp0 = el_bar_per0_list[i], el_bar_amp0_list[i]
    
    el_hat_data = el_hat_data_list_list[i]
    el_hat_p_o, el_hat_a_o = el_hat_p_o_list[i], el_hat_a_o_list[i]
    el_hat_per0, el_hat_amp0 = el_hat_per0_list[i], el_hat_amp0_list[i]
    
    spectrum_p = [el_p_o, el_bar_p_o, el_hat_p_o]
    spectrum_a = [el_a_o, el_bar_a_o, el_hat_a_o]
    spectrum_data_specs = [[0.75, "r", el_symb_list[i]],
                           [0.75, "b", el_bar_list[i]],
                           [0.75, "g", el_hat_list[i]]]
    spectrum_per = [el_bar_per0, el_hat_per0]
    spectrum_amp = [el_bar_amp0, el_hat_amp0]
    marker_spec_array = [[(7.5, -7.5), "s", "r", "peaks used for " + el_bar_list[i], el_symb_list[i]],
                         [(-7.5, -20), "s", "b", "peaks used for " + el_hat_list[i], el_bar_list[i]]]
    Δt_list = [el_bar_Δt_n, el_hat_Δt_n]
    Δt_spec_array = [[(0, (4, 4)), "r", el_symb_list[i]], [(0, (4, 4)), "b", el_bar_list[i]]]
    spectrum_tit = "Amplitude - Period - Spectrum of " + el_symb_list[i] + ", " + N_n_title
    file_name_fft = "spec/spec_" + ["x", "y", "z"][i] + "_" + N_n + ".png"
    file_name_fft = file_name + file_name_fft
    fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                       spectrum_per, spectrum_amp, marker_spec_array,
                       Δt_list, Δt_spec_array, spectrum_tit, 1, [10, 100], 0,
                       [1, file_name_fft])
    # plot hat
    data_array_list = [el_hat_data]
    data_spec_array = [[0.8, "g", el_hat_list[i], 1]]
    y_label = el_symb_list[i] + " " + el_unit
    ref_spec_array = [[0.8, "chocolate", ref_bar, 1]]
    ref_y_spec_list = [ref_symb + " " + ref_unit, "chocolate"]
    tit = el_hat_list[i] + " and " + ref_bar + ", " + N_n_title
    ylimits = [0, 0]
    vline_specs = ["gold", 0.75, 1]
    z_order = 1
    nongra_fac = 1
    file_name_hat = "hat/hat_" + ["x", "y", "z"][i] + "_" + N_n + ".png"
    file_name_hat = file_name + file_name_hat
    plot_func_6(data_array_list, data_spec_array, y_label,
                [ref_smooth], ref_spec_array, ref_y_spec_list,
                tit, MJD_0, MJD_end, [0, 0], ylimits,
                vline_specs, z_order, nongra_fac, 0,
                vline_list1, vline_list_specs1, vline_list2, vline_list_specs2,
                [1, file_name_hat])
    # diff
    data_array_list = [np.vstack((el_bar_data[:,0], (el_bar_data - el_hat_data)[:, 1])).T]
    data_spec_array = [[0.5, "turquoise", el_bar_list[i] + '-' + el_hat_list[i], 1]]
    tit = el_bar_list[i] + '-' + el_hat_list[i] + " and " + ref_bar + ", " + N_n_title
    file_name_diff = "diff/diff_" + ["x", "y", "z"][i] + "_" + N_n + ".png"
    file_name_diff = file_name + file_name_diff
    plot_func_6(data_array_list, data_spec_array, y_label,
                [ref_smooth], ref_spec_array, ref_y_spec_list,
                tit, MJD_0, MJD_end, [0, 0], ylimits,
                vline_specs, z_order, nongra_fac, 0,
                vline_list1, vline_list_specs1, vline_list2, vline_list_specs2,
                [1, file_name_diff])
# %%
