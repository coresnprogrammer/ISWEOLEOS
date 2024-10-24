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
from functions import step_data_generation, fft, sec_to_day
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
#%%
print(os.getcwd())
os.chdir('/Users/levin/Documents/Uni/Bachelorthesis new')
print(os.getcwd())
#%%
foldername = "newoscele/Akzelerometer"
foldername_ref = "newoscele/GRACE_LST"
file_name_list = ["nongra"]
ref_str = "rho"
step_fac = 1
MJD_interval = [58301, 58451]
q = 1
n_partition = 0.5
lsa_method = 3
ε = 10 ** (-3)
lcm_list_el = [1, 50, 10, 0.09, 0.95, 10]
lcm_list_el_bar = [0, 50, 1, 0.09, 0.95, 15]
lcm_list_el_hat = [0, 50, 1, 1, 0.95, 15]
lcm_list_ref = [0, 0.1, 1, 0.09, 0.25, 10]
interval_list_el = [[0, 0]] # [[0.05, 0.06], [0.07, 0.09], [0.015, 0.025]]
interval_list_el_bar = [[0,0]]
interval_list_el_hat = [[0.35, 0.45], [0.55, 0.56]]
interval_list_ref = [[0.01, 0.0125], [0.0125, 0.015], [0.015, 0.02], [0.02, 0.025]]
xlimit_list = [[0, 0]]
#%%
MJD_0, MJD_end = MJD_interval[0], MJD_interval[1]
n_days_tot = MJD_end - MJD_0
str_normal_nongra = ["normal", "nongra"]

ref_num = 9
ref_str = 'rho'
ref_word, ref_symb = 'air density', r'$\varrho$'
ref_unit, ref_bar = r'[$kg m^{-3}$]', r'$\bar{\varrho}$'
ref_tild = r'$\tilde{\varrho}$'

ref_name = foldername_ref + 2 * ("/year_" + str_normal_nongra[1]) + "_" + ref_str + ".txt"
data_ref = np.loadtxt(ref_name, skiprows=1)
data_ref = array_denormalize(data_ref)
data_ref = array_modifier(data_ref, MJD_0, n_days_tot)
data_ref = array_normalize(data_ref, 0)
data_ref = data_ref[1:]
#%%
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
fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                   spectrum_per, spectrum_amp, marker_spec_array,
                   Δt_list, Δt_spec_array, spectrum_tit, 1, [1], 4)
print("DONE!!!")
#%%
acc_file_name = foldername + '/year/all.txt'
data = np.loadtxt(acc_file_name, skiprows = 0)
data = sec_to_day(data) # now with days
data[0, 0] = data[0, 0] + 51544 + 0.5
data = array_denormalize(data)
data = array_modifier(data, MJD_0, n_days_tot)
data = array_normalize(data, 0)

data = data[1:]
#%%
α = 0.8
lw = 1
data_spec_array_tot = [[α, 'r', 'S', lw],
                       [α, 'b', 'W', lw],
                       [α, 'g', 'R', lw]]
xyz_list = ['S', 'W', 'R']
acc_str_list = [[r'$a_S$', r'$\bar{a}_S$'],
                [r'$a_W$', r'$\bar{a}_W$'],
                [r'$a_R$', r'$\bar{a}_R$']]

color_array = [['royalblue', 'navy'],
               ['lawngreen', 'darkgreen'],
               ['orangered', 'crimson']]
rws_data_list_list = []
rsw_smooth_list_list = []
for i in range(0, 3):
    acc_symb = acc_str_list[i][0]
    acc_bar = acc_str_list[i][1]
    
    x_data = data[:, 0]
    y_data = data[:, 1 + i]
    if (i == 0):
        y_data = - y_data
    rsw_data = np.vstack((x_data, y_data)).T
    rws_data_list_list.append(rsw_data)
    # data_spectrum
    rsw_data_spectrum_list = spectrum(rsw_data, 0, lcm_list_el, interval_list_el)
    rsw_p_o, rsw_a_o = rsw_data_spectrum_list[0], rsw_data_spectrum_list[1]
    rsw_per, rsw_amp = rsw_data_spectrum_list[2], rsw_data_spectrum_list[3]
    
    # data smoothing
    rsw_smooth_list = smoother(rsw_data, rsw_per, rsw_amp, lcm_list_el, 0, q)
    rsw_smooth = rsw_smooth_list[0]
    rsw_per0, rsw_amp0 = rsw_smooth_list[1], rsw_smooth_list[2]
    rsw_Δt_n = rsw_smooth_list[4]
    rsw_smooth_list_list.append(rsw_smooth)
    # smoothed spectrum
    rsw_smooth_spectrum_list = spectrum(rsw_smooth, 0, lcm_list_el_bar, interval_list_el_bar)
    rsw_smooth_p_o, rsw_smooth_a_o = rsw_smooth_spectrum_list[0], rsw_smooth_spectrum_list[1]
    rsw_smooth_per, rsw_smooth_amp = rsw_smooth_spectrum_list[2], rsw_smooth_spectrum_list[3]
    
    # acc spectrum
    spectrum_p = [rsw_p_o, rsw_smooth_p_o]
    spectrum_a = [rsw_a_o, rsw_smooth_a_o]
    spectrum_data_specs = [[0.75, color_array[i][0], acc_symb],
                           [0.75, color_array[i][1], acc_bar]]
    spectrum_per = [rsw_per0, rsw_smooth_per]
    spectrum_amp = [rsw_amp0, rsw_smooth_amp]
    #spectrum_per = []
    #spectrum_amp = []
    marker_spec_array = [[(7.5, -7.5), 's', color_array[i][0], 'peaks used for ' + acc_bar, acc_symb],
                         [(-7.5, 20), 'D', color_array[i][1], 'peaks of ' + acc_bar, acc_bar]]
    #marker_spec_array = []
    Δt_list = [rsw_Δt_n]
    Δt_spec_array = [[(0, (4, 4)), 'k', acc_symb]]
    spectrum_tit = 'amplitude - period spectrum of ' + acc_symb
    fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                       spectrum_per, spectrum_amp, marker_spec_array,
                       Δt_list, Δt_spec_array, spectrum_tit, 1, [1], 4)
#%%
# data plot
vline_list1 = [58372, 58375, 58383, 58399]
vline_list2 = [58164, 58186, 58309, 58354]
ylimit_fac_array = [[2.5, 0.75], [0.065, 0.015], [0.75, 0.5], [0, 7]]
nongra_fac_list = [1, 1, 1, 1]
xlimit_list = [[54, 92]]
for i in range(0, 3):
    rsw_data = rws_data_list_list[i]
    rsw_smooth = rsw_smooth_list_list[i]
    data_array_list = [rsw_data, rsw_smooth]
    
    
    data_spec_array = [[α, color_array[i][0], acc_symb, lw],
                       [α, color_array[i][1], acc_bar, lw]]
        
    y_label = r'acceleration [$m s^{-2}$]'
    ref_spec_array = [[1, 'k', ref_bar, 1]]
    ref_y_spec_list = [ref_bar, 'k']
    tit = 'GRACE data: acceleration in ' + xyz_list[i]
    mean = np.mean(rsw_data[:, 1])
    std = np.std(rsw_data[:, 1])
    fac_low, fac_upp = ylimit_fac_array[i][0], ylimit_fac_array[i][1]
    ylimits = [mean - fac_low * std, mean + fac_upp * std, 1]
    ylimit_list = [ylimits, [0, 0, 0]]
    vline_col = 'gold'
    z_order = 1
    nongra_fac = 1
    
    for j in range(0, len(xlimit_list)):
        plot_func_6(data_array_list, data_spec_array, y_label,
                    [ref_smooth], ref_spec_array, ref_y_spec_list,
                    tit, MJD_0, MJD_end, xlimit_list[j], ylimits,
                    vline_col, z_order, nongra_fac, 0, vline_list1, 0.5, vline_list2, 1)
#%%
def acc_analysis(foldername, foldername_ref, interval, ref_str,
                 q, lcm_list_el, lcm_list_el_smooth, lcm_list_ref,
                 interval_list_el, interval_list_el_smooth, interval_list_ref,
                 ylimit_array, xlimits_list):
    # foldername: name of folder of data, string like
    # MJD_interval: data interval
    # lcm_list = [p_spec, prec, l, m, p_max, threshold_quotient, limit]:
    #            p_spec: manuel chosen period
    #            prec: precision of lcm
    #            l: if there is only one dominant period -> l * period as Δt
    #            m: lcm * m as Δt
    #            p_max: only periods smaller than p_max are detected
    #            threshold_quotient: periods with amplitude thresh_quot * amp_max are detected
    #            limit: lcm * m has to be smaller than limit
    # ε: precision for least squares adjustment
    # interval_list_el: intervals in data spectrum to look for peaks
    # interval_list_ref: intervals in ref spectrum to look for peaks
    t_tot_a = tt.time()
    MJD_0 = interval[0]
    MJD_end = interval[1]
    n_days_tot = MJD_end - MJD_0
    #n_days_tot = MJD_end - MJD_0
    
    string_list = ["u_sat", "beta_sun", "u_sun", "a", "e", "i", "Omega_upp", "omega_low", "T0", "rho", "r", "h_ell"]
    name_list = [["satellite anomaly", r'$u_{sat}$', r'[$^{\circ}$]', r'$\bar{u}_{sat}$', r'$\tilde{u}_{sat}$'],
                 ["beta angle", r'$\beta_{\odot}$', r'[$^{\circ}$]', r'$\bar{\beta}_{\odot}$', r'$\tilde{\beta}_{\odot}$'],
                 ["sun anomaly", r'$u_{\odot}$', r'[$^{\circ}$]', r'$\bar{u}_{\odot}$', r'$\tilde{u}_{\odot}$'],
                 ['semi-major axis', r'$a$', r'[$m$]', r'$\bar{a}$', r'$\tilde{a}$'],
                 ['eccentricity', r'$e$', r'[$-$]', r'$\bar{e}$', r'$\tilde{e}$'],
                 ['inclination', r'$i$', r'[$^{\circ}$]', r'$\bar{i}$', r'$\tilde{i}$'],
                 ['longitude of ascending node', r'$\Omega$', r'[$^{\circ}$]', r'$\bar{\Omega}$', r'$\tilde{\Omega}$'],
                 ['argument of periapsis', r'$\omega$', r'[$^{\circ}$]', r'$\bar{\omega}$', r'$\tilde{\omega}$'],
                 ['time of periapsis passage', r'$T_0$', r'[$s$]', r'$\bar{T}_0$', r'$\tilde{T}_0$'],
                 ['air density', r'$\varrho$', r'[$kg m^{-3}$]', r'$\bar{\varrho}$', r'$\tilde{\varrho}$'],
                 ['radius', r'$r$', r'[$m$]', r'$\bar{r}$', r'$\tilde{r}$'],
                 ['ell. height', r'$h_{ell}$', r'[$m$]', r'$\bar{h}_{ell}$', r'$\tilde{h}_{ell}$']]
    
    ref_num = arg_finder(string_list, ref_str)
    ref_word, ref_symb = name_list[ref_num][0], name_list[ref_num][1]
    ref_unit, ref_bar = name_list[ref_num][2], name_list[ref_num][3]
    ref_tild = name_list[ref_num][4]
    
    # prepare data
    acc_file_name = foldername + '/year/all.txt'
    data = np.loadtxt(acc_file_name, skiprows = 0)
    data = sec_to_day(data) # now with days
    data[0, 0] = data[0, 0] + 51544 + 0.5
    data = array_denormalize(data)
    data = array_modifier(data, MJD_0, n_days_tot)
    data = array_normalize(data, 0)
    
    data = data[1:]
    
    ref_name = foldername_ref + 2 * ("/year_" + 'nongra') + '_' + ref_str + ".txt"
    data_ref = np.loadtxt(ref_name, skiprows = 1)
    data_ref = array_denormalize(data_ref)
    data_ref = array_modifier(data_ref, MJD_0, n_days_tot)
    data_ref = array_normalize(data_ref, 0)
    data_ref = data_ref[1:]
    
    # spectrum(data, el_num, lcm_list, interval_list)
    # spectrum_list = [p_o_list, a_o_list, per_list, amp_list]
    
    # smoother(data, per_list, amp_list, lcm_list, el_num, q)
    # smooth_list = [data_smoothed, per0_list, amp0_list, n_Δt, Δt_n, Δt_dt]
    
    # fitter(data, per0_list, amp0_list, n_partition, lsa_method, ε)
    # fit_list = [data_fitted, para_fit_array, para_e_fit_array, m0_list]
    
    # PROCESSING REF
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
    ref_smooth_p_o, ref_smooth_a_o = ref_smooth_spectrum_list[0], ref_smooth_spectrum_list[1]
    ref_smooth_per, ref_smooth_amp = ref_smooth_spectrum_list[2], ref_smooth_spectrum_list[3]
    # ref spectrum
    spectrum_p = [ref_p_o, ref_smooth_p_o]
    spectrum_a = [ref_a_o, ref_smooth_a_o]
    spectrum_data_specs = [[0.75, 'r', ref_symb],
                           [0.75, 'b', ref_bar]]
    spectrum_per = [ref_per0]
    spectrum_amp = [ref_amp0]
    marker_spec_array = [[(7.5, 0), 's', 'r', 'peaks used for ' + ref_bar, ref_symb]]
    Δt_list = [ref_Δt_n]
    Δt_spec_array = [[(0, (4, 4)), 'tab:green', ref_symb]]
    spectrum_tit = 'amplitude - period spectrum of non gravitational \ndata of ' + ref_word
    fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                       spectrum_per, spectrum_amp, marker_spec_array,
                       Δt_list, Δt_spec_array, spectrum_tit, 1)
    α = 0.8
    lw = 1
    data_spec_array_tot = [[α, 'r', 'S', lw],
                           [α, 'b', 'W', lw],
                           [α, 'g', 'R', lw]]
    xyz_list = ['S', 'W', 'R']
    acc_str_list = [[r'$a_S$', r'$\bar{a}_S$'],
                    [r'$a_W$', r'$\bar{a}_W$'],
                    [r'$a_R$', r'$\bar{a}_R$']]
    
    color_array = [['lawngreen', 'darkgreen'],
                   ['royalblue', 'navy'],
                   ['orangered', 'crimson']]
    
    for i in range(0, 3):
        acc_symb = acc_str_list[i][0]
        acc_bar = acc_str_list[i][1]
        
        x_data = data[:, 0]
        y_data = data[:, 1 + i]
        if (i == 0):
            y_data = - y_data
        rsw_data = np.vstack((x_data, y_data)).T
        # data_spectrum
        rsw_data_spectrum_list = spectrum(rsw_data, 0, lcm_list_el, interval_list_el)
        rsw_p_o, rsw_a_o = rsw_data_spectrum_list[0], rsw_data_spectrum_list[1]
        rsw_per, rsw_amp = rsw_data_spectrum_list[2], rsw_data_spectrum_list[3]
        
        # data smoothing
        rsw_smooth_list = smoother(rsw_data, rsw_per, rsw_amp, lcm_list_el, 0, q)
        rsw_smooth = rsw_smooth_list[0]
        rsw_per0, rsw_amp0 = rsw_smooth_list[1], rsw_smooth_list[2]
        rsw_Δt_n = rsw_smooth_list[4]
        # smoothed spectrum
        rsw_smooth_spectrum_list = spectrum(rsw_smooth, 0, lcm_list_el_smooth, interval_list_el_smooth)
        rsw_smooth_p_o, rsw_smooth_a_o = rsw_smooth_spectrum_list[0], rsw_smooth_spectrum_list[1]
        rsw_smooth_per, rsw_smooth_amp = rsw_smooth_spectrum_list[2], rsw_smooth_spectrum_list[3]
        
        # acc spectrum
        spectrum_p = [rsw_p_o, rsw_smooth_p_o]
        spectrum_a = [rsw_a_o, rsw_smooth_a_o]
        spectrum_data_specs = [[0.75, color_array[i][0], acc_symb],
                               [0.75, color_array[i][1], acc_bar]]
        spectrum_per = [rsw_per0, rsw_smooth_per]
        spectrum_amp = [rsw_amp0, rsw_smooth_amp]
        marker_spec_array = [[(7.5, -7.5), 's', color_array[i][0], 'peaks used for ' + acc_bar, acc_symb],
                             [(-7.5, 20), 'D', color_array[i][1], 'peaks of ' + acc_bar, acc_bar]]
        Δt_list = [rsw_Δt_n]
        Δt_spec_array = [[(0, (4, 4)), 'k', acc_symb]]
        spectrum_tit = 'amplitude - period spectrum of ' + acc_symb
        fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                           spectrum_per, spectrum_amp, marker_spec_array,
                           Δt_list, Δt_spec_array, spectrum_tit, 1)
        
        # data plot
        data_array_list = [rsw_data, rsw_smooth]
        
        data_spec_array = [[α, color_array[i][0], acc_symb, lw],
                           [α, color_array[i][1], acc_bar, lw]]
        
        y_label = r'acceleration [$m s^{-2}$]'
        ref_spec_array = [[1, 'k', ref_bar, 1]]
        ref_y_spec_list = [ref_bar, 'k']
        tit = 'GRACE data: acceleration in ' + xyz_list[i]
        mean = np.mean(rsw_data[:, 1])
        std = np.std(rsw_data[:, 1])
        fac_low, fac_upp = ylimit_array[i][0], ylimit_array[i][1]
        ylimits = [mean - fac_low * std, mean + fac_upp * std, 1]
        ylimit_list = [ylimits, [0, 0, 0]]
        vline_col = 'gold'
        z_order = 1
        nongra_fac = 1
        
        for i in range(0, len(xlimits_list)):
            plot_func_6(data_array_list, data_spec_array, y_label,
                        [ref_smooth], ref_spec_array, ref_y_spec_list,
                        tit, MJD_0, MJD_end, xlimits_list[i], ylimit_list[i],
                        vline_col, z_order, nongra_fac)
    print("YYYYEEEEAAAAHHHH!!!")

#storm: 58356
#acc_analysis('newoscele/Akzelerometer', 'newoscele/GRACE_LST', [58346, 58366],'rho',
#             1, [0, 10, 1, 0.09, 0.75, 10], [0, 10, 1, 0.09, 0.75, 10],  [0, 10, 1, 0.09, 0.75, 10],
#             [[0, 0]], [[0, 0]],
#             [[0.01, 0.0125], [0.0125, 0.015], [0.015, 0.02], [0.02, 0.025]],
#             [[5, 3], [8, 4], [3, 5]],
#             [[0, 0, 0, 0], [58355.5, 58358.5, 1, 0]])

#acc_analysis('newoscele/Akzelerometer', 'newoscele/GRACE_LST', [58276, 58436],'rho',
#             1, [0, 0.1, 1, 0.09, 0.25, 10],
#             [[0.01, 0.0125], [0.0125, 0.015], [0.015, 0.02], [0.02, 0.025]],
#             [[2, 8], [3, 4], [4, 4]])