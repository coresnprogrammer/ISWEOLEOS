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
from functions import step_data_generation, fft
from functions import ele_gen_txt, step_data_generation
#Δa_day(n_rev_day, fac, a, ρ)
#%%
print(os.getcwd())
os.chdir('/Users/levin/Documents/Uni/Bachelorthesis new')
print(os.getcwd())
#%%
#ele_gen_txt("sat24/sentinel", 1)
#%%
foldername = "sat24/sentinel"
foldername_ref = "sat24/sentinel"
file_name_list = ["normal"]
ref_str = "rho"
step_fac = 1
MJD_interval = [60440, 60442]
q = 1
n_partition = 0.5
lsa_method = 3
ε = 10 ** (-3)
lcm_list_el = [10, 50, 10, 0.09, 0.95, 10]
lcm_list_el_bar = [0, 50, 1, 0.09, 0.95, 15]
lcm_list_el_hat = [0, 50, 1, 1, 0.95, 15]
lcm_list_ref = [1, 0.1, 1, 0.09, 0.25, 10]
interval_list_el = [[0, 0]] # [[0.05, 0.06], [0.07, 0.09], [0.015, 0.025]]
interval_list_el_bar = [[0.015, 0.025], [6, 8]]
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
"""
ref_name = foldername_ref + 2 * ("/year_" + str_normal_nongra[1]) + "_" + ref_str + ".txt"
data_ref = np.loadtxt(ref_name, skiprows=1)
data_ref = array_denormalize(data_ref)
data_ref = array_modifier(data_ref, MJD_0, n_days_tot)
data_ref = array_normalize(data_ref, 0)
data_ref = data_ref[1:]
"""
data_ref = np.vstack((np.arange(0,1,0.0001), np.sin(np.arange(0, 10000)))).T
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
v_line_specs = ['b', 0.5, 1]
xlimits = [0, 0]
ylimits = [0, 0]
ylimits = [0, 0]
v_line_list = []
fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                   spectrum_per, spectrum_amp, marker_spec_array,
                   Δt_list, Δt_spec_array, spectrum_tit, 1, [1], 4,
                   v_line_list, v_line_specs,
                   xlimits, ylimits, [0])
print("DONE!!!")
#%%
# import normalized data ##############################
n_files = len(file_name_list)
if (file_name_list[0] == 'normal' and n_files == 1):
    file_number = 0
if (file_name_list[0] == 'nongra' and n_files == 1):
    file_number = 1
else:
    file_number = 0
data_list = []
data_step_list = []
data_abs_list = []
p_o_list = []
a_o_list = []
MJD_0_list = []
for i in range(0, n_files):
    name = file_name_list[i]
    data_0 = np.loadtxt(foldername + '/year_' + name + '/year_' + name + '.txt')
    data = array_denormalize(data_0)
    data_cropped = array_modifier(data, MJD_0, n_days_tot)
    data_cropped_0 = array_normalize(data_cropped, 0)
    MJD_0 = data_cropped_0[0, 0]
    data = data_cropped_0[1:]
    data_step = step_data_generation(data, step_fac)
    
    data_abs = np.array([[0, 0]])
    for i in range(0, len(data)):
        acc_abs_i = 0
        for j in range(0, 3):
            acc_abs_i += (data[i, j +1])**2
        acc_abs_i = np.sqrt(acc_abs_i)
        row_i = np.array([[data[i, 0], acc_abs_i]])
        data_abs = np.vstack((data_abs, row_i))
    data_abs = data_abs[1:]
    
    orig_list = fft(data)
    p_o, a_o = orig_list[0], orig_list[1]
    
    data_list.append(data)
    data_step_list.append(data_step)
    data_abs_list.append(data_abs)
    p_o_list.append(p_o)
    a_o_list.append(a_o)
    MJD_0_list.append(MJD_0)
#%%
vline_list1 = [59958.915, 59961.875, 59967.125, 59988.71,
               59995.416, 59999.5, 60001.78125, 60002.427,
               60017.17, 60018.17, 60023.9375, 60026.375, 60053.389]
vline_list2 = []
vline_list_specs1 = ['k', 0.25, 1, "CME"]
vline_list_specs2 = ['r', 0.25, 1, "MAN"]
vline_specs = ["gold", 0.75, 1]
ylimit_fac_array = [[12, 10], [1, 15], [3, 14], [0, 7]]
nongra_fac_list = [1, 1, 1, 1]
xlimit_list = [[0, 0]]
print("CREATING PLOTS ...")
col_list = np.array([['darkred', 'orangered'],
                     ['darkblue', 'dodgerblue'],
                     ['darkolivegreen', 'lime'],
                     ['darkmagenta', 'hotpink']])
direction_list = np.array(['R', 'S', 'W'])
normal_nongra_text = np.array(['normal', 'non grav.'])

ref_spec_array = [[0, 'k', ref_symb, 1]]
ref_y_spec_list = [ref_symb + ' ' + ref_unit, 'k']

for i in range(0, 3):
    data_spec_array = []
    new_data_list = []
    y_label = r"acceleration [$m s^{-2}$]"
    ylimit_list = []
    for j in range(0, n_files):
        tit = 0
        if (n_files == 2):
            tit = 'normal and nongra acceleration in ' + direction_list[i]
        else:
            tit = file_name_list[0] + ' acceleration in ' + direction_list[i]
        data_spec_array.append([0.8, col_list[i][j + file_number],
                                normal_nongra_text[j + file_number], 1])
        per_list_list, amp_list_list, marker_spec_array = [], [], []
        Δt_list, Δt_spec_array = [], []
        data_j = np.vstack((data_list[j][:, 0], data_list[j][:, i + 1])).T
        data_j = step_data_generation(data_j, 100)
        new_data_list.append(data_j)
        ylimit = np.std(data_list[j][:, i + 1])
        ylimit_list.append(ylimit)

        #fft_logplot_spec_3(p_o_list, a_o_list, data_spec_array,
        #                   per_list_list, amp_list_list, marker_spec_array,
        #                   Δt_list, Δt_spec_array, tit, log)
        ylimit = max(ylimit_list)
        fac_low = ylimit_fac_array[i][0]
        fac_upp = ylimit_fac_array[i][1]
        ylimits = [- fac_low * ylimit, fac_upp * ylimit, 1]
        plot_func_6(new_data_list, data_spec_array[:3], y_label,
                    [data_ref], ref_spec_array, ref_y_spec_list,
                    tit, MJD_0, MJD_end, xlimit_list[0], ylimits,
                    vline_specs, 1, nongra_fac_list[j], 0,
                    vline_list1, vline_list_specs1,
                    vline_list2, vline_list_specs2, [0])
# %%