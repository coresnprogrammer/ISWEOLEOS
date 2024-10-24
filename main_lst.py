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
from functions import get_n_s, lst_sqrs_adjstmnt_adv, insert
from functions import fit_func_1, fit_func_1_derivs
from functions import fit_func_2, fit_func_2_derivs
from functions import fit_func_3, fit_func_3_derivs
from functions import lst_sqrs_adjstmnt, ravel, fit_func_adv, n_objects
from functions import O_minus_C_adv, new_Δx, m0, quotient, encode_list, list_code, N_inv_co, round_list
#Δa_day(n_rev_day, fac, a, ρ)
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
foldername = "All Data new/osceleswarm"
file_name_list = ["normal", "nongra"]
el_str = "a"
ref_str = "rho"
MJD_interval = [59950, 60050]
q = 1
n_partition = 1
lsa_method = 5
ε = 10 ** (-3)
# lcm_list = [p_spec, prec, N, p_max, thresh_quot, limit]
lcm_list_el = [0, 50, 1, 0.09, 0.95, 10]
lcm_list_el_bar = [0, 10, 1, 0.09, 0.95, 15]
lcm_list_el_hat = [0, 50, 1, 1, 0.95, 15]
lcm_list_ref = [0, 0.1, 1, 0.09, 0.25, 10]
interval_list_el = [[0.056, 0.06], [0.072, 0.075], [0.08, 0.085]] # [[0.58, 0.5825]]
interval_list_el_bar = [[0.021, 0.023], [0.0725, 0.0775], [0.08, 0.085]]
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

el_num = arg_finder(string_list, el_str)
el_word, el_symb = name_list[el_num][0], name_list[el_num][1]
el_unit, el_bar = name_list[el_num][2], name_list[el_num][3]
el_hat, el_tilde = name_list[el_num][4], name_list[el_num][5]

ref_num = arg_finder(string_list, ref_str)
ref_word, ref_symb = name_list[ref_num][0], name_list[ref_num][1]
ref_unit, ref_bar = name_list[ref_num][2], name_list[ref_num][3]
ref_hat, ref_tilde = name_list[ref_num][4], name_list[ref_num][5]

# prepare data
str_normal_nongra = ["normal", "nongra"]
data_both = []
for i in range(0, len(file_name_list)):
    file_name = file_name_list[i]
    name_i = foldername + 2 * ("/year_" + file_name) + "_" + el_str + ".txt"
    data_i = np.loadtxt(name_i, skiprows=1)  # load data
    data_i = array_denormalize(data_i)  # denormalize data
    data_i = array_modifier(data_i, MJD_0, n_days_tot)  # trim data
    data_i = array_normalize(data_i, 0)  # normalize data
    data_i = data_i[1:]  # cut MJD_0
    data_both.append(data_i)

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
                   Δt_list, Δt_spec_array, spectrum_tit, 1, [1], 4)
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
for i in range(0, 1):#len(data_both)
    name = file_name_list[i]
    el_data = data_both[i]
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
    spectrum_data_specs = [[0.75, "r", el_symb]]
    spectrum_per = [el_per]
    spectrum_amp = [el_amp]
    marker_spec_array = [[(7.5, -7.5), "s", "r", "peaks used for " + el_bar, el_symb]]
    Δt_list = []
    Δt_spec_array = [[(0, (4, 4)), "r", el_symb]]
    spectrum_tit = "test"
    # [0.58, 0.5825], [0.43, 0.445], [0.301, 0.305], [0.23, 0.24]
    # [0.15, 0.17], [0.095, 0.1], [0.085, 0.09], [0.08, 0.085],
    # [0.072, 0.075], [0.068, 0.072], [0.063, 0.068], [0.06, 0.063],
    # [0.056, 0.06], [0.053, 0.056], [0.051, 0.053]
    # auswahl: [[0.056, 0.06], [0.072, 0.075], [0.08, 0.085]]
    v_line_specs = ['b', 0.5, 1]
    #xlimits = [0.01, 0.5]
    xlimits = [0,0]
    #ylimits = [1, 10]
    ylimits = [0, 0]
    v_line_list = [0.08, 0.072, 0.056]
    fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                       spectrum_per, spectrum_amp, marker_spec_array,
                       Δt_list, Δt_spec_array, spectrum_tit, 1, [], 0,
                       v_line_list, v_line_specs,
                       xlimits, ylimits, [0])
    
print("DONE!!!")
#%%
el_bar_data_list_list = []
el_bar_per0_list = []
el_bar_amp0_list = []
el_bar_Δt_n_list = []
el_bar_p_o_list = []
el_bar_a_o_list = []
el_bar_per_list = []
el_bar_amp_list = []
for i in range(0, len(data_both)):
    el_data = data_both[i]
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
    spectrum_data_specs = [[0.75, "r", el_symb], [0.75, "b", el_bar]]
    spectrum_per = [el_bar_per0, el_bar_per]
    spectrum_amp = [el_bar_amp0, el_bar_amp]
    marker_spec_array = [[(7.5, -7.5), "s", "r", "peaks used for " + el_bar, el_symb],
                         [(-7.5, 20), "s", "b", "peaks used for " + el_tilde, el_bar]]
    Δt_list = [el_bar_Δt_n]
    Δt_spec_array = [[(0, (4, 4)), "r", el_symb]]
    spectrum_tit = "test"
    """
    fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                       spectrum_per, spectrum_amp, marker_spec_array,
                       Δt_list, Δt_spec_array, spectrum_tit, 1, [5], 0)
    """
    """
    data_spec_array = [[0.8, "b", el_bar, 1]]
    y_label = el_symb + " " + el_unit
    ref_spec_array = [[0.8, "k", ref_symb, 1], [0.8, "chocolate", ref_bar, 1]]
    ref_y_spec_list = [ref_symb + " " + ref_unit, "chocolate"]
    tit = "test"
    for j in range(0, len(xlimit_list)):
        ylimits = [0, 0, 0]
        vline_col = "gold"
        z_order = 1
        nongra_fac = 1
        plot_func_6([el_bar_data], data_spec_array, y_label,
                    [ref_data, ref_smooth], ref_spec_array, ref_y_spec_list,
                    tit, MJD_0, MJD_end, xlimit_list[j], ylimits,
                    vline_col, z_order, nongra_fac, 0, [], 1, [], 1)
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
#%%
el_hat_data_list_list = []
el_hat_per0_list = []
el_hat_amp0_list = []
el_hat_Δt_n_list = []
el_hat_p_o_list = []
el_hat_a_o_list = []
el_hat_per_list = []
el_hat_amp_list = []
for i in range(0, len(data_both)):
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
    spectrum_data_specs = [[0.75, "r", el_symb], [0.75, "b", el_bar], [0.75, "g", el_hat]]
    spectrum_per = [el_bar_per0, el_hat_per0, el_hat_per]
    spectrum_amp = [el_bar_amp0, el_hat_amp0, np.array(el_hat_amp)]
    marker_spec_array = [[(7.5, -7.5), "s", "r", "peaks used for " + el_bar, el_symb],
                         [(-7.5, -20), "s", "b", "peaks used for " + el_hat, el_bar],
                         [(-7.5, -20), "s", "g", "peaks used for " + el_tilde, el_hat]]
    Δt_list = [el_bar_Δt_n, el_hat_Δt_n]
    Δt_spec_array = [[(0, (4, 4)), "r", el_symb], [(0, (4, 4)), "b", el_bar]]
    spectrum_tit = "test"
    
    fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                       spectrum_per, spectrum_amp, marker_spec_array,
                       Δt_list, Δt_spec_array, spectrum_tit, 1, [4, 16], 0,
                       [], [], [0, 0], [0, 0], [0])
    
    data_spec_array = [[0.8, "b", el_bar, 1], [0.8, "g", el_hat, 1]]
    y_label = el_symb + " " + el_unit
    ref_spec_array = [[0.8, "k", ref_symb, 1], [0.8, "chocolate", ref_bar, 1]]
    ref_y_spec_list = [ref_symb + " " + ref_unit, "chocolate"]
    tit = "test"
    for j in range(0, len(xlimit_list)):
        yoffset = 6.8695 * 10**6
        ylimits = [j * (yoffset+40), j * (yoffset + 65)]
        vline_col = "gold"
        z_order = 1
        nongra_fac = 1
        """
        plot_func_6([el_bar_data, el_hat_data], data_spec_array, y_label,
                    [ref_data, ref_smooth], ref_spec_array, ref_y_spec_list,
                    tit, MJD_0, MJD_end, xlimit_list[j], ylimits,
                    vline_col, z_order, nongra_fac, 0, [], 1, [], 1)
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
def A_adv(t_array, para_lists_list, Δt, n_s):
    # not dependend on para_array
    # t_array normalized
    # (n_s + 1) * 6 parameters
    
    s_pd_s_mat = np.zeros((len(t_array), n_s + 1))
    print("Δt", Δt)
    print("n_s", n_s)
    i = 1
    t_i_1 = 0
    t_i = Δt
    print("begin A_mat:")
    for k in range(0, len(t_array)):
        if (t_array[k] >= i * Δt):
            i += 1
            t_i = i * Δt
            t_i_1 = (i - 1) * Δt
            #print("i = ", i)
            #print("k = ", k)
        
        fac_i_1 = (t_i - t_array[k])
        fac_i = (t_array[k] - t_i_1)
        s_pd_s_mat[k][i - 1] = fac_i_1
        s_pd_s_mat[k][i] = fac_i
    
    A_mat = s_pd_s_mat / Δt
    print("A_mat finished")
    return(A_mat)


def N_inv_co(P_mat, A_mat):
    print("calculate A_mat.T @ P_mat @ A_mat")
    inv = A_mat.T @ P_mat @ A_mat
    print("calculate det")
    det = np.linalg.det(inv)
    if (det == 0):
        print("WARNING: DET = 0")
        print("A_mat:")
        print(A_mat)
        print("P_mat:")
        print(P_mat)
        print("A_mat.T @ P_mat @ A_mat:")
        print(inv)
    term1 = np.linalg.inv(inv)
    term2 = A_mat.T @ P_mat
    return(term1 @ term2)

def N_inv_co_new(A_mat, n_s):
    length = int(len(A_mat) / n_s)
    side_diag = np.array([])
    for i in range(0, n_s):
        col_i = A_mat[i * length : (i + 1) * length, i]
        col_i_1 = A_mat[i * length : (i + 1) * length, i + 1]
        side_diag = np.append(side_diag, np.sum(col_i * col_i_1))
    diag = np.array([])
    for i in range(0, n_s+1):
        col_i = A_mat[i * length : (i + 1) * length, i]
        diag = np.append(diag, np.sum(col_i * col_i))
    inv = np.diag(side_diag, -1) + np.diag(side_diag, 1) + np.diag(diag)
    print("shape is ", np.shape(inv))
    print("now inverting")
    term1 = np.linalg.inv(inv)
    print("inverted")
    term2 = A_mat.T
    return(term1 @ term2)

def N_inv_co_new_new(A_mat, P_mat, n_s):
    length = int(len(A_mat) / n_s)
    N = np.zeros((n_s + 1, n_s + 1))
    print("n_s = ", n_s)
    for i in range(0, n_s + 1):
        A_part = A_mat[i * length : (i + 1) * length, :]
        P_part = P_mat[i * length : (i + 1) * length, i * length : (i + 1) * length]
        N += A_part.T @ P_part @ A_part
        #print("i = ", i)
    print("N calculated")
    term1 = np.linalg.inv(N)
    print("N inverted")
    term2 = A_mat.T
    return(term1 @ term2)


def ATP(P_mat, n_s, p_list, t_list, τ_list):
    # P_mat: weight matrix
    # n_s: number of intervals
    # p_list: list of periods
    # 
    K = len(t_list)
    R = len(p_list)
    Q = 2 * R + 1
    l = int(K / n_s)
    
    ATP_mat = np.zeros((Q * (n_s + 1), K))
    s = 0
    for j in range(0, Q):
        s += 1
    return(s)


def m0_adv(A_mat, P_mat, Δx_vec, Δl_vec, n_s): # calculation of m0
    n = len(A_mat)
    u = len(A_mat.T)
    print("Δx_vec:")
    print(Δx_vec)
    length = int(len(A_mat) / n_s)
    A_Δx_vec = np.array([[0]])
    for i in range(0, n_s):
        mini_A = A_mat[i * length : (i + 1) * length, i : i + 2]
        mini_Δx_vec = Δx_vec[i : i + 2]
        mini_A_Δx_vec = mini_A @ mini_Δx_vec
        A_Δx_vec = np.vstack((A_Δx_vec, mini_A_Δx_vec))
    A_Δx_vec = A_Δx_vec[1:]
    v = A_Δx_vec - Δl_vec
    m0 = np.sqrt((v.T @ v) / (n - u))
    return(m0)

def lst_sqrs_adjstmnt_adv(ε, obs_array, para_lists_list_0, Δt, n_s): # least squares adjustment
    # ε: precision
    # obs_array = [[t_i, x_i, y_i]]
    t_list = obs_array[:, 0]
    
    para_lists_list = para_lists_list_0
    length = len(para_lists_list_0)
    para_fit_code = list_code(para_lists_list)
    σ0 = 1
    #σ_list = np.ones(len(obs_array))
    #P_mat = P(σ0, σ_list)
    P_mat = np.diag(np.ones(len(obs_array)))
    counter = 0 # to count and to avoid accidental infinite loop
    quotient_list = np.array([1])
    m_0 = 10
    A_mat = A_adv(t_list, para_lists_list_0, Δt, n_s)
    print("################################################")
    print("################################################")
    print("################################################")
    print("A_mat_T:", A_mat.T)
    print("A_mat_T[:3, n_s - 5 : n_s + 5]")
    print(A_mat.T[:3, 287 : 291])
    print("################################################")
    print("################################################")
    print("################################################")
    
    print(A_mat)
    print("A_mat shape: ", np.shape(A_mat))
    #half_Δx = N_inv_co(P_mat, A_mat)
    #half_Δx = N_inv_co_new(A_mat, n_s)
    print("000000000000000000000000000")
    print("000000000000000000000000000")
    print("000000000000000000000000000")
    half_Δx = N_inv_co_new_new(A_mat, P_mat, n_s)
    print("000000000000000000000000000")
    print("000000000000000000000000000")
    print("000000000000000000000000000")
    print("now while loop")
    while (max(quotient_list) > ε) and (counter < 100):
        
        x_vec = []
        for i in range(0, length):
            obj_i = para_lists_list[i]
            typ = type(obj_i)
            if (typ == int or typ == float or typ == np.float64):
                x_vec.append(obj_i)
            elif (typ == list or typ == np.ndarray):
                x_vec.extend(obj_i)
            else:
                print("!!!ERROR!!!\n!!!ERROR!!!\n!!!ERROR!!!\n!!!ERROR!!!\n!!!ERROR!!!\n")
        
        x_vec = np.array([x_vec]).T
        Δl_vec = O_minus_C_adv(obs_array, para_lists_list, Δt, n_s)
        print("Δl_vec:")
        print(Δl_vec)
        print("shape of Δl_vec:", np.shape(Δl_vec))
        print("shape of half_Δx:", np.shape(half_Δx))
        Δx_vec = new_Δx(half_Δx, Δl_vec)
        print("Δx_vec done")
        m_0 = m0_adv(A_mat, P_mat, Δx_vec, Δl_vec, n_s)
        print("m_0 done")
        quotient_list = quotient(x_vec, Δx_vec)
        print("quotient_list done")
        x_vec = x_vec + Δx_vec
        print("x_vec done")
        para_lists_list = x_vec.T[0]
        print("para_lists_list done")
        para_lists_list = encode_list(para_lists_list, para_fit_code)
        #e_para_fit = encode_list(e_para_fit, para_fit_code)
        
        counter += 1
        
        print("Iteration step: %3d | m0 = " % (counter), m_0)
        print("para_fit")
        print(round_list(para_lists_list, 3))
        #print("e_para_fit")
        #print(round_list(e_para_fit, 3))
        print("quotient_list")
        print(round_list(quotient_list, 3))
        print("------------------------------------------------------------")
    e_para_fit = encode_list(np.zeros(len(para_lists_list[0])), para_fit_code)
    return([para_lists_list, e_para_fit, m_0])

#%%
from functions import encode_list

def C_n_q_tilde(n, q_tilde, τ_n_list, t_n_list, # depend on n and q_tilde
                Δτ, L, n_s, K, # depend on data
                p_list): # depend on modelling
    # t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #     with τ_n_1 <= t_n_l < τ_n
    # τ_n_list = [τ_n_1, τ_n]
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    R = len(p_list) # 2 * R + 1 = Q
    τ_n_1, τ_n = τ_n_list[0], τ_n_list[1]
    t_n_list = np.array(t_n_list)
    
    if (n == n_s):
        L_tilde = K % n_s
        if (L_tilde != 0):
            L = L_tilde
            print("L_tilde != 0, L_tilde = ", L_tilde)
            
        
    
    mat_t = np.vstack((-t_n_list, t_n_list)).T
    mat_τ = np.vstack((τ_n * np.ones(L), - τ_n_1 * np.ones(L))).T
    mat = (mat_t + mat_τ) / Δτ
    
    if (q_tilde >= 1 and q_tilde <= R): # [pd_μ_r_(n-1), pd_μ_r_n]_t=t_k
        r = q_tilde
        p_r = p_list[r - 1] # r = 1, ..., R
        mat_t_symmetric = np.vstack((t_n_list, t_n_list)).T
        mat_μ_r = np.sin(p_r * mat_t_symmetric)
        mat *= mat_μ_r
    elif (q_tilde >= R + 1 and q_tilde <= 2*R): # [pd_η_r_(n-1), pd_η_r_n]_t=t_k
        r = int(q_tilde / 2)
        p_r = p_list[r - 1] # r = 1, ..., R
        mat_t_symmetric = np.vstack((t_n_list, t_n_list)).T
        mat_η_r = np.cos(p_r * mat_t_symmetric)
        mat *= mat_η_r
    # else (q_tilde == 0): mat = mat
    return(mat)

def E_n_q_q_prime(n, q_tilde_list, τ_n_list, t_n_list, # depend on n and q_tilde
                  Δτ, L, n_s, K, # depend on data
                  p_list, P_n): # depend on modelling
    # is actually E_(n,(q_tilde),(q_tilde_prime))
    # q_tilde_list = [q_tilde, q_tilde_prime]
    # t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #     with τ_n_1 <= t_n_l < τ_n
    # τ_n_list = [τ_n_1, τ_n]
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    # P_n: block matrix in weighting matrix P
    q_tilde, q_tilde_prime = q_tilde_list[0], q_tilde_list[1]
    C_n_q = C_n_q_tilde(n, q_tilde, τ_n_list, t_n_list,
                        Δτ, L, n_s, K, p_list)
    C_n_q_prime = C_n_q_tilde(n, q_tilde_prime, τ_n_list, t_n_list,
                              Δτ, L, n_s, K, p_list)
    #print("shape C_n_q", np.shape(C_n_q))
    #print("shape P_n", np.shape(P_n))
    mat_center_center = C_n_q.T @ P_n @ C_n_q_prime
    mat_center_left = np.zeros((2, n - 1))
    mat_center_right = np.zeros((2, n_s - n))
    
    mat_top = np.zeros((n - 1, n_s + 1))
    mat_center = np.hstack((mat_center_left, mat_center_center, mat_center_right))
    mat_bottom = np.zeros((n_s - n, n_s + 1))
    
    mat = np.vstack((mat_top, mat_center, mat_bottom))
    return(mat)

def F_n(n, τ_n_list, t_n_list, # depend on n
        Δτ, L, n_s, K, # depend on data
        p_list, P_n): # depend on modelling
    # F_n = (A_n)^T P_n A_n
    # q_tilde_list = [q_tilde, q_tilde_prime]
    # t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #     with τ_n_1 <= t_n_l < τ_n
    # τ_n_list = [τ_n_1, τ_n]
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    # P_n: block matrix in weighting matrix P
    R = len(p_list)
    Q = 2 * R + 1
    Q_tilde = Q - 1
    
    mat = np.zeros((1, Q * (n_s + 1))) # to be deleted
    for q_tilde in range(0, Q_tilde + 1): # stack row for row
        mat_help = np.zeros((n_s + 1, 1)) # to be deleted
        for q_tilde_prime in range(0, Q_tilde + 1): # stack column for column
            q_tilde_list = [q_tilde, q_tilde_prime]
            E_mat = E_n_q_q_prime(n, q_tilde_list, τ_n_list, t_n_list,
                                  Δτ, L, n_s, K, p_list, P_n)
            mat_help = np.hstack((mat_help, E_mat))
        mat_help = mat_help[:, 1:] # delete start
        mat = np.vstack((mat, mat_help))
    mat = mat[1:] # delete start
    return(mat)

def N_eff(τ_n_list_list, t_n_list_list, # depend on n
          Δτ, L, n_s, K, # depend on data
          p_list, P_n_list): # depend on modelling
    # N = A^T P A, eff -> should be efficient
    # τ_n_list_list = [τ_1_list, ..., τ_n_list, ..., τ_n_s_list]
    #     τ_n_list = [τ_n_1, τ_n]
    # t_n_list_list = [t_1_list, ..., t_n_list, ..., t_n_s_list]
    #     t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #         τ_n_1 <= t_n_l < τ_n
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    # P_n_list = [P_1, ..., P_n, ..., P_n_s]; P_n block matrix in weighting matrix P
    R = len(p_list)
    Q = 2 * R + 1
    
    mat = np.zeros((Q * (n_s + 1), Q * (n_s + 1))) # start
    for n in range(1, n_s + 1):
        τ_n_list = τ_n_list_list[n - 1] # index starts at 0
        t_n_list = t_n_list_list[n - 1] # index starts at 0
        P_n = P_n_list[n - 1] # index starts at 0
        F_n_mat = F_n(n, τ_n_list, t_n_list,
                      Δτ, L, n_s, K,p_list, P_n)
        mat += F_n_mat
    return(mat)

def G_n_q_tilde(n, q_tilde, τ_n_list, t_n_list, # depend on n and q_tilde
                Δτ, L, n_s, K, # depend on data
                p_list, P_n): # depend on modelling
    # t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #     with τ_n_1 <= t_n_l < τ_n
    # τ_n_list = [τ_n_1, τ_n]
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    C_n_q = C_n_q_tilde(n, q_tilde, τ_n_list, t_n_list,
                        Δτ, L, n_s, K, p_list)
    mat_center = C_n_q.T @ P_n
    
    if (n == n_s):
        L_tilde = K % n_s
        if (L_tilde != 0):
            L = L_tilde
    
    mat_top = np.zeros((n - 1, L))
    mat_bottom = np.zeros((n_s - n, L))
    
    mat = np.vstack((mat_top, mat_center, mat_bottom))
    return(mat)

def H_n(n, τ_n_list, t_n_list, # depend on n and q_tilde
        Δτ, L, n_s, K, # depend on data
        p_list, P_n): # depend on modelling
    # t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #     with τ_n_1 <= t_n_l < τ_n
    # τ_n_list = [τ_n_1, τ_n]
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    R = len(p_list)
    Q = 2 * R + 1
    Q_tilde = Q - 1
    
    mat = np.zeros((1, L)) # to be deleted
    for q_tilde in range(0, Q_tilde + 1):
        G_mat = G_n_q_tilde(n, q_tilde, τ_n_list, t_n_list,
                            Δτ, L, n_s, K, p_list, P_n)
        mat = np.vstack((mat, G_mat))
    mat = mat[1:] # delete start
    return(mat)

def AT_P(τ_n_list_list, t_n_list_list, # depend on n
         Δτ, L, n_s, K, # depend on data
         p_list, P_n_list): # depend on modelling
    # τ_n_list_list = [τ_1_list, ..., τ_n_list, ..., τ_n_s_list]
    #     τ_n_list = [τ_n_1, τ_n]
    # t_n_list_list = [t_1_list, ..., t_n_list, ..., t_n_s_list]
    #     t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #         τ_n_1 <= t_n_l < τ_n
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    # P_n_list = [P_1, ..., P_n, ..., P_n_s]; P_n block matrix in weighting matrix P
    R = len(p_list)
    Q = 2 * R + 1
    
    mat = np.zeros((Q * (n_s + 1), 1)) # to be deleted
    for n in range(1, n_s + 1):
        τ_n_list = τ_n_list_list[n - 1] # index starts at 0
        t_n_list = t_n_list_list[n - 1] # index starts at 0
        P_n = P_n_list[n - 1] # index starts at 0
        H_mat = H_n(n, τ_n_list, t_n_list,
                    Δτ, L, n_s, K, p_list, P_n)
        mat = np.hstack((mat, H_mat))
    mat = mat[:, 1:] # delete start
    return(mat)

def x_q_n_vec(x_vec, n, q_tilde, n_s):
    # x_vec = [a_0, ..., a_n, ..., a_(n_s),
    #          μ_(1,0), ..., μ_(1,n), ..., μ_(1,n_s), ...,
    #          μ_(r,0), ..., μ_(r,n), ..., μ_(r,n_s), ...,
    #          μ_(R,0), ..., μ_(R,n), ..., μ_(R,n_s),
    #          η_(1,0), ..., η_(1,n), ..., η_(1,n_s), ...,
    #          η_(r,0), ..., η_(r,n), ..., η_(r,n_s), ...,
    #          η_(R,0), ..., η_(R,n), ..., η_(R,n_s)]
    vec = x_vec[n_s * q_tilde + n - 1 : n_s * q_tilde + n + 1]
    return(vec)

def I_n(n, τ_n_list, t_n_list, # depend on n and q_tilde
        Δτ, L, n_s, K, # depend on data
        p_list, x_vec): # depend on modelling
    # t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #     with τ_n_1 <= t_n_l < τ_n
    # τ_n_list = [τ_n_1, τ_n]
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    R = len(p_list) # 2 * R + 1 = Q
    Q = 2 * R + 1
    Q_tilde = Q - 1
    
    vec = np.zeros((L, 1))
    for q_tilde in range(0, Q_tilde + 1):
        C_mat = C_n_q_tilde(n, q_tilde, τ_n_list, t_n_list,
                            Δτ, L, n_s, K, p_list)
        x_q_n = x_q_n_vec(x_vec, n, q_tilde, n_s)
        vec += C_mat @ x_q_n
    return(vec)

def A_x(τ_n_list_list, t_n_list_list, # depend on n and q_tilde
        Δτ, L, n_s, K, # depend on data
        p_list, x_vec):
    # τ_n_list_list = [τ_1_list, ..., τ_n_list, ..., τ_n_s_list]
    #     τ_n_list = [τ_n_1, τ_n]
    # t_n_list_list = [t_1_list, ..., t_n_list, ..., t_n_s_list]
    #     t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #         τ_n_1 <= t_n_l < τ_n
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    # x_vec = [a_n, μ_(r,n), η_(r,n)] (see def of x_q_n_vec)
    vec = np.zeros((1, 1)) # to be deleted
    for n in range(1, n_s + 1):
        τ_n_list = τ_n_list_list[n - 1] # index starts at 0
        t_n_list = t_n_list_list[n - 1] # index starts at 0
        I_vec = I_n(n, τ_n_list, t_n_list,
                    Δτ, L, n_s, K, p_list, x_vec)
        vec = np.vstack((vec, I_vec))
    vec = vec[1:] # delete start
    return(vec)

def compute_data(τ_n_list_list, t_n_list_list,
                 Δτ, L, n_s, K, p_list, x_vec):
    R = len(p_list)
    Q = 2 * R + 1
    Q_tilde = Q - 1
    comp_list = np.array([])
    for n in range(1, n_s + 1):
        τ_n_list = τ_n_list_list[n - 1]
        t_n_list = t_n_list_list[n - 1]
        if (n == n_s):
            L_tilde = K % n_s
            if (L_tilde != 0):
                L = L_tilde
        
        τ_n_1 = τ_n_list[0]
        τ_n = τ_n_list[1]
        #print("x_vec = ", x_vec)
        for l in range(0, L):
            t = t_n_list[l]
            t_covec = np.array([[τ_n - t, t - τ_n_1]]) / Δτ
            #print("shape t_covec", np.shape(t_covec))
            tot = 0
            for q_tilde in range(0, Q_tilde + 1):
                vec = x_q_n_vec(x_vec, n, q_tilde, n_s)
                #print("q_tilde = ", q_tilde)
                #print("vec shape", np.shape(vec))
                #print("vec = ", vec)
                if (q_tilde >= 1 and q_tilde <= R): # μ
                    r = q_tilde
                    p_r = p_list[r - 1]
                    vec *= np.sin(p_r * t)
                elif (q_tilde >= R + 1 and q_tilde <= 2 * R): # η
                    r = int(q_tilde / 2)
                    p_r = p_list[r - 1]
                    vec *= np.cos(p_r * t)
                # else: vec = vec # a_bar
                tot += t_covec @ vec
            comp_list = np.append(comp_list, tot)
    
    t_n_list_list = np.array(ravel(t_n_list_list))
    comp_data = np.vstack((t_n_list_list, comp_list)).T
    return(comp_data)

def O_C(obs_data, τ_n_list_list, t_n_list_list,
        Δτ, L, n_s, K, p_list, x_vec):
    obs_vec = np.array([obs_data[:, 1]]).T
    comp_data = compute_data(τ_n_list_list, t_n_list_list,
                             Δτ, L, n_s, K, p_list, x_vec)
    comp_vec = np.array([comp_data[:, 1]]).T
    return(obs_vec - comp_vec)

def v(τ_n_list_list, t_n_list_list,
      Δτ, L, n_s, K, p_list, Δx_vec, O_C_vec):
    # Ax - l
    A_x_vec = A_x(τ_n_list_list, t_n_list_list,
                  Δτ, L, n_s, K, p_list, Δx_vec)
    return(A_x_vec - O_C_vec)

def splitter(liste, L, n_s, K):
    # with help of encode_list split liste in several lists
    L_tilde = K % n_s
    if (L_tilde == 0):
        L_tilde = L
    code = (n_s - 1) * [L]
    code.append(L_tilde)
    list_list = encode_list(liste, code)
    return(list_list)

def m0_eff(τ_n_list_list, t_n_list_list,
           Δτ, L, n_s, K, p_list, P_n_list, Δx_vec, O_C_vec):
    R = len(p_list)
    Q = 2 * R + 1
    v_vec = v(τ_n_list_list, t_n_list_list,
              Δτ, L, n_s, K, p_list, Δx_vec, O_C_vec)
    v_list = ravel(v_vec)
    v_splitted = splitter(v_list, L, n_s, K)
    tot = 0
    for n in range(1, n_s + 1):
        v_n_list = v_splitted[n - 1]
        P_n = P_n_list[n - 1]
        
        v_n_vec = np.array([v_n_list]).T
        tot += v_n_vec.T @ P_n @ v_n_vec
    tot /= (K - Q * (n_s + 1)) # n = K, u = Q * (n_s + 1)
    return(tot)

def lst_sqrs_adjstmnt_eff(ε, obs_data, τ_n_list_list, t_n_list_list,
                          Δτ, L, n_s, K, p_list, P_n_list, x_vec):
    print("compute N")
    N = N_eff(τ_n_list_list, t_n_list_list,
              Δτ, L, n_s, K, p_list, P_n_list)
    print("compute N_inv")
    N_inv = np.linalg.inv(N)
    print("compute ATP")
    ATP = AT_P(τ_n_list_list, t_n_list_list,
               Δτ, L, n_s, K, p_list, P_n_list)
    print("compute N_inv_ATP")
    N_inv_ATP = N_inv @ ATP
    counter = 0 # to count and to avoid accidental infinite loop
    quotient_list = np.array([1])
    m0 = 10
    print("begin iteration process")
    while (max(quotient_list) > ε) and (counter < 100):
        print("compute O-C")
        O_C_vec = O_C(obs_data, τ_n_list_list, t_n_list_list,
                      Δτ, L, n_s, K, p_list, x_vec)
        print("compute Δx")
        Δx_vec = N_inv_ATP @ O_C_vec
        print("compute m0")
        m0 = m0_eff(τ_n_list_list, t_n_list_list,
                    Δτ, L, n_s, K, p_list, P_n_list, Δx_vec, O_C_vec)
        quotient_list = quotient(x_vec, Δx_vec)
        
        x_vec += Δx_vec
        counter += 1
        print("Iteration step: %3d | m0 = " % (counter), m0)
        print("x_vec:")
        print(round_list(ravel(x_vec), 3))
        print("quotient_list:")
        print(round_list(quotient_list, 3))
        print("------------------------------------------------------------")
    print("compute fitted data")
    fitted_data = compute_data(τ_n_list_list, t_n_list_list,
                               Δτ, L, n_s, K, p_list, x_vec)
    fit_list = [fitted_data, ravel(x_vec), ravel(m0)]
    return(fit_list)

def pre_lst_sqrs_adjstmt(obs_data, Δτ, K, p_list, P, apriori_list):
    # generate τ_n_list_list, t_n_list_list, n_s, L, P_n_list, x_vec
    # apriori_list = [a_(apriori), μ_(apriori)_list, η_(apriori)_list]
    t_tot = obs_data[-1, 0] - obs_data[0, 0]
    n_s = int(np.ceil(t_tot / Δτ))
    L = int(K / n_s)
    t_n_list_list = splitter(obs_data[:, 0], L, n_s, K)
    τ_n_list_list = []
    for n in range(1, n_s):
        t_n_1_list = t_n_list_list[n - 1 - 1]
        t_n_list = t_n_list_list[n - 1]
        
        τ_n_1 = t_n_1_list[0]
        τ_n = t_n_list[0]
        
        τ_n_list_list.append([τ_n_1, τ_n])
    τ_n_list_list.append([t_n_list_list[-1][0], t_n_list_list[-1][-1]])
    
    P_n_list = []
    for n in range(1, n_s + 1):
        P_n = P[(n - 1) * L : n * L, (n - 1) * L : n * L]
        P_n_list.append(P_n)
    
    R = len(p_list)
    Q = 2 * R + 1
    
    a_apriori = apriori_list[0]
    μ_apriori_list = apriori_list[1]
    η_apriori_list = apriori_list[2]
    
    x_a_list = np.array((n_s + 1) * [a_apriori])
    print("shape x_a_list", np.shape(x_a_list))
    x_μ_list = np.array([])
    x_η_list = np.array([])
    for r in range(0, R):
        x_μ_r = np.array([(n_s + 1) * [μ_apriori_list[r]]])
        x_η_r = np.array([(n_s + 1) * [η_apriori_list[r]]])
        x_μ_list = np.append(x_μ_list, x_μ_r)
        x_η_list = np.append(x_η_list, x_η_r)
    print("R", R)
    
    x_vec = np.array([np.hstack((x_a_list, x_μ_list, x_η_list))]).T
    print("x_vec shape", np.shape(x_vec))
    big_list = [τ_n_list_list, t_n_list_list, n_s, L, P_n_list, x_vec]
    return(big_list)

def fitter(data, per0_list, amp0_list, n_partition, lsa_method, ε):
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
    #        τ_fit_list: τ for each fitted parameter
    
    if (lsa_method == 1): # give t0 and periods; fit h, s, amplitudes and phases
        para = insert([0], per0_list) # given parameters
        para_fit_0 = insert([0, 0], amp0_list) # first guesses for parmeters
        para_fit_0 = insert(para_fit_0, np.zeros(len(per0_list))) # to be fitted
        fit_func = fit_func_1
        fit_func_derivs = fit_func_1_derivs
    elif (lsa_method == 2): # give t0; fit h, s, periods, amplitudes and phases
        para = [0] # given parameters
        para_fit_0 = insert([0, 0], per0_list[:, 0]) # first guesses for parmeters
        para_fit_0 = insert(para_fit_0, amp0_list[:, 1]) # to be fitted
        para_fit_0 = insert(para_fit_0, np.zeros(len(per0_list)))
        fit_func = fit_func_2
        fit_func_derivs = fit_func_2_derivs
    elif (lsa_method == 3): # give t0; fit h and s (linear regression)
        para = [0] # given parameters
        para_fit_0 = [0, 0] # first guesses for parmeters to be fitted
        fit_func = fit_func_3
        fit_func_derivs = fit_func_3_derivs
    elif (lsa_method == 4):
        print("lsa_method == 4")
        t_array = data[:, 0]
        Δt = n_partition
        print("n_partition = ", n_partition)
        n_s = get_n_s(data, Δt)
        para_lists_list_0 = 1 * [(n_s + 1) * [6*10**6]]
        print("fitting")
        lsa_list = lst_sqrs_adjstmnt_adv(ε, data, para_lists_list_0, Δt, n_s)
        print("fitted")
        para_fit, e_para_fit, m_0 = lsa_list[0], lsa_list[1], lsa_list[2]
        τ_fit_list = np.append(Δt * np.arange(n_s - 1), data[-1, 0])
        obj_list = np.array([])
        n = 1
        for i in range(0, len(t_array)):
            t_i = t_array[i]
            if (t_i >= n * Δt):
                n += 1
            t_list = [(n - 1) * Δt, n * Δt, Δt]
            obj = fit_func_adv(t_i, para_fit, t_list, n)
            obj_list = np.append(obj_list, obj)
        data_fitted = np.vstack((t_array, obj_list)).T
        fit_list = [data_fitted, para_fit, e_para_fit, m_0, τ_fit_list]
        print("shape of data: ", np.shape(data))
        print("shape of fitted data:", np.shape(data_fitted))
        print("fitter end")
        return(fit_list)
    elif (lsa_method == 5):
        print("lsa_method = 5")
        K = len(data)
        P = np.diag(np.ones(K))
        Δτ = n_partition
        p_list = per0_list
        apriori_list = [data[0, 1], amp0_list, amp0_list]
        print("prepare for fit")
        big_list = pre_lst_sqrs_adjstmt(data, Δτ, K, p_list,
                                        P, apriori_list)
        τ_n_list_list, t_n_list_list = big_list[0], big_list[1]
        n_s, L = big_list[2], big_list[3]
        P_n_list, x_vec = big_list[4], big_list[5]
        print("n_s", n_s)
        print("execute lsa")
        fit_list = lst_sqrs_adjstmnt_eff(ε, data, τ_n_list_list, t_n_list_list,
                                         Δτ, L, n_s, K, p_list, P_n_list, x_vec)
        fitted_data = fit_list[0]
        x_list = fit_list[1]
        m0 = fit_list[2]
        
        e_x_list = np.zeros(len(x_list))
        τ_fit_list = np.array([])
        for n in range(0, n_s):
            τ_n_1 = τ_n_list_list[n][0]
            τ_fit_list = np.append(τ_fit_list, τ_n_1)
        τ_fit_list = np.append(τ_fit_list, τ_n_list_list[-1][1])
        
        fit_list = [fitted_data, x_list, e_x_list, ravel(m0), τ_fit_list]
        return(fit_list)
    else:
        print("ERROR!!! SPECIFY LSA-METHOD!!!")
        return(1)
    
    # create subdata
    start, end = data[0, 0], data[-1, 0]
    subdata_list = []
    while (start <= end):
        subdata = array_modifier(data, start, n_partition)
        # normalizing ???
        subdata_list.append(subdata)
        start += n_partition
    
    data_fitted = np.zeros((1, 2)) # to be deleted
    n_para = n_objects(para_fit_0)
    para_fit_array = np.zeros((1, n_para)) # to be deleted
    para_e_fit_array = np.zeros((1, n_para)) # to be deleted
    m0_list = []
    τ_fit_list = np.array([])
    for i in range(0, len(subdata_list)):
        subdata = subdata_list[i]
        lsa_list = lst_sqrs_adjstmnt(ε, subdata, fit_func,
                                     fit_func_derivs, para, para_fit_0)
        para_fit, para_e_fit, m0 = lsa_list[0], lsa_list[1], lsa_list[2]
        
        τ_i = subdata[0, 0] + n_partition / 2
        
        subdata_fitted = np.zeros((1, 2)) # to be deleted
        for j in range(0, len(subdata)):
            t_j = subdata[j, 0]
            row_j = np.array([t_j, fit_func(t_j, para, para_fit)])
            subdata_fitted = np.vstack((subdata_fitted, row_j))
        subdata_fitted = subdata_fitted[1:]
        
        para_fit_list = ravel(para_fit)
        para_e_fit_list = ravel(para_e_fit)
        m0 = ravel(m0)
        data_fitted = np.vstack((data_fitted, subdata_fitted))
        para_fit_array = np.vstack((para_fit_array, para_fit_list))
        para_e_fit_array = np.vstack((para_e_fit_array, para_e_fit_list))
        m0_list.append(m0)
        τ_fit_list = np.append(τ_fit_list, τ_i)
    data_fitted = data_fitted[1:]
    para_fit_array = para_fit_array[1:]
    para_e_fit_array = para_e_fit_array[1:]
    
    fit_list = [data_fitted, para_fit_array, para_e_fit_array, m0_list, τ_fit_list]
    return(fit_list)
#%%
el_fit_list_list = []
el_para_fit_list = []
el_para_e_fit_list = []
el_m0_list = []
τ_fit_list_list = []
el_amp_fit_list = []
for i in range(0, len(data_both)):
    el_hat_data = el_hat_data_list_list[i]
    el_hat_per = el_hat_per_list[i]
    el_hat_amp = el_hat_amp_list[i]
    # fit
    el_fit_list = fitter(el_hat_data, el_hat_per, el_hat_amp, n_partition, lsa_method, ε)
    el_fit = el_fit_list[0]
    el_para_fit, el_para_e_fit = el_fit_list[1], el_fit_list[2]
    el_m0, τ_fit_list = el_fit_list[3], el_fit_list[4]
    if (lsa_method != 3 and lsa_method != 4):
        el_amp_fit = column_abs_averager(el_para_fit, np.arange(2, 2 + len(el_hat_per)))
    else:
        el_amp_fit = el_hat_amp
    el_fit_list_list.append(el_fit)
    el_para_fit_list.append(el_para_fit)
    el_para_e_fit_list.append(el_para_e_fit)
    el_m0_list.append(el_m0)
    τ_fit_list_list.append(τ_fit_list)
    el_amp_fit_list.append(el_amp_fit)
print("DONE!!!")
#%%
N_n_n = "n_" + str(lcm_list_el[2]) + "_" + str(lcm_list_el_bar[2]) + "_" + str(lcm_list_el_hat[2])
N_n_title = "N = (" + str(lcm_list_el[2]) + "," + str(lcm_list_el_bar[2]) + "," + str(lcm_list_el_hat[2]) + ")"
vline_list1 = [59958.915, 59961.875, 59967.125, 59988.71,
               59995.416, 59999.5, 60001.78125, 60002.427,
               60017.17, 60018.17, 60023.9375, 60026.375, 60053.389]
vline_list_specs1 = ['k', 0.25, 1, "CME"]
vline_list_specs2 = ['r', 0.25, 1, "MAN"]
file_name = "updates/update 8/lst_data/swarml23/" 
path = os.getcwd() + "/" + file_name[:-1]
if (os.path.exists(path) == False):
    os.mkdir(os.getcwd() + "/" + file_name[:-1])
xlimit_list = [[0, 0]]
log_fac_list = [2, 4]
el_fit_p_o_list = []
el_fit_a_o_list = []
for i in range(0, len(data_both)):
    name = file_name_list[i]
    el_fit = el_fit_list_list[i]
    print("el_fit", el_fit)
    # for plots
    el_p_o, el_a_o = el_p_o_list[i], el_a_o_list[i]
    el_bar_p_o, el_bar_a_o = el_bar_p_o_list[i], el_bar_a_o_list[i]
    el_hat_p_o, el_hat_a_o = el_hat_p_o_list[i], el_hat_a_o_list[i]
    el_bar_per0, el_bar_amp0 = el_bar_per0_list[i], el_bar_amp0_list[i]
    el_hat_per0, el_hat_amp0 = el_hat_per0_list[i], el_hat_amp0_list[i]
    el_hat_per, el_amp_fit = el_hat_per_list[i], el_amp_fit_list[i]
    el_bar_Δt_n = el_bar_Δt_n_list[i]
    el_hat_Δt_n = el_hat_Δt_n_list[i]
    # spectrum fit
    el_fit_spectrum_list = spectrum(el_fit, el_num, lcm_list_el, [[0, 0]])
    el_fit_p_o, el_fit_a_o = el_fit_spectrum_list[0], el_fit_spectrum_list[1]

    # el spectrum
    spectrum_p = [el_p_o, el_bar_p_o, el_hat_p_o]
    spectrum_a = [el_a_o, el_bar_a_o, el_hat_a_o]
    spectrum_data_specs = [[0.75, "r", el_symb], [0.75, "b", el_bar], [0.75, "g", el_hat]]
    spectrum_per = [el_bar_per0, el_hat_per0, el_hat_per]
    spectrum_amp = [el_bar_amp0, el_hat_amp0, np.array(el_amp_fit)]
    print(el_hat_a_o)
    print(el_amp_fit)
    marker_spec_array = [[(7.5, -7.5), "s", "r", "peaks used for " + el_bar, el_symb],
                         [(-7.5, 20), "s", "b", "peaks used for " + el_hat, el_bar],
                         [(5, -20), "s", "k", "averaged peaks of " + el_tilde, el_hat]]
    if (lsa_method == 3 or lsa_method == 4):
        spectrum_per = spectrum_per[:-1]
        spectrum_amp = spectrum_amp[:-1]
        marker_spec_array = marker_spec_array[:-1]
    Δt_list = [el_bar_Δt_n, el_hat_Δt_n]
    Δt_spec_array = [[(0, (4, 4)), "r", el_symb], [(0, (4, 4)), "b", el_bar]]
    vlines = []
    vline_specs = ['k', 0.5, 0.5]
    xlimits = [0, 0]
    ylimits = [0, 0]
    spectrum_tit = "Spectrum of " + name + " data of " + el_symb + ", " + N_n_title
    file_name_fft = "spec/spec_" + name + "_" + N_n_n + ".png"
    file_name_fft = file_name + file_name_fft
    fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                       spectrum_per, spectrum_amp, marker_spec_array,
                       Δt_list, Δt_spec_array, spectrum_tit, 1,
                       log_fac_list, 4,
                       vlines, vline_specs, xlimits, ylimits,
                       [0, 0])
    
    el_fit_p_o_list.append(el_fit_p_o)
    el_fit_a_o_list.append(el_fit_a_o)
    print("yes")
print("DONE!!!")
#%%
from functions import fft
vline_specs = ["gold", 0.75, 1]
# DATA PLOTS
for i in range(0, 1):
    for j in range(0, len(xlimit_list)):
        name = file_name_list[i]
        el_data = data_both[i]
        el_bar_data = el_bar_data_list_list[i]
        el_hat_data = el_hat_data_list_list[i]
        el_fit =  el_fit_list_list[i]
        # all
        data_array_list = [el_data, el_bar_data, el_hat_data, el_fit]
        data_spec_array = [[0.8, "r", el_symb, 1],
                           [0.8, "b", el_bar, 1],
                           [0.8, "g", el_hat, 1],
                           [0.8, 'm', el_tilde, 1]]
        y_label = el_symb + " " + el_unit
        ref_array_list = [ref_data, ref_smooth]
        ref_spec_array = [[0.8, "k", ref_symb, 1], [0.8, "chocolate", ref_bar, 1]]
        ref_y_spec_list = [ref_symb + " " + ref_unit, "chocolate"]
        tit = name + " data of " + el_word + " with " + ref_bar
        
        ylimits = [0, 0]
        z_order = 1
        nongra_fac = 1
        plot_func_6(data_array_list, data_spec_array, y_label,
                    ref_array_list, ref_spec_array, ref_y_spec_list,
                    tit, MJD_0, MJD_end, xlimit_list[j], ylimits,
                    vline_specs, z_order, nongra_fac, 0,
                    vline_list1, vline_list_specs1,
                    vline_list2, vline_list_specs2,
                    [0, 0])
        # only mean
        ylim_offset = 6.83 * 10**6 + 3000
        ylimits = [ylim_offset-1000, ylim_offset+5000+500]
        ref_y_spec_list = [ref_bar + " " + ref_unit, "chocolate"]
        plot_func_6(data_array_list[1:], data_spec_array[1:], y_label,
                    ref_array_list[1:], ref_spec_array[1:], ref_y_spec_list,
                    tit, MJD_0, MJD_end, xlimit_list[j], ylimits,
                    vline_specs, z_order, nongra_fac, 0,
                    vline_list1, vline_list_specs1,
                    vline_list2, vline_list_specs2,
                    [0, 0])
        
        # difference hat - fit
        ylimits = [-0.1, 0.1]
        data_array_list = [np.vstack((el_hat_data[:, 0], el_hat_data[:, 1] - el_fit[:, 1])).T]
        data_spec_array = [[0.8, "turquoise", el_hat + " - " + el_tilde, 1]]
        y_label = el_hat + " - " + el_tilde + " " + el_unit
        tit = (name + " data: comparison smooth and fit of " + el_word + ",\n with " + ref_word)
        plot_func_6(data_array_list, data_spec_array, y_label,
                    ref_array_list[1:], ref_spec_array[1:], ref_y_spec_list,
                    tit, MJD_0, MJD_end, xlimit_list[j], ylimits,
                    vline_specs, z_order, nongra_fac, 0,
                    vline_list1, vline_list_specs1,
                    vline_list2, vline_list_specs2,
                    [0, 0])
        # spectrum of hat - fit
        
        orig_list_diff = fft(data_array_list[0])
        fig = plt.figure(figsize = (10, 5), dpi = 300)
        plt.title("Spectrum of " + el_hat + " - " + el_tilde, fontsize = 20)
        plt.loglog(orig_list_diff[0], orig_list_diff[1], 'c-')
        plt.xlabel("Period [d]", fontsize = 15)
        plt.ylabel("Amplitude", fontsize = 15)
        plt.grid()
        plt.show(fig)
        plt.close(fig)
        
print("HELL YES!!!")
# %%
slope_data_list_list = []
add_data_list_list = []
n_rev_day_list = []
ijk = 10
for i in range(0, len(data_both)):
    el_hat_data = el_hat_data_list_list[i]
    el_para_fit = el_para_fit_list[i]
    el_para_e_fit = el_para_e_fit_list[i]
    τ_fit_list = τ_fit_list_list[i]
    n_rev_day = 1 / (2 * el_per_list[i][0])
    
    slope_list = el_para_fit[:, 1]
    slope_error_list = el_para_e_fit[:, 1]
    slope_data = np.vstack((τ_fit_list, slope_list, slope_error_list)).T
    
    add_data = np.array([0, 0])
    lmn = 0
    while (lmn * ijk < len(el_hat_data)):
        el_lmn = el_hat_data[lmn * ijk, 1]
        ref_lmn = ref_smooth[lmn * ijk, 1]
        t_lmn = el_hat_data[lmn * ijk, 0]
        Δa_day_lmn = Δa_day_spec(n_rev_day, el_lmn, ref_lmn)
        add_data = np.vstack((add_data, np.array([t_lmn, Δa_day_lmn])))
        lmn += ijk
    add_data = add_data[1:]
    
    slope_data_list_list.append(slope_data)
    add_data_list_list.append(add_data)
    n_rev_day_list.append(n_rev_day)
    print("succes")
print("DONE!!!")
#%%
xlimit_list = [[0, 0]]
error_fac_list = [20]
fac = 0.006
for i in range(0, 1):
    add_el = 0
    slope_data = slope_data_list_list[i]
    add_data = add_data_list_list[i]
    name = file_name_list[i]

    for j in range(0, len(xlimit_list)):
        xlimits = xlimit_list[j]
        slope_data_list = [slope_data]
        
        slope_specs_list = [[error_fac_list[j], 1.,  1., 0, "b", "b", "r"]]
        add_data_list = [add_data]
        add_data_specs_list = [[0.5, 'g', 'theoretical (f = ' + str(fac) + ')', 1, 0]]
        el_label = el_symb
        el_unit = el_unit
        ref_data_list = [ref_smooth]
        ref_specs_list = [[0.75, "chocolate", ref_bar, 1]]
        ref_lab_list = [ref_bar, ref_unit]
        ylimits = [-110, -30]
        title = name + " decrease of " + el_symb
        decrease_plot_hyperadv(slope_data_list, slope_specs_list,
                               add_data_list, add_data_specs_list,
                               el_label, el_unit,
                               ref_data_list, ref_specs_list, ref_lab_list,
                               MJD_0, MJD_end, xlimits, ylimits, title, 0,
                               fac, 4, (1., -0.3), vline_list1, 0.25,
                               vline_list2, 0.5) # 1, (1, 1)
#%%
if (len(data_both) == 2):
    el_normal = data_both[0]
    el_nongra = data_both[1]
    el_bar_normal = el_bar_data_list_list[0]
    el_bar_nongra = el_bar_data_list_list[1]
    el_hat_normal = el_hat_data_list_list[0]
    el_hat_nongra = el_hat_data_list_list[1]
    el_fit_normal = el_fit_list_list[0]
    el_fit_nongra = el_fit_list_list[1]
    
    t_list = el_normal[:, 0]
    el_diff = el_normal[:, 1] - el_nongra[:, 1]
    el_bar_diff = el_bar_normal[:, 1] - el_bar_nongra[:, 1]
    el_hat_diff = el_hat_normal[:, 1] - el_hat_nongra[:, 1]
    el_fit_diff = el_fit_normal[:, 1] - el_fit_nongra[:, 1]
    
    el_diff = np.vstack((t_list, el_diff)).T
    el_bar_diff = np.vstack((t_list, el_bar_diff)).T
    el_hat_diff = np.vstack((t_list, el_hat_diff)).T
    el_fit_diff = np.vstack((t_list, el_fit_diff)).T
    
    data_array_list = [el_diff, el_bar_diff, el_hat_diff, el_fit_diff]
    data_spec_array = [[0.125, "r", 'Δ' + el_symb, 1],
                       [0.25, "b", 'Δ' + el_bar, 1],
                       [0.5, "g", 'Δ' + el_hat, 1],
                       [1, 'm', 'Δ' + el_tilde, 1]]
    y_label = 'Δ' + el_symb + " " + el_unit
    ref_array_list = [ref_smooth]
    ref_spec_array = [[0.5, "chocolate", ref_bar, 1]]
    ref_y_spec_list = [ref_bar + " " + ref_unit, "chocolate"]
    tit = "data difference of " + el_symb + " with " + ref_bar
        
    ylimits = [-0.5, 0.5] # [-0.1, 0.1] [-0.025, 0.025]
    vline_col = "gold"
    z_order = 1
    nongra_fac = 1
    for j in range(0, len(xlimit_list)):
        print(vline_list2)
        plot_func_6(data_array_list, data_spec_array, y_label,
                   ref_array_list, ref_spec_array, ref_y_spec_list,
                   tit, MJD_0, MJD_end, [0, 0], ylimits,
                   vline_specs, z_order, nongra_fac, 0,
                   vline_list1, vline_list_specs1,
                   vline_list2, vline_list_specs2,
                   [0, 0])
    
    slope_normal = slope_data_list_list[0]
    slope_nongra = slope_data_list_list[1]
    slope_t = slope_normal[:, 0]
    slope_diff = slope_normal[:, 1] - slope_nongra[:, 1]
    slope_diff = np.vstack((slope_t, slope_diff)).T
    
    xlimits = [0, 0]
    slope_diff = np.vstack((slope_diff.T, np.zeros(len(slope_diff)))).T
    slope_data_list = [slope_diff]
    slope_specs_list = [[0, 1.5,  1., 0, "b", "b", "r"]]
    add_data_list = []
    add_data_specs_list = []
    el_label = el_symb
    el_unit = el_unit
    ref_data_list = [ref_smooth]
    ref_specs_list = [[0.75, "chocolate", ref_bar, 1]]
    ref_lab_list = [ref_bar, ref_unit]
    ylimits = [-0.5, 0.5]
    title = "difference of decrease of " + el_symb
    decrease_plot_hyperadv(slope_data_list, slope_specs_list,
                           add_data_list, add_data_specs_list,
                           el_label, el_unit,
                           ref_data_list, ref_specs_list, ref_lab_list,
                           MJD_0, MJD_end, xlimits, ylimits, title, 0,
                           fac, 2, (0, 1), vline_list1, 0.25, vline_list2, 0.5) # 1, (1, 1)
    
print("HELL YES!!!")
#%%
"""
foldername = "newoscele/SWARM_LST"
file_name_list = ["nongra"]
el_str = "a"
ref_str = "rho"
MJD_interval = [58306, 58406]
q = 1
n_partition = 0.5
lsa_method = 3
ε = 10 ** (-3)
lcm_list_el = [0, 10, 10, 0.09, 0.5, 5]
lcm_list_el_bar = [0, 10, 1, 0.09, 0.95, 10]
lcm_list_el_hat = [0, 10, 1, 3, 0.95, 10]
lcm_list_ref = [0, 0.1, 1, 0.09, 0.25, 10]
interval_list_el = [[0.05, 0.06], [0.07, 0.09], [0.015, 0.025]]
interval_list_el_bar = [[0.065, 0.075]]
interval_list_el_hat = [[0.35, 0.45], [0.55, 0.65]]
interval_list_ref = [[0.01, 0.0125], [0.0125, 0.015], [0.015, 0.02], [0.02, 0.025]]
xlimit_list = [[0, 0], [45, 45]]
log_fac_list = [2, 4]
xlimit_list = [[0, 0],[40, 5], [40, 40]]
error_fac_list = [10, 10, 10]
fac = 0.003
vline_list = [] #[58372, 58375, 58383, 58399]
"""
#%%
"""
foldername = "newoscele/GRACE_LST"
file_name_list = ["nongra"]
el_str = "a"
ref_str = "rho"
MJD_interval = [58301, 58451]
q = 1
n_partition = 0.125
lsa_method = 3
ε = 10 ** (-3)
lcm_list_el = [0, 50, 1, 0.09, 0.9, 10]
lcm_list_el_bar = [0, 50, 1, 0.09, 0.9, 10]
lcm_list_el_hat = [0, 10, 1, 1, 0.95, 10]
lcm_list_ref = [0, 0.1, 1, 0.09, 0.25, 10]
interval_list_el = [[2.775, 2.78], [5.55, 5.56]] # [[0.05, 0.06], [0.07, 0.09], [0.015, 0.025]]
interval_list_el_bar = [[0.05, 0.06], [0.07, 0.09], [0.015, 0.025], [0.09, 0.1], [0.4, 0.5]]
interval_list_el_hat = [[0.35, 0.45], [0.55, 0.56]]
interval_list_ref = [[0.01, 0.0125], [0.0125, 0.015], [0.015, 0.02], [0.02, 0.025]]
xlimit_list = [[0, 0]]
log_fac_list = [2, 4]
error_fac_list = [20]
fac = 0.003
vline_list = [58309] #[58372, 58375, 58383, 58399]
"""
#%%
"""
foldername = "All Data new/oscelesentinel"
file_name_list = ["normal", "nongra"]
el_str = "a"
ref_str = "rho"
MJD_interval = [59950, 60050]
q = 1
n_partition = 0.5
lsa_method = 3
ε = 10 ** (-3)
lcm_list_el = [10, 50, 10, 0.09, 0.95, 10]
lcm_list_el_bar = [0, 50, 1, 0.09, 0.95, 15]
lcm_list_el_hat = [0, 50, 1, 1, 0.95, 15]
lcm_list_ref = [0, 0.1, 1, 0.09, 0.25, 10]
interval_list_el = [[0, 0]] # [[0.05, 0.06], [0.07, 0.09], [0.015, 0.025]]
interval_list_el_bar = [[0.015, 0.025], [6, 8]]
interval_list_el_hat = [[0.35, 0.45], [0.55, 0.56]]
interval_list_ref = [[0.01, 0.0125], [0.0125, 0.015], [0.015, 0.02], [0.02, 0.025]]
xlimit_list = [[0, 0]]
log_fac_list = [2, 4]
vline_list1 = [59995, 59988, 59967, 59961, 59958,
               60023]
vline_list2 = [60017, 60018]
error_fac_list = [20]
fac = 0.006
"""
#%%