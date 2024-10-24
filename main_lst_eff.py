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
from functions import arg_finder, fft
from functions import decrease_plot_hyperadv
from functions import file_extreme_day_new
from functions import get_n_s, lst_sqrs_adjstmnt_adv, insert
from functions import fit_func_1, fit_func_1_derivs
from functions import fit_func_2, fit_func_2_derivs
from functions import fit_func_3, fit_func_3_derivs
from functions import lst_sqrs_adjstmnt, ravel, fit_func_adv, n_objects
from functions import quotient, encode_list, list_code, N_inv_co, round_list
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
#file_extreme_day_new('sat24/sentinel/lst', 1)
#%%
foldername = "All Data new/osceleswarm"
file_name_list = ["normal", "nongra"]
el_str = "a"
ref_str = "rho"
MJD_interval = [60017, 60037]
save_on = 0
path = os.getcwd() + "/updates/update 9/grace/c/"
q = 1
n_partition = 1
lsa_method = 5
ε_I = 10**(-3)
# lcm_list = [p_spec, prec, N, p_max, thresh_quot, limit]
lcm_list_el = [0, 50, 1, 0.09, 0.95, 10]
lcm_list_el_bar = [0, 10, 1, 0.09, 0.95, 15]
lcm_list_el_hat = [0, 50, 1, 1, 0.95, 15]
lcm_list_ref = [0, 0.1, 1, 0.09, 0.25, 10]
interval_list_el = [[0.42, 0.45], [0.3, 0.305], [0.23, 0.24], [0.15, 0.17],
                    [0.135, 0.138], [0.118, 0.122], [0.106, 0.109], [0.09, 0.1],
                    [0.085, 0.09], [0.08, 0.085], [0.073, 0.077], [0.067, 0.073],
                    [0.063, 0.067], [0.0605, 0.063], [0.059, 0.0606], [0.057, 0.059],
                    [0.054, 0.055], [0.051, 0.052], [0.048, 0.05], [0.0465, 0.048],
                    #[0.036, 0.0363], [0.0348, 0.035], [0.0305, 0.0308], [0.0295, 0.0298],
                    [0.021, 0.0215], [0.0215, 0.022], [0.022, 0.0225], [0.015, 0.018]]
#interval_list_el = [[0, 0]]          # 1860.38624283, naja, (-110, 135), 0 
#interval_list_el = [[0.575, 0.579]]  # 1788.59694374, naja, (-110, 120), 17
#interval_list_el = [[0.3, 0.305]]    # 1789.4587139 , naja, (-110, 120), 18
#interval_list_el = [[0.23, 0.24]]    # 1778.43402988, naja, (-110, 120), 16
#interval_list_el = [[0.15, 0.17]]    # 1814.33498169, naja, (-110, 120), 20
#interval_list_el = [[0.135, 0.138]]  # 1800.72440716, naja, (-110, 125), 19
#interval_list_el = [[0.118, 0.122]]  # 1757.1201348 , naja, (-110, 125), 15
#interval_list_el = [[0.106, 0.109]]  # 1698.7879499 , naja, (-110, 125), 13
#interval_list_el = [[0.09, 0.1]]     # 1646.63773616, naja, (-110, 125), 10
#interval_list_el = [[0.085, 0.09]]   # 1610.95118153, naja, (-110, 125), 8
#interval_list_el = [[0.08, 0.085]]   # 1585.59589638, naja, (-110, 125), 6
#interval_list_el = [[0.073, 0.077]]  # 1559.15766753, naja, (-105, 125), 4
#interval_list_el = [[0.067, 0.073]]  # 1537.71735759, naja, (-100, 125), 2
#interval_list_el = [[0.063, 0.067]]  # 1540.03521013, naja, (-100, 135), 3
#interval_list_el = [[0.0605, 0.063]] # 1574.87026955, naja, (-115, 140), 5
#interval_list_el = [[0.059, 0.0606]] # 1607.44184737, naja, (-120, 140), 7
#interval_list_el = [[0.057, 0.059]]  # 1632.30098931, naja, (-125, 140), 9
#interval_list_el = [[0.054, 0.055]]  # 1681.09302066, naja, (-130, 140), 12
#interval_list_el = [[0.051, 0.052]]  # 1712.06724228, naja, (-125, 140), 14
#interval_list_el = [[0.048, 0.05]]   # 1907
#interval_list_el = [[0.0465, 0.048]] # 1926
#interval_list_el = [[0.0437, 0.045]] # 1929
#interval_list_el = [[0.046, 0.0465]] # 1927
#interval_list_el = [[0.042, 0.0425]] # 1927
#interval_list_el = [[0.036, 0.0363]] # 1866
#interval_list_el = [[0.0348, 0.035]] # 1863
#interval_list_el = [[0.0305, 0.0308]]# 1880
#interval_list_el = [[0.0295, 0.0298]]# 1889
#interval_list_el = [[0.022, 0.0225]] # 1025
#interval_list_el = [[0.02, 0.025]]   # 824.63265317 , good, (-75 , 100), 1
#interval_list_el = [[0.021, 0.0215]] # 1116
#interval_list_el = [[0.015, 0.018]]  # 1669.88299847, naod, (-110, 120), 11
#interval_list_el = [[0.02, 0.025], [0.067, 0.073], [0.063, 0.067],
#                    [0.073, 0.077], [0.0605, 0.063], [0.08, 0.085]]

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
print(len(data_both[0])/20)
#data_ref = np.vstack((np.arange(0,2,0.0001), np.sin(np.arange(0, 10000)))).T
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
for i in range(0, len(data_both)):
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
    marker_spec_array = [[(7.5, -7.5), "o", "k", "peaks used for " + el_bar, el_symb]]
    Δt_list = []
    Δt_spec_array = [[(0, (4, 4)), "r", el_symb]]
    spectrum_tit = "test"
    # [0.58, 0.5825], [0.43, 0.445], [0.301, 0.305], [0.23, 0.24]
    # [0.15, 0.17], [0.095, 0.1], [0.085, 0.09], [0.08, 0.085],
    # [0.072, 0.075], [0.068, 0.072], [0.063, 0.068], [0.06, 0.063],
    # [0.056, 0.06], [0.053, 0.056], [0.051, 0.053]
    # auswahl: [[0.056, 0.06], [0.072, 0.075], [0.08, 0.085]]
    
    v_line_specs = ['b', 0.5, 1]
    xlimits = [0, 0]
    xlimits = [0.01, 1.5]
    ylimits = [0, 0]
    ylimits = [1, 1000]
    #v_line_list = []
    # [0.021, 0.0215], [0.0215, 0.022], [0.022, 0.0225]
    v_line_list = [0.42, 0.45]#, 0.021, 0.0215, 0.015, 0.018]
    fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                       spectrum_per, spectrum_amp, marker_spec_array,
                       Δt_list, Δt_spec_array, spectrum_tit, 1, [], 0,
                       v_line_list, v_line_specs,
                       xlimits, ylimits, [0])
    
print("DONE!!!")
#%%
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
    
    #if (n == n_s):
    #    L_tilde = K - (n_s - 1) * L
    #    if (L_tilde != 0):
    #        L = L_tilde
            
    mat_t = np.vstack((-t_n_list, t_n_list)).T
    mat_τ = np.vstack((τ_n * np.ones(L), - τ_n_1 * np.ones(L))).T
    mat = (mat_t + mat_τ) / Δτ
    
    if (q_tilde >= 1 and q_tilde <= R): # [pd_μ_r_(n-1), pd_μ_r_n]_t=t_k
        r = q_tilde
        p_r = p_list[r - 1] # r = 1, ..., R
        mat_t_symmetric = np.vstack((t_n_list, t_n_list)).T
        mat_μ_r = np.sin(2 * np.pi / p_r * mat_t_symmetric)
        mat = mat * mat_μ_r
    elif (q_tilde >= R + 1 and q_tilde <= 2 * R): # [pd_η_r_(n-1), pd_η_r_n]_t=t_k
        r = q_tilde - R
        p_r = p_list[r - 1] # r = 1, ..., R
        mat_t_symmetric = np.vstack((t_n_list, t_n_list)).T
        mat_η_r = np.cos(2 * np.pi / p_r * mat_t_symmetric)
        mat = mat * mat_η_r
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
        mat = mat + F_n_mat
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
    
    #if (n == n_s):
    #    L_tilde = K - (n_s - 1) * L
    #    if (L_tilde != 0):
    #        L = L_tilde
    
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
    
    #if (n == n_s):
    #    L_tilde = K - (n_s - 1) * L
    #    if (L_tilde != 0):
    #        mat = np.zeros((1, L_tilde))
    #else:
    #    mat = np.zeros((1, L)) # to be deleted
    mat = np.zeros((1, L))
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
    vec = x_vec[(n_s + 1) * q_tilde + n - 1 : (n_s + 1) * q_tilde + n + 1]
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
    
    #if (n == n_s):
    #    L_tilde = K - (n_s - 1) * L
    #    if (L_tilde != 0):
    #        vec = np.zeros((L_tilde, 1))
    #else:
    #    vec = np.zeros((L, 1))
    vec = np.zeros((L, 1))
    
    for q_tilde in range(0, Q_tilde + 1):
        C_mat = C_n_q_tilde(n, q_tilde, τ_n_list, t_n_list,
                            Δτ, L, n_s, K, p_list)
        x_q_n = x_q_n_vec(x_vec, n, q_tilde, n_s)
        vec = vec + C_mat @ x_q_n
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
        #if (n == n_s):
        #    L_tilde = K - (n_s - 1) * L
        #    if (L_tilde != 0):
        #        L = L_tilde
        τ_n_1 = τ_n_list[0]
        τ_n = τ_n_list[1]
        for l in range(0, L):
            t = t_n_list[l]
            t_covec = np.array([[τ_n - t, t - τ_n_1]]) / Δτ
            tot = 0
            for q_tilde in range(0, Q_tilde + 1):
                vec = x_q_n_vec(x_vec, n, q_tilde, n_s)
                if (q_tilde >= 1 and q_tilde <= R): # μ
                    r = q_tilde
                    p_r = p_list[r - 1]
                    vec = vec * np.sin(2 * np.pi / p_r * t)
                elif (q_tilde >= R + 1 and q_tilde <= 2 * R): # η
                    r = q_tilde - R
                    p_r = p_list[r - 1]
                    vec = vec * np.cos(2 * np.pi / p_r * t)
                # else: vec = vec # a_bar
                tot = tot + t_covec @ vec
            comp_list = np.append(comp_list, tot)
    t_list = np.array(ravel(t_n_list_list))
    comp_data = np.vstack((t_list, comp_list)).T
    return(comp_data)

def v(τ_n_list_list, t_n_list_list,
      Δτ, L, n_s, K, p_list, Δx_vec, O_C_vec):
    # Ax - l
    A_x_vec = A_x(τ_n_list_list, t_n_list_list,
                  Δτ, L, n_s, K, p_list, Δx_vec)
    return(A_x_vec - O_C_vec)

def encode_list(liste, code_list):
    # liste = [a, b, c, d, e, f, g]
    # code_list = [0, 0, 2, 3]
    # wanted: [a, b, [c, d], [e,f,g]]
    new_list = []
    s = 0
    for i in range(0, len(code_list)):
        n_objects = code_list[i]
        if (n_objects == 0):
            new_list.append(liste[s])
            s = s + 1
        else:
            new_list.append(liste[s : s + n_objects])
            s = s + n_objects
    return(new_list)

def splitter(liste, L, n_s, K):
    # with help of encode_list split liste in several lists
    #L_tilde = K - (n_s - 1) * L
    code = (n_s - 1) * [L]
    #if (L_tilde != 0):
    #    L = L_tilde
    code.append(L)
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
    tot = tot / (K - Q * (n_s + 1)) # n = K, u = Q * (n_s + 1)
    return(np.sqrt(tot))

def L_constr(n_s, Q):
    # shape(L_mat) = (Q * (n_s + 1), Q * (n_s - 1))
    diag_upp = np.ones(n_s - 1)
    diag_cen = - 2 * np.ones(n_s)
    diag_low = np.ones(n_s + 1)
    a_bar_mat = np.diag(diag_low) + np.diag(diag_cen, 1) + np.diag(diag_upp, 2)
    a_bar_mat = a_bar_mat[: -2]
    zero_mat_1 = np.zeros(((Q - 1) * (n_s - 1), 1 * (n_s + 1)))
    zero_mat_2 = np.zeros((Q * (n_s - 1), (Q - 1) * (n_s + 1)))
    L_mat = np.hstack((np.vstack((a_bar_mat, zero_mat_1)), zero_mat_2))
    return(L_mat)

def L_constr_all(n_s, Q):
    # shape(L_mat) = (Q * (n_s + 1), Q * (n_s - 1))
    diag_upp = np.ones(n_s - 1)
    diag_cen = - 2 * np.ones(n_s)
    diag_low = np.ones(n_s + 1)
    constr_mat = np.diag(diag_low) + np.diag(diag_cen, 1) + np.diag(diag_upp, 2)
    constr_mat = constr_mat[: -2]
    
    shape = np.shape(constr_mat)
    hlist = []
    for i in range(0, Q):
        before = i * [np.zeros(shape)]
        after = (Q - i - 1) * [np.zeros(shape)]
        before.append(constr_mat)
        before.extend(after)
        obj = np.block([before])
        hlist.append(obj)
    L_mat = np.vstack(tuple(hlist))
    return(L_mat)

def lsa_cond(ε_I, quotient_list, counter):
    # ε_I = {ε if ε_I < 1, I if ε_I is integer}
    # quotient_list: relative improvements of each parameter
    # counter: counter for iterations
    # if ε_I is < 1 -> iterate until max(quotient_list) < ε_I
    #     ex: ε_I = 1e-3 -> iterate until max(quotient_list) < 1e-3
    # if ε_I is int -> iterate until counter = ε_I
    #     ex: ε_I = 5 -> iterate until counter = 5 (5 iterations)
    if (type(ε_I) == int):
        return(counter <= ε_I)
    else:
        return(max(quotient_list) > ε_I)

def lst_sqrs_adjstmnt_eff_adv(ε_I, obs_data, τ_n_list_list, t_n_list_list,
                              Δτ, L, n_s, K, p_list, P_n_list, x_vec, constr):
    print("compute N")
    N = N_eff(τ_n_list_list, t_n_list_list,
              Δτ, L, n_s, K, p_list, P_n_list)
    print("compute L_mat")
    if (constr == 0):
        L_mat = L_constr(n_s, 1 + 2 * len(p_list))
    elif (constr == 1):
        L_mat = L_constr_all(n_s, 1 + 2 * len(p_list))
    else:
        L_mat = np.zeros(np.shape(N))
    print("compute new N")
    N = N + ε_I * L_mat.T @ L_mat
    print("compute N_inv")
    N_inv = np.linalg.inv(N)
    print("compute ATP")
    ATP = AT_P(τ_n_list_list, t_n_list_list,
               Δτ, L, n_s, K, p_list, P_n_list)
    print("compute N_inv_ATP")
    N_inv_ATP = N_inv @ ATP
    l_vec = np.array([obs_data[:, 1]]).T
    print("compute x_vec")
    x_vec = N_inv_ATP @ l_vec
    print("compute m0")
    m0 = m0_eff(τ_n_list_list, t_n_list_list,
                Δτ, L, n_s, K, p_list, P_n_list, x_vec, l_vec)
    print("m0 = %.3e" % m0[0][0])
    x_list = ravel(x_vec)
    print("compute Kxx")
    Kxx = m0**2 * N_inv
    #print(np.diag(Kxx))
    print("lsajdfklajsldkfjklsdjfklasdfasdjfajkdfjks")
    dx_list = np.sqrt(np.diag(Kxx))
    print("compute fitted data")
    fitted_data = compute_data(τ_n_list_list, t_n_list_list,
                               Δτ, L, n_s, K, p_list, x_vec)
    print("fitted data computed")
    fit_list = [fitted_data, x_list, dx_list, m0[0][0]]
    return(fit_list)

def pre_lst_sqrs_adjstmt(obs_data, n_s, K, p_list, P, apriori_list):
    # generate τ_n_list_list, t_n_list_list, L, P_n_list, x_vec, Δτ
    # apriori_list = [a_(apriori), μ_(apriori)_list, η_(apriori)_list]
    t_tot = obs_data[-1, 0] - obs_data[0, 0]
    #Δτ = np.ceil(t_tot) / n_s
    Δτ = t_tot / n_s
    L = int(K / n_s)
    print("L = ", L)
    #L_tilde = K - (n_s - 1) * L
    #print("L = ", L, " | L_tilde = ", L_tilde)
    #if (L_tilde == 0):
    #    n_s = n_s - 1
    
    t_n_list_list = splitter(obs_data[:, 0], L, n_s, K)
    τ_n_list_list = []
    for n in range(1, n_s + 1):
        τ_n_1 = (n - 1) * Δτ
        τ_n = n * Δτ
        τ_n_list_list.append([τ_n_1, τ_n])
    
    P_n_list = []
    for n in range(1, n_s + 1):
        P_n = P[(n - 1) * L : n * L, (n - 1) * L : n * L]
        P_n_list.append(P_n)
    
    R = len(p_list)
    a_apriori = apriori_list[0]
    μ_apriori_list = apriori_list[1]
    η_apriori_list = apriori_list[2]
    
    x_a_list = np.array((n_s + 1) * [a_apriori])
    x_μ_list, x_η_list = np.array([]), np.array([])
    for r in range(0, R):
        x_μ_r = np.array([(n_s + 1) * [μ_apriori_list[r]]])
        x_η_r = np.array([(n_s + 1) * [η_apriori_list[r]]])
        x_μ_list = np.append(x_μ_list, x_μ_r)
        x_η_list = np.append(x_η_list, x_η_r)
    x_vec = np.array([np.hstack((x_a_list, x_μ_list, x_η_list))]).T
    #x_vec = np.ones(((n_s + 1) * (2 * R + 1), 1)) # would also work
    
    big_list = [τ_n_list_list, t_n_list_list, L, P_n_list, x_vec, Δτ]
    return(big_list)

def fitter(data, per0_list, amp0_list, n_partition, lsa_method, ε_I, constr):
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
    if (lsa_method != 5):
        n_partition = 1 / n_partition
        ε = ε_I
    
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
        n_s = n_partition
        K = len(data)
        P = np.diag(np.ones(K))
        p_list = per0_list
        apriori_list = [data[0, 1], amp0_list, amp0_list]
        print("prepare for fit")
        big_list = pre_lst_sqrs_adjstmt(data, n_s, K, p_list,
                                        P, apriori_list)
        τ_n_list_list, t_n_list_list = big_list[0], big_list[1]
        L, P_n_list = big_list[2], big_list[3]
        x_vec, Δτ = big_list[4], big_list[5]
        print("preparations finished")
        print("n_s = %4d | Δτ = %3.6f" % (n_s, Δτ))
        print("execute lsa")
        fit_list = lst_sqrs_adjstmnt_eff_adv(ε_I, data, τ_n_list_list, t_n_list_list,
                                             Δτ, L, n_s, K, p_list, P_n_list, x_vec, constr)
        fitted_data = fit_list[0]
        x_list = fit_list[1]
        dx_list = fit_list[2]
        m0 = fit_list[3]
        
        τ_fit_list = np.array(τ_n_list_list)[:, 0] + Δτ / 2
        R = len(p_list)
        Q = 2 * R + 1
        Q_tilde = Q - 1
        para_list = []
        para_e_list = []
        for q_tilde in range(0, Q_tilde + 1):
            sub_list = []
            e_sub_list = []
            for n in range(1, n_s + 1):
                t = τ_fit_list[n - 1]
                a_q_n_1 = x_q_n_vec(x_list, n, q_tilde, n_s)[0]
                a_q_n = x_q_n_vec(x_list, n, q_tilde, n_s)[1]
                da_q_n_1 = x_q_n_vec(dx_list, n, q_tilde, n_s)[0]
                da_q_n = x_q_n_vec(dx_list, n, q_tilde, n_s)[1]
                
                slope = (a_q_n - a_q_n_1) / Δτ
                d_sope = (da_q_n - da_q_n_1) / Δτ
                if (q_tilde >= 1 and q_tilde <= R): # μ
                    r = q_tilde
                    p_r = p_list[r - 1]
                    slope = slope * np.sin(2 * np.pi / p_r * t)
                    d_sope = d_sope * np.sin(2 * np.pi / p_r * t)
                elif (q_tilde >= R + 1 and q_tilde <= 2 * R): # η
                    r = q_tilde - R
                    p_r = p_list[r - 1]
                    slope = slope * np.cos(2 * np.pi / p_r * t)
                    d_sope = d_sope * np.cos(2 * np.pi / p_r * t)
                # else: vec = vec # a_bar
                sub_list.append(slope)
                e_sub_list.append(d_sope)
            para_list.append(sub_list)
            para_e_list.append(e_sub_list)
        fit_list = [fitted_data, para_list, para_e_list, m0, τ_fit_list]
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
# interesting: R = 5, ε_I = 10000, constr = 0, n_partition = 16 * n_days_tot
# -> the zoom: data and fit is interesting
t_fit_a = tt.time()
R = 5 # 11 is limit # 7 -> 17.338
ε_I = 10000
constr = 1
print(len(data_both[0]))
print(el_per[:R])
n_partition = 10 * n_days_tot #0.125
el_fit_list_list = []
el_para_fit_list = []
el_para_e_fit_list = []
el_m0_list = []
τ_fit_list_list = []
for i in range(0, 1):
    el_data = data_both[i]
    el_per = el_per_list[i]
    el_amp = el_amp_list[i]
    
    # fit
    el_fit_list = fitter(el_data, el_per[:R], el_amp[:R], n_partition, lsa_method, ε_I, constr)
    el_fit = el_fit_list[0]
    el_para_fit, el_para_e_fit = el_fit_list[1], el_fit_list[2]
    el_m0, τ_fit_list = el_fit_list[3], el_fit_list[4]
    
    el_fit_list_list.append(el_fit)
    el_para_fit_list.append(el_para_fit)
    el_para_e_fit_list.append(el_para_e_fit)
    el_m0_list.append(el_m0)
    τ_fit_list_list.append(τ_fit_list)
t_fit_b = tt.time()
print("Time: ", t_fit_b - t_fit_a)
print("DONE!!!")
#%%
el_fit_p_o_list = []
el_fit_a_o_list = []
el_fit_per_list = []
el_fit_amp_list = []
for i in range(0, 1):#len(data_both)
    name = file_name_list[i]
    el_fit = el_fit_list_list[i]
    # data_spectrum
    el_fit_spectrum_list = spectrum(el_fit, el_num, lcm_list_el, [[0, 0]])
    el_fit_p_o, el_fit_a_o = el_fit_spectrum_list[0], el_fit_spectrum_list[1]
    el_fit_per, el_fit_amp = el_fit_spectrum_list[2], el_fit_spectrum_list[3]
    
    el_fit_p_o_list.append(el_fit_p_o)
    el_fit_a_o_list.append(el_fit_a_o)
    el_fit_per_list.append(el_fit_per)
    el_fit_amp_list.append(el_fit_amp)
#%%
#N_n_n = "n_" + str(lcm_list_el[2]) + "_" + str(lcm_list_el_bar[2]) + "_" + str(lcm_list_el_hat[2])
#N_n_title = "N = (" + str(lcm_list_el[2]) + "," + str(lcm_list_el_bar[2]) + "," + str(lcm_list_el_hat[2]) + ")"
vline_list1 = [59958.915, 59961.875, 59967.125, 59988.71,
               59995.416, 59999.5, 60001.78125, 60002.427,
               60017.17, 60018.17, 60023.9375, 60026.375, 60053.389]
vline_list_specs1 = ['k', 0.25, 1, "CME"]
vline_list_specs2 = ['r', 0.25, 1, "MAN"]
path_add = "_" + str(R) + "_" + str(ε_I)
#if (os.path.exists(path) == False):
#    os.mkdir(os.getcwd() + "/" + file_name[:-1])
xlimit_list = [[0, 0]]
log_fac_list = [10]
for i in range(0, 1):
    name = file_name_list[i]
    m0 = el_m0_list[i]
    # for plots
    el_p_o, el_a_o = el_p_o_list[i], el_a_o_list[i]
    el_fit_p_o, el_fit_a_o = el_fit_p_o_list[i], el_fit_a_o_list[i]
    
    el_per, el_amp = el_per_list[i], el_amp_list[i]
    el_fit_per, el_fit_amp = el_fit_per_list[i], el_fit_amp_list[i]
    
    # spectrum
    spectrum_p = [el_p_o, el_fit_p_o]
    spectrum_a = [el_a_o, el_fit_a_o]
    spectrum_data_specs = [[0.75, "r", el_symb],
                           [1, "g", el_tilde]]
    spectrum_per = [el_per[:R]]
    spectrum_amp = [el_amp[:R]]
    marker_spec_array = [[(7.5, -7.5), "o", "r", "peaks used for " + el_tilde, el_symb]]
    #if (lsa_method == 3 or lsa_method == 4):
    #    spectrum_per = spectrum_per[:-1]
    #    spectrum_amp = spectrum_amp[:-1]
    #    marker_spec_array = marker_spec_array[:-1]
    Δt_list = []
    Δt_spec_array = []
    vlines = []
    vline_specs = ['k', 0.5, 0.5]
    xlimits = [0, 0]
    ylimits = [0, 0]
    string = r'$\tilde{a}_{y,z}$'.replace('y', str(R)).replace('z', str(ε_I))
    spectrum_tit_1 = name + " data: spectrum of " + el_symb + " and " + string
    spectrum_tit_2 = r", $m_0 = $ %.3f" % m0
    spectrum_tit = spectrum_tit_1 + spectrum_tit_2
    file_name_fft = "spec/spec_" + name + "_" + el_symb + ".png"
    file_name_fft = file_name + file_name_fft
    fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                       spectrum_per, spectrum_amp, marker_spec_array,
                       Δt_list, Δt_spec_array, spectrum_tit, 1,
                       log_fac_list, 4,
                       vlines, vline_specs, xlimits, ylimits,
                       [save_on, path + "spec" + path_add])
    print("yes")
print("DONE!!!")
#for i in range(0, len(el_per_list[0])):
#    print(el_per_list[0][i], el_per_list[1][i])
#%%
xlimit_list = [[0.1, 0.1]]
vline_specs = ["gold", 0.75, 1]
# DATA PLOTS
for i in range(0, 1):
    for j in range(0, len(xlimit_list)):
        name = file_name_list[i]
        m0 = el_m0_list[i]
        str_0 = name + " data: "
        str_1 = r'$\tilde{a}_{y,z}$'.replace('y', str(R)).replace('z', str(ε_I))
        str_2 =  r" , $m_0 = $ %.3f m" % m0
        
        el_data = data_both[i]
        el_fit =  el_fit_list_list[i]
        # all
        data_array_list = [el_data, el_fit]
        data_spec_array = [[1, "r", el_symb, 1],
                           [0.5, 'b', el_tilde, 1]]
        y_label = el_symb + " " + el_unit
        ref_array_list = [ref_data, ref_smooth]
        ref_spec_array = [[0.8, "k", ref_symb, 1], [0.8, "chocolate", ref_bar, 1]]
        ref_y_spec_list = [ref_symb + " " + ref_unit, "chocolate"]
        
        tit = str_0 + el_symb + " and " + str_1 + str_2
        
        z_order = 1
        nongra_fac = 1
        # only mean
        ylim_offset = np.mean(el_data[:, 1])
        ylimits = [ylim_offset - 11150, ylim_offset + 10850]
        ref_y_spec_list = [ref_bar + " " + ref_unit, "chocolate"]
        xlimit_zoom = [9, 4]
        
        plot_func_6(data_array_list, data_spec_array, y_label,
                    ref_array_list[2:], ref_spec_array[2:], ref_y_spec_list,
                    tit, MJD_0, MJD_end, xlimit_zoom, ylimits,
                    vline_specs, z_order, nongra_fac, 1,
                    vline_list1, vline_list_specs1,
                    vline_list2, vline_list_specs2,
                    [0, 0])
        
        ylimits = [ylim_offset - 10000, ylim_offset - 9450]
        
        #xlimit_zoom = [0, 13] #################
        #xlimit_zoom = [6.5, 6.5] #################
        #xlimit_zoom = [13, 0] #################
        
        
        #ylimits = [ylim_offset - 10150+500, ylim_offset - 9450+500] #########
        #ylimits = [ylim_offset - 10150+150, ylim_offset - 9450+150] #########
        #ylimits = [ylim_offset - 10150-200, ylim_offset - 9450-200] #########
        plot_func_6(data_array_list, data_spec_array, y_label,
                    ref_array_list[2:], ref_spec_array[2:], ref_y_spec_list,
                    tit, MJD_0, MJD_end, xlimit_zoom, ylimits,
                    vline_specs, z_order, nongra_fac, 1,
                    vline_list1, vline_list_specs1,
                    vline_list2, vline_list_specs2,
                    [0, 0])
        
        ylimits = [ylim_offset + 9150, ylim_offset + 9850]
        
        #ylimits = [ylim_offset + 9150+500, ylim_offset + 9850+500] ############
        #ylimits = [ylim_offset + 9150+150, ylim_offset + 9850+150] ############
        #ylimits = [ylim_offset + 9150-200, ylim_offset + 9850-200] ############
        plot_func_6(data_array_list, data_spec_array, y_label,
                    ref_array_list[2:], ref_spec_array[2:], ref_y_spec_list,
                    tit, MJD_0, MJD_end, xlimit_zoom, ylimits,
                    vline_specs, z_order, nongra_fac, 1,
                    vline_list1, vline_list_specs1,
                    vline_list2, vline_list_specs2,
                    [0, 0])
        
        # difference data - fit
        ylimits = [-50, 50]
        data_array_list = [np.vstack((el_data[:, 0], el_data[:, 1] - el_fit[:, 1])).T]
        data_spec_array = [[0.5, "c", el_symb + " - " + el_tilde, 1]]
        y_label = el_symb + " - " + el_tilde + " " + el_unit
        str_3 = r", $\langle Δa \rangle = $ %.3e m" % np.mean(el_data[:, 1] - el_fit[:, 1])
        tit = str_0 + el_symb + " - " + str_1 + str_2 + str_3
        
        plot_func_6(data_array_list, data_spec_array, y_label,
                    ref_array_list[1:], ref_spec_array[1:], ref_y_spec_list,
                    tit, MJD_0, MJD_end, xlimit_zoom, ylimits,
                    vline_specs, z_order, nongra_fac, 1,
                    vline_list1, vline_list_specs1,
                    vline_list2, vline_list_specs2,
                    [save_on, path + "diff" + path_add])
        
        # spectrum of data - fit
        orig_list_diff = fft(data_array_list[0])
        fig = plt.figure(figsize = (10, 5), dpi = 300)
        plt.title(str_0 + "spectrum of " + el_symb + " - " + str_1 + str_2,
                  fontsize = 17.5)
        plt.loglog(orig_list_diff[0], orig_list_diff[1], 'c-')
        #plt.axvline(1.5, color = 'k', alpha = 0.5)
        #plt.axvline(0.5, color = 'k', alpha = 0.5)
        #plt.xlim(1, 3)
        #plt.ylim(10**(-2), 1)
        plt.xlabel("Period [d]", fontsize = 15)
        plt.ylabel("Amplitude", fontsize = 15)
        plt.grid()
        if (save_on == 1):
            fig.savefig(path + "diff_spec" + path_add, transparent = None, dpi = 'figure',
                        format = None, metadata = None, bbox_inches = 'tight',
                        facecolor = 'auto', pad_inches = 'layout',
                        edgecolor = 'auto', backend = None)
        plt.show(fig)
        plt.close(fig)
        
        
print("HELL YES!!!")
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

xlimit_list = [[-0.1, -0.1]]
vline_specs = ["gold", 0.75, 1]
e_fac = 1
fac = 1
for i in range(0, 1):
    for j in range(0, len(xlimit_list)):
        name = file_name_list[i]
        str_0 = name + " data: "
        str_1 = r'$\tilde{a}_{y,z}$'.replace('y', str(R)).replace('z', str(ε_I))
        str_2 =  r" , $m_0 = $ %.3f m" % m0
        
        para_list = el_para_fit_list[i]
        e_para_list = el_para_e_fit_list[i]
        τ_list = τ_fit_list_list[i]
        m0 = el_m0_list[i]
        
        slope_data = np.vstack((τ_list, para_list[0], np.abs(e_para_list[0]))).T
        slope_data_list = [slope_data]
        slope_specs_list = [[e_fac, 1.5,  1., 0, "b", "b", "r"]]
        add_data_list = []
        add_data_specs_list = []
        el_label = el_symb
        el_unit = el_unit
        ref_data_list = [ref_smooth]
        ref_specs_list = [[0.75, "chocolate", ref_bar, 1]]
        ref_lab_list = [ref_bar, ref_unit]
        title = str_0 + "decrease of " + str_1 + str_2
        
        ylimits = [-150, -30]
        z_order = 1
        nongra_fac = 1
        ref_y_spec_list = [ref_bar + " " + ref_unit, "chocolate"]
        
        decrease_plot_hyperadv(slope_data_list, slope_specs_list,
                               add_data_list, add_data_specs_list,
                               el_label, el_unit,
                               ref_data_list, ref_specs_list, ref_lab_list,
                               MJD_0, MJD_end, xlimit_list[0], ylimits,
                               title, 1, fac, 1, (1., 1.),
                               vline_list1, vline_list_specs1,
                               vline_list2, vline_list_specs2,
                               [save_on, path + "decrease" + path_add])
# %%
xlimits = [-0.1, 0.1]
if (len(data_both) == 2):
    string = r'$\tilde{a}_{y,z}$'.replace('y', str(R)).replace('z', str(ε_I))
    el_normal = data_both[0]
    el_nongra = data_both[1]
    el_fit_normal = el_fit_list_list[0]
    el_fit_nongra = el_fit_list_list[1]
    
    t_list = el_normal[:, 0]
    el_diff = el_normal[:, 1] - el_nongra[:, 1]
    el_fit_diff = el_fit_normal[:, 1] - el_fit_nongra[:, 1]
    
    el_diff = np.vstack((t_list, el_diff)).T
    el_fit_diff = np.vstack((t_list, el_fit_diff)).T
    
    data_array_list = [el_diff, el_fit_diff]
    data_spec_array = [[1, "r", 'Δ' + el_symb, 1],
                       [0.5, 'b', 'Δ' + string, 1]]
    y_label = 'Δ' + el_symb + " " + el_unit
    ref_array_list = [ref_smooth]
    ref_spec_array = [[0.5, "chocolate", ref_bar, 1]]
    ref_y_spec_list = [ref_bar + " " + ref_unit, "chocolate"]
    tit = "data difference of " + el_symb + " with " + ref_bar
        
    ylimits = [-0.2, 0.2] # [-0.1, 0.1] [-0.025, 0.025]
    vline_col = "gold"
    z_order = 1
    nongra_fac = 1
    for j in range(0, len(xlimit_list)):
        plot_func_6(data_array_list, data_spec_array, y_label,
                   ref_array_list, ref_spec_array, ref_y_spec_list,
                   tit, MJD_0, MJD_end, [0, 0], ylimits,
                   vline_specs, z_order, nongra_fac, 0,
                   vline_list1, vline_list_specs1,
                   vline_list2, vline_list_specs2,
                   [save_on, path + "data_diff" + path_add])
    
    para_normal = el_para_fit_list[0]
    para_nongra = el_para_fit_list[1]
    e_para_normal = el_para_e_fit_list[0]
    e_para_nongra = el_para_e_fit_list[1]
    τ_list = τ_fit_list_list[0]
    
    slope_normal = np.vstack((τ_list, para_normal[0], np.abs(e_para_normal[0]))).T
    slope_nongra = np.vstack((τ_list, para_nongra[0], np.abs(e_para_nongra[0]))).T
    
    slope_diff = slope_normal[:, 1] - slope_nongra[:, 1]
    slope_diff = np.vstack((τ_list, slope_diff)).T
    
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
    ylimits = [-0.1, 0.1]
    title = "difference of decrease of " + string
    decrease_plot_hyperadv(slope_data_list, slope_specs_list,
                           add_data_list, add_data_specs_list,
                           el_label, el_unit,
                           ref_data_list, ref_specs_list, ref_lab_list,
                           MJD_0, MJD_end, xlimits, ylimits,
                           title, 0, fac, 1, (1, 1),
                           vline_list1, vline_list_specs1, 
                           vline_list2, vline_list_specs2,
                           [save_on, path + "diff_decrease" + path_add]) # 1, (1, 1)
    
print("HELL YES!!!")
# %%
