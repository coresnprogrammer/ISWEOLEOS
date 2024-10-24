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
from functions import flx_get_data, flx_get_data_ymd, array_columns
from functions import cl_lin
from functions import file_extreme_day_new
from functions import get_n_s, lst_sqrs_adjstmnt_adv, insert
from functions import fit_func_1, fit_func_1_derivs
from functions import fit_func_2, fit_func_2_derivs
from functions import fit_func_3, fit_func_3_derivs
from functions import lst_sqrs_adjstmnt, ravel, fit_func_adv, n_objects
from functions import quotient, encode_list, list_code, N_inv_co, round_list
from functions import mjd_to_mmdd, mjd_to_ddmm, xaxis_year
from functions import lab_gen, plot_bar
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
#file_extreme_day_new('sentinel_collection/se3a', 1)
#%%
foldername = "grace18newlst"
path_flx = 'FLXAP_P.FLX'
sat_name = 'GFOC'
file_name_list = ["nongra"]
el_str = "a"
ref_str = "rho"

#MJD_interval = [59975, 60045]
#MJD_interval = [59995, 60015]
#MJD_interval = [58352, 58362]

MJD_interval = [58340, 58413]
save_on = 0
path = os.getcwd() + "/updates/update 10/swarm with flx/"
q = 1
n_partition = 1
lsa_method = 5
ε_I = 10**(-3)
# lcm_list = [p_spec, prec, N, p_max, thresh_quot, limit]
lcm_list_el = [0, 50, 1, 0.09, 0.95, 10]
lcm_list_el_bar = [0, 10, 1, 0.09, 0.95, 15]
lcm_list_el_hat = [0, 50, 1, 1, 0.95, 15]
lcm_list_ref = [0, 0.1, 1, 0.09, 0.25, 10]

interval_list_el = [[0.0205,0.024], [0.05, 0.075], [0.075, 0.1]]
interval_list_el_bar = [[0.021, 0.023], [0.0725, 0.0775], [0.08, 0.085]]
interval_list_el_hat = [[0, 0]]
interval_list_ref = [[0.01, 0.0125], [0.0125, 0.015], [0.015, 0.02], [0.02, 0.025]]

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
ele = string_list[el_num]

# prepare data
str_normal_nongra = ["normal", "nongra"]
data_both = []
for i in range(0, len(file_name_list)):
    file_name = file_name_list[i]
    name_i = foldername + 2 * ("/year_" + file_name) + "_" + el_str + ".txt"
    data_i = np.loadtxt(name_i, skiprows=1) # load data
    data_i = array_denormalize(data_i) # denormalize data
    data_i = array_modifier(data_i, MJD_0, n_days_tot) # trim data
    data_i = array_normalize(data_i, 0) # normalize data
    data_i = data_i[1:] # cut MJD_0
    data_both.append(data_i)

data_flx_apday, data_ap = flx_get_data_ymd(path_flx)
data_ap = array_modifier(data_ap, MJD_0, n_days_tot)
data_ap = array_normalize(data_ap, 0) # normalize data
data_ap = data_ap[1:] # cut MJD_0

#data_ap_t = np.arange(0, n_days_tot, 1)
#data_ap_ap = np.random.rand(len(data_ap_t), 8)
#data_ap = []
#print(data_ap)
#%%
########################################################################################
# PROCESSING EL ########################################################################
########################################################################################
el_p_o_list = []
el_a_o_list = []
el_per_list = []
el_amp_list = []
for i in range(0,1):
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
    marker_spec_array = [[(7.5, -7.5), "o", "k",
                          "peaks used for " + el_tilde, el_symb]]
    Δt_list = []
    Δt_spec_array = [[(0, (4, 4)), "r", el_symb]]
    spectrum_tit = "test"
    v_line_specs = ['b', 0.5, 1]
    xlimits = [0, 0]
    xlimits = [0.001, 1]
    ylimits = [0, 0]
    ylimits = [0.01, 100]
    #[0.0145, 0.0185], [0.019, 0.026], [0.045, 0.1]
    v_line_list = [0.019, 0.026, 0.05, 0.075, 0.1]
    unit = 'a'
    fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                       spectrum_per, spectrum_amp, marker_spec_array,
                       Δt_list, Δt_spec_array, spectrum_tit, unit, 1, [], 0,
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
    σ2_list = np.diag(Kxx) # σ^2
    print("compute fitted data")
    fitted_data = compute_data(τ_n_list_list, t_n_list_list,
                               Δτ, L, n_s, K, p_list, x_vec)
    print("compute data free of oscillations")
    bar_data = compute_data(τ_n_list_list, t_n_list_list,
                            Δτ, L, n_s, K, [], x_vec)
    fit_list = [fitted_data, x_list, σ2_list, m0[0][0], bar_data]
    return(fit_list)

def pre_lst_sqrs_adjstmt(obs_data, n_s, K, p_list, P, apriori_list):
    # generate τ_n_list_list, t_n_list_list, L, P_n_list, x_vec, Δτ
    # apriori_list = [a_(apriori), μ_(apriori)_list, η_(apriori)_list]
    t_tot = obs_data[-1, 0] - obs_data[0, 0]
    t0 = obs_data[0, 0]
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
        τ_n_1 = t0 + (n - 1) * Δτ
        τ_n = t0 + n * Δτ
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
        σ2_list = fit_list[2]
        m0 = fit_list[3]
        bar_data = fit_list[4]
        
        τ_fit_list = np.array(τ_n_list_list)[:, 0]
        τ_fit_list = np.append(τ_fit_list, τ_n_list_list[-1][1])
        R = len(p_list)
        Q = 2 * R + 1
        Q_tilde = Q - 1
        para_list = []
        s2_para_list = []
        for q_tilde in range(0, Q_tilde + 1):
            sub_list = []
            s2_sub_list = []
            for n in range(0, n_s + 1):
                a_q_n = x_list[(n_s + 1) * q_tilde + n]
                σ2_q_n = σ2_list[(n_s + 1) * q_tilde + n]
                para = 0
                s2_para = 0
                if (q_tilde == 0): # a_bar
                    para = a_q_n
                    s2_para = σ2_q_n
                elif (q_tilde >= 1 and q_tilde <= R): # μ
                    τ_n = τ_fit_list[n]
                    r = q_tilde
                    p_r = p_list[r - 1]
                    para = a_q_n * np.sin(2 * np.pi / p_r * τ_n)
                    s2_para = σ2_q_n * (np.sin(2 * np.pi / p_r * τ_n))**2
                else: # q_tilde >= R + 1 and q_tilde <= 2 * R # η
                    τ_n = τ_fit_list[n]
                    r = q_tilde - R
                    p_r = p_list[r - 1]
                    para = a_q_n * np.cos(2 * np.pi / p_r * τ_n)
                    s2_para = σ2_q_n * (np.cos(2 * np.pi / p_r * τ_n))**2
                sub_list.append(para)
                s2_sub_list.append(s2_para)
            para_list.append(sub_list)
            s2_para_list.append(s2_sub_list)
        
        a_bar_list = x_list[: n_s + 1]
        σ2_a_bar_list = σ2_list[: n_s + 1]
        slope_gen_list = [τ_fit_list, Δτ, a_bar_list, σ2_a_bar_list]
        
        fit_list = [fitted_data, para_list, s2_para_list, m0,
                    τ_fit_list, slope_gen_list, bar_data]
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
print(len(data_both[0]))
#%%
# configuration_list = [[n_fac, R, ω, constr]]
t_fit_a = tt.time()
""" configuration_list = [[5, 1, 100, 1],
                      [10, 1, 100, 1],
                      [15, 1, 100, 1],
                      [20, 1, 100, 1],
                      [5, 2, 100, 1],
                      [10, 2, 100, 1],
                      [15, 2, 100, 1],
                      [20, 2, 100, 1],
                      [5, 3, 100, 1],
                      [10, 3, 100, 1],
                      [15, 3, 100, 1],
                      [20, 3, 100, 1],
                      [5, 1, 1000, 1],
                      [10, 1, 1000, 1],
                      [15, 1, 1000, 1],
                      [20, 1, 1000, 1],
                      [5, 2, 1000, 1],
                      [10, 2, 1000, 1],
                      [15, 2, 1000, 1],
                      [20, 2, 1000, 1],
                      [5, 3, 1000, 1],
                      [10, 3, 1000, 1],
                      [15, 3, 1000, 1],
                      [20, 3, 1000, 1],
                      [5, 1, 10000, 1],
                      [10, 1, 10000, 1],
                      [15, 1, 10000, 1],
                      [20, 1, 10000, 1],
                      [5, 2, 10000, 1],
                      [10, 2, 10000, 1],
                      [15, 2, 10000, 1],
                      [20, 2, 10000, 1],
                      [5, 3, 10000, 1],
                      [10, 3, 10000, 1],
                      [15, 3, 10000, 1],
                      [20, 3, 10000, 1]] """

configuration_list = [[5, 1, 1000, 1],
                      [10, 1, 10000, 1],
                      [15, 1, 100000, 1]]


print("len of data: ", len(data_both[0]))

#configuration0a = "n_" + str(n_fac) + "/"
#configuration1 = "-n_" + str(n_fac) + "-R"
#configuration3 = "-w_%.0e.png" % ω

el_fit_list_list = []
el_para_fit_list = []
el_para_e_fit_list = []
el_m0_list = []
τ_fit_list_list = []
slope_gen_list_list = []
for i in range(0, len(configuration_list)):
    configurations = configuration_list[i]
    n_fac = configurations[0]
    R = configurations[1]
    ω = configurations[2]
    constr = configurations[3]
    
    print("periods: ", el_per[:R])
    
    n_partition = n_fac * n_days_tot
    
    el_data = data_both[0]
    el_per = el_per_list[0]
    el_amp = el_amp_list[0]
    
    # fit
    el_fit_list = fitter(el_data, el_per[:R], el_amp[:R], n_partition,
                         lsa_method, ω, constr)
    el_fit = el_fit_list[0]
    el_para_fit, el_para_e_fit = el_fit_list[1], el_fit_list[2]
    el_m0, τ_fit_list = el_fit_list[3], el_fit_list[4]
    slope_gen_list = el_fit_list[5]
    
    el_fit_list_list.append(el_fit)
    el_para_fit_list.append(el_para_fit)
    el_para_e_fit_list.append(el_para_e_fit)
    el_m0_list.append(el_m0)
    τ_fit_list_list.append(τ_fit_list)
    slope_gen_list_list.append(slope_gen_list)
t_fit_b = tt.time()
print("Time: ", t_fit_b - t_fit_a)
print("DONE!!!")
#%%
comp_data_list = []
for i in range(0, len(configuration_list)):
    configuration = configuration_list[i]
    n_fac = configuration[0]
    R = configuration[1]
    el_para_fit = el_para_fit_list[i]
    
    x_vec = np.array([el_para_fit[0]]).T
    n_s = n_fac * n_days_tot
    K = len(data_both[0])
    P = np.diag(np.ones(K))
    p_list = el_per_list[0][:0]
    apriori_list = [data_both[0][0, 1], [], []]
    
    big_list = pre_lst_sqrs_adjstmt(data_both[0], n_s, K, p_list,
                                    P, apriori_list)
    τ_n_list_list, t_n_list_list = big_list[0], big_list[1]
    L = big_list[2]
    Δτ = big_list[5]
    comp_data = compute_data(τ_n_list_list, t_n_list_list,
                             Δτ, L, n_s, K, p_list, x_vec)
    comp_data_list.append(comp_data)
    
#%%
comp_array = comp_data_list[0][:, 0]
for i in range(0, len(comp_data_list)):
    comp_list = comp_data_list[i]
    comp_array = np.vstack((comp_array, comp_list[:, 1]))
comp_array = comp_array.T

header_array = np.zeros((4, 1))
for i in range(0, len(configuration_list)):
    config_array = np.array([configuration_list[i]]).T
    header_array = np.hstack((header_array, config_array))

m0_row = np.append(np.array([0]), np.array(el_m0_list))

comp_array = np.vstack((header_array, m0_row, comp_array))

comp_name = foldername + '/run_1.txt'
np.savetxt(comp_name, comp_array)
#%%
el_fit_p_o_list = []
el_fit_a_o_list = []
el_fit_per_list = []
el_fit_amp_list = []
for i in range(0, len(configuration_list)):#len(data_both)
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
vline_list_specs1 = ['k', 0.75, 1, "CME"]
vline_list_specs2 = ['r', 0.25, 1, "MAN"]
path_add = "_" + str(0) + "_" + str(ω)
#if (os.path.exists(path) == False):
#    os.mkdir(os.getcwd() + "/" + file_name[:-1])
xlimit_list = [[0, 0]]
log_fac_list = [10, 100, 1000, 10000]
name = file_name_list[0]
# for plots
el_per, el_amp = el_per_list[0], el_amp_list[0]
col_add = 0
if (len(configuration_list) == 1):
    col_add = 1
col_list = cl_lin(np.arange(len(configuration_list) + col_add), mat.colormaps['brg'])
# spectrum
spectrum_p = el_p_o_list + el_fit_p_o_list
spectrum_a = el_a_o_list + el_fit_a_o_list

spectrum_data_specs = [[0.75, "r", el_symb]]
for i in range(0, len(configuration_list)):
    spec_list = [0.75, col_list[i],
                 lab_gen(ele, configuration_list[i])]
    spectrum_data_specs.append(spec_list)

spectrum_per = [el_per[:3]]
spectrum_amp = [el_amp[:3]]
marker_spec_array = [[(7.5, -7.5), "o", "k", "periods used", el_symb]]
Δt_list = []
Δt_spec_array = []
vlines = []
vline_specs = ['k', 0.5, 0.5]
xlimits = [0, 0]
ylimits = [0, 0]
spectrum_tit_1 = sat_name + ' - ' + name + " data: spectrum of " + el_symb + " and " + el_tilde
spectrum_tit_2 = r", $m_0 = $ [" + ", ".join("%.1e" % m0 for m0 in el_m0_list) + "] m"
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
        ax1.plot(x_data + MJD_0, y_data, color = col, ls = '-', lw = width,
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
    
    plt.figlegend(fontsize = 12.5, markerscale = 5, loc = 1,
                  bbox_to_anchor = (1, 1), bbox_transform = ax1.transAxes,
                  labelspacing = 0)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    
    plt.show(fig)
    plt.close(fig)

spring = mat.colormaps['spring']
rainbow = mat.colormaps['gist_rainbow']
plasma = mat.colormaps['plasma']
jet = mat.colormaps['jet']
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


vline_list1 = [59958.915, 59961.875, 59967.125, 59988.71,
               59995.416, 59999.5, 60001.78125, 60002.427,
               60017.17, 60018.17, 60023.9375, 60026.375, 60053.389]
vline_list_specs1 = ['k', 0.75, 1, "CME"]
vline_list_specs2 = ['r', 0.25, 1, "MAN"]
path_add = "_n" + str(n_fac) + "_w" + str(ω)
#spring, rainbow, plasma, jet
xlimits = [0, 0]
vline_specs = ["gold", 0.75, 1]
# DATA PLOTS
for i in range(0, len(configuration_list)):
    configuration = configuration_list[i]
    n_fac = configuration[0]
    R = configurations[1]
    ω = configuration[2]
    constr = configuration[3]
    
    name = file_name_list[0]
    m0 = el_m0_list[i]
    str_0 = sat_name + ' - ' + name + " data: "
    str_1 = lab_gen(ele, configuration)
    str_2 =  r" , $m_0 = $ %.1e m" % m0
    el_data = data_both[0]
    el_fit =  el_fit_list_list[i]
    # all
    data_array_list = [el_data, el_fit]
    data_spec_array = [[1, "r", el_symb, 1],
                       [0.5, 'b', el_tilde, 1]]
    y_label = el_symb + " " + el_unit
    flx_data = data_ap
    flx_spec_list = [0.5, 0, 'summer']
    flx_y_spec_list = []

    tit = str_0 + el_symb + " and " + str_1 + str_2
    
    z_order = 1
    nongra_fac = 1
    # only mean
    ylim_offset = np.mean(el_data[:, 1])
    ylimits = [ylim_offset - 11150, ylim_offset + 10850]
    xlimit_zoom = [0, 0]
    
    configuration0b = "all"
    path_add = "0"
    plot_bar(data_array_list, data_spec_array, y_label,
             flx_data, flx_spec_list, flx_y_spec_list,
             tit, MJD_0, MJD_end, xlimit_zoom, ylimits,
             vline_specs, z_order, nongra_fac, 0,
             vline_list1, vline_list_specs1,
             vline_list2, vline_list_specs2,
             [save_on, path + path_add])
    
    ylimits = [ylim_offset - 10150, ylim_offset - 9450]
    
    #xlimit_zoom = [0, 13] #################
    #xlimit_zoom = [6.5, 6.5] #################
    #xlimit_zoom = [13, 0] #################
    
    
    #ylimits = [ylim_offset - 10150+500, ylim_offset - 9450+500] #########
    #ylimits = [ylim_offset - 10150+150, ylim_offset - 9450+150] #########
    #ylimits = [ylim_offset - 10150-200, ylim_offset - 9450-200] #########
    #"""
    ylimits = [ylim_offset - 10150-700, ylim_offset - 9450+1000]
    configuration0b = "zoomlow"
    path_add = "0"
    plot_bar(data_array_list, data_spec_array, y_label,
             flx_data, flx_spec_list, flx_y_spec_list,
             tit, MJD_0, MJD_end, xlimit_zoom, ylimits,
             vline_specs, z_order, nongra_fac, 0,
             vline_list1, vline_list_specs1,
             vline_list2, vline_list_specs2,
             [save_on, path + path_add])
    
    ylimits = [ylim_offset + 9150, ylim_offset + 9850]
    
    #ylimits = [ylim_offset + 9150+500, ylim_offset + 9850+500] ############
    #ylimits = [ylim_offset + 9150+150, ylim_offset + 9850+150] ############
    #ylimits = [ylim_offset + 9150-200, ylim_offset + 9850-200] ############
    #"""
    ylimits = [ylim_offset + 9150-600, ylim_offset + 9850+1000]
    configuration0b = "zoomupp"
    path_add = "0"
    plot_bar(data_array_list, data_spec_array, y_label,
             flx_data, flx_spec_list, flx_y_spec_list,
             tit, MJD_0, MJD_end, xlimit_zoom, ylimits,
             vline_specs, z_order, nongra_fac, 0,
             vline_list1, vline_list_specs1,
             vline_list2, vline_list_specs2,
             [save_on, path + path_add])
    
    # difference data - fit
    ylimits = [-150, 150]
    data_array_list = [np.vstack((el_data[:, 0], el_data[:, 1] - el_fit[:, 1])).T]
    data_spec_array = [[0.5, "c", el_symb + " - " + el_tilde, 1]]
    y_label = el_symb + " - " + el_tilde + " " + el_unit
    str_3 = r", $\langle Δa \rangle = $ %.3e m" % np.mean(el_data[:, 1] - el_fit[:, 1])
    tit = str_0 + el_symb + " - " + str_1 + str_2 + str_3
    configuration0b = "diff"
    path_add = "0"
    plot_bar(data_array_list, data_spec_array, y_label,
             flx_data, flx_spec_list, flx_y_spec_list,
             tit, MJD_0, MJD_end, xlimits, ylimits,
             vline_specs, z_order, nongra_fac, 0,
             vline_list1, vline_list_specs1,
             vline_list2, vline_list_specs2,
             [save_on, path + path_add])
    
    # spectrum of data - fit
    orig_list_diff = fft(data_array_list[0])
    fig = plt.figure(figsize = (10, 5), dpi = 300)
    plt.title(str_0 + "spectrum of " + el_symb + " - " + str_1 + str_2,
              fontsize = 17.5)
    plt.loglog(orig_list_diff[0], orig_list_diff[1], 'c-',
               label = r'$\Delta a$')
    for i in range(0, len(el_per[:R])):
        p_i = el_per[:R][i]
        if (i == 0):
            lab = 'fitted periods'
        else:
            lab = None
        plt.axvline(p_i, color = 'r', alpha = 0.5,
                    ls = (0, (5, 10)), lw = 2,
                    label = lab)
    #plt.axvline(1.5, color = 'k', alpha = 0.5)
    #plt.axvline(0.5, color = 'k', alpha = 0.5)
    #plt.xlim(1, 3)
    #plt.ylim(10**(-2), 1)
    plt.xlabel("Period [d]", fontsize = 15)
    plt.ylabel("Amplitude", fontsize = 15)
    plt.legend(fontsize = 10, loc = 4)
    plt.grid()
    if (save_on == 1):
        configuration0b = "diffspec"
        path_add = "0"
        fig.savefig(path + path_add, transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)
    #"""

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
                 efac, ave, el_label, el_unit,
                 flx_data, flx_spec_list,
                 MJD_0, MJD_end, xlimits, ylimits, title, grid,
                 fac, location, anchor, n_cols,
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
        
        if (ave == 1):
            mean = np.mean(y_slope)
            y_slope = y_slope / mean
            y_low_slope = y_low_slope / mean
            y_upp_slope = y_upp_slope / mean
            mean_list.append(mean)
        
        ax1.plot(x_slope + MJD_0, y_slope,
                 ls = '-', lw = l_w, color = lcol,
                 alpha =  α, label = lab, zorder = 10)
        ax1.fill_between(x_slope + MJD_0, y_low_slope, y_upp_slope,
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
        n_ap = len(flx_data.T) - 1
        α = flx_spec_list[0]
        col_map_stat = flx_spec_list[1]
        
        ax2 = ax1.twinx()
        bar_width = 1 / (n_ap)
        if (col_map_stat == 1):
            col_map = flx_spec_list[2]
            colmap = mat.colormaps[col_map]
            col_list = cl_lin(np.linspace(0, 1, n_ap), colmap)
            norming = colors.BoundaryNorm(np.arange(1 / 2, n_ap, 1),
                                          colmap.N) # discrete bins
            
            for i in range(1, n_ap + 1):
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
            for i in range(1, n_ap + 1):
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
    
    plt.figlegend(fontsize = 12.5, markerscale = 1, loc = location,
                  bbox_to_anchor = anchor, bbox_transform = ax1.transAxes,
                  labelspacing = 0.1, ncols = n_cols, columnspacing = 1)
    
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)


vline_list1 = [59958.915, 59961.875, 59967.125, 59988.71,
               59995.416, 59999.5, 60001.78125, 60002.427,
               60017.17, 60018.17, 60023.9375, 60026.375, 60053.389]
vline_list_specs1 = ['k', 0.75, 1, "CME"]
vline_list_specs2 = ['r', 0.25, 1, "MAN"]

xlimit_list = [[-0.1, -0.1]]
vline_specs = ["gold", 0.75, 1]
e_fac = 1
fac = 1

name = file_name_list[0]
str_0 = sat_name + ' - ' + name + " data: "
str_1 = r'$\tilde{a}$'
if (len(el_m0_list) > 1):
    str_2 = r", $m_0 = $ [" + ", ".join("%.1e" % m0 for m0 in el_m0_list) + "] m"
else:
    str_2 = r", $m_0 = $ %.1e m" % el_m0_list[0]

n_slope = n_days_tot * 1
slope_data_list = []
slope_specs_list = []
#color_group_list = [['magenta', 'deeppink', 'mediumvioletred'],
#                    ['darkturquoise', 'royalblue', 'darkblue']]
#color_group_list = [['r', 'r'],
#                    ['b', 'b']]
#color_group_list = [['g', 'g'],
#                    ['b', 'b']]
#color_group_list = [['yellow', 'yellow'],
#                    ['red', 'red']]
color_group_list = [['limegreen', 'royalblue']]
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
    lab = lab_gen(ele, configuration)
    e_lab = lab_gen('σ', configuration)
    color_index = 0
    lcol = color_group_list[color_index][0]
    #mcol = color_group_list[color_index][1]
    #ecol = color_group_list[color_index][2]
    ecol = color_group_list[color_index][1]
    #slope_specs = [2, lcol, ecol, 1 - i / 2.5, 0.25, lab, e_lab]
    slope_specs = [2, lcol, ecol, 1 - i / 2.5, 0.75, lab, e_lab]
    slope_data_list.append(slope_data)
    slope_specs_list.append(slope_specs)
    
    name = '%d %d %f %d' % (n_fac, R, ω, constr)
    name = name.replace(' ', '_') + '.txt'
    head = str(m0)
    
    np.savetxt(foldername + '/' + name, slope_data,
               header = head, comments = '')
    



#slope_specs = [5.,  0, "deeppink", 1.5, 1]
add_data_list = []
add_data_specs_list = []
el_label = ele
el_unit = r'$\frac{m}{d}$'
flx_data = data_ap
#flx_spec_list = [0.75, 1, 'plasma']
flx_spec_list = [1, 0, 'crimson']
title = str_0 + "decrease of " + str_1 #+ str_2
#title = 'test'
ylimits = [0, 0]
#ylimits = [-30, 30]
#ylimits = [-20, 20]
z_order = 1
nongra_fac = 1
configuration0b = "decr"
path_add = "0"
ave = 0
n_cols = 1
print(title)
decr_bar_adv(slope_data_list, slope_specs_list,
             e_fac, ave, el_label, el_unit,
             flx_data, flx_spec_list,
             MJD_0, MJD_end, xlimit_list[0], ylimits,
             title, 1, fac, 3, (0., 0.), n_cols,
             vline_list1, vline_list_specs1,
             vline_list2, vline_list_specs2,
             [save_on, path + path_add])
# %%
xlimits = [-0.1, 0.1]
if (len(data_both) == 2):
    string = r'$\tilde{a}_{y,z}$'.replace('y', str(R)).replace('z', str(ω))
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
    flx_data = data_ap
    flx_spec_list = [0.5, 0, 'summer']
    flx_y_spec_list = []
    tit = "data difference of " + el_symb
        
    ylimits = [-0.2, 0.2] # [-0.1, 0.1] [-0.025, 0.025]
    vline_col = "gold"
    z_order = 1
    nongra_fac = 1
    for j in range(0, len(xlimit_list)):
        plot_bar(data_array_list, data_spec_array, y_label,
                   flx_data, flx_spec_list, flx_y_spec_list,
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
    ylimits = [-0.1, 0.1]
    title = "difference of decrease of " + string
    decr_bar_new(slope_data_list, slope_specs_list,
                 add_data_list, add_data_specs_list,
                 el_label, el_unit,
                 flx_data, flx_spec_list, flx_y_spec_list,
                 MJD_0, MJD_end, xlimits, ylimits,
                 title, 0, fac, 1, (1, 1),
                 vline_list1, vline_list_specs1, 
                 vline_list2, vline_list_specs2,
                 [save_on, path + "diff_decrease" + path_add]) # 1, (1, 1)
    
print("HELL YES!!!")
# %%
def ϕ_amb(ϕ_list):
    for i in range(1, len(ϕ_list)):
        ϕ_i_1 = ϕ_list[i - 1]
        ϕ_i = ϕ_list[i]
        if (np.abs(ϕ_i - ϕ_i_1) > np.pi / 2):
            if (ϕ_i - ϕ_i_1 < np.pi / 2):
                for j in range(i, len(ϕ_list)):
                    ϕ_list[j] = ϕ_list[j] + np.pi
            elif (ϕ_i - ϕ_i_1 > np.pi / 2):
                for j in range(i, len(ϕ_list)):
                    ϕ_list[j] = ϕ_list[j] - np.pi
    return(ϕ_list)

def ϕ_arctan(list1, list2):
    ϕ_list = np.arctan(np.array(list1) / np.array(list2))
    return(ϕ_list)

def Δ_el_list(el_list):
    new_list = np.array([])
    for i in range(1, len(el_list)):
        el_i_1 = el_list[i - 1]
        el_i = el_list[i]
        Δ_el_i = el_i - el_i_1
        new_list = np.append(new_list, Δ_el_i)
    return(new_list)

def τ_mid(τ_fit_list):
    τ_mid_list = np.array([])
    for i in range(1, len(τ_fit_list)):
        τ_i_1 = τ_fit_list[i - 1]
        τ_i = τ_fit_list[i]
        τ_mid_i = τ_i_1 + (τ_i - τ_i_1) / 2
        τ_mid_list = np.append(τ_mid_list, τ_mid_i)
    return(τ_mid_list)

def Δp(para_list, τ_fit_list, p_list):
    R = len(p_list)
    τ_mid_list = τ_mid(τ_fit_list)
    Δp_array = τ_mid_list
    Δτ_list = Δ_el_list(τ_fit_list)
    ϕ_r_array = τ_fit_list
    ϕ_r_amb_array = τ_fit_list
    for r in range(0, R):
        p_r = p_list[r]
        μ_r_list = para_list[1 + r]
        η_r_list = para_list[1 + R + r]
        
        ϕ_r_list_amb = ϕ_arctan(μ_r_list, η_r_list)
        ϕ_r_list_amb2 = ϕ_r_list_amb.copy()
        ϕ_r_list = ϕ_amb(ϕ_r_list_amb)
        
        Δϕ_r_list = Δ_el_list(ϕ_r_list)
        
        Δp_r_list = - p_r**2 / (2 * np.pi) * Δϕ_r_list / Δτ_list
        
        Δp_array = np.vstack((Δp_array, Δp_r_list))
        ϕ_r_array = np.vstack((ϕ_r_array, ϕ_r_list))
        ϕ_r_amb_array = np.vstack((ϕ_r_amb_array, ϕ_r_list_amb2))
    Δp_array = Δp_array.T
    ϕ_r_array = ϕ_r_array.T
    ϕ_r_amb_array = ϕ_r_amb_array.T
    return(Δp_array, ϕ_r_array, ϕ_r_amb_array)

for i in range(0, len(configuration_list)):
    configuration = configuration_list[i]
    n_fac = configuration[0]
    R = configurations[1]
    ω = configuration[2]
    constr = configuration[3]
    
    m0 = el_m0_list[i]
    
    para_list = el_para_fit_list[i]
    τ_fit_list = τ_fit_list_list[i]
    p_list = el_per_list[0][:R]
    
    Δp_array, ϕ_r_array, ϕ_r_amb_array = Δp(para_list, τ_fit_list, p_list)
    
    col_list = cl_lin(np.arange(0, R), mat.colormaps['jet'])
    dt = Δp_array[1, 0] - Δp_array[0, 0]
    
    fig, ax1 = plt.subplots(figsize = (11, 5), dpi = 300)
    fig.suptitle(r'Time series of $\phi_r$ (amb)', fontsize = 17.5)
    for i in range(0, R):
        lab = r'$ϕ_i$ = %.2e d' % p_list[i]
        lab = lab.replace('i', str(i + 1))
        ax1.plot(ϕ_r_amb_array[:, 0] + MJD_0, ϕ_r_amb_array[:, i + 1],
                 ls = 'solid', lw = 1, color = col_list[i],
                 marker = 'o', mfc = col_list[i], ms = 1.5,
                 alpha = 0.5, label = lab)
    ax1.xaxis.set_major_formatter(mjd_to_mmdd)
    ax1.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
    ax1.tick_params(axis = 'x', which = 'major',
                    width = 2, length = 8,
                    direction = 'out', top = False)
    ax1.tick_params(axis = 'x', which = 'minor',
                    width = 1, length = 4,
                    direction = 'out', top = False)
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    ax1.set_ylabel(r'$\phi$ [rad]', fontsize = 15)
    ax1.legend(fontsize = 12.5, labelspacing = 0.1, loc = 1)
    ax1.set_xlim(0 + MJD_0, 2 + MJD_0)
    ax1.set_ylim(-2,2)
    ax1.axhline(- np.pi / 2)
    ax1.axhline(np.pi / 2)
    ax1.grid()
    plt.show(fig)
    plt.close(fig)
    
    
    fig, ax1 = plt.subplots(figsize = (11, 5), dpi = 300)
    fig.suptitle(r'Time series of $\phi_r$', fontsize = 17.5)
    for i in range(0, R):
        lab = r'$ϕ_i$ = %.2e d' % p_list[i]
        lab = lab.replace('i', str(i + 1))
        ax1.plot(ϕ_r_array[:, 0] + MJD_0, ϕ_r_array[:, i + 1],
                 ls = 'solid', lw = 1, color = col_list[i],
                 marker = 'o', mfc = col_list[i], ms = 1.5,
                 alpha = 0.5, label = lab)
    ax1.xaxis.set_major_formatter(mjd_to_mmdd)
    ax1.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
    ax1.tick_params(axis = 'x', which = 'major',
                    width = 2, length = 8,
                    direction = 'out', top = False)
    ax1.tick_params(axis = 'x', which = 'minor',
                    width = 1, length = 4,
                    direction = 'out', top = False)
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    ax1.set_ylabel(r'$\phi$ [rad]', fontsize = 15)
    ax1.legend(fontsize = 12.5, labelspacing = 0.1, loc = 1)
    #ax1.set_xlim(0 + MJD_0, 2 + MJD_0)
    #ax1.set_ylim(-2,2)
    #ax1.axhline(- np.pi / 2)
    #ax1.axhline(np.pi / 2)
    ax1.grid()
    plt.show(fig)
    plt.close(fig)
    
    fig, ax1 = plt.subplots(figsize = (11, 5), dpi = 300)
    fig.suptitle("relative period changes", fontsize = 17.5)
    for i in range(0, R):
        lab = r'$p_i$ = %.2e d' % p_list[i]
        lab = lab.replace('i', str(i + 1))
        ax1.plot(Δp_array[:, 0] + MJD_0, Δp_array[:, i + 1] / p_list[i],
                 ls = 'solid', lw = 1.5, color = col_list[i],
                 marker = 'o', mfc = col_list[i], ms = 2.5,
                 alpha = 0.5, label = lab)
    ax1.xaxis.set_major_formatter(mjd_to_mmdd)
    ax1.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
    ax1.tick_params(axis = 'x', which = 'major',
                    width = 2, length = 8,
                    direction = 'out', top = False)
    ax1.tick_params(axis = 'x', which = 'minor',
                    width = 1, length = 4,
                    direction = 'out', top = False)
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    ax1.set_ylabel(r'$\frac{\Delta p_i}{p_i}$', fontsize = 15)
    ax1.legend(fontsize = 12.5, labelspacing = 0.1)
    ax1.grid()
    plt.show(fig)
    plt.close(fig)
    
    p_o_list = []
    a_o_list = []
    for i in range(0, R):
        array = array_columns(np.abs(Δp_array), [0, i + 1])
        orig_list = fft(array)
        p_o_list.append(orig_list[0])
        a_o_list.append(orig_list[1])
    
    fig = plt.figure(figsize = (11, 5), dpi = 300)
    plt.title("spectrum of relative period changes", fontsize = 17.5)
    for i in range(0, R):
        lab = r'$p_i$ = %.2e d' % p_list[i]
        lab = lab.replace('i', str(i + 1))
        plt.loglog(p_o_list[i], a_o_list[i] / (5**i),
                   ls = 'solid', lw = 1.5, color = col_list[i],
                   alpha = 0.5, label = lab)
    plt.xlabel("Period [d]", fontsize = 15)
    plt.ylabel("Amplitude", fontsize = 15)
    plt.legend(fontsize = 12.5, labelspacing = 0.1)
    plt.grid()
    plt.show(fig)
    plt.close(fig)
# %%