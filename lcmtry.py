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
#%%
def round_dot5(x):
    # round 0.5 to 1
    if (x - int(x) == 0.5):
        return(round(x + 0.5, 0))
    else:
        return(round(x, 0))

def round_uneven(x):
    # round x to the nearest uneven number
    # if x is an even number, then it should return x + 1
    x_int = 0
    x_round = int(round_dot5(x)) # round to nearest integer
    if (x_round % 2 == 1): # nearest integer is uneven
        x_int = x_round
    elif (x % 2 == 0): # x is even
        x_int = int(x + 1)
    else: # nearest integer is even -> find out if rounded up or down
        diff = x - x_round
        if (diff < 0): # it was rounded up
            x_int = int(x)
        else: # it was rounded down
            x_int = int(x + 1)
    return(x_int)

def mod_abs(u, v):
    return(min(u % v, v - (u % v)))

def lcm_status_1(k, lcm_copy, n, tilde_p_i, hut_ε_i):
    str1 = "k = %3d | " % k
    str2 = "lcm = %12.3f | " % lcm_copy
    str3 = "p_%1d = %9.3f | " % (n, tilde_p_i)
    str4 = "ε_%1d = %6.1f" % (n, hut_ε_i)
    print(str1 + str2 + str3 + str4)

def lcm_status_2(n, hut_ε_i, hut_ε):
    str1 = "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    str2 = "----------------- "
    str3 = "ε_%1d = %6.1f < %6.1f = ε" % (n, hut_ε_i, hut_ε)
    str4 = " -----------------"
    print(str1)
    print(str2 + str3 + str4)
    print(str1)

def lcm_as_Δt(ε, N, limit, dt, per_list):
    tilde_per_list = per_list / dt
    tilde_limit = limit / dt
    hut_ε = ε / N
    tilde_lcm = tilde_per_list[0]
    n = 1
    n_tot = len(per_list)
    
    print("TO ALIGN: ", per_list)
    print("eff periods: ", tilde_per_list)
    print("hut_ε = %9.3f | tilde_limit = %9.3f" % (hut_ε, tilde_limit))
    
    k = 1
    while (k * N * tilde_lcm < tilde_limit and n < n_tot):
        lcm_copy = k * tilde_lcm
        
        tilde_p_i = tilde_per_list[n]
        hut_ε_i = mod_abs(lcm_copy, tilde_p_i)
        lcm_status_1(k, lcm_copy, n, tilde_p_i, hut_ε_i)
        if (hut_ε_i < hut_ε):
            lcm_status_2(n, hut_ε_i, hut_ε)
            tilde_lcm *= k
            n += 1
            k = 0
        
        k += 1
    print("result:", N * tilde_lcm * dt, per_list[:n])
    return(N * tilde_lcm * dt, per_list[:n])

def testfunc(ε, N, limit, dt, per_list):
    Δt, per0_list = lcm_as_Δt(ε, N, limit, dt, per_list)
    n_Δt = round_uneven(Δt / dt) # window width for filtering
    Δt_n = n_Δt * dt # corresponding time interval for filtering
    Δt_dt = Δt_n % dt # "error"
    print("n_Δt = ", n_Δt)
    print("Δt_n = ", Δt_n)
    print("Δt_dt = ", Δt_dt)
    
ε=2
N=1
limit=10000
dt=15
per_list=np.array([121,343,441,152])
print(testfunc(ε, N, limit, dt, per_list))
# %%
print(round_uneven(3.9))
# %%
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
    print(fac_str, exp_str)
    if (fac_str == '1'):
        if (exp_str == '0'):
            new_str = r'$1$'
        elif (exp_str == '1'):
            new_str = r'$10$'
        else:
            new_str = r'$10^{%s}$' % (exp_str)
    else:
        if (exp_str == '0'):
            new_str = r'$%s$' % fac_str
        else:
            new_str = r'$%s\cdot10^{%s}$' % (fac_str, exp_str)
    return(new_str)

scientific_to_exponent(0.3, 1)
# %%
