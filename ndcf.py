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
from astropy.time import Time
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
os.chdir('/Users/levin/Documents/Uni/Bachelorthesis new')
print(os.getcwd())
#%%
path = 'magnetometer/oe_m1m_dscovr_s20180812000000_e20180812235959_p20180813032247_pub.nc'

def yyyymmddhhmm_to_mjd(string):
    t_obj = Time(string, format = 'isot', scale = 'utc')
    t_mjd = t_obj.mjd
    mjd_float = float(t_mjd)
    return(mjd_float)

t0 = yyyymmddhhmm_to_mjd("1970-01-01T00:00:00Z")

from netCDF4 import Dataset
data = Dataset(path, "r", format="NETCDF4")
time = t0 + (data['time'][:])/(24*60*60*1000) # miliseconds since 1970-01-01T00:00:00Z
bz = data['bz_gsm'][:] # nT
data.close()
#%%
fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.plot(time, bz, 'r.')
plt.show()
plt.close()
# %%
