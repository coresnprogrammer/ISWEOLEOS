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
os.chdir('/Users/levin/Documents/Uni/Bachelorthesis new')
print(os.getcwd())
#%%
#file_extreme_day_new('sat24/sentinel/lst', 1)
#%%
satname = 'SENTINEL-1A'
year = '2023'

image_folder = 'Thesis/LATEX/Images/Results/' + satname + '_' + year + '/'

#foldername = "All Data new/oscelesentinel"
#foldername = "newoscele/GRACE_LST"
#foldername = "All Data new/osceleswarm"
#foldername = "oscele24/swma"

foldername = 'gpsquality'
filename = 'Phase_RMS_S1A_2023.txt'

#file_name = "nongra"

path_flx = 'All Data new/FLXAP_P_MJD.FLX'
file_name_list = ["nongra"]
el_str = "a"
#MJD_interval = [58340, 58413] # year 2018 (2018.08.10 - 2018.10.22)
MJD_interval = [59990, 60040] # year 2023 (2023.02.15 - 2023.04.06)
#MJD_interval = [60431, 60445] # year 2024 (2024.05.01 - 2024.05.14)
#MJD_interval = [59945, 60309]
MJD_0 = MJD_interval[0]
n_days_tot = MJD_interval[1] - MJD_0
save_on = 0
path = os.getcwd() + "/updates/update 10/swarm with flx/"
q = 0
n_partition = 1
lsa_method = 5

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

vline_list_specs1 = ['gold', 1, 1, "CME", (0, (4, 4)), 2]


vline_list2 = []
vline_list_specs2 = ['saddlebrown', 1, 1, "MAN", (0, (4, 4)), 2]
# put as comments if it is not sentinel-1a!!!
#"""
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
vline_list_specs1 = ['gold', 1, 1.5, "CME", (0, (4, 12)), 2]
vline_list_specs2 = ['saddlebrown', 1, 1.5, "MAN", (-6, (2, 4,2,8)), 2]
#"""
#%%
from astropy.time import Time
def gen_quality_file(foldername, filename):
    file = open(foldername + '/' + filename, 'r')
    lines = file.readlines()
    file.close()
    
    data = np.zeros((1, 2))
    print(data)
    for i in range(0, len(lines)):
        row = lines[i].strip()
        yday = '20%s:%s' % (row[:2], row[2:5])
        t_obj = Time(yday, format = 'yday', scale = 'utc')
        mjd = float(t_obj.mjd)
        if (len(row) == 5):
            rms = -1 # no value available
        else:
            rms = float(row[38:46])
        data = np.vstack((data, np.array([mjd, rms])))
    data = data[1:]
    
    np.savetxt(foldername + '/gps_rms.txt', data)
    
    print("Done!!!")

#gen_quality_file(foldername, filename)
gps_quality_data = np.loadtxt(foldername + '/gps_rms.txt')
gps_quality_data = array_modifier(gps_quality_data, MJD_0, n_days_tot)
#%%

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
            new_str = r'$%s \cdot 10^{%s}$' % (fac_str, exp_str)
    return(new_str)


def plot_bar(data, data_specs, ylab,
             ylimits, grid, fosi,
             location, anchor, n_cols, labspa,
             vline_list1, vline_list_specs1,
             vline_list2, vline_list_specs2,
             save_specs):
    xstart, xend = min(data[:, 0]), max(data[:, 0])
    fig, ax = plt.subplots(figsize = (12, 5), dpi = 300)
    ax.xaxis.set_major_formatter(mjd_to_mmdd)
    ax.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
    ax.tick_params(axis = 'x', which = 'major',
                   width = 2, length = 8,
                   direction = 'out', top = False)
    ax.tick_params(axis = 'x', which = 'minor',
                   width = 1, length = 4,
                   direction = 'out', top = False)
    ax.set_xlabel(xaxis_year([xstart], 'mm.dd'), fontsize = 15)
    ax.set_ylabel(ylab, fontsize = 15)
    if (grid == 1):
        ax.grid()
    y_low, y_upp = ylimits[0], ylimits[1]
    if (y_low != y_upp):
        ax.set_ylim(y_low, y_upp)
    ax.set_xlim(xstart, xend)
    
    α = data_specs[0]
    col = data_specs[1]
    bar_width = 1
    ax.bar(data[:, 0] + 0.5, data[:, 1]*1000,
           width = bar_width, alpha = α,
           color = col, lw = 1, label = 'RMS')
    
    c = 0
    for i in range(0, len(vline_list1)):
        vline = vline_list1[i]
        col = vline_list_specs1[0]
        α = vline_list_specs1[1]
        width = vline_list_specs1[2]
        style = vline_list_specs1[4]
        lab = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                lab = vline_list_specs1[3]
                c += 1
            ax.axvline(vline, color = col, alpha = α,
                       ls = style, lw = width,
                       label = lab, zorder = 200)
    c = 0
    for i in range(0, len(vline_list2)):
        vline = vline_list2[i]
        col = vline_list_specs2[0]
        α = vline_list_specs2[1]
        width = vline_list_specs2[2]
        style = vline_list_specs2[4]
        lab = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                lab = vline_list_specs2[3]
                c += 1
            ax.axvline(vline, color = col, alpha = α,
                       ls = style, lw = width,
                       label = lab, zorder = 20)
    
    plt.figlegend(fontsize = fosi, markerscale = 1, loc = location,
                  bbox_to_anchor = anchor, bbox_transform = ax.transAxes,
                  labelspacing = labspa, ncols = n_cols, columnspacing = 1)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)

data = gps_quality_data
data_specs = [0.5, 'turquoise']
ylab = 'RMS [mm]'
ylimits = [0, 0]
grid = 1
fosi = 15
location, anchor = 1, (1, 1)
n_cols = 1
labspa = 0.5

plot_bar(data, data_specs, ylab,
         ylimits, grid, fosi,
         location, anchor, n_cols, labspa,
         vline_list1, vline_list_specs1,
         vline_list2, vline_list_specs2,
         [0, ])
print(data[-1,0]-data[0,0])
# %%
print(Time('2023-02-25', format = 'iso', scale = 'utc').yday)
# %%
data = np.loadtxt('se1a_att/year/year_ATT.txt')
MJD_0 = data[0,0]

xstart, xend = 0, 0

fig, ax = plt.subplots(figsize=(10,5),dpi=300)
#ax.plot(data[:,0],data[:,1], 'r.', label = 'Qx')
#ax.plot(data[:,0],data[:,2], 'b.', label = 'Qy')
#ax.plot(data[:,0],data[:,3], 'g.', label = 'Qz')
ax.plot(data[:,0],data[:,4], 'm.', label = 'Qw')
ax.xaxis.set_major_formatter(mjd_to_mmdd)
ax.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
ax.tick_params(axis = 'x', which = 'major',
                    width = 2, length = 8,
                    direction = 'out', top = False)
ax.tick_params(axis = 'x', which = 'minor',
                    width = 1, length = 4,
                    direction = 'out', top = False)
ax.get_yaxis().get_major_formatter().set_useMathText(True)
ax.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
ax.set_ylabel('',rotation='horizontal', fontsize = 20)
c = 0
for i in range(0, len(vline_list1)):
    vline = vline_list1[i]
    col = vline_list_specs1[0]
    α = vline_list_specs1[1]
    width = vline_list_specs1[2]
    style = vline_list_specs1[4]
    lab = None
    if (xstart <= vline and xend >= vline):
        if (c == 0):
            lab = vline_list_specs1[3]
            c += 1
        ax.axvline(vline, color = col, alpha = α,
                   ls = style, lw = width,
                      label = lab, zorder = 200)
c = 0
for i in range(0, len(vline_list2)):
    vline = vline_list2[i]
    col = vline_list_specs2[0]
    α = vline_list_specs2[1]
    width = vline_list_specs2[2]
    style = vline_list_specs2[4]
    lab = None
    if (xstart <= vline and xend >= vline):
        if (c == 0):
            lab = vline_list_specs2[3]
            c += 1
        ax.axvline(vline, color = col, alpha = α,
                   ls = style, lw = width,
                   label = lab, zorder = 20)
ax.set_xlim(MJD_0 + 8, MJD_0 + 15)
ax.grid()
ax.legend()
plt.show()
plt.close()
# %%
