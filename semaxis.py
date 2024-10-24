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
foldername = "oscele24/swma"
file_name = "normal"

path_flx = 'All Data new/FLXAP_P_MJD.FLX'
file_name_list = ["nongra"]
el_str = "a"
#MJD_interval = [58340, 58413] # year 2018 (2018.08.10 - 2018.10.22)
#MJD_interval = [59990, 60040] # year 2023 (2023.02.15 - 2023.04.06)
MJD_interval = [60431, 60445-9] # year 2024 (2024.05.01 - 2024.05.14)
MJD_0 = MJD_interval[0]
n_days_tot = MJD_interval[1] - MJD_0
save_on = 0
path = os.getcwd() + "/updates/update 10/swarm with flx/"
q = 0
n_partition = 1
lsa_method = 5

name_el = foldername + 2 * ("/year_" + file_name) + "_" + el_str + ".txt"
el_data = np.loadtxt(name_el, skiprows=1)  # load data
el_data = array_denormalize(el_data)  # denormalize data
el_data = array_modifier(el_data, MJD_0, n_days_tot)  # trim data
#el_data = array_normalize(el_data, 0)  # normalize data
#el_data = el_data[1:]  # cut MJD_0
# %%
fig = plt.figure(figsize = (10,5), dpi = 300)
plt.plot(el_data[:,0],el_data[:,1],'r.')
plt.show(fig)
plt.close(fig)
print("range = %.0f, %.0f" % (min(el_data[:,1])/1000, max(el_data[:,1])/1000))
# %%
from astropy.time import Time
def mjd_to_mmdd(t, pos):
    t_obj = Time(str(t), format = 'mjd')
    t_iso = t_obj.iso
    string = t_iso[5 : 10].replace('-', '.')
    return(string)

def mjd_to_ddmm(t, pos):
    t_obj = Time(str(t), format = 'mjd')
    t_iso = t_obj.iso
    mm = t_iso[5 : 7]
    dd = t_iso[8 : 10]
    string = dd + "." + mm
    return(string)

def xaxis_year(t_list, time_format):
    t0 = t_list[0]
    t_obj = Time(str(t0), format = 'mjd')
    t_iso = t_obj.iso
    string = t_iso[0 : 4]
    string = string + " + " + time_format
    return(string)

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

def plot_data(data_list, data_spec_array, y_label,
              ref_list, ref_spec_array, ref_y_spec_list,
              tit, xlimits, ylimits, grid,
              location, anchor, n_cols, fosi, labspa,
              vline_list1, vline_list_specs1,
              vline_list2, vline_list_specs2,
              save_specs):
    """ description of input parameters
    data_list        : [data_1, data_2, ...]
    data_spec_array  : [alpha_i, col_i, lab_i, lw_i]
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
    
    fig, ax1 = plt.subplots(figsize = (10, 5), dpi = 300)
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
    #ax1.set_xticks(MJD_0 + np.linspace(0,24,25)/24)
    ax1.get_yaxis().get_major_formatter().set_useMathText(True)
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    ax1.set_ylabel(y_label, fontsize = 15)
    ax1.axhline(6.85525e6, color = 'g')
    ax1.axhline(6.8362e6, color = 'b')
    y_low, y_upp = ylimits[0], ylimits[1]
    if (xstart != xend):
        ax1.set_xlim(xstart, xend)
    if (y_low != y_upp):
        ax1.set_ylim(y_low, y_upp)
    if (grid == 1):
        ax1.grid()
    
    if (len(ref_list) != 0):
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
        
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.set_frame_on(False)
    
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
    
    #plt.figlegend(fontsize = fosi, markerscale = 5, loc = location,
    #              bbox_to_anchor = anchor, bbox_transform = ax1.transAxes,
    #              labelspacing = labspa, ncols = n_cols)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    
    plt.show(fig)
    plt.close(fig)

print(el_data[-1,0]-el_data[0,0])

data_list = [el_data] # a_data
data_spec_array = [#[0.5, 'y', r'$a_{\text{LST}}$', 0.1, 0.25, 1],
                   [1, 'chocolate', r'$a_{\text{PCA}}$', 1, 0.25, 1e4, r'$\sigma$']]
a_unit =r'[$\text{m}$]'
y_label = r'$a$' + " " + a_unit
ref_list, ref_spec_array, ref_y_spec_list = [], [], []
tit = None #r'semi-major axis from LST-data and numerical integration'
xlimits = [0, 0]
mean = np.mean(el_data[:, 1])
std = np.std(el_data[:, 1])
fac = 1.5
ylimits = [mean - fac * std, mean + fac * std]
ylimits = [6.8358e6, 6.8556e6]
#6.85525e6, 6.8362e6
print(6.8362e6-6.8358e6, 6.8556e6-6.85525e6)
#ylimits = [0, 0]
grid = 0
save_specs = [0]
location, anchor = 1, (1, 1)
n_cols = 1
fosi = 15
labspa = 0.5
vline_list1, vline_list2 = [], []
vline_list_specs1, vline_list_specs2 = [], []
plot_data(data_list, data_spec_array, y_label,
          ref_list, ref_spec_array, ref_y_spec_list,
          tit, xlimits, ylimits, grid,
          location, anchor, n_cols, fosi, labspa,
          vline_list1, vline_list_specs1,
          vline_list2, vline_list_specs2,
          [0, 'Thesis/LATEX/Images/osc_ele_fullsignal.png'])
# %%
fft_list = fft(el_data)
p_data, a_data = fft_list[0], fft_list[1]

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
                       modus, save_specs):
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
    if (modus == 'd'): # day
        γ = 1
        p_lab = '[d]'
    elif (modus == 'h'): # hour
        γ = 24
        p_lab = '[h]'
    elif (modus == 'm'): # minute
        γ = 24 * 60
        p_lab = '[min]'
    elif (modus == 's'): # second
        γ = 24 * 60 * 60
        p_lab = '[s]'
    p = p_lab[1:-1]
    print("fft_logplot_spec_3 version 8")
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
            plt.semilogx(p_o_list[i]*γ, new_a_o_list[i],
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
            plt.loglog(p_o_list[i]*γ, new_a_o_list[i],
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
        plt.scatter(per_list_list[i]*γ, new_amp_list_list[i],
                    color = col, s = size,
                    marker = mark,
                    fc = 'None', lw = 0.5,
                    alpha = 1, label = lab)
        for j in range(0, len(per_list_list[i])):
            str_lab = dollar_remove(text)
            string = r'$p_{0}$'.replace('0', str(j + 1))
            plt.annotate(string, (per_list_list[i][j]*γ, new_amp_list_list[i][j]),
                         coords, textcoords = 'offset points',
                         fontsize = 10)
            string_add = ' = %.6e ' % (per_list_list[i][j]*γ)
            #plt.figtext(0.92, 0.875 - k, string + string_add,
            #            fontsize = 10, ha = 'left')
            print(string + string_add + p)
            k += 0.05
        k += 0.03
    
    for i in range(0, n_Δt):
        lab1 = r'$Δt_{xyz}$ = '.replace('xyz', dollar_remove(Δt_spec_array[i][2]*γ))
        lab2 = str(np.round(Δt_list[i], 3)) + ' ' + p
        plt.axvline(Δt_list[i]*γ, color = Δt_spec_array[i][1], alpha = 0.5,
                    ls = Δt_spec_array[i][0], lw = 5,
                    label = lab1 + lab2)
    
    for i in range(0, len(vlines_list)):
        vline = vlines_list[i]
        plt.axvline(vline*γ, color = vlines_specs[0],
                    alpha = vlines_specs[1], lw = vlines_specs[2],
                    ls = 'solid')
    
    if (xlimits[0] != xlimits[1]):
        plt.xlim(xlimits[0], xlimits[1])
    
    if (ylimits[0] != ylimits[1]):
        plt.ylim(ylimits[0], ylimits[1])
    
    plt.xlabel(r'period ' + p_lab, fontsize = 15)
    plt.ylabel(r'amplitude ' + unit, fontsize = 15)
    #plt.legend(fontsize = fosi, loc = location,
    #           labelspacing = labspa, ncols = n_cols)
    plt.grid()
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)

lcm_list_el = [0, 50, 1, 0.09, 0.95, 10]
interval_list_el = [[0,0]]#[[1.75,2]]#for se1a

#n_days_tot_lst = 61

el_p_o_list = []
el_a_o_list = []
el_per_list = []
el_amp_list = []

# data_spectrum
el_spectrum_list = spectrum(el_data, -1, lcm_list_el, interval_list_el)
el_p_o, el_a_o = el_spectrum_list[0], el_spectrum_list[1]
el_per, el_amp = el_spectrum_list[2], el_spectrum_list[3]

el_p_o_list.append(el_p_o)
el_a_o_list.append(el_a_o)
el_per_list.append(el_per)
el_amp_list.append(el_amp)


spectrum_p = [el_p_o]
spectrum_a = [el_a_o]
spectrum_data_specs = [[1, "chocolate", 'a']]
spectrum_per = [el_per]
spectrum_amp = [el_amp]
marker_spec_array = [[(5, -5), 10, "o", "k",
                      "peaks used for " + r'$a_{\text{Fit}}$', 'a']]
#for se1a
#marker_spec_array = [[(-15, 0), 10, "o", "k",
#                      "peaks used for " + r'$a_{\text{Fit}}$', el_symb]]
Δt_list = []
Δt_spec_array = [[(0, (4, 4)), "r", 'a']]
spectrum_tit = None
v_line_specs = ['b', 0.5, 1]
xlimits = [1, 1e4]
#xlimits = [0.01, 0.1]
ylimits = [0, 0]
ylimits = [0.01, 1e5]
v_line_list = []#[0.019, 0.026, 0.05, 0.1, 5.55]
log_fac_list = []
location = 4
fosi = 15
labspa = 0.5
n_cols = 1
modus = 'm'
fft_logplot_spec_3(spectrum_p, spectrum_a, spectrum_data_specs,
                   spectrum_per, spectrum_amp, marker_spec_array,
                   Δt_list, Δt_spec_array, spectrum_tit, "[m]", 1,
                   log_fac_list, location, fosi, labspa, n_cols,
                   v_line_list, v_line_specs,
                   xlimits, ylimits, modus,
                   [0])
# %%
