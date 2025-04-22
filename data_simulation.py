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
#from scipy.optimize import curve_fit
from scipy import special
import os
import random
import matplotlib.image as mpimg
#from scipy.integrate import solve_ivp as ode_solve
#from scipy.optimize import curve_fit
plt.style.use('classic')
mat.rcParams['figure.facecolor'] = 'white'
print(np.random.randint(1,9))
# print(u'\u03B1') # alpha
# print(u'\u03C3') # sigma
# %%
from functions import master_lst, master_ele, master_acc, master_man
from functions import flx_get_data_ymd, yyyymmddhhmm_to_mjd
from functions import array_modifier, array_columns, round_dot5
from functions import smoother, spectrum, fitter, integrator, fft, master_integrator
from functions import down_sample, step_data_generation, gen_a_dot_array
from functions import cl_lin, mjd_to_mmdd, xaxis_year, decr_bar_ult
from functions import smoother_give_n, low_pass_filter

from functions2 import fitter_new, get_a_dot, get_a_list, get_abar_list
#Δa_day(n_rev_day, fac, a, ρ)
#%%
os.chdir('/Users/levin/Documents/Uni/Master/Semester 2/Mysterious/wien/plots/Code')
print(os.getcwd())
os.chdir('..')
#os.chdir('FurtherResearch/')
print(os.getcwd())
print(os.listdir())
#%%
def perturbation(t, t0, height_fac, width_fac):
    # t_start: beginning of data
    # data_mean and data_std are important for scaling
    # height_fac and width_fac for properties of gaussian perturbation
    # 1/2 * (1 + erf((x - μ)/(sqrt(2)σ)))
    argument = (t - t0) * np.sqrt(np.pi) / (width_fac / 2)
    return(- height_fac * width_fac / 2 * (1 + special.erf(argument)) / 2)

def perturbation_slope(t, t0, height_fac, width_fac):
    argument = - 4 * np.pi * ((t - t0) / width_fac)**2
    return(- height_fac * np.exp(argument))
#%%
generator = np.random.default_rng()
def data_simulator(status_list, hf_list, lf_list,
                   p_list, a_list, ϕ_list,
                   noise_fac, pert_list_list,
                   off, lin, t_dep_fac, t_list):
    status_lin = status_list[0]
    status_pert = status_list[1]
    #status_noise = status_list[2]
    status_p_t = status_list[2]
    status_hf = status_list[3]
    status_lf = status_list[4]
    
    hf_n = hf_list[0]
    hf_p_low, hf_p_high = hf_list[1], hf_list[2]
    hf_a_low, hf_a_high = hf_list[3], hf_list[4]
    lf_n = lf_list[0]
    lf_p_low, lf_p_high = lf_list[1], lf_list[2]
    lf_a_low, lf_a_high = lf_list[3], lf_list[4]
    
    p_hf_list = 10**generator.uniform(low = hf_p_low, high = hf_p_high,
                                      size = status_hf * hf_n)
    a_hf_list = generator.uniform(low = hf_a_low, high = hf_a_high,
                                  size = status_hf * hf_n)
    ϕ_hf_list = generator.random(status_hf * hf_n) * 2 * np.pi

    p_lf_list = 10**generator.uniform(low = lf_p_low, high = lf_p_high,
                                      size = status_lf * lf_n)
    a_lf_list = generator.uniform(low = lf_a_low, high = lf_a_high,
                                  size = status_lf * lf_n)
    ϕ_lf_list = generator.random(status_lf * lf_n) * 2 * np.pi
    
    k_pert = len(pert_list_list)
    
    p_all = np.concatenate((p_list, p_hf_list, p_lf_list))
    a_all = np.concatenate((a_list, a_hf_list, a_lf_list))
    ϕ_all = np.concatenate((ϕ_list, ϕ_hf_list, ϕ_lf_list))
    
    el_list = np.array([])
    for i in range(0, len(t_list)):
        t_i = t_list[i]
        term1 = off + status_lin * lin * t_i
        term2 = 0
        for j in range(0, len(p_all)):
            p_j = p_all[j]
            a_j = a_all[j]
            ϕ_j = ϕ_all[j]
            if (j < len(p_list)):
                p_t_fac = status_p_t * t_dep_fac * lin / off * t_i
                t_term = 2 * np.pi * t_i / (p_j * (1 - p_t_fac)**(3 / 2))
            else:
                t_term = 2 * np.pi / p_j * t_i
            term2 += a_j * np.sin(t_term + ϕ_j)
        #term2 += status_noise * random.uniform(-noise_fac, noise_fac)
        if (status_pert != 0):
            for k in range(0, k_pert):
                t0 = pert_list_list[k][0]
                height_fac = pert_list_list[k][1]
                width_fac = pert_list_list[k][2]
                term2 += perturbation(t_i, t0, height_fac, width_fac)
        el_list = np.append(el_list, term1 + term2)
    
    sim_data = np.vstack((t_list, el_list)).T
    return(sim_data)
#%%
def execute_fit(sim_data, fit_specs, R):
    time_start = tt.time()
    n_days = fit_specs[0]
    p_list = fit_specs[1][:R]
    e_fac = fit_specs[2]
    n_fac = fit_specs[3]
    ω = fit_specs[4]
    
    n_s = n_fac * n_days
    fit_list = fitter(sim_data, p_list, np.zeros(len(p_list)), n_s, ω, 1)
    fit_data = fit_list[0]
    para_list = fit_list[1]
    para_e_list = fit_list[2]
    m0 = fit_list[3]
    τ_fit_list = fit_list[4]
    slope_gen_list = fit_list[5]
    bar_data = fit_list[6]
    
    n_slope = 10 * n_days * n_fac
    slope_data = gen_a_dot_array(n_slope, slope_gen_list)
    
    bar_error = np.sqrt(bar_data[:, 2])
    bar_wo_error = bar_data[:, : 2]
    bar_data = np.vstack((bar_wo_error.T, bar_error)).T
    
    big_fit_list = [fit_data, para_list, para_e_list,
                    m0, τ_fit_list, slope_data, bar_data]
    time_end = tt.time()
    duration = time_end - time_start
    print(duration)
    return(big_fit_list)
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
        elif (exp_str == '1'):
            new_str = r'$%s \cdot 10$' % (fac_str)
        else:
            new_str = r'$%s \cdot 10^{%s}$' % (fac_str, exp_str)
    return(new_str)

def show_fit_old(sim_data, fit_specs, e_fac_bar, e_fac_slope, show_list, zoom_list, big_fit_list,
             off, lin, pert_list_list, save_specs):
    # show_list = [spectrum, data_fit, data_fit_zoom, decrease]
    # zoom_list = [xlima, xlimb, zoom_fac_max, zoom_fac_min]
    n_days = fit_specs[0]
    p_list = fit_specs[1]
    #e_fac = fit_specs[2]
    n_fac = fit_specs[3]
    ω = fit_specs[4]
    
    n_s = n_fac * n_days
    fit_data = big_fit_list[0]
    para_list = big_fit_list[1]
    para_e_list = big_fit_list[2]
    m0 = big_fit_list[3]
    τ_fit_list = big_fit_list[4]
    slope_data = big_fit_list[5]
    bar_data = big_fit_list[6]
    
    mean = np.mean(sim_data[:, 1])
    std = np.std(sim_data[:, 1])
    
    fft_list = fft(sim_data)
    p_o, a_o = fft_list[0], fft_list[1]
    
    fit_fft_list = fft(fit_data)
    fit_p_o, fit_a_o = fit_fft_list[0], fit_fft_list[1]
    
    bar_fft_list = fft(bar_data)
    bar_p_o, bar_a_o = bar_fft_list[0], bar_fft_list[1]
    
    diff_data = np.vstack((sim_data[:, 0], sim_data[:, 1] - fit_data[:, 1])).T
    diff_fft_list = fft(diff_data)
    diff_p_o, diff_a_o = diff_fft_list[0], diff_fft_list[1]
    
    dec_fft_list = fft(slope_data[:, :2])
    dec_p_o, dec_a_o = dec_fft_list[0], dec_fft_list[1]
    
    pert_amp_list = []
    for i in range(0, len(pert_list_list)):
        pert_amp = pert_list_list[i][1]
        pert_amp_list.append(pert_amp)
    
    t1 = tt.time()
    bar_t = np.linspace(0, n_days, 1000)
    bar_sim_data = np.array([0 ,0])
    bar_slope_sim_data = np.array([0, 0])
    t0 = pert_list_list[0][0]
    height_fac = pert_list_list[0][1]
    width_fac = pert_list_list[0][2]
    for i in range(0, len(bar_t)):
        t_i = bar_t[i]
        bar_sim = off + lin * t_i + perturbation(t_i, t0, height_fac, width_fac)
        bar_slope_sim = lin + perturbation_slope(t_i, t0, height_fac, width_fac)
        bar_sim_data = np.vstack((bar_sim_data, np.array([t_i, bar_sim])))
        bar_slope_sim_data = np.vstack((bar_slope_sim_data, np.array([t_i, bar_slope_sim])))
    bar_sim_data = bar_sim_data[1:]
    bar_slope_sim_data = bar_slope_sim_data[1:]
    t2 = tt.time()
    print("time: %.3f" % (t2 - t1))
    
    
    bar_sim_fft_list = fft(bar_sim_data)
    bar_sim_p_o, bar_sim_a_o = bar_sim_fft_list[0], bar_sim_fft_list[1]
    
    
    save_on = save_specs[0]
    path = save_specs[1]
    settings = "w_%.1e" % ω
    
    print("ω = %.1e" % ω)
    print("m_0 = %.3e m" % m0)
    
    if (show_list[0] == 1): # show spectrum
        fig = plt.figure(figsize = (10, 5), dpi = 300)
        #plt.title("Spectrum", fontsize = 20)
        plt.loglog(p_o, a_o, 'r-', alpha = 1, lw = 1, label = r'$a_{\text{Sim}}$')
        plt.loglog(fit_p_o, fit_a_o/4, 'b-', alpha = 1, lw = 1,
                   label = r'$a_\text{Fit} / 4$')
        
        plt.loglog(bar_sim_p_o, bar_sim_a_o/16, 'g-', alpha = 1, lw = 1,
                   label = r'$\bar{a}_{\text{Sim}} / 16$')
        plt.loglog(bar_p_o, bar_a_o/16, 'g-', alpha = 1, lw = 1,
                   label = r'$\bar{a}_{\text{Fit}} / 64$')
        #plt.loglog(bar_p_o, bar_a_o*10**8, 'g-', alpha = 1, lw = 1, label = r'trend_fit $\cdot 10^8$')
        #plt.ylim(1e-6,1e4)
        
        plt.xlabel("Period [d]", fontsize = 20)
        plt.ylabel("Amplitude", fontsize = 20)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 15)
        plt.legend(fontsize = 17.5, labelspacing = 0.25, loc = 4)
        plt.grid()
        if (save_on == 1):
            type_fig = "spec_both_" 
            name = path + type_fig + settings + ".png"
            fig.savefig(name, transparent = None, dpi = 'figure',
                        format = None, metadata = None, bbox_inches = 'tight',
                        facecolor = 'auto', pad_inches = 'layout',
                        edgecolor = 'auto', backend = None)
        plt.show(fig)
        plt.close(fig)
    if (show_list[1] == 1): # show all
        xlima, xlimb = zoom_list[0], zoom_list[1]
        fig, ax = plt.subplots(figsize = (10, 5), dpi = 300)
        #plt.title("Zoom", fontsize = 20)
        #ax.plot(sim_data[:, 0], sim_data[:, 1],
        #        'r-', alpha = 1, lw = 1, label = r'$a_{\text{Sim}}$')
        #ax.plot(fit_data[:, 0], fit_data[:, 1],
        #        'b-', alpha = 0.5, lw = 1, label = r'$a_\text{Fit}$')
        ax.plot(bar_sim_data[:, 0], bar_sim_data[:, 1],
                'r-', alpha = 1, lw = 5,
                label = r'$\bar{a}_{\text{Sim}}$', zorder = 2)
        ax.plot(bar_data[:, 0], bar_data[:, 1],
                'g-', alpha = 1, lw = 1.5,
                label = r'$\bar{a}_{\text{Fit}}$', zorder = 3)
        lab_fac = scientific_to_exponent(e_fac_bar, 1)
        e_lab = r'$\sigma$'
        if (e_fac_bar != 1):
            e_lab = e_lab + r' $\times$ ' + lab_fac
        s_y =  bar_data[:, 2] * e_fac_bar
        ax.fill_between(bar_data[:, 0], bar_data[:, 1] - s_y, bar_data[:, 1] + s_y,
                        color = 'g', alpha = 0.25,
                        label = e_lab, zorder = 1)
        
        ax.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
        ax.tick_params(axis = 'x', which = 'major',
                        width = 2, length = 8,
                        direction = 'out', top = False)
        ax.tick_params(axis = 'x', which = 'minor',
                        width = 1, length = 4,
                        direction = 'out', top = False)
        ax.get_yaxis().get_major_formatter().set_useMathText(True)
        ax.set_xlabel(r'$t$ [d]', fontsize = 20)
        ax.set_ylabel(r'$a$ [m]', fontsize = 20)
        
        ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
        fig.legend(fontsize = 20, labelspacing = 0.5,
                   loc = 4, bbox_to_anchor = (1, 0),bbox_transform = ax.transAxes)
        ax.grid()
        #ax.set_xlim(xlima, xlimb)
        if (save_on == 1):
            type_fig = "all_" 
            name = path + type_fig + settings + ".jpg"
            fig.savefig(name, transparent = None, dpi = 'figure',
                        format = None, metadata = None, bbox_inches = 'tight',
                        facecolor = 'auto', pad_inches = 'layout',
                        edgecolor = 'auto', backend = None)
        plt.show(fig)
        plt.close(fig)
    if (show_list[2] == 1): # show zoom
        xlima, xlimb = zoom_list[0], zoom_list[1]
        zoom_fac_max = zoom_list[2]
        zoom_fac_min = zoom_list[3]
        fig = plt.figure(figsize = (10, 5), dpi = 300)
        plt.title("lower part: ")
        plt.plot(sim_data[:, 0], sim_data[:, 1],
                 'r-', alpha = 1, lw = 1, label = 'sim_data')
        plt.plot(fit_data[:, 0], fit_data[:, 1],
                 'b-', alpha = 0.5, lw = 1, label = 'fit_data')
        plt.xlabel("t [d]")
        plt.ylabel("a [m]")
        plt.legend()
        plt.grid()
        plt.xlim(xlima, xlimb)
        plt.ylim(mean - zoom_fac_max * std, mean - zoom_fac_min * std)
        if (save_on == 1):
            type_fig = "spec_zoom-low_" 
            name = path + type_fig + settings + ".png"
            fig.savefig(name, transparent = None, dpi = 'figure',
                        format = None, metadata = None, bbox_inches = 'tight',
                        facecolor = 'auto', pad_inches = 'layout',
                        edgecolor = 'auto', backend = None)
        plt.show(fig)
        plt.close(fig)
        fig = plt.figure(figsize = (10, 5), dpi = 300)
        plt.title("upper part: ")
        plt.plot(sim_data[:, 0], sim_data[:, 1],
                         'r-', alpha = 1, lw = 1, label = 'sim_data')
        plt.plot(fit_data[:, 0], fit_data[:, 1],
                 'b-', alpha = 0.5, lw = 1, label = 'fit_data')
        plt.xlabel("t [d]")
        plt.ylabel("a [m]")
        plt.legend(loc = 3)
        plt.grid()
        plt.xlim(xlima, xlimb)
        plt.ylim(mean + zoom_fac_min * std, mean + zoom_fac_max * std)
        if (save_on == 1):
            type_fig = "spec_zoom-upp_" 
            name = path + type_fig + settings + ".jpg"
            fig.savefig(name, transparent = None, dpi = 'figure',
                        format = None, metadata = None, bbox_inches = 'tight',
                        facecolor = 'auto', pad_inches = 'layout',
                        edgecolor = 'auto', backend = None)
        plt.show(fig)
        plt.close(fig)
    if (show_list[3] == 1): # show difference
        μ_Δa = np.mean(diff_data[:, 1])
        σ_Δa = np.std(diff_data[:, 1])
        print("μ_Δa = %.3e m" % μ_Δa)
        print("σ_Δa = %.3e m" % σ_Δa)
        fig = plt.figure(figsize = (10, 5), dpi = 300)
        plt.title("Difference", fontsize = 20)
        plt.plot(sim_data[:, 0], sim_data[:, 1] - fit_data[:, 1],
                 'm-', alpha = 1, lw = 1)
        plt.xlabel("t [d]", fontsize = 20)
        plt.ylabel("Δa [m]", fontsize = 20)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 15)
        plt.grid()
        if (save_on == 1):
            type_fig = "diff_" 
            name = path + type_fig + settings + ".jpg"
            fig.savefig(name, transparent = None, dpi = 'figure',
                        format = None, metadata = None, bbox_inches = 'tight',
                        facecolor = 'auto', pad_inches = 'layout',
                        edgecolor = 'auto', backend = None)
        plt.show(fig)
        plt.close(fig)
    if (show_list[4] == 1): # show difference spectrum
        fig = plt.figure(figsize = (10, 5), dpi = 300)
        plt.title("Spectrum of Difference", fontsize = 20)
        plt.loglog(diff_p_o, diff_a_o,
                 'm-', alpha = 1, lw = 1)
        plt.xlabel("Period [d]", fontsize = 20)
        plt.ylabel("Amplitude", fontsize = 20)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 15)
        plt.grid()
        if (save_on == 1):
            type_fig = "spec_diff_" 
            name = path + type_fig + settings + ".png"
            fig.savefig(name, transparent = None, dpi = 'figure',
                        format = None, metadata = None, bbox_inches = 'tight',
                        facecolor = 'auto', pad_inches = 'layout',
                        edgecolor = 'auto', backend = None)
        plt.show(fig)
        plt.close(fig)
    if (show_list[5] == 1): # show decrease
        yupp_fac, ylow_fac = zoom_list[4], zoom_list[5]
        dec_min, dec_max = lin, lin - max(pert_amp_list) #* 0 # deactivate
        μ_dec = np.mean(slope_data[:, 1])
        σ_dec = np.std(slope_data[:, 1])
        print("μ_dec = %.3f m/d" % μ_dec)
        print("σ_dec = %.3e m/d" % σ_dec)
        fig = plt.figure(figsize = (10, 5), dpi = 300)
        plt.title("Slope", fontsize = 20)
        plt.errorbar(slope_data[:, 0], slope_data[:, 1], yerr = e_fac_slope * slope_data[:, 2],
                     ls = 'none', marker = 'o', ms = 1, mew = 0,
                     capsize = 0, mfc = 'b', mec = 'b', ecolor = 'r',
                     elinewidth = 1.5,
                     label = r'$\dot{\bar{a}}$ (%.2e $\cdot$ error)' % e_fac_slope,
                     zorder = 1)
        plt.plot(slope_data[:, 0], slope_data[:, 1],
                 'b-', lw = 1.5, alpha = 1, zorder = 3)
        #"""
        plt.axhline(lin, ls = 'solid', lw = 10,
                    color = 'gold', alpha = 1,
                    label = "target value (slope)",
                    zorder = 2)
        #"""
        #"""
        c = 0
        for i in range(0, len(pert_amp_list)):
            y = pert_amp_list[i]
            lab = None
            if (c == 0):
                lab = "target value (perturbation)"
                c += 1
            plt.axhline(lin - y, ls = 'solid', lw = 5,
                        color = 'g', alpha = 1,
                        label = lab, zorder = 1)
        #"""
        plt.xlabel("t [d]", fontsize = 25)
        plt.ylabel(r"$\dot{\bar{a}}$ [$\frac{m}{d}$]", fontsize = 25)
        
        plt.ylim(ylow_fac * dec_max, yupp_fac * dec_min)
        #plt.gca().ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0))
        #plt.ylim(-2e-6, 2e-6)
        
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
        plt.legend(fontsize = 25, labelspacing = 0.25, loc = 4, markerscale = 3)
        plt.grid()
        if (save_on == 1):
            type_fig = "dec_" 
            name = path + type_fig + settings + ".png"
            fig.savefig(name, transparent = None, dpi = 'figure',
                        format = None, metadata = None, bbox_inches = 'tight',
                        facecolor = 'auto', pad_inches = 'layout',
                        edgecolor = 'auto', backend = None)
        plt.show(fig)
        plt.close(fig)
    if (show_list[6] == 1): # spec of decrease
        fig = plt.figure(figsize = (10, 5), dpi = 300)
        plt.title("Spectrum of Slope", fontsize = 20)
        plt.loglog(dec_p_o, dec_a_o, 'b-', alpha = 1, lw = 1)
        plt.xlabel("Period [d]", fontsize = 25)
        plt.ylabel("Amplitude", fontsize = 25)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
        plt.grid()
        if (save_on == 1):
            type_fig = "spec_dec_" 
            name = path + type_fig + settings + ".png"
            fig.savefig(name, transparent = None, dpi = 'figure',
                        format = None, metadata = None, bbox_inches = 'tight',
                        facecolor = 'auto', pad_inches = 'layout',
                        edgecolor = 'auto', backend = None)
        plt.show(fig)
        plt.close(fig)

def show_fit(sim_data, fit_specs, e_fac_bar, e_fac_slope, show_list, zoom_list, big_fit_list,
             off, lin, pert_list_list, save_specs, R):
    # show_list = [spectrum, data_fit, data_fit_zoom, decrease]
    # zoom_list = [xlima, xlimb, zoom_fac_max, zoom_fac_min]
    n_days = fit_specs[0]
    p_list = fit_specs[1]
    #e_fac = fit_specs[2]
    n_fac = fit_specs[3]
    ω = fit_specs[4]
    
    n_s = n_fac * n_days
    fit_data = big_fit_list[0]
    para_list = big_fit_list[1]
    para_e_list = big_fit_list[2]
    m0 = big_fit_list[3]
    τ_fit_list = big_fit_list[4]
    slope_data = big_fit_list[5]
    bar_data = big_fit_list[6]
    
    mean = np.mean(sim_data[:, 1])
    std = np.std(sim_data[:, 1])
    
    fft_list = fft(sim_data)
    p_o, a_o = fft_list[0], fft_list[1]
    
    fit_fft_list = fft(fit_data)
    fit_p_o, fit_a_o = fit_fft_list[0], fit_fft_list[1]
    
    bar_fft_list = fft(bar_data)
    bar_p_o, bar_a_o = bar_fft_list[0], bar_fft_list[1]
    
    diff_data = np.vstack((sim_data[:, 0], sim_data[:, 1] - fit_data[:, 1])).T
    diff_fft_list = fft(diff_data)
    diff_p_o, diff_a_o = diff_fft_list[0], diff_fft_list[1]
    
    dec_fft_list = fft(slope_data[:, :2])
    dec_p_o, dec_a_o = dec_fft_list[0], dec_fft_list[1]
    
    pert_amp_list = []
    for i in range(0, len(pert_list_list)):
        pert_amp = pert_list_list[i][1]
        pert_amp_list.append(pert_amp)
    
    t1 = tt.time()
    bar_t = np.linspace(0, n_days, 10000)
    bar_sim_data = np.array([0 ,0])
    bar_slope_sim_data = np.array([0, 0])
    t0 = pert_list_list[0][0]
    height_fac = pert_list_list[0][1]
    width_fac = pert_list_list[0][2]
    for i in range(0, len(bar_t)):
        t_i = bar_t[i]
        bar_sim = off + lin * t_i + perturbation(t_i, t0, height_fac, width_fac)
        bar_slope_sim = lin + perturbation_slope(t_i, t0, height_fac, width_fac)
        bar_sim_data = np.vstack((bar_sim_data, np.array([t_i, bar_sim])))
        bar_slope_sim_data = np.vstack((bar_slope_sim_data, np.array([t_i, bar_slope_sim])))
    bar_sim_data = bar_sim_data[1:]
    bar_slope_sim_data = bar_slope_sim_data[1:]
    t2 = tt.time()
    print("time: %.3f" % (t2 - t1))
    
    
    bar_sim_fft_list = fft(bar_sim_data)
    bar_sim_p_o, bar_sim_a_o = bar_sim_fft_list[0], bar_sim_fft_list[1]
    
    
    save_on = save_specs[0]
    path = save_specs[1]
    settings = "R_%d-w_%.1e" % (R, ω)
    
    print("ω = %.1e" % ω)
    print("m_0 = %.3e m" % m0)
    
    #col = ['crimson', 'midnightblue', 'forestgreen'][i]
    #col = ['tomato', 'deepskyblue', 'lime'][i]
    to_min = 24 * 60
    
    if (show_list[0] == 1): # show spectrum data and bar
        fig = plt.figure(figsize = (16, 5), dpi = 300)
        #plt.title("Spectrum", fontsize = 20)
        plt.loglog(p_o * to_min, a_o,
                   'saddlebrown', ls = 'solid', alpha = 1, lw = 2,
                   label = r'$a_{\text{Sim}}$')
        #plt.loglog(fit_p_o * to_min, fit_a_o/2,
        #           'goldenrod', ls = 'solid', alpha = 1, lw = 1,
        #           label = r'$a_\text{Fit} / 2$')
        
        #plt.loglog(bar_sim_p_o * to_min, bar_sim_a_o/200,
        #           'red', ls = 'solid', alpha = 1, lw = 1,
        #           label = r'$\bar{a}_{\text{Sim}} / 200$')
        #plt.loglog(bar_p_o * to_min, bar_a_o/400,
        #           'blue', ls = 'solid', alpha = 1, lw = 1,
        #           label = r'$\bar{a}_{\text{Fit}} / 400$')
        #plt.axvline(115, color = 'g', alpha = 0.25)
        #plt.axvline(127, color = 'g', alpha = 0.25)
        plt.xlabel("period [min]", fontsize = 25)
        plt.ylabel("amplitude [m]", fontsize = 25)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
        plt.legend(fontsize = 30, labelspacing = 0.5, loc = 4,
                   ncols = 2, columnspacing = 0.5)
        plt.xlim(1, 1e5)
        plt.ylim(1e-2, 1e4)
        plt.grid()
        if (save_on == 1):
            type_fig = "spec_" 
            name = path + type_fig + settings + ".png"
            fig.savefig(name, transparent = None, dpi = 'figure',
                        format = None, metadata = None, bbox_inches = 'tight',
                        facecolor = 'auto', pad_inches = 'layout',
                        edgecolor = 'auto', backend = None)
        plt.show(fig)
        plt.close(fig)
    if (show_list[1] == 1): # show bar
        fig, ax = plt.subplots(figsize = (10, 5), dpi = 300)
        #plt.title("Zoom", fontsize = 20)
        #ax.plot(sim_data[:, 0], sim_data[:, 1],
        #        'r-', alpha = 1, lw = 1, label = r'$a_{\text{Sim}}$')
        #ax.plot(fit_data[:, 0], fit_data[:, 1],
        #        'b-', alpha = 0.5, lw = 1, label = r'$a_\text{Fit}$')
        ax.plot(bar_sim_data[:, 0], bar_sim_data[:, 1],
                'red', ls = 'solid', alpha = 1, lw = 5,
                label = r'$\bar{a}_{\text{Sim}}$', zorder = 2)
        ax.plot(bar_data[:, 0], bar_data[:, 1],
                'blue', ls = 'solid', alpha = 1, lw = 1.5,
                label = r'$\bar{a}_{\text{Fit}}$', zorder = 3)
        lab_fac = scientific_to_exponent(e_fac_bar, 1)
        e_lab = r'$\sigma_{\text{Fit}}$'
        if (e_fac_bar != 1):
            e_lab = e_lab + r' $\times$ ' + lab_fac
        s_y =  bar_data[:, 2] * e_fac_bar
        ax.fill_between(bar_data[:, 0], bar_data[:, 1] - s_y, bar_data[:, 1] + s_y,
                        color = 'blue', alpha = 0.25,
                        label = e_lab, zorder = 1)
        
        ax.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
        ax.tick_params(axis = 'x', which = 'major',
                        width = 4, length = 12,
                        direction = 'out', top = False)
        ax.tick_params(axis = 'x', which = 'minor',
                        width = 1, length = 8,
                        direction = 'out', top = False)
        ax.tick_params(axis = 'y', which = 'major',
                       width = 2, length = 8,
                       direction = 'inout', labelsize = 20)
        ax.get_yaxis().get_major_formatter().set_useMathText(True)
        ax.set_xlabel(r'$t$ [d]', fontsize = 25)
        ax.set_ylabel(r'$a$ [m]', fontsize = 25)
        ax.set_ylim(6.836e6,6.8378e6)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
        fig.legend(fontsize = 25, labelspacing = 0.5,
                   loc = 1, bbox_to_anchor = (1, 1),bbox_transform = ax.transAxes)
        ax.grid()
        if (save_on == 1):
            type_fig = "bar_" 
            name = path + type_fig + settings + ".png"
            fig.savefig(name, transparent = None, dpi = 'figure',
                        format = None, metadata = None, bbox_inches = 'tight',
                        facecolor = 'auto', pad_inches = 'layout',
                        edgecolor = 'auto', backend = None)
        plt.show(fig)
        plt.close(fig)
    if (show_list[2] == 1): # show spectrum of slope
        fig = plt.figure(figsize = (10, 5), dpi = 300)
        #plt.title("Spectrum", fontsize = 20)
        plt.loglog(bar_sim_p_o * to_min, bar_sim_a_o,
                   'green', ls = 'solid', alpha = 1, lw = 1,
                   label = r'$\dot{\bar{a}}_{\text{Sim}}$')
        plt.loglog(dec_p_o * to_min, dec_a_o,
                   'purple', ls = 'solid', alpha = 1, lw = 1,
                   label = r'$\dot{\bar{a}}_\text{Fit}$')
        
        plt.xlabel("period [min]", fontsize = 20)
        plt.ylabel("amplitude [m]", fontsize = 20)
        plt.xlim(1e0, 1e5)
        #plt.ylim(1e-4, 1e3)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 15)
        plt.legend(fontsize = 20, labelspacing = 0.5, loc = 4)
        plt.grid()
        if (save_on == 1):
            type_fig = "spec_slope_" 
            name = path + type_fig + settings + ".png"
            fig.savefig(name, transparent = None, dpi = 'figure',
                        format = None, metadata = None, bbox_inches = 'tight',
                        facecolor = 'auto', pad_inches = 'layout',
                        edgecolor = 'auto', backend = None)
        plt.show(fig)
        plt.close(fig)
    if (show_list[3] == 1): # show slope
        fig, ax = plt.subplots(figsize = (16, 5), dpi = 300)
        #plt.title("Slope", fontsize = 20)
        ax.plot(bar_slope_sim_data[:, 0], bar_slope_sim_data[:, 1],
                'red', ls = 'solid', alpha = 0.5, lw = 7.5,
                label = r'$\dot{\bar{a}}_{\text{Sim}}$', zorder = 2)
        ax.plot(slope_data[:, 0], slope_data[:, 1],
                color = 'blue', ls = 'solid', alpha = 1, lw = 2.5,
                label = r'$\dot{\bar{a}}_{\text{Fit}}$', zorder = 3)
        lab_fac = scientific_to_exponent(e_fac_slope, 1)
        e_lab = r'$\sigma_{\text{Fit}}$'
        if (e_fac_slope != 1):
            e_lab = e_lab + r' $\times$ ' + lab_fac
        s_y = e_fac_slope * slope_data[:, 2]
        #ax.fill_between(slope_data[:, 0], slope_data[:, 1] - s_y, slope_data[:, 1] + s_y,
        #                color = 'blue', alpha = 0.25,
        #                label = e_lab, zorder = 1)
        
        ax.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
        ax.tick_params(axis = 'x', which = 'major',
                        width = 4, length = 12,
                        direction = 'out', top = False)
        ax.tick_params(axis = 'x', which = 'minor',
                        width = 1, length = 8,
                        direction = 'out', top = False)
        ax.yaxis.set_minor_locator(mat.ticker.MultipleLocator(10))
        ax.yaxis.set_major_locator(mat.ticker.MultipleLocator(20))
        ax.tick_params(axis = 'y', which = 'major',
                       width = 4, length = 12,
                       direction = 'inout', right = False)
        ax.tick_params(axis = 'y', which = 'minor',
                       width = 1, length = 8,
                       direction = 'inout', right = False)
        ax.get_yaxis().get_major_formatter().set_useMathText(True)
        ax.set_xlabel(r'$t$ [d]', fontsize = 40)
        ax.set_ylabel(r'$\dot{a}$ [md$^{-1}$]', fontsize = 40)
        ax.set_ylim(zoom_list[0], zoom_list[1])
        ax.tick_params(axis = 'both', which = 'major', labelsize = 30)
        fig.legend(fontsize = 40, labelspacing = 0.25, ncols = 1, columnspacing = 0.5,
                   loc = 4, bbox_to_anchor = (1, 0), bbox_transform = ax.transAxes)
        ax.grid()
        if (save_on == 1):
            type_fig = "slope_" 
            name = path + type_fig + settings + ".jpg"
            fig.savefig(name, transparent = None, dpi = 'figure',
                        format = None, metadata = None, bbox_inches = 'tight',
                        facecolor = 'auto', pad_inches = 'layout',
                        edgecolor = 'auto', backend = None)
        plt.show(fig)
        plt.close(fig)
#%%
# status = 0 -> deactivated, status = 1 -> activated
status_lin = 1 # linear decay
status_pert = 1 # perturbation (to simulate cme event)
status_p_t = 0 # time dependence of main periods
status_hf = 0 # additional hf periods
status_lf = 0 # additional lf periods
status_list = [status_lin, status_pert,
               status_p_t, status_hf, status_lf]
status_str_list = ["1-lin", "2-pert", "3-p_t", "4-hf", "5-lf"]
#%%
n_fac = 10
R = 1
ω = 1e6
path_status = status_str_list[sum(status_list)-1] + "/"
path = os.getcwd() + "/Thesis/Presentation/images/Simulation/" + path_status
save_specs = [0, path]
#%%
sec_to_day = 1 / (24 * 60 * 60)
G  = 6.674e-11 / (sec_to_day**2)
Me = 5.972e24
μ = G * Me

n_days = 20
n_data = 57600
t_list = np.arange(0, n_days, n_days / n_data)

off = 6837491
lin = -61.324
p1 = 2 * np.pi * np.sqrt(off**3 / μ)
p2 = p1 / 2
p3 = p1 / 3
p4 = p1 * 1.149925

pert_list_list = [[6.5, 42, 1.5]] # t0, height_fac, width_fac
noise_fac = 1
p_list = [p2, p4, p3]
a_list = [7342, 12.21, 43.32]
ϕ_list = [0.3, 1.2, 0.73]

t_dep_fac = 1.01
hf_list = [50, -2.5, -1.8, 0.5, 5] # [n, p_low, p_upp, a_low, a_upp]
lf_list = [10, -1.15, -0.5, 1, 10]
sim_data = data_simulator(status_list, hf_list, lf_list,
                          p_list, a_list, ϕ_list,
                          noise_fac, pert_list_list, off, lin,
                          t_dep_fac, t_list)

print("len of data: ", len(sim_data))
print("number of days: ", (sim_data[-1, 0] - sim_data[0, 0]))
print("sampling in days: ", sim_data[1, 0] - sim_data[0, 0])
print("sampling in seconds: ", (sim_data[1, 0] - sim_data[0, 0]) * 24*60*60)
#%%
fit_specs = [n_days, p_list, 1, n_fac, ω] # e_fac, n_s, ω
big_fit_list = execute_fit(sim_data, fit_specs, R)
# %%
e_fac_bar = 2e2
e_fac_slope = 1
#show_list = [1, 1, 0, 0, 0, 0, 0]
#zoom_list = [8, 12, 1.45, 1.375, 0.85, 1.05]
#show_fit(sim_data, fit_specs, e_fac_bar, e_fac_slope, show_list, zoom_list,
#         big_fit_list, off, lin, pert_list_list, save_specs)
show_list = [1, 0, 0, 1]
zoom_list = [-120, -40]
show_fit(sim_data, fit_specs, e_fac_bar, e_fac_slope, show_list, zoom_list,
         big_fit_list, off, lin, pert_list_list, save_specs, R)
# %%
print(p2 * 24 * 60)
print(p3 * 24 * 60)
print(p4 * 24 * 60)
# %%
def fft(array):
    """ perform Fast Fourier Transformation
    array = [[t_1, x_1], [t_2, x_2], ...]
    gives: orig_list = [period_orig, amp_orig] (periods, amplitudes)
    """
    t_array = array[:,0]
    y_array = array[:,1]
    size = np.shape(y_array)[-1]
    
    # original data orig
    fourier_orig = np.fft.fft(y_array) / size # preparing the coefficients
    print(fourier_orig)
    new_fourier_orig = fourier_orig[1 : int(len(fourier_orig) / 2 + 1)] # only want a1, ..., an
    freq_orig = np.fft.fftfreq(size, d = t_array[1] - t_array[0]) # preparing the frequencies
    new_freq_orig = freq_orig[1 : int(len(freq_orig) / 2 + 1)] # only want frequencies to a1, ..., an
    period_orig = 1 / new_freq_orig # period
    amp_orig = 2 * np.abs(new_fourier_orig) # amplitude
    amp_tild_orig = amp_orig**2 / sum(amp_orig) # amplitude for the power spectrum
    
    orig_list = [period_orig, amp_orig]
    return(orig_list)

fft(sim_data)