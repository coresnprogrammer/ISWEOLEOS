# %% import stuff
# python3 -m pip install module
# python -m pip install numpy
# python -m pip install matplotlib
# python -m pip install astropy

# IMPORT STANDARD MODULES
import time as tt
import numpy as np
import math
import matplotlib.pyplot as plt
from astropy.time import Time
import os
import random
import matplotlib as mat
from matplotlib import colors

from functions import master_lst, master_ele, master_acc, master_man
from functions import flx_get_data_ymd, yyyymmddhhmm_to_mjd
from functions import array_modifier, array_columns, round_dot5
from functions import smoother, spectrum, fitter, integrator, fft, master_integrator
from functions import down_sample, step_data_generation, gen_a_dot_array
from functions import cl_lin, mjd_to_mmdd, xaxis_year, decr_bar_ult
from functions import smoother_give_n, low_pass_filter
plt.style.use('classic')
plt.rcParams['text.usetex'] = True
mat.rcParams['figure.facecolor'] = 'white'

print(np.random.randint(1,9))
print(u'\u03B1') # alpha

# general constants
day_sec = 24 * 60 * 60
sec_to_day = 1 / day_sec
G  = 6.674e-11 / (sec_to_day**2)
Me = 5.972e24
μ = G * Me
# %% change directory
"""
Your master directory should contain:
    - folder containing "functions.py" and "main.py"
    - folders containing data
    - folders for results (act as failsafes)
    - if necessary, folders to save plots
Change to this master directory.
In my case it is called "TestFolder" and contains:
    - "Code" (contains "functions.py" and "main.py")
    - "Data" (contains "ACC", "ELE", "LST", "MAN" and "FLXAP_P.FLX")
    - "Results"
    - "Plots"
"""
print(os.getcwd())
os.chdir('/Users/levin/Documents/Uni/Master/Semester 2/Mysterious/wien/plots/Code')
print(os.getcwd())
os.chdir('..')
#os.chdir('FurtherResearch/')
print(os.getcwd())
print(os.listdir())
# %% create master files if not already done
#""" # set "#" before to create master files
# create master LST file
osc_list = [["a", 20], ["e", 21], ["omega_small", 24],
            ["u", 17]]
osc_list_rho = [["rho", 26],["beta", 18]]
#master_lst('swarm24/LST', 1, osc_list, 'swarm24/')

# old_osc_list = ["a", "e", "omega_low", "u_sat"]
# for i in range(0, len(old_osc_list)):
#     path_old = 'se2a24/LST/year_normal_' + old_osc_list[i] + '.txt'
#     path_new = 'se2a24/' + osc_list[i][0] + '.txt'
#     data = np.loadtxt(path_old, skiprows = 1)
#     offset = data[0]
#     newdata = data[1:] + offset
#     np.savetxt(path_new, newdata)


# create master ELE and error file
#master_ele('grace18/ELE', 'grace18/')

# create master ACC file, time step should be identical to the one of LST data
#master_acc('Data/ACC', 'BIAS_SCALE', 'RSW', 1, 5, 1)

# create master MAN file
#master_man('se2a24/MAN', 'se2a24/')
#""" # set "#" before to create master files
# %% Get data location
satellite = ['grace18', 'swarm24', 'se2a24'][0]
path = satellite + '/'
#path_lst = "Data/"
#path_ele = "Data/PCA.txt"
#path_ele_masters = "Data/"
#path_ele_error = "Data/errors.txt"
#path_acc = 0#"Data/ACC/ALL/" # if no ACC available put 0
#path_man = 0#"Data/MAN/all.txt" # if no MAN available put 0
#file_type = "NL"
path_flx = path + 'FLXAP_P.FLX'
rho_beta = 0 # if rho and beta should also be analysed (only with NG possible)

oslist = os.listdir(path)
print(oslist)
if ('ACC.txt' in oslist):
    print('yes')
# %% set time interval for lsa (least squares adjustment) and int (integration)

# on 58356 is a big storm
MJD_interval_lsa = [58330, 58420] # [58340, 58401] was what I used for GRACE-FO-1 in 2018
MJD_interval_int = [58352, 58412] # [58352, 58412] was what I used for GRACE-FO-1 in 2018

#MJD_interval_lsa = [60431, 60445]
#MJD_interval_int = [60431, 60445] 

MJD_0_lsa, MJD_end_lsa = MJD_interval_lsa[0], MJD_interval_lsa[1]
MJD_0_int, MJD_end_int = MJD_interval_int[0], MJD_interval_int[1]

n_days_tot_lsa = MJD_end_lsa - MJD_0_lsa
n_days_tot_int = MJD_end_int - MJD_0_int
n_days_tot_max = max([MJD_end_lsa, MJD_end_int]) - min([MJD_0_lsa, MJD_0_int])

MJD_0_list = [MJD_0_lsa, MJD_0_int]
MJD_end_list = [MJD_end_lsa, MJD_end_int]
MJD_0 = min(MJD_0_list)
MJD_end = max(MJD_end_list)
# %% get LST data for lsa and int
osc_str_list = ["a", "e", "omega_small", "u"]

a_for_lsa_data = 0 # for lsa
osc_for_int_data_list = []
for osc_element in osc_str_list:
    osc_name = path + osc_element + '.txt' # file name
    osc_data = np.loadtxt(osc_name) # load data
    if (osc_element == "a"): # for lsa
        a_for_lsa_data = array_modifier(osc_data, MJD_0_lsa, n_days_tot_lsa) # trim data
    osc_data = array_modifier(osc_data, MJD_0_int, n_days_tot_int) # trim data
    osc_for_int_data_list.append(osc_data)
a_for_int_data = osc_for_int_data_list[0]
e_for_int_data = osc_for_int_data_list[1]
omega_small_for_int_data = osc_for_int_data_list[2]
u_for_int_data = osc_for_int_data_list[3]

rho_data, beta_data = 0, 0
if (rho_beta == 1):
    rho_name = path + 'rho.txt' # file name
    beta_name = path + 'beta.txt' # file name
    
    rho_data = np.loadtxt(rho_name) # load data
    rho_data = array_modifier(rho_data, MJD_0_lsa, n_days_tot_lsa) # trim data
    
    beta_data = np.loadtxt(beta_name) # load data
    beta_data = array_modifier(beta_data, MJD_0_lsa, n_days_tot_lsa) # trim data
#%%
plt.figure()
plt.plot(a_for_lsa_data[:, 0], a_for_lsa_data[:, 1], 'r-')
plt.show()
plt.close()
# %% get ELE data
ele_data = np.loadtxt(path + 'PCA.txt')
ele_data = array_modifier(ele_data, MJD_0_int, n_days_tot_int)

ele_R_data = array_columns(ele_data, [0, 1])
ele_S_data = array_columns(ele_data, [0, 2])
ele_W_data = array_columns(ele_data, [0, 3])
ele_A_data = array_columns(ele_data, [0, 4])
#%%
plt.figure()
plt.plot(ele_S_data[:, 0], ele_S_data[:, 1], 'r-')
plt.show()
plt.close()
# %% get ACC data and make shifted ACC data
acc_R_data, acc_S_data = 0, 0
acc_W_data, acc_A_data = 0, 0

acc_R_shifted_data = 0 
acc_S_shifted_data = 0 
acc_W_shifted_data = 0 

if ('ACC.txt' in oslist):
    acc_name = path + "ACC.txt"
    acc_data = np.loadtxt(acc_name)
    acc_data = array_modifier(acc_data, MJD_0_int, n_days_tot_int)
    
    # this is how to get mean, scale and bias
    acc_mean_data = np.loadtxt(path + "ACC_MEAN.txt")
    acc_mean_data = array_modifier(acc_mean_data, MJD_0_int, n_days_tot_int)
    
    acc_scale_data = np.loadtxt(path + "ACC_SCALE.txt")
    acc_scale_data = array_modifier(acc_scale_data, MJD_0_int, n_days_tot_int)
    
    acc_bias_data = np.loadtxt(path + "ACC_BIAS.txt")
    acc_bias_data = array_modifier(acc_bias_data, MJD_0_int, n_days_tot_int)

    acc_R_data = array_columns(acc_data, [0, 1])
    acc_S_data = array_columns(acc_data, [0, 2])
    acc_W_data = array_columns(acc_data, [0, 3])
    acc_A_data = array_columns(acc_data, [0, 4])
    
    # now make shifted ACC data
    mean_acc_R = np.mean(acc_R_data[:, 1])
    mean_acc_S = np.mean(acc_S_data[:, 1])
    mean_acc_W = np.mean(acc_W_data[:, 1])

    mean_ele_R = np.mean(ele_data[:, 1])
    mean_ele_S = np.mean(ele_data[:, 2])
    mean_ele_W = np.mean(ele_data[:, 3])

    Δ_mean_R = mean_ele_R - mean_acc_R
    Δ_mean_S = mean_ele_S - mean_acc_S
    Δ_mean_W = mean_ele_W - mean_acc_W

    acc_R_data_shifted = acc_R_data + 0 # avoid pointer problems
    acc_S_data_shifted = acc_S_data + 0 # avoid pointer problems
    acc_W_data_shifted = acc_W_data + 0 # avoid pointer problems

    acc_R_data_shifted[:, 1] += Δ_mean_R
    acc_S_data_shifted[:, 1] += Δ_mean_S
    acc_W_data_shifted[:, 1] += Δ_mean_W
#%%
def mjd_to_mmdd(t, pos): # for having mjd scale
    t_obj = Time(str(t), format = 'mjd')
    t_iso = t_obj.iso
    string = t_iso[5 : 10].replace('-', '.')
    return(string)

def xaxis_year(t_list, time_format): # get year
    t0 = t_list[0]
    t_obj = Time(str(t0), format = 'mjd')
    t_iso = t_obj.iso
    string = t_iso[0 : 4]
    string = string + " + " + time_format
    return(string)

def t_min_t_max(data_list): # get borders
    t_min_list = []
    t_max_list = []
    for i in range(0, len(data_list)):
        t_min_list.append(min(data_list[i][:, 0]))
        t_max_list.append(max(data_list[i][:, 0]))
    return(min(t_min_list), max(t_max_list))

def scientific_to_exponent(number, coma_digits): # for labels of scaled errors
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
        new_str = r'$10^{%s}$' % (exp_str)
    else:
        new_str = r'$%s \cdot 10^{%s}$' % (fac_str, exp_str)
    return(new_str)

fig, ax = plt.subplots()
ax.plot(acc_S_data[:, 0], acc_S_data[:, 1], 'g.')
ax.xaxis.set_major_formatter(mjd_to_mmdd)
ax.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
ax.tick_params(axis = 'x', which = 'major',
                width = 4, length = 12,
                direction = 'out', top = False,
                labelsize = 15)
ax.tick_params(axis = 'x', which = 'minor',
                width = 2, length = 8,
                direction = 'out', top = False,
                labelsize = 15)
ax.tick_params(axis = 'y', labelcolor = 'k', labelsize = 15)
ax.axvline(58352, color = 'red', ls = 'solid')
ax.axvline(58412, color = 'red', ls = 'solid')
plt.show(fig)
plt.close(fig)
# %% get MAN data
man_data = 0
vline_list2 = []

if ('MAN.txt' in oslist):
    man_data = np.loadtxt(path + 'MAN.txt')
    man_data = array_modifier(man_data, MJD_0, n_days_tot_max)
    vline_list2 = man_data[:, 0] # want only start times
# %% add CME arrival times
vline_list1 = []
# I just took the CME arrival times from the website https://kauai.ccmc.gsfc.nasa.gov/CMEscoreboard/
add_cme_list = ['2018-07-10T11:25Z', '2018-08-24T05:50Z',
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
for i in range(0, len(add_cme_list)):
    vline_list1.append(yyyymmddhhmm_to_mjd(add_cme_list[i]))
# %% get ap index
data_flx_apday, data_ap = flx_get_data_ymd(path_flx)
data_ap = array_modifier(data_ap, MJD_0, n_days_tot_max)
# %% generate step data of ele data with corresponding sampling

pca_step = 6 * 60 # 6 min, sampling of ele data
lst_step = (a_for_lsa_data[-1, 0] - a_for_lsa_data[0, 0]) / (len(a_for_lsa_data) - 1) * day_sec # sampling of LST data
step_fac = int(pca_step / round_dot5(lst_step))
print(lst_step)
print((a_for_lsa_data[1, 0] - a_for_lsa_data[0, 0]) * day_sec)
"""
ele_data_step = step_data_generation(ele_data, step_fac) # may take 30 s

ele_R_data = array_columns(ele_data_step, [0, 1])
ele_S_data = array_columns(ele_data_step, [0, 2])
ele_W_data = array_columns(ele_data_step, [0, 3])
ele_A_data = array_columns(ele_data_step, [0, 4])
"""
# now the sampling of osc elements, PCAs and ACC data should be the same
# %% integration (depending on amount of data may take some time)
step_high = 30 # should be an integer multiple of lst_step
sample_fac_high = int(round_dot5(step_high / lst_step))
u_for_int_data_high = down_sample(u_for_int_data, sample_fac_high)

step_low = 60 # should be an integer multiple of lst_step
sample_fac_low = int(round_dot5(step_low / lst_step))
u_for_int_data_low = down_sample(u_for_int_data, sample_fac_low)
#%%
a_int_ele_list = master_integrator(u_for_int_data_low, path, MJD_interval_int, 'PCA')
print("pca low integration complete")
a_int_ele_list_high = master_integrator(u_for_int_data_high, path, MJD_interval_int, 'PCA')
print("pca high integration complete")

if ('ACC.txt' in oslist):
    acc_name = path + "ACC.txt"
    a_int_acc_list = master_integrator(u_for_int_data_low, path, MJD_interval_int, 'ACC')
    print("acc low integration complete")
    a_int_acc_list_high = master_integrator(u_for_int_data_high, path, MJD_interval_int, 'ACC')
    print("acc high integration complete")
    
    # not yet working
    a_int_acc_shifted_list = 0#master_integrator(u_for_int_data_low, path, MJD_interval_int, 'ACC')
    a_int_acc_shifted_list_high = 0#master_integrator(u_for_int_data_high, path, MJD_interval_int, 'ACC')
# %% split into a and a_dot
a_int_ele, a_dot_ele = a_int_ele_list[0], a_int_ele_list[1]
a_int_ele_high, a_dot_ele_high = a_int_ele_list_high[0], a_int_ele_list_high[1]
if ('ACC.txt' in oslist):
    a_int_acc, a_dot_acc = a_int_acc_list[0], a_int_acc_list[1]
    a_int_acc_high, a_dot_acc_high = a_int_acc_list_high[0], a_int_acc_list_high[1]
    a_int_acc_shifted, a_dot_acc_shifted = 0,0#a_int_acc_shifted_list[0], a_int_acc_shifted_list[1]
# %% save/retrieve results (failsafe -> load results later on to not compute everything again)
folder_collection = path + 'Results/'
run = 3
add = folder_collection + 'run_' + str(run) + '-'

name_a_int_ele = add + 'a_int_ele' + '.txt'
name_a_int_ele_high = add + 'a_int_ele_high' + '.txt'
name_a_dot_ele = add + 'a_dot_ele' + '.txt'
name_a_dot_ele_high = add + 'a_dot_ele_high' + '.txt'

name_list = [name_a_int_ele, name_a_int_ele_high,
             name_a_dot_ele, name_a_dot_ele_high]

if ('ACC.txt' in oslist):
    name_a_int_acc = add + 'a_int_acc' + '.txt'
    name_a_int_acc_high = add + 'a_int_acc_high' + '.txt'
    name_a_dot_acc = add + 'a_dot_acc' + '.txt'
    name_a_dot_acc_high = add + 'a_dot_acc_high' + '.txt'
    
    #name_a_int_acc_shifted = add + 'a_int_acc_shifted' + '.txt'
    #name_a_dot_acc_shifted = add + 'a_dot_acc_shifted' + '.txt'

    name_list = name_list + [name_a_int_acc, name_a_int_acc_high,
                             name_a_dot_acc, name_a_dot_acc_high]

""" # set "#" before to save results
array_list = [a_int_ele, a_int_ele_high, a_dot_ele, a_dot_ele_high]

if ('ACC.txt' in oslist):
    array_list = array_list + [a_int_acc, a_int_acc_high,
                               a_dot_acc, a_dot_acc_high]

for i in range(0, len(name_list)):
    name_i = name_list[i]
    array_i = array_list[i]
    np.savetxt(name_i, array_i)
""" # set "#" before to save results

#""" # set "#" before to retrieve results
a_int_ele = np.loadtxt(name_a_int_ele)
a_int_ele_high = np.loadtxt(name_a_int_ele_high)
a_dot_ele = np.loadtxt(name_a_dot_ele)
a_dot_ele_high = np.loadtxt(name_a_dot_ele_high)

if ('ACC.txt' in oslist):
    a_int_acc = np.loadtxt(name_a_int_acc)
    a_int_acc_high = np.loadtxt(name_a_int_acc_high)
    a_dot_acc = np.loadtxt(name_a_dot_acc)
    a_dot_acc_high = np.loadtxt(name_a_dot_acc_high)

#a_int_acc_shifted = np.loadtxt(name_a_int_acc_shifted)
#a_dot_acc_shifted = np.loadtxt(name_a_dot_acc_shifted)
#""" # set "#" before to retrieve results
#%%
a_dot_ele_high_new = np.array([0, 0, 0])
a_int_ele_high_new = np.array([0, 0, 0])

dt_ele_high = (a_int_ele_high[-1, 0] - a_int_ele_high[0, 0]) / (len(a_int_ele_high) - 1)
dt_ele = (a_int_ele[-1, 0] - a_int_ele[0, 0]) / (len(a_int_ele) - 1)
dt_ele_fac = int(np.round(dt_ele / dt_ele_high))

for i in range(0, len(a_dot_ele_high)):
    if (i % dt_ele_fac == 0):
        a_dot_ele_high_new = np.vstack((a_dot_ele_high_new, a_dot_ele_high[i]))
        a_int_ele_high_new = np.vstack((a_int_ele_high_new, a_int_ele_high[i]))
a_dot_ele_high_new = a_dot_ele_high_new[1:]
a_int_ele_high_new = a_int_ele_high_new[1:]

if ('ACC.txt' in oslist):
    a_dot_acc_high_new = np.array([0, 0, 0])
    a_int_acc_high_new = np.array([0, 0, 0])
    
    dt_acc_high = (a_int_acc_high[-1, 0] - a_int_acc_high[0, 0]) / (len(a_int_acc_high) - 1)
    dt_acc = (a_int_acc[-1, 0] - a_int_acc[0, 0]) / (len(a_int_acc) - 1)
    dt_acc_fac = int(np.round(dt_acc / dt_acc_high))
    
    for i in range(0, len(a_dot_acc_high)):
        if (i % dt_acc_fac == 0):
            a_dot_acc_high_new = np.vstack((a_dot_acc_high_new, a_dot_acc_high[i]))
            a_int_acc_high_new = np.vstack((a_int_acc_high_new, a_int_acc_high[i]))
    a_dot_acc_high_new = a_dot_acc_high_new[1:]
    a_int_acc_high_new = a_int_acc_high_new[1:]
#%%
vals1_ele = a_dot_ele[:, 1]
vals2_ele = a_dot_ele_high_new[:, 1]

new_errors_pca = (vals2_ele - vals1_ele) / 30
print("max error h half: ", max(np.abs(new_errors_pca)))
print("max error propagation: ", max(a_dot_ele[:, 2]))
print("Order O(h^4): ", 60/(24*60*60)**4)

plt.figure()
plt.title('error h half')
plt.plot(a_dot_ele[:, 0], vals2_ele - vals1_ele, 'r-')
plt.ylim(-5e-9,5e-9)
plt.show()
plt.close()

plt.figure()
plt.title('difference adot')
plt.plot(a_dot_ele_high_new[:, 0],
         a_dot_ele_high_new[:, 1] - a_dot_ele[:, 1], 'g-')
plt.show()
plt.close()

plt.figure()
plt.title('difference a')
plt.plot(a_int_ele_high_new[:, 0],
         a_int_ele_high_new[:, 1] - a_int_ele[:, 1], 'k-')
plt.show()
plt.close()


if ('ACC.txt' in oslist):
    vals1_acc = a_dot_acc[:, 1]
    vals2_acc = a_dot_acc_high_new[:, 1]

    print(len(vals2_acc))
    print(len(vals1_acc))
    new_errors_acc = (vals2_acc - vals1_acc) / 30
    print("max error h half: ", max(np.abs(new_errors_acc)))
    print("max error propagation: ", max(a_dot_acc[:, 2]))
    print("Order O(h^4): ", 60/(24*60*60)**4)

    plt.figure()
    plt.title('error h half')
    plt.plot(a_dot_acc[:, 0], vals2_acc - vals1_acc, 'r-')
    plt.ylim(-5e-9,5e-9)
    plt.show()
    plt.close()

    plt.figure()
    plt.title('difference adot')
    plt.plot(a_dot_acc_high_new[:, 0],
            a_dot_acc_high_new[:, 1] - a_dot_acc[:, 1], 'g-')
    plt.show()
    plt.close()

    plt.figure()
    plt.title('difference a')
    plt.plot(a_int_acc_high_new[:, 0],
            a_int_acc_high_new[:, 1] - a_int_acc[:, 1], 'k-')
    plt.show()
    plt.close()
# %% smoothing of integration results
# lcm parameters [manual period (0 if not manual), ε, N, p_max, thresh_fac, limit]
a_dot_ele_lcm_list = [0, 10, 1, 0.1, 0.9, 10]
a_dot_acc_lcm_list = [0, 10, 1, 0.1, 0.9, 10]
a_dot_lcm_list = [a_dot_ele_lcm_list, a_dot_acc_lcm_list]

# search intervals [[p_min, p_max, n (how many to detect in interval [p_min, p_max])], ...]
a_dot_ele_interval_list = [[0, 0, 0]]
a_dot_acc_interval_list = [[0, 0, 0]]
a_dot_interval_list = [a_dot_ele_interval_list, a_dot_acc_interval_list]

unsmoothed_list = [a_dot_ele]
if ('ACC.txt' in oslist):
    unsmoothed_list = [a_dot_ele, a_dot_acc]

a_dot_p_o_list, a_dot_a_o_list = [], [] # periods, amplitudes
a_dot_per_list, a_dot_amp_list = [], [] # peak periods, peak amplitudes

# remark: when the time interval contains many manoeuvres
#   it may be advantageous to perform a spectral analysis
#   for a shorter time interval void of manoeuvres to find
#   the optimal smoothing period

for i in range(0, len(unsmoothed_list)):
    a_dot = array_columns(unsmoothed_list[i],[0,1])
    lcm_list = a_dot_lcm_list[i]
    interval_list = a_dot_interval_list[i]
    
    spectrum_list = spectrum(a_dot, lcm_list, interval_list)
    p_o_list, a_o_list = spectrum_list[0], spectrum_list[1]
    per_list, amp_list = spectrum_list[2], spectrum_list[3]
    
    a_dot_p_o_list.append(p_o_list)
    a_dot_a_o_list.append(a_o_list)
    a_dot_per_list.append(per_list)
    a_dot_amp_list.append(amp_list)

smoothed_list = [] # smoothed data list
per0_list_list = [] # periods of peaks used to smoothed data
amp0_list_list = [] # amplitudes of peaks used to smoothed data
Δt_n_list = [] # window width for smoothing
q = 0 # polynomial order
for i in range(0, len(unsmoothed_list)):
    full_a_dot_data = unsmoothed_list[i]
    per_list = a_dot_per_list[i]
    amp_list = a_dot_amp_list[i]
    lcm_list = a_dot_lcm_list[i]
    
    a_dot_data = array_columns(full_a_dot_data, [0, 1])
    σ_a_dot_data = array_columns(full_a_dot_data, [0, 2])
    
    a_dot_smooth_list = smoother(a_dot_data, per_list, amp_list,
                                 lcm_list, q)
    σ_a_dot_smooth_list = σ_a_dot_data
    # what I did before and was unnecessary:
    #σ_a_dot_smooth_list = smoother(σ_a_dot_data, per_list, amp_list,
    #                               lcm_list, q)
    
    a_dot_smoothed = a_dot_smooth_list[0]
    σ_a_dot_smoothed = σ_a_dot_smooth_list
    a_dot_data_smoothed = np.vstack((a_dot_smoothed.T,
                                     σ_a_dot_smoothed[:, 1])).T
    
    per0_list = a_dot_smooth_list[1]
    amp0_list = a_dot_smooth_list[2]
    Δt_n = a_dot_smooth_list[4]
    
    smoothed_list.append(a_dot_data_smoothed)
    per0_list_list.append(per0_list)
    amp0_list_list.append(amp0_list)
    Δt_n_list.append(Δt_n)

# plot to see how it went
for i in range(0, len(unsmoothed_list)):
    p_list = a_dot_p_o_list[i] # periods of unsmoothed
    a_list = a_dot_a_o_list[i] # amplitudes of unsmoothed
    per_list = a_dot_per_list[i] # periods of detected peaks of unsmoothed
    amp_list = a_dot_amp_list[i] # amplitudes of detected peaks of unsmoothed
    per0_list = per0_list_list[i] # periods of detected peaks of unsmoothed used to evaluate smoothing period
    amp0_list = amp0_list_list[i] # amplitudes of detected peaks of unsmoothed used to evaluate smoothing period
    Δt_n = Δt_n_list[i] # smoothing period
    
    smoothed_orig_list = fft(smoothed_list[i])
    p_smoothed_list = smoothed_orig_list[0] # periods of smoothed
    a_smoothed_list = smoothed_orig_list[1] / 10 # amplitudes of smoothed (scaled)
    
    col_list = ['crimson', 'tomato', 'dodgerblue', 'deepskyblue', 'forestgreen', 'lime']
    col_1 = col_list[2 * i + 0]
    col_2 = col_list[2 * i + 1]
    label_list = ['PCA', 'ACC', 'ACC,shifted']
    
    fig = plt.figure(figsize = (12, 5), dpi = 300)
    
    plt.loglog(p_list, a_list,
               color = col_1, ls = 'solid', lw = 1,
               label = label_list[i] + ' unsmoothed')
    plt.loglog(p_smoothed_list, a_smoothed_list,
               color = col_2, ls = 'solid', lw = 1,
               label = label_list[i] + ' smoothed / 10')
    
    plt.scatter(per_list, amp_list,
                color = 'goldenrod', s = 150, lw = 1.5,
                marker = 'o', fc = 'None',
                label = 'detected peaks')
    plt.scatter(per_list, amp_list,
                color = 'violet', s = 25, lw = 1.5,
                marker = 'D', fc = 'None',
                label = 'peaks used for smoothing')
    plt.axvline(Δt_n, color = 'purple', ls = (0, (2, 5)), lw = 1.5,
                label = r'$\Delta t$ (smoothing period)')

    plt.xlabel(r'period [d]', fontsize = 15)
    plt.ylabel(r'amplitude [m]', fontsize = 15)
    plt.legend(fontsize = 15, loc = 0,
               labelspacing = 0.5, ncols = 1)
    plt.grid()
    plt.show()
    plt.close()
# %% analyse spectrum of osculating semi-major axis to get peak periods
# lcm parameters [manual period (0 if not manual), ε, N, p_max, thresh_fac, limit]
lcm_list_a_for_lsa = [0, 50, 1, 0.09, 0.95, 10]
# search intervals [[p_min, p_max, n (how many to detect in interval [p_min, p_max])], ...]
interval_list_a_for_lsa = [[5,6,1]] # to disable: [[0,0]]
# remark: for GRACE-FO-1 in 2018 a period between 5d and 6d
#   is well visible if n_days_tot_lsa = 61d


#[58340, 58401]
a_for_lsa_data_spectrum = array_modifier(a_for_lsa_data, 58340, 61)
#a_for_lsa_data_spectrum = a_for_lsa_data

a_for_lsa_spectrum_list = spectrum(a_for_lsa_data_spectrum, lcm_list_a_for_lsa, interval_list_a_for_lsa)
a_for_lsa_p_o, a_for_lsa_a_o = a_for_lsa_spectrum_list[0], a_for_lsa_spectrum_list[1]
a_for_lsa_per, a_for_lsa_amp = a_for_lsa_spectrum_list[2], a_for_lsa_spectrum_list[3]
print(a_for_lsa_per)
# plot to see spectrum and detected peaks
fig = plt.figure(figsize = (12, 5), dpi = 300)
plt.loglog(a_for_lsa_p_o, a_for_lsa_a_o,
           color = 'red', ls = 'solid', lw = 1,
           label = r'$a_{osculating}$')
plt.scatter(a_for_lsa_per, a_for_lsa_amp,
            color = 'blue', s = 50, lw = 1,
            marker = 'o', fc = 'None',
            label = 'detected peaks')
for i in range(0, len(interval_list_a_for_lsa)):
    interval = interval_list_a_for_lsa[i]
    if (interval[0] != interval[1]):
        lab = None
        if (i == 0):
            lab = 'search intervals'
        plt.axvline(interval[0], color = 'green',
                    ls = (0, (2, 5)), lw = 1.5,
                    label = lab)
        plt.axvline(interval[1], color = 'green',
                    ls = (0, (2, 5)), lw = 1.5)
plt.xlabel(r'period [d]', fontsize = 15)
plt.ylabel(r'amplitude [m]', fontsize = 15)
plt.legend(fontsize = 15, loc = 0,
           labelspacing = 0.5, ncols = 1)
plt.grid()
plt.show()
plt.close()
#%%
def mjd_to_mmdd_spec(t, pos): # for having mjd scale
    t_obj = Time(str(t), format = 'mjd')
    t_iso = t_obj.iso
    string = t_iso[11 : 13].replace('-', '.')
    return(string)

def xaxis_year_spec(t_list, time_format): # get year
    t0 = t_list[0]
    t_obj = Time(str(t0), format = 'mjd')
    t_iso = t_obj.iso
    string = t_iso[0 : 10]
    string = string + " + " + time_format
    return(string)


fig, ax = plt.subplots(figsize = (10,5), dpi = 300)
ax.plot(a_for_lsa_data_spectrum[:, 0], a_for_lsa_data_spectrum[:, 1]/1000,
        color = 'chocolate', lw = 1.5)
ax.xaxis.set_major_formatter(mjd_to_mmdd_spec)
ax.xaxis.set_major_locator(mat.ticker.MultipleLocator(1/12))
ax.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1/24))
ax.tick_params(axis = 'x', which = 'major',
                width = 4, length = 12,
                direction = 'out', top = False,
                labelsize = 20)
ax.tick_params(axis = 'x', which = 'minor',
                width = 1, length = 8,
                direction = 'out', top = False,
                labelsize = 20)
ax.set_xlabel(xaxis_year_spec([60431], 'hh'), fontsize = 25)

ax.tick_params(axis = 'y', which = 'major',
                width = 2, length = 8, right = False,
                direction = 'inout', labelsize = 20)
ax.set_ylabel(r'$a$ [km]', fontsize = 25)
ax.set_xlim(60431, 60431.5)
ax.set_ylim(6.8e3+35, 6.8e3+57)
ax.grid()
#ax.axvline(58352, color = 'red', ls = 'solid')
#ax.axvline(58412, color = 'red', ls = 'solid')
plt.show(fig)
plt.close(fig)


a_for_lsa_data_spectrum_new = array_modifier(a_for_lsa_data_spectrum, 60431, 10)
a_for_lsa_spectrum_list_new = spectrum(a_for_lsa_data_spectrum_new, lcm_list_a_for_lsa, interval_list_a_for_lsa)
a_for_lsa_p_o_new, a_for_lsa_a_o_new = a_for_lsa_spectrum_list_new[0], a_for_lsa_spectrum_list_new[1]
a_for_lsa_per_new, a_for_lsa_amp_new = a_for_lsa_spectrum_list_new[2], a_for_lsa_spectrum_list_new[3]

fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.loglog(a_for_lsa_p_o_new * 24 * 60, a_for_lsa_a_o_new / 1000,
           color = 'chocolate', ls = 'solid', lw = 1,
           label = r'$a_{osculating}$')
#plt.scatter(a_for_lsa_per_new*24*60, a_for_lsa_amp_new/1000,
#            color = 'blue', s = 50, lw = 1,
#            marker = 'o', fc = 'None',
#            label = 'detected peaks')
#ax.xaxis.set_major_formatter(mjd_to_mmdd_spec)
#ax.xaxis.set_major_locator(mat.ticker.MultipleLocator(1/12))
#ax.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1/24))
ax.tick_params(axis = 'x', which = 'major',
                width = 4, length = 12,
                direction = 'out', top = False,
                labelsize = 20)
ax.tick_params(axis = 'x', which = 'minor',
                width = 1, length = 8,
                direction = 'out', top = False,
                labelsize = 20)

ax.tick_params(axis = 'y', which = 'major',
                width = 2, length = 8, right = False,
                direction = 'inout', labelsize = 20)

plt.tick_params(axis = 'x', which = 'major',
                labelsize = 20)
plt.tick_params(axis = 'y', which = 'major',
                labelsize = 20)
plt.xlim(1, 2e4)
plt.ylim(1e-4,20)
plt.xlabel(r'period [min]', fontsize = 25)
plt.ylabel(r'amplitude [km]', fontsize = 25)

#plt.legend(fontsize = 15, loc = 0,
#           labelspacing = 0.5, ncols = 1)
plt.grid()
plt.show()
plt.close()







#%%
def splitter(n_s, liste):
    return(np.split(liste,n_s))

def get_tau_list(n_s, tlist):
    start, end = tlist[0], tlist[-1]
    return(np.linspace(start, end, n_s + 1))

def get_deltatau(n_s, tlist):
    start, end = tlist[0], tlist[-1]
    return((end - start) / n_s)

def B_n_theta(n_s, n, theta, taulist, deltatau, tlist, periodlist, R):
    subtlist = splitter(n_s, tlist)[n-1]
    tau_n = taulist[n]
    tau_n_1 = taulist[n-1]
    if (theta == 0): # abar
        leftcolumn = (tau_n - subtlist) / deltatau
        rightcolumn = (subtlist - tau_n_1) / deltatau
    elif (theta >=1 and theta <= R):
        r = theta
        p_r = periodlist[r - 1]
        leftcolumn = (tau_n - subtlist) / deltatau * np.sin(2 * np.pi * subtlist / p_r)
        rightcolumn = (subtlist - tau_n_1) / deltatau * np.sin(2 * np.pi * subtlist / p_r)
    elif (theta >= R + 1 and theta <= 2 * R):
        r = theta
        p_r = periodlist[r - 1 - R]
        leftcolumn = (tau_n - subtlist) / deltatau * np.cos(2 * np.pi * subtlist / p_r)
        rightcolumn = (subtlist - tau_n_1) / deltatau * np.cos(2 * np.pi * subtlist / p_r)
    else:
        print("ERROR B_n_theta!!!")
        return(0)
    matrix = np.vstack((leftcolumn, rightcolumn)).T
    return(matrix)

def H_n_theta1_theta2(n_s, n, theta1, theta2, taulist, deltatau, tlist, periodlist, R):
    B_n_theta1_mat = B_n_theta(n_s, n, theta1, taulist, deltatau, tlist, periodlist, R)
    B_n_theta2_mat = B_n_theta(n_s, n, theta2, taulist, deltatau, tlist, periodlist, R)
    
    toprow = np.zeros((n - 1, n_s + 1))
    bottomrow = np.zeros((n_s - n, n_s + 1))
    
    middlerowleft = np.zeros((2, n - 1))
    middlerowmiddle = B_n_theta1_mat.T @ B_n_theta2_mat
    middlerowright = np.zeros((2, n_s - n))
    middlerow = np.hstack((middlerowleft, middlerowmiddle, middlerowright))
    
    matrix = np.vstack((toprow, middlerow, bottomrow))
    return(matrix)

def H_n(n_s, n, taulist, deltatau, tlist, periodlist, R):
    matrix = np.zeros((0, (n_s + 1) * (2 * R + 1)))
    for theta1 in range(0, 2 * R + 1):
        rowtheta1 = np.zeros((n_s + 1, 0))
        for theta2 in range(0, 2 * R + 1):
            H_n_theta1_theta2_mat = H_n_theta1_theta2(n_s, n, theta1, theta2, taulist, deltatau, tlist, periodlist, R)
            rowtheta1 = np.hstack((rowtheta1, H_n_theta1_theta2_mat))
        matrix = np.vstack((matrix, rowtheta1))
    return(matrix)

def N(n_s, taulist, deltatau, tlist, periodlist, R):
    matrix = np.zeros(((n_s + 1) * (2 * R + 1), (n_s + 1) * (2 * R + 1)))
    for n in range(1, n_s + 1):
        H_n_matrix = H_n(n_s, n, taulist, deltatau, tlist, periodlist, R)
        matrix = matrix + H_n_matrix
    return(matrix)

def D_n_theta(n_s, n, theta, taulist, deltatau, tlist, periodlist, R, kappa):
    U_n_theta_matrix = np.zeros((kappa, n - 1))
    V_n_theta_matrix = np.zeros((kappa, n_s - n))
    B_n_theta_matrix = B_n_theta(n_s, n, theta, taulist, deltatau, tlist, periodlist, R)
    matrix = np.hstack((U_n_theta_matrix, B_n_theta_matrix, V_n_theta_matrix))
    return(matrix)

def D_n(n_s, n, taulist, deltatau, tlist, periodlist, R, kappa):
    matrix = np.zeros((kappa, 0))
    for theta in range(0, 2 * R + 1):
        D_n_theta_matrix = D_n_theta(n_s, n, theta, taulist, deltatau, tlist, periodlist, R, kappa)
        matrix = np.hstack((matrix, D_n_theta_matrix))
    return(matrix)

def D_full(n_s, taulist, deltatau, tlist, periodlist, R, kappa):
    matrix = np.zeros((1, (n_s + 1) * (2 * R + 1)))
    for n in range(1, n_s + 1):
        D_n_row = D_n(n_s, n, taulist, deltatau, tlist, periodlist, R, kappa)
        matrix = np.vstack((matrix, D_n_row))
    matrix = matrix[1:]
    return(matrix)

def other_b(n_s, taulist, deltatau, tlist, periodlist, R, kappa, alist):
    D_matrix = D_full(n_s, taulist, deltatau, tlist, periodlist, R, kappa)
    l_vec = np.array([alist]).T
    return(D_matrix.T @ l_vec)

def b_vec(n_s, taulist, deltatau, tlist, periodlist, R, kappa, alist):
    l_vec_list = splitter(n_s, alist)
    matrix = np.zeros(((n_s + 1) * (2 * R + 1), 1))
    for n in range(1, n_s + 1):
        D_n_matrix = D_n(n_s, n, taulist, deltatau, tlist, periodlist, R, kappa)
        l_vec_n = np.array([l_vec_list[n-1]]).T
        #print("--------------")
        #print(np.shape(D_n_matrix.T))
        matrix = matrix + (D_n_matrix.T @ l_vec_n)
    #print(matrix)
    return(matrix)

def C_theta(n_s):
    matrix = np.zeros((n_s - 1, n_s + 1))
    for i in range(0, n_s - 1):
        matrix[i, i] = 1
        matrix[i, i + 1] = -2
        matrix[i, i + 2] = 1
    return(matrix)

def C(n_s, R):
    matrix = np.zeros((0, (n_s + 1) * (2 * R + 1)))
    for theta in range(0, 2 * R + 1):
        leftzeros = np.zeros((n_s - 1, theta * (n_s + 1)))
        rightzeros = np.zeros((n_s - 1, (2 * R - theta) * (n_s + 1)))
        C_theta_matrix = C_theta(n_s)
        rowtheta = np.hstack((leftzeros, C_theta_matrix, rightzeros))
        matrix = np.vstack((matrix, rowtheta))
    return(matrix)

def lsa(n_s, taulist, deltatau, tlist, periodlist, R, kappa, alist, psi):
    N_matrix = N(n_s, taulist, deltatau, tlist, periodlist, R)
    C_matrix = C(n_s, R)
    N_eff = N_matrix + psi * C_matrix.T @ C_matrix
    
    N_inv = np.linalg.inv(N_eff)
    b_vec_vector = b_vec(n_s, taulist, deltatau, tlist, periodlist, R, kappa, alist)
    x_vec = N_inv @ b_vec_vector
    return(x_vec, N_inv)

def get_abar_list(n_s, taulist, deltatau, tlist, xvec):
    xlist = np.ravel(xvec)
    tsublistlist = splitter(n_s, tlist)
    abarlist = np.array([])
    for n in range(1, n_s+1):
        tsublist = tsublistlist[n - 1]
        tau_n = taulist[n]
        tau_n_1 = taulist[n - 1]
        x_n_1 = xlist[n - 1]
        x_n = xlist[n]
        abar_n = ((tau_n - tsublist) * x_n_1 + (tsublist - tau_n_1) * x_n) / deltatau
        abarlist = np.append(abarlist, abar_n)
    abardata = np.vstack((tlist, abarlist)).T
    return(abardata)

def pwlf(deltatau, tau_n_1, tau_n, x_n_1, x_n, tlist):
    return(((tau_n - tlist) * x_n_1 + (tlist - tau_n_1) * x_n) / deltatau)

def get_a_list(n_s, taulist, deltatau, tlist, periodlist, R, xvec):
    xlist = np.ravel(xvec)
    t_n_list_list = splitter(n_s, tlist)
    alist = np.array([])
    for n in range(1, n_s + 1):
        t_n_list = t_n_list_list[n - 1]
        tau_n = taulist[n]
        tau_n_1 = taulist[n - 1]
        
        a_bar_n_1 = xlist[n - 1]
        a_bar_n = xlist[n]
        a = pwlf(deltatau, tau_n_1, tau_n, a_bar_n_1, a_bar_n, t_n_list)
        
        for r in range(1, R + 1):
            p_r = periodlist[r - 1]
            omega_r = 2 * np.pi / p_r
            ot = omega_r * t_n_list
            
            mu_r_n_1 = xlist[(n_s + 1) * (r) + n - 1]
            mu_r_n = xlist[(n_s + 1) * (r) + n]
            mu_r = pwlf(deltatau, tau_n_1, tau_n, mu_r_n_1, mu_r_n, t_n_list)
            a = a + mu_r * np.sin(ot)
            
            eta_r_n_1 = xlist[(n_s + 1) * (R+r) + n - 1]
            eta_r_n = xlist[(n_s + 1) * (R+r) + n]
            eta_r = pwlf(deltatau, tau_n_1, tau_n, eta_r_n_1, eta_r_n, t_n_list)
            a = a + eta_r * np.cos(ot)
        alist = np.append(alist, a)
    a_data = np.vstack((tlist, alist)).T
    return(a_data)

def get_a_dot(n_s, taulist, deltatau, xvec, Kxx):
    #lowtlist = np.linspace(tlist)
    xlist = np.ravel(xvec)
    #tsublistlist = splitter(n_s, tlist)
    
    σ2_list = np.diag(Kxx)
    
    a_dot_list = np.array([])
    t_dot_list = np.array([])
    s_a_dot_list = np.array([])
    slopefac = 10
    for n in range(1, n_s+1):
        tsublist = np.linspace(taulist[n - 1], taulist[n], slopefac)
        x_n_1 = xlist[n - 1]
        x_n = xlist[n]
        σ2_n_1 = σ2_list[n - 1]
        σ2_n = σ2_list[n]
        
        a_dot_n = (x_n - x_n_1) / deltatau * np.ones(len(tsublist))
        s_a_dot_n = math.sqrt((σ2_n + σ2_n_1) / (deltatau * deltatau)) * np.ones(len(tsublist))
        
        a_dot_list = np.append(a_dot_list, a_dot_n)
        t_dot_list = np.append(t_dot_list, tsublist)
        s_a_dot_list = np.append(s_a_dot_list, s_a_dot_n)
    a_dot_data = np.vstack((t_dot_list, a_dot_list, s_a_dot_list)).T
    return(a_dot_data)

def get_muretar_list(n_s, taulist, deltatau, tlist, R, xvec):
    xlist = np.ravel(xvec)
    t_n_list_list = splitter(n_s, tlist)
    mu_list = []
    eta_list = []
    for r in range(1, R + 1):
        mur_list = np.array([])
        etar_list = np.array([])
        for n in range(1, n_s + 1):
            t_n_list = t_n_list_list[n - 1]
            tau_n = taulist[n]
            tau_n_1 = taulist[n - 1]
            
            mu_r_n_1 = xlist[(n_s + 1) * (r) + n - 1]
            mu_r_n = xlist[(n_s + 1) * (r) + n]
            mu_r = pwlf(deltatau, tau_n_1, tau_n, mu_r_n_1, mu_r_n, t_n_list)
            mur_list = np.append(mur_list, mu_r)
            
            eta_r_n_1 = xlist[(n_s + 1) * (R+r) + n - 1]
            eta_r_n = xlist[(n_s + 1) * (R+r) + n]
            eta_r = pwlf(deltatau, tau_n_1, tau_n, eta_r_n_1, eta_r_n, t_n_list)
            etar_list = np.append(etar_list, eta_r)
        mu_list.append(mur_list)
        eta_list.append(etar_list)
    return(mu_list, eta_list)

def x_theta_n(n_s, n, theta, R, xvec):
    abc = (n_s + 1) * (theta)
    return(xvec[abc + n - 1 : abc + n + 1])

def J_n_theta(n_s, n, theta, taulist, deltatau, tlist, periodlist, R, xvec):
    B_n_theta_matrix = B_n_theta(n_s, n, theta, taulist, deltatau, tlist, periodlist, R)
    x_theta_n_vec = x_theta_n(n_s, n, theta, R, xvec)
    return(B_n_theta_matrix @ x_theta_n_vec)

def J_theta(n_s, theta, taulist, deltatau, tlist, periodlist, R, xvec):
    vec = np.array([[0]])
    for n in range(1, n_s + 1):
        J_n_theta_vec = J_n_theta(n_s, n, theta, taulist, deltatau, tlist, periodlist, R, xvec)
        vec = np.vstack((vec, J_n_theta_vec))
    vec = vec[1:]
    return(vec)

def Dx(n_s, taulist, deltatau, tlist, periodlist, R, xvec):
    vec = np.zeros((len(tlist), 1))
    for theta in range(0, 2 * R + 1):
        vec = vec + J_theta(n_s, theta, taulist, deltatau, tlist, periodlist, R, xvec)
    return(vec)

def v(n_s, taulist, deltatau, tlist, periodlist, R, xvec, alist):
    l_vec = np.array([alist]).T
    Dx_vec = Dx(n_s, taulist, deltatau, tlist, periodlist, R, xvec)
    return(Dx_vec - l_vec)

def get_m0(n_s, taulist, deltatau, tlist, periodlist, R, xvec, alist):
    v_vec = v(n_s, taulist, deltatau, tlist, periodlist, R, xvec, alist)
    oben = (v_vec.T @ v_vec)[0][0]
    unten = len(tlist) - (n_s + 1) * (2 * R + 1)
    return(np.sqrt(oben / unten))

def get_Kxx(N_inv, m0_el):
    return(m0_el**2 * N_inv)

def fitter_new(data, periodlist, n_fac, R, psi):
    ta = tt.time()
    tlist = data[:, 0]
    alist = data[:, 1]
    
    n_days = int(np.round(tlist[-1] - tlist[0]))
    print("n_days = ", n_days)
    
    ns = int(np.round(n_fac * n_days))
    print("ns = ", ns)
    
    taulist = get_tau_list(ns, tlist)
    
    deltatau = get_deltatau(ns, tlist)
    print("deltatau = ", deltatau)
    
    R = len(periodlist)
    
    kappa = int(np.round(len(tlist) / ns))
    print("kappa = ", kappa)
    
    print("executing lsa ...")
    t1 = tt.time()
    x_vec, N_inv = lsa(ns, taulist, deltatau, tlist, periodlist, R, kappa, alist, psi)
    t2 = tt.time()
    print("lsa finished, duration: %.3f" % (t2 - t1))
    
    m0 = get_m0(ns, taulist, deltatau, tlist, periodlist, R, x_vec, alist)
    Kxx = get_Kxx(N_inv, m0)
    
    # print("reproduced data ...")
    # t1 = tt.time()
    # fitdata = get_a_list(ns, taulist, deltatau, tlist, periodlist, R, x_vec)
    # t2 = tt.time()
    # print("reproduced data finished, duration: %.3f" % (t2 - t1))
    
    # print("abar data ...")
    # t1 = tt.time()
    # abardata = get_abar_list(ns, taulist, deltatau, tlist, x_vec)
    # t2 = tt.time()
    # print("abar data finished, duration: %.3f" % (t2 - t1))
    
    # print("get slope data ...")
    # t1 = tt.time()
    # slopedata = get_a_dot(ns, taulist, deltatau, periodlist, R, x_vec)
    # t2 = tt.time()
    # print("slope data finished, duration: %.3f" % (t2 - t1))
    
    tb = tt.time()
    print("fit executed, duration: %.3f" % (tb - ta))
    
    fit_list = [ns, taulist, deltatau, R, kappa, x_vec, m0, Kxx]
    
    return(fit_list)

# %% down sample lst data for lsa
lsa_step = 30 # should be an integer multiple of lst_step
sample_fac = int(round_dot5(lsa_step / lst_step))
a_data_lst_old = a_for_lsa_data
a_for_lsa_data_sampled = down_sample(a_for_lsa_data, sample_fac)
print((a_for_lsa_data_sampled[1,0] - a_for_lsa_data_sampled[0,0])*24*60*60)
# %% choose configurations for fit
# [n_fac, R, ψ, constr (0 -> only constrain a_bar, 1 -> constrain also amplitudes)]
configuration_list = [#[10, 1, 1e3, 1],
                      #[20, 1, 1e3, 2],
                      [10, 2, 1e4, 1],
                      #[20, 1, 1e4, 2],
                      [10, 2, 1e5, 1],
                      #[20, 1, 1e5, 2]
                      ]
el_data_list = [a_for_lsa_data_sampled]
print(len(configuration_list))

ns_list = [] # n_s for each configuration
taulist_list = [] # taulist for each configuration
deltatau_list = [] # deltatau for each configuration
R_list = [] # R for each configuration
kappa_list = [] # kappa for each configuration
xvec_list = [] # x_vec for each configuration
m0_list = [] # m0 for each configuration
Kxx_list = [] # Kxx for each configuration
# fitdata_list = [] # reproduced data for each configuration
# abardata_list = [] # clean data for each configuration
# slopedata_list = [] # slope data for each configuration
# %% execute fit with given configurations (depending on configurations may take some time)

# the fit only works if len(data) is an integer multiple of n_s
# in other words: each subinterval contains the same number of data points
# and the number of data points is an integer
# reason: code was easier to implement this way

for i in range(0, len(configuration_list)):
    configurations = configuration_list[i]
    n_fac = configurations[0] # amount of subintervals per day
    R = configurations[1] # number of periods to be modelled
    ψ = configurations[2] # constraining factor
    fperio = configurations[3] # constr
    
    el_data = a_for_lsa_data_sampled
    el_per = a_for_lsa_per
    
    # fit
    t1 = tt.time() # measure time for fun
    fit_list = fitter_new(el_data, el_per[:R], n_fac, R, ψ)
    t2 = tt.time()
    print("time: %.3f" % (t2 - t1))
    
    ns = fit_list[0]
    taulist = fit_list[1]
    deltatau = fit_list[2]
    R = fit_list[3]
    kappa = fit_list[4]
    xvec = fit_list[5]
    m0 = fit_list[6]
    Kxx = fit_list[7]
    # fitdata = fit_list[8]
    # abardata = fit_list[9]
    # slopedata = fit_list[10]
    
    ns_list.append(ns)
    taulist_list.append(taulist)
    deltatau_list.append(deltatau)
    R_list.append(R)
    kappa_list.append(kappa)
    xvec_list.append(xvec)
    m0_list.append(m0)
    Kxx_list.append(Kxx)
    # fitdata_list.append(fitdata)
    # abardata_list.append(abardata)
    # slopedata_list.append(slopedata)
print(len(ns_list))
# %% save/retrieve results (failsafe -> load results later on to not compute everything again)
folder_collection = path + 'Results/'
run = 3
add = folder_collection + 'run_' + str(run) + '-'

""" # set "#" before to save results
for i in range(0, len(configuration_list)):
    configurations = configuration_list[i]
    n_fac = configurations[0]
    R = configurations[1]
    ψ = configurations[2]
    constr = configurations[3]
    config = 'n_%d-R_%d-p_%.1E' % (n_fac, R, ψ)
    
    ns_name = add + 'ns-' + config + '.txt'
    taulist_name = add + 'taulist-' + config + '.txt'
    deltatau_name = add + 'deltatau-' + config + '.txt'
    R_name = add + 'R-' + config + '.txt'
    kappa_name = add + 'kappa-' + config + '.txt'
    xvec_name = add + 'xvec' + config + '.txt'
    m0_name = add + 'm0-' + config + '.txt'
    Kxx_name = add + 'Kxx-' + config + '.txt'
    
    ns = ns_list[i]
    taulist = taulist_list[i]
    deltatau = deltatau_list[i]
    R = R_list[i]
    kappa = kappa_list[i]
    xvec = xvec_list[i]
    m0 = m0_list[i]
    Kxx = Kxx_list[i]
    
    np.savetxt(ns_name, np.array([ns]))
    np.savetxt(taulist_name, taulist)
    np.savetxt(deltatau_name, np.array([deltatau]))
    np.savetxt(R_name, np.array([R]))
    np.savetxt(kappa_name, np.array([kappa]))
    np.savetxt(xvec_name, xvec)
    np.savetxt(m0_name, np.array([m0]))
    np.savetxt(Kxx_name, Kxx)
""" # set "#" before to save results

#""" # set "#" before to retrieve results
for i in range(0, len(configuration_list)):
    configurations = configuration_list[i]
    n_fac = configurations[0]
    R = configurations[1]
    ψ = configurations[2]
    constr = configurations[3]
    config = 'n_%d-R_%d-p_%.1E' % (n_fac, R, ψ)
    
    ns_name = add + 'ns-' + config + '.txt'
    taulist_name = add + 'taulist-' + config + '.txt'
    deltatau_name = add + 'deltatau-' + config + '.txt'
    R_name = add + 'R-' + config + '.txt'
    kappa_name = add + 'kappa-' + config + '.txt'
    xvec_name = add + 'xvec' + config + '.txt'
    m0_name = add + 'm0-' + config + '.txt'
    Kxx_name = add + 'Kxx-' + config + '.txt'
    
    ns = int(np.loadtxt(ns_name))
    taulist = np.loadtxt(taulist_name)
    deltatau = np.loadtxt(deltatau_name)
    R = int(np.loadtxt(R_name))
    kappa = int(np.loadtxt(kappa_name))
    xvec = np.loadtxt(xvec_name)
    m0 = np.loadtxt(m0_name)
    Kxx = np.loadtxt(Kxx_name)
    
    ns_list.append(ns)
    taulist_list.append(taulist)
    deltatau_list.append(deltatau)
    R_list.append(R)
    kappa_list.append(kappa)
    xvec_list.append(xvec)
    m0_list.append(m0)
    Kxx_list.append(Kxx)
#""" # set "#" before to retrieve results
# %%
len_filter = 0
filterlist = [1,2]
filtered_adot = []
"""
filterlist = 5*np.arange(1,6)
print(filterlist)
len_filter = len(filterlist)
new_a_data = [a_data_lst_old, a_for_lsa_data][1]
filterperiod = el_per[-1] * 1

def superfilter(data, n, iterations):
    filtered_data = data
    for i in range(0, iterations):
        filtered_data = low_pass_filter(filtered_data, n, 2)
    return(filtered_data)

n_Δt, Δt_n, Δt_dt = smoother_give_n(new_a_data, [filterperiod], [1], lcm_list_a_for_lsa, 0)
print("filterperiod: %.6f, Δt_n: %.6f" % (filterperiod * 24 * 60, Δt_n * 24 * 60))
print(n_Δt, Δt_dt)

filtered_a = []
filtered_adot = []
for n_c in range(0, len(filterlist)):
    n = filterlist[n_c]
    n0 = 0
    if (n_c > 0) and (n > filterlist[n_c - 1]):
        data = filtered_a[-1]
        n0 = filterlist[n_c - 1]
        n = n - n0
    else:
        data = new_a_data
             
    filtered_a_n = superfilter(data, n_Δt, n)
    
    dt = (filtered_a_n[-1,0] - filtered_a_n[0,0]) / (len(filtered_a_n) - 1)
    
    adot_t = (filtered_a_n[1:, 0] + filtered_a_n[:-1, 0]) / 2
    adot_a = np.array([])
    adot_error = np.zeros(len(adot_t))
    for k in range(0, len(filtered_a_n) - 1):
        a_k = filtered_a_n[k, 1]
        a_kp1 = filtered_a_n[k + 1, 1]
        adot_k = (a_kp1 - a_k) / dt
        adot_a = np.append(adot_a, adot_k)
    adot_n = np.vstack((adot_t, adot_a, adot_error)).T
    
    filtered_a.append(filtered_a_n)
    filtered_adot.append(adot_n)
    print("done: ", n + n0)
"""
# %% prepare for plotting
slope_data_list = [] # data list
slope_specs_list = [] # plotting specifications

special_color_list = ['crimson', 'tomato', 'dodgerblue', 'deepskyblue',
                      'forestgreen', 'lime']
special_color_list = ['red', 'blue', 'green']
#color_list = ['maroon', 'tomato', 'mediumblue', 'cyan']
#color_list = 2 * ['darkorchid', 'fuchsia']
#color_list = 3 * ['darkorchid', 'fuchsia'] + cl_lin(np.arange(len_filter), mat.colormaps['gist_rainbow'])
#color_list = special_color_list + color_list

#grace18
col_list_fit = ['deepskyblue', 'forestgreen']
col_list_int = ['red', 'blue', 'tomato', 'deepskyblue']
col_list_filter = []#cl_lin(np.arange(len_filter), mat.colormaps['gist_rainbow'])

#swma24
# col_list_fit = ['red', 'blue', 'green']
# col_list_int = ['fuchsia', 'darkorchid']
# col_list_filter = []#cl_lin(np.arange(len_filter), mat.colormaps['gist_rainbow'])

color_list = col_list_fit + col_list_int + col_list_filter

#grace18
len_fit = len(configuration_list)
len_int = len(unsmoothed_list)
α_list = (len_fit + 2 * len_int + len_filter) * [1]
lab_list = [2, 3] + [r'PCA', r'ACC'] + [r'PCA', r'ACC'] + list(filterlist)

#swma24
# len_fit = len(configuration_list)
# len_int = len(unsmoothed_list)
# α_list = (len_fit + 2 * len_int + len_filter) * [1]
# lab_list = [1, 2, 3] + [r'PCA', r'PCA'] + list(filterlist)

for i in range(0, len_fit): # fit
    configurations = configuration_list[i]
    n_fac = configurations[0]
    R = configurations[1]
    ψ = configurations[2]
    constr = configurations[3]
    
    ns = ns_list[i]
    taulist = taulist_list[i]
    deltatau = deltatau_list[i]
    R = R_list[i]
    kappa = kappa_list[i]
    xvec = xvec_list[i]
    m0 = m0_list[i]
    Kxx = Kxx_list[i]
    
    slope_data = get_a_dot(ns, taulist, deltatau, xvec, Kxx) # generate step data
    
    add = lab_list[i]
    lab = r'$\dot{\bar{a}}_{\psi_{%d}}$' % add
    e_lab = r'$\sigma_{\psi_{%d}}$' % add
    lcol = color_list[i]
    ecol = color_list[i]
    slope_specs = [2, lcol, ecol, α_list[i], 0.25, lab, e_lab]
    
    slope_data_list.append(slope_data)
    slope_specs_list.append(slope_specs)

for i in range(len_fit, len_fit + len_int): # unsmoothed int
    add = lab_list[i]
    lab = r'$\dot{a}_{%s}$' % add
    e_lab = r'$\sigma_{%s}$' % add
    lcol = color_list[i]
    ecol = color_list[i]
    slope_specs = [2, lcol, ecol, α_list[i], 0.25, lab, e_lab]
    
    slope_data_list.append(unsmoothed_list[i - len_fit])
    slope_specs_list.append(slope_specs)

for i in range(len_fit + len_int, len_fit + 2 * len_int): # smoothed int
    add = lab_list[i]
    lab = r'$\tilde{\dot{a}}_{%s}$' % add
    e_lab = r'$\tilde{\sigma}_{%s}$' % add
    lcol = color_list[i]
    ecol = color_list[i]
    slope_specs = [2, lcol, ecol, α_list[i], 0.25, lab, e_lab]
    
    slope_data_list.append(smoothed_list[i - len_fit - len_int])
    slope_specs_list.append(slope_specs)

for i in range(len_fit + 2 * len_int, len_fit + 2 * len_int + len_filter): # smoothed int
    add = lab_list[i]
    lab = r'$\hat{\dot{a}}_{%s}$' % add
    e_lab = r'$\hat{\sigma}_{%s}$' % add
    lcol = color_list[i]
    ecol = color_list[i]
    slope_specs = [2, lcol, ecol, α_list[i], 0.25, lab, e_lab]
    
    slope_data_list.append(filtered_adot[i - len_fit - 2 * len_int])
    slope_specs_list.append(slope_specs)
# %%
def cl_lin(liste, cmap): # linear scaling
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

def mjd_to_mmdd(t, pos): # for having mjd scale
    t_obj = Time(str(t), format = 'mjd')
    t_iso = t_obj.iso
    string = t_iso[5 : 10].replace('-', '.')
    return(string)

def xaxis_year(t_list, time_format): # get year
    t0 = t_list[0]
    t_obj = Time(str(t0), format = 'mjd')
    t_iso = t_obj.iso
    string = t_iso[0 : 4]
    string = string + " + " + time_format
    return(string)

def t_min_t_max(data_list): # get borders
    t_min_list = []
    t_max_list = []
    for i in range(0, len(data_list)):
        t_min_list.append(min(data_list[i][:, 0]))
        t_max_list.append(max(data_list[i][:, 0]))
    return(min(t_min_list), max(t_max_list))

def scientific_to_exponent(number, coma_digits): # for labels of scaled errors
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
        new_str = r'$10^{%s}$' % (exp_str)
    else:
        new_str = r'$%s \cdot 10^{%s}$' % (fac_str, exp_str)
    return(new_str)

def decr_bar_ult(slope_data_list, slope_specs_list, e_fac_list,
                 flx_data, flx_spec_list,
                 vline_list1, vline_list_specs1,
                 vline_list2, vline_list_specs2,
                 xlims, ylims, save_specs):
    # ultimative plot: slope of semi-major axis, ap index, CMEs, MANs
    t_min, t_max = t_min_t_max(slope_data_list)
    xstart = t_min
    xend = t_max
    n_days = xend - xstart
    new_slope_data_list = slope_data_list
    flx_data = array_modifier(flx_data, xstart, n_days)
    colcounter = 0
    fig, ax1 = plt.subplots(figsize = (20,5), dpi = 300)
    for i in range(0, len(new_slope_data_list)):
        slope_data = new_slope_data_list[i]
        slope_specs = slope_specs_list[i]
        x_slope = slope_data[:, 0]
        y_slope = slope_data[:, 1]
        
        s_fac = e_fac_list[i]
        s_y_slope = slope_data[:, 2] * s_fac
        lab_fac = scientific_to_exponent(s_fac, 1)
        # slope_specs = [[l_w, lcol, ecol, α, e_α, lab, e_lab]]
        l_w = slope_specs[0]
        lcol = slope_specs[1]
        ecol = slope_specs[2]
        α = slope_specs[3]
        e_α = slope_specs[4]
        lab = slope_specs[5]
        e_lab = slope_specs[6]
        if (s_fac != 1):
            e_lab = e_lab + r' $\times$ ' + lab_fac
        
        ax1.plot(x_slope, y_slope,
                 ls = '-', lw = l_w, color = lcol,
                 alpha = α, label = lab, zorder = 10)
        colcounter += 0.501
        if (s_fac != 0):
            ax1.fill_between(x_slope, y_slope - s_y_slope, y_slope + s_y_slope,
                             color = ecol, alpha = e_α, zorder = 9,
                             label = e_lab)
            colcounter += 0.501
        
        print(lab, min(y_slope))
    
    ax1.xaxis.set_major_formatter(mjd_to_mmdd)
    ax1.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
    ax1.tick_params(axis = 'x', which = 'major',
                    width = 4, length = 12,
                    direction = 'out', top = False,
                    labelsize = 20)
    ax1.tick_params(axis = 'x', which = 'minor',
                    width = 1, length = 8,
                    direction = 'out', top = False,
                    labelsize = 20)
    ax1.set_xlabel(xaxis_year([xstart], 'mm.dd'), fontsize = 25)
    
    ax1.tick_params(axis = 'y', which = 'major',
                    width = 2, length = 8,
                    direction = 'inout', labelsize = 20)
    #ax1.tick_params(axis = 'y', labelcolor = 'k', labelsize = 20)
    y_lab = r'$\dot{a}$ [md$^{-1}$]'
    ax1.set_ylabel(y_lab, fontsize = 25)
    
    ax1.grid()
    if (xstart != xend):
        ax1.set_xlim(xstart, xend)
    
    if (len(flx_data) != 0):
        len_flx_data = len(flx_data.T) - 1
        α = flx_spec_list[0]
        
        ax2 = ax1.twinx()
        bar_width = 1 / (len_flx_data)
        
        col = flx_spec_list[2]
        for i in range(1, len_flx_data + 1):
            t_ap = flx_data[:, 0]
            ap = flx_data[:, i]
            
            ax2.bar(t_ap + (i - 1 / 2) * bar_width, ap,
                    width = bar_width, alpha = α,
                    color = col, lw = 0)
        
        ax2.tick_params(axis = 'y', which = 'major',
                    width = 2, length = 8,
                    direction = 'inout', labelsize = 20)
        flx_ylabel =  r'$ap$'
        ax2.set_ylabel(flx_ylabel, color = col, fontsize = 25, rotation = 0)
    
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
            ax1.axvline(vline, color = col, alpha = α,
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
            ax1.axvline(vline, color = col, alpha = α,
                        ls = style, lw = width,
                        label = lab, zorder = 20)
    
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.set_frame_on(False)
    
    ax1.set_ylim(ylims[0], ylims[1])
    ax1.set_xlim(xlims[0], xlims[1])
    
    plt.figlegend(fontsize = 25, markerscale = 1,
                  #9, (0.5, 1.15)
                  loc = 9, bbox_to_anchor = (0.5, 1.2),
                  bbox_transform = ax1.transAxes, labelspacing = 1,
                  ncols = len(slope_data) + 1, columnspacing = 1,
                  frameon = True, borderpad = 0.2)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)
# %% plot results
flx_spec_list = [1, 0, 'k']
vline_list_specs1 = ['goldenrod', 1, 5, "CME", (4, (4, 4)), 2]
vline_list_specs2 = ['saddlebrown', 1, 5, "MAN", (-4, (4, 4, 4, 4)), 2]
e_fac_list = len(configuration_list) * [0] + 4 * [0] + len_filter * [0] # error scaling
d = 0 # fit
xlims = MJD_interval_int
ylims = [-8, 0]
#ylims = [-250,0]
#ylims = [-35,20]
saveon = 0
decr_bar_ult(slope_data_list[d : d + len_fit],
             slope_specs_list[d : d + len_fit],
             e_fac_list[d : d + len_fit],
             data_ap, flx_spec_list,
             vline_list1, vline_list_specs1,
             vline_list2, vline_list_specs2,
             xlims, ylims, [saveon, 'Plots/' + satellite + '-fit.png'])

d = len_fit + len_int
decr_bar_ult(slope_data_list[d : d + len_int],
             slope_specs_list[d : d + len_int],
             e_fac_list[d : d + len_int],
             data_ap, flx_spec_list,
             vline_list1, vline_list_specs1,
             vline_list2, vline_list_specs2,
             xlims, ylims, [saveon, 'Plots/' + satellite + '-int.png'])

# list1 = slope_data_list[0 : 3] + slope_data_list[4 : 5]
# list2 = slope_specs_list[0 : 3] + slope_specs_list[4 : 5]
# list3 = e_fac_list[0 : 3] + e_fac_list[4 : 5]
# decr_bar_ult(list1, list2, list3,
#              data_ap, flx_spec_list,
#              vline_list1, vline_list_specs1,
#              vline_list2, vline_list_specs2,
#              xlims, ylims, [saveon, 'Plots/' + satellite + '-both.png'])

"""
d = 5 # unsmoothed int, smoothed int
decr_bar_ult(slope_data_list[d : d + len_filter], slope_specs_list[d : d + len_filter], e_fac_list[d : d + nfilter],
             data_ap, flx_spec_list,
             vline_list1, vline_list_specs1,
             vline_list2, vline_list_specs2,
             [0])
"""
# %%
fft_data_list = []
fft_specs_list = []

for i in range(0, len(slope_data_list)):
    slope_data = slope_data_list[i]
    fft_list = fft(slope_data)
    periods, amplitudes = fft_list[0], fft_list[1]
    fft_data_list.append([periods, amplitudes])
    specs = [1, color_list[i], lab_list[i]]
    fft_specs_list.append(specs)

def logplot(data_list, specs_list, xlims, ylims, n_cols, vlist):
    fig = plt.figure(figsize = (12, 5), dpi = 300)
    for i in range(0, len(data_list)):
        data = data_list[i]
        specs = specs_list[i]
        lwidth = specs[0]
        lcol = specs[1]
        lab = specs[2]
        
        plt.loglog(data[0], data[1],
                   color = lcol, ls = 'solid', lw = lwidth,
                   label = lab)
    plt.xlabel(r'period [d]', fontsize = 15)
    plt.ylabel(r'amplitude [m]', fontsize = 15)
    plt.legend(fontsize = 15, loc = 0,
               labelspacing = 0.5, ncols = n_cols)
    plt.vlines(vlist, ylims[0], ylims[1], linestyle = 'solid', color = 'k')
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    plt.grid()
    plt.show()
    plt.close()

print(len(fft_data_list), len(fft_specs_list))
vlist = 1/np.arange(1,13)*a_for_lsa_per[-1]*2
d = 0
xlims = [1e-2, 1e1]
ylims = [1e-3, 1e2]
n_cols = len_fit
logplot(fft_data_list[d : d + len_fit], fft_specs_list[d : d + len_fit],
        xlims, ylims, n_cols, vlist)
d = len_fit
xlims = [1e-2, 1e1]
ylims = [1e-4, 1e2]
n_cols = len_fit
logplot(fft_data_list[d : d + 2 * len_int], fft_specs_list[d : d + 2 * len_int],
        xlims, ylims, n_cols, vlist)
d = len_fit + 2 * len_int
xlims = [1e-2, 1e1]
ylims = [1e-4, 1e2]
n_cols = len_fit
logplot(fft_data_list[d :], fft_specs_list[d :],
        xlims, ylims, 1, vlist)
# %% plot the air density and the beta angle
fig, ax1 = plt.subplots(figsize = (12,5), dpi = 300)
ax1.plot(rho_data[:, 0], rho_data[:, 1], 'r-', label = r'$\varrho$')
ax1.xaxis.set_major_formatter(mjd_to_mmdd)
ax1.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
ax1.tick_params(axis = 'x', which = 'major',
                width = 4, length = 12,
                direction = 'out', top = False,
                labelsize = 15)
ax1.tick_params(axis = 'x', which = 'minor',
                width = 2, length = 8,
                direction = 'out', top = False,
                labelsize = 15)
ax1.tick_params(axis = 'y', labelcolor = 'r', color = 'r', labelsize = 15)
ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
y_lab1 = r'$\varrho$ [kgm$^{-3}$]'
ax1.set_ylabel(y_lab1, fontsize = 15, color = 'r')
ax2 = ax1.twinx()
ax2.plot(beta_data[:, 0], beta_data[:, 1], 'b-', label = r'$\beta$')
y_lab2 = r'$\beta$ [°]'
ax2.set_ylabel(y_lab2, color = 'b', fontsize = 15, rotation = 0)
ax2.tick_params(axis = 'y', labelcolor = 'b', color = 'b', labelsize = 15)
fig.legend()
plt.show(fig)
plt.close(fig)
# %% plot the amplitudes from fit model
n_s = int(np.round(n_fac * n_days_tot_lsa))
τ_fit_list = τ_fit_list_list[0]

μ_r_list_1 = np.array(x_list_list[0][1 * (n_s + 1) : (1 + 1) * (n_s + 1)])
η_r_list_1 = np.array(x_list_list[0][(1 + 1) * (n_s + 1) : (1 + 1 + 1) * (n_s + 1)])

μ_r_list_2 = np.array(x_list_list[1][1 * (n_s + 1) : (1 + 1) * (n_s + 1)])
η_r_list_2 = np.array(x_list_list[1][(1 + 1) * (n_s + 1) : (1 + 1 + 1) * (n_s + 1)])

μ_r_list_3 = np.array(x_list_list[2][1 * (n_s + 1) : (1 + 1) * (n_s + 1)])
η_r_list_3 = np.array(x_list_list[2][(1 + 1) * (n_s + 1) : (1 + 1 + 1) * (n_s + 1)])

fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.plot(τ_fit_list, μ_r_list_1,
         'o-', color = 'crimson', lw = 3, ms = 3, mew = 0,
         label = r'$\mu_1(\psi_1)$')
plt.plot(τ_fit_list, μ_r_list_2,
         'o-', color = 'dodgerblue', lw = 2, ms = 2, mew = 0,
         label = r'$\mu_1(\psi_2)$')
plt.plot(τ_fit_list, μ_r_list_3,
         'o-', color = 'forestgreen', lw = 1, ms = 1, mew = 0,
         label = r'$\mu_1(\psi_3)$')

plt.plot(τ_fit_list, η_r_list_1,
         'o-', color = 'tomato', lw = 3, ms = 3, mew = 0,
         label = r'$\eta_1(\psi_1)$')
plt.plot(τ_fit_list, η_r_list_2,
         'o-', color = 'deepskyblue', lw = 2, ms = 2, mew = 0,
         label = r'$\eta_1(\psi_2)$')
plt.plot(τ_fit_list, η_r_list_3,
         'o-', color = 'lime', lw = 1, ms = 1, mew = 0,
         label = r'$\eta_1(\psi_3)$')
plt.ylabel('Amplitude [m]')
plt.grid()
plt.legend(ncols = 2,loc = 0)
plt.show(fig)
plt.close(fig)
# %% plot the covariance matrix of the fit model
for k in range(0, len(configuration_list)):
    configurations = configuration_list[k]
    n_fac = configurations[0]
    R = configurations[1]
    
    length = int(np.round(n_fac * n_days_tot_lsa))
    half = length / 2
    ticks_list = []
    for q in range(1, 2 * R + 1 + 1):
        ticks_list.append(half + (q - 1) * length)
    
    a_bar_list = [r'$\bar{a}$']
    μ_r_list, η_r_list = [], []
    for r in range(1, R + 1):
        μ_r_list.append(r'$\mu_{%d}$' % r)
        η_r_list.append(r'$\eta_{%d}$' % r)
    labels_list = a_bar_list + μ_r_list + η_r_list
    
    Kxx = Kxx_list[k]
    negative = np.ma.masked_array(Kxx, Kxx > 0)
    positive = np.ma.masked_array(Kxx, Kxx < 0)
    
    fig, ax = plt.subplots(dpi = 300)
    ax.set_title(r'$K_{xx}(\psi_{%d})$' % (k + 1), pad = 25)
    im_negative = ax.imshow(-negative, interpolation = 'nearest',
                            norm = colors.LogNorm(vmin = -negative.min(),
                                                  vmax = -negative.max()),
                            cmap = mat.colormaps['cool'])
    im_positive = ax.imshow(positive, interpolation = 'nearest',
                            norm = colors.LogNorm(vmin = positive.min(),
                                                  vmax = positive.max()),
                            cmap = mat.colormaps['hot'])
    colbar_negative = plt.colorbar(im_negative, shrink = 0.75)
    colbar_positive = plt.colorbar(im_positive, shrink = 0.75)
    ax.set_xticks(ticks_list, labels_list, minor = False)
    ax.set_yticks(ticks_list, labels_list, minor = False)
    ax.tick_params(top = False, left = False, labeltop = True,
                   bottom = False, labelbottom = False)
    colbar_negative.set_label('negative magnitude')
    colbar_positive.set_label('positive magnitude')
    plt.show()
    plt.close()
# %%