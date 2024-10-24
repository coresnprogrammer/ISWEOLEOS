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
from astropy.time import Time
plt.style.use('classic')
mat.rcParams['figure.facecolor'] = 'white'
print(np.random.randint(1,9))
# print(u'\u03B1') # alpha
# print(u'\u03C3') # sigma
#%%
print(os.getcwd())
os.chdir('/Users/levin/Documents/Uni/Bachelorthesis new')
print(os.getcwd())
#%%
from functions import array_normalize, array_denormalize, array_modifier
#%%
def find_string_length(string, target_string):
    # gives length of whole string until with target string
    # ex: string = "123456789", target_string = "456"
    # --> length = 6
    target_length = len(target_string)
    length = -1
    for i in range(0, len(string) - target_length + 1):
        scanned_string = string[i : i + target_length]
        if (scanned_string == target_string):
            length = i + target_length
            break
    if (length == -1):
        print("WARNING: TARGET STRING NOT FOUND!!!")
    return(length)

def read_bias_or_scale(line):
    string = line[find_string_length(line, "(R,S,W)"):]
    string_list = string.split(' ')
    string_list = [obj for obj in string_list if obj != ''] # remove ''
    new_string_list = [float(obj) for obj in string_list]
    return(new_string_list)

def file_number_sorter(liste, time_format, yday0):
    # files: 'RDAVFING183650.LST', 'RDAVFLNG180010.LST'
    # sort by number
    # liste = [el1, el2, el3] <-> [ind_1, ind_2, ind_3]
    if (time_format == "mjd"):
        # for example acc_classic_orbitFit4Levin_svn964_58270.txt
        a = - 9
        b = - 4
    elif (time_format == "yday"):
        # for example GFOC_ACT_T18152.ACC
        a = - 9 - yday0
        b = - 4 - yday0
    else:
        print("ERROR!!! SPECIFY TIME FORMAT!!!")
        return(0)
    number_list = np.array([])
    for i in range(0, len(liste)):
        name = liste[i]
        number = int(name[a : b])
        number_list = np.append(number_list, number)
    sorting_list = np.argsort(number_list) # [ind_2, ind_3, ind_1]
    new_list = np.array([])
    for i in sorting_list:
        new_list = np.append(new_list, liste[i])
    return(new_list)

def duo_file_reader(path_1, path_2, dt, corr):
    # path_i = foldername_i/file_name_i
    # path_1 -> txt file with bias and scale; filename with mjd
    # path_2 -> acc file with data; filename with yday
    file1 = open(path_1, 'r')
    lines1 = file1.readlines()
    file1.close()
    c1 = 0
    while (c1 <= 10):
        row1 = lines1[c1].strip()
        if (row1[:6] == "# bias"):
            break
        if (c1 == 10):
            print("!!! ERROR 1 !!!")
            break
        c1 += 1
    bias_list = read_bias_or_scale(row1)
    scale_list = read_bias_or_scale(lines1[c1 + 1].strip())
    
    file2 = open(path_2, 'r')
    lines2 = file2.readlines()
    file2.close()
    c2 = 0
    while (c2 <= 20):
        row2 = lines2[c2].strip()
        if (row2[:3] == "RSW"):
            break
        if (c2 == 20):
            print("!!! ERROR 1 !!!")
            break
        c2 += 1
    mjd_0 = float(lines2[c2 + 1].strip())
    
    data = np.loadtxt(path_2, skiprows = c2 + 2, usecols = (0, 1, 2, 3))
    data[:, 0] += mjd_0
    
    if (dt != 0): # sampling from 1s to dt s
        sampled_data = np.zeros((1, 4))
        length = len(data)
        c = 0
        while (c < length):
            sample = data[c]
            sampled_data = np.vstack((sampled_data, sample))
            c += dt
        sampled_data = sampled_data[1:]
        data = sampled_data
    
    mean_array = np.array([mjd_0])
    scale_array = np.array([mjd_0])
    bias_array = np.array([mjd_0])
    
    for i in range(1, 3 + 1):
        mean = np.mean(data[:, i])
        data[:, i] -= mean
        mean_array = np.append(mean_array, mean)
    
    for i in range(1, 3 + 1):
        scale = scale_list[i - 1]
        bias = bias_list[i - 1]
        
        if (corr == 1):
            data[:, i] = data[:, i] * scale + bias * 1000 # m/s^2 -> mm/s^2
        
        data[:, i] /= 1000 # mm/s^2 -> m/s^2
        
        scale_array = np.append(scale_array, scale)
        bias_array = np.append(bias_array, bias)
    
    a_acc_list = np.array([])
    for i in range(0, len(data)):
        abs = 0
        for j in range(1, 4):
            abs += (data[i, j])**2
        a_acc_list = np.append(a_acc_list, np.sqrt(abs))
    data = np.hstack((data, np.array([a_acc_list]).T))
    
    m_s_b_list = [mean_array, scale_array, bias_array]
    return(data, m_s_b_list)

def file_list_generator(foldername, file_format, time_format, yday0):
    # foldername: string like
    # file_format: ".txt" or ".ACC"
    # time_format: "mjd" or "yday"
    entries_a = os.listdir(str(foldername))
    entries_b = np.array([])
    for i in range(0, len(entries_a)):
        file_i = entries_a[i]
        if (file_i[-4:] == file_format):
            entries_b = np.append(entries_b, file_i)
    entries_b = file_number_sorter(entries_b, time_format, yday0)
    entries_c = np.array([])
    for i in range(0, len(entries_b)): # get rid of '.DS_Store'
        file_i = entries_b[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 10000): # avoid empty files
            entries_c = np.append(entries_c, file_i)
    return(entries_c)

def duo_file_generator(foldername, subfoldername_1, subfoldername_2,
                       yday0, dt, corr):
    # foldername_1 -> txt files with bias and scale; filenames with mjd
    # foldername_2 -> acc files with data; filenames with yday
    # yday0: 0 -> it has no zero, 1 -> it has a zero at the end of date
    #       183240: 18 -> year 2018, 324 -> day number, 0 -> daily file (bernese add on)
    # dt: if 10 -> 10s sampling
    # corr: if 0 -> no correction, if 1 -> correction
    entries_1 = file_list_generator(foldername + "/" + subfoldername_1,
                                    ".txt", "mjd", yday0) # bias, scale
    entries_2 = file_list_generator(foldername + "/" + subfoldername_2,
                                    ".ACC", "yday", yday0) # data
    
    entries_2_mjd_list = np.array([])
    for i in range(0, len(entries_2)):
        file_name = entries_2[i]
        yday = file_name[-9 - yday0 : - 4 - yday0]
        yday = '20' + yday[:2] + ':' + yday[-3:]
        yday = Time(yday, format = 'yday')
        mjd_date = yday.mjd # warning: is float
        entries_2_mjd_list = np.append(entries_2_mjd_list, mjd_date)
    
    data_year = np.zeros((1, 5)) # [t, R, S, W, A] (A = absolute)
    mean_year = np.zeros((1, 4))
    scale_year = np.zeros((1, 4))
    bias_year = np.zeros((1, 4))
    for i in range(0, len(entries_1)):
        file_1 = entries_1[i]
        file_1_mjd = float(file_1[- 9 : - 4])
        pos = np.argsort(np.abs(entries_2_mjd_list - file_1_mjd))[0]
        if (entries_2_mjd_list[pos] == file_1_mjd):
            file_2 = entries_2[pos]
            path_1 = foldername + "/" + subfoldername_1 + "/" + file_1
            path_2 = foldername + "/" + subfoldername_2 + "/" + file_2
            data_i, m_s_b_i_list = duo_file_reader(path_1, path_2, dt, corr)
            
            mean_array = m_s_b_i_list[0]
            scale_array = m_s_b_i_list[1]
            bias_array = m_s_b_i_list[2]
            
            data_year = np.vstack((data_year, data_i))
            mean_year = np.vstack((mean_year, mean_array))
            scale_year = np.vstack((scale_year, scale_array))
            bias_year = np.vstack((bias_year, bias_array))
    data_year = data_year[1:]
    mean_year = mean_year[1:]
    scale_year = scale_year[1:]
    bias_year = bias_year[1:]
    #data_year = array_normalize(data_year, 0)
    
    all_name = "/all.txt"
    if (corr == 0):
        all_name = "/all_uncorr.txt"
    else:
        all_name = "/all_corr.txt"
    name_year = foldername + all_name
    name_mean = foldername + "/mean.txt"
    name_scale = foldername + "/scale.txt"
    name_bias = foldername + "/bias.txt"
    np.savetxt(name_year, data_year, delimiter = ' ')
    #np.savetxt(name_mean, mean_year, delimiter = ' ')
    #np.savetxt(name_scale, scale_year, delimiter = ' ')
    #np.savetxt(name_bias, bias_year, delimiter = ' ')
    print("DDDDOOOONNNNEEEE")

duo_file_generator("ultimatedata/GF-C", "GF-C_old", "gf-c", 1, 5, 0)
# %%
MJD_0 = 58270
n_days_tot = 1
name_acc = "ultimatedata/GF-C" + "/all.txt"
acc_data = np.loadtxt(name_acc)
acc_data = array_denormalize(acc_data)
acc_data = array_modifier(acc_data, MJD_0, n_days_tot)
acc_data = array_normalize(acc_data, 0)
acc_data = acc_data[1:]
# %%
fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.plot(acc_data[:, 0], acc_data[:, 0], 'r.')
plt.xlim(0,0.01)
plt.ylim(0,0.01)
plt.show(fig)
plt.close(fig)
# %%
foldername = "ultimatedata/GF-C"
accname1 = "/gf-c/GFOC_ACT_T182120.ACC"
accname2 = "/gf-c/GFOC_ACT_T182130.ACC"
txtname1 = "/GF-C_old/acc_classic_orbitFit4Levin_svn964_58330.txt"
txtname2 = "/GF-C_old/acc_classic_orbitFit4Levin_svn964_58331.txt"

accfile1 = np.loadtxt(foldername + accname1, skiprows = 9)
accfile2 = np.loadtxt(foldername + accname2, skiprows = 9)
accfile2[:, 0] += 1
accfile = np.vstack((accfile1, accfile2))

txtfile1 = np.loadtxt(foldername + txtname1, skiprows = 0)
#txtfile1[:, 4] *= 0.6460617149014
#txtfile1[:, 4] += -0.7620224771808E-07
txtfile2 = np.loadtxt(foldername + txtname2, skiprows = 0)
#txtfile2[:, 4] *= 0.8787635959766
#txtfile2[:, 4] += -0.7680077292690E-07
txtfile = np.vstack((txtfile1, txtfile2))
txtfile[:, 0] -= 58330

#%%
n = 3
rsw_list = ["R", "S", "W"]
fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.title("acc file: " + rsw_list[n - 1] + " mean: %.3e" % np.mean(accfile[:, n]))
plt.plot(accfile[:, 0], accfile[:, n], 'r.', ms = 1)
plt.show(fig)
plt.close(fig)
fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.title("txt file: " + rsw_list[n - 1] + " mean: %.3e" % np.mean(txtfile[:, 3 + n]))
plt.plot(txtfile[:, 0], txtfile[:, 3 + n], 'r.', ms = 1)
plt.show(fig)
plt.close(fig)
# %%
print(-0.7680077292690E-07)
# %%
foldername = "ultimatedata/GF-C"
all_corr = np.loadtxt(foldername + "/all_corr.txt",
                      usecols = (0, 1, 2, 3))
MJD_0_corr = all_corr[0, 0]
all_corr = all_corr[1:]

all_uncorr = np.loadtxt(foldername + "/all_uncorr.txt",
                        usecols = (0, 1, 2, 3))
MJD_0_uncorr = all_uncorr[0, 0]
all_uncorr = all_uncorr[1:]
#%%
n = 3
mean_corr = np.mean(all_corr[:, n])
std_corr = np.std(all_corr[:, n])
mean_uncorr = np.mean(all_uncorr[:, n])
std_uncorr = np.std(all_uncorr[:, n])
title_list = ["R", "S", "W"]

fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.title("corrected: " + title_list[n - 1])
plt.plot(all_corr[:, 0], all_corr[:, n], 'r.', ms = 1)
plt.axvline(58356 - MJD_0_corr)
plt.xlabel(str(MJD_0_corr) + " + t [d]")
plt.ylabel("m / s^2")
plt.xlim(all_corr[:, 0][0], all_corr[:, 0][-1])
plt.xlim(82, 90)
#plt.xlim(0,5)
plt.ylim(mean_corr - 1 * std_corr, mean_corr + 1 * std_corr)
plt.show(fig)
plt.close(fig)

fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.title("uncorrected: " + title_list[n - 1])
plt.plot(all_uncorr[:, 0], all_uncorr[:, n], 'r.', ms = 1)
plt.ylim(mean_uncorr - 1 * std_uncorr, mean_uncorr + 1 * std_uncorr)
plt.xlim(82, 90)
plt.ylim(-1.2*10**(-5), -1.1*10**(-5))
plt.axvline(58356 - MJD_0_uncorr)
plt.ylabel("m / s^2")
plt.xlabel(str(MJD_0_uncorr) + " + t [d]")
plt.show(fig)
plt.close(fig)
# %%
fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.title("corrected: " + title_list[n - 1])
plt.plot(all_corr[:, 0], all_corr[:, 1],
         'r.', ms = 1, label = "R")
plt.plot(all_corr[:, 0], all_corr[:, 2],
         'b.', ms = 1, label = "S")
plt.plot(all_corr[:, 0], all_corr[:, 3],
         'g.', ms = 1, label = "W")
plt.axvline(58356 - MJD_0_corr)
plt.xlabel(str(MJD_0_corr) + " + t [d]")
plt.ylabel("m / s^2")
plt.ylim(-0.000015, 0.000001)
plt.xlim(82, 90)
#plt.xlim(0,5)
plt.show(fig)
plt.close(fig)

fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.title("uncorrected: " + title_list[n - 1])
plt.plot(all_uncorr[:, 0], all_uncorr[:, 1],
         'r.', ms = 1, label = "R")
plt.plot(all_uncorr[:, 0], all_uncorr[:, 2],
         'b.', ms = 1, label = "S")
plt.plot(all_uncorr[:, 0], all_uncorr[:, 3],
         'g.', ms = 1, label = "W")
plt.xlim(82, 90)
plt.ylim(-0.000013, 0.000001)
plt.axvline(58356 - MJD_0_uncorr)
plt.ylabel("m / s^2")
plt.xlabel(str(MJD_0_uncorr) + " + t [d]")
plt.show(fig)
plt.close(fig)
# %%
#################################
#################################
#################################
#################################
#################################
#################################
foldername = "ultimatedata/GF-C"
all_corr = np.loadtxt(foldername + "/all_corr.txt",
                      usecols = (0, 1, 2, 3))
MJD_0_corr = all_corr[0, 0]
all_corr = all_corr[1:]
# %%
mean_corr = 0
std_corr = 0
fac = 1/20
for n in range(0, 3):
    mean_corr += np.mean(all_corr[:, n + 1]) / 3
    std_corr += np.std(all_corr[:, n + 1]) / 3
fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.title("corrected acc data")
plt.plot(all_corr[:, 0], all_corr[:, 1],
         'r.', ms = 1, label = "R")
plt.plot(all_corr[:, 0], all_corr[:, 2],
         'b.', ms = 1, label = "S")
plt.plot(all_corr[:, 0], all_corr[:, 3],
         'g.', ms = 1, label = "W")
#plt.xlim(82, 90)
#plt.ylim(-0.000013, 0.000001)

plt.xlim(all_corr[:, 0][0], all_corr[:, 0][-1])
plt.axvline(58356 - MJD_0_corr, color = 'k', ls = (0, (4, 4)), zorder = 1,
            label = '26. August')
plt.legend(markerscale = 10, fontsize = 12.5, ncols = 2)
plt.ylabel("m / s^2")
plt.grid()
plt.ylim(mean_corr - fac * std_corr, mean_corr + fac * std_corr)
plt.xlabel(str(MJD_0_corr) + " + t [d]")
plt.show(fig)
plt.close(fig)

fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.title("corrected acc data")
plt.plot(all_corr[:, 0], all_corr[:, 1],
         'r.', ms = 1, label = "R")
plt.plot(all_corr[:, 0], all_corr[:, 2],
         'b.', ms = 1, label = "S")
plt.plot(all_corr[:, 0], all_corr[:, 3],
         'g.', ms = 1, label = "W")
#plt.xlim(82, 90)
#plt.ylim(-0.000013, 0.000001)

plt.xlim(58355 - MJD_0_corr, 58358 - MJD_0_corr)
plt.axvline(58356 - MJD_0_corr, color = 'k', ls = (0, (4, 4)), zorder = 1,
            label = '26. August')
plt.legend(markerscale = 10, fontsize = 12.5, ncols = 2)
plt.ylabel("m / s^2")
plt.grid()
plt.ylim(mean_corr - fac * std_corr, mean_corr + fac * std_corr)
plt.xlabel(str(MJD_0_corr) + " + t [d]")
plt.show(fig)
plt.close(fig)
#%%