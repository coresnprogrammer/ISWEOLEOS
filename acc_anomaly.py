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
#%%
print(os.getcwd())
os.chdir('/Users/levin/Documents/Uni/Bachelorthesis new')
print(os.getcwd())
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

def acc_hyper_read(foldername, filename):
    # returns [acc_data_uncorr, bias_list, scale_list]
    file = open(foldername + '/' + filename, 'r')
    lines = file.readlines()
    file.close()
    
    c = 0
    while (c <= 10):
        row = lines[c].strip()
        if (row[:6] == "# bias"):
            break
        if (c == 10):
            print("!!! ERROR 1 !!!")
            break
        c += 1
    bias_list = read_bias_or_scale(row)
    scale_list = read_bias_or_scale(lines[c + 1].strip())
    c += 2
    acc_data = np.loadtxt(foldername + '/' + filename,
                          skiprows = c, usecols = (0, 4, 5, 6))
    return([acc_data, bias_list, scale_list])

def acc_corr(acc_data_uncorr, bias_list, scale_list):
    # returns acc_data_corr
    cor_acc_data = acc_data_uncorr[:, 0] # add corrected accelerations
    for i in range(0, 3):
        acc_col = acc_data_uncorr[:, i + 1]
        cor_acc_col = bias_list[i] + scale_list[i] * acc_col
        cor_acc_data = np.vstack((cor_acc_data, cor_acc_col))
    cor_acc_data = cor_acc_data.T
    return(cor_acc_data)
#%%
data_txt_1_list = acc_hyper_read("hyperdata/hyperdata1/GF-1",
                                 "acc_classic_orbitFit4Levin_svn964_58270.txt")

data_txt_1_uncorr = data_txt_1_list[0]
data_txt_1_corr = acc_corr(data_txt_1_uncorr, data_txt_1_list[1], data_txt_1_list[2])
#%%
fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.plot(data_txt_1_uncorr[:, 0], data_txt_1_uncorr[:, 2], 'r.', ms = 1,
         label = "uncorrected")
plt.plot(data_txt_1_corr[:, 0], data_txt_1_corr[:, 2], 'b.', ms = 1,
         label = "corrected")
plt.legend(markerscale = 5)
plt.show(fig)
plt.close(fig)
print(np.mean(data_txt_1_uncorr[:, 2]))#m/s^2
print(np.mean(data_txt_1_corr[:, 2]))#m/s^2
# %%
data1_new = np.loadtxt("ultimatedata/GF-C/GFOC_ACT_T18152.ACC",
                       skiprows = 9, usecols = (0, 1, 2, 3))
data2_new = np.loadtxt("ultimatedata/GF-D/GFOD_ACT_T18152.ACC",
                       skiprows = 9, usecols = (0, 1, 2, 3))
#print(data1_new[0])
#print(data1_new[0, 1])
# %%
fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.plot(data1_new[:, 0], data1_new[:, 2]*1000, 'k.', ms = 1)
plt.show(fig)
plt.close(fig)
print(np.mean(data1_new[:, 1])*1000)# m/s^2
# %%
