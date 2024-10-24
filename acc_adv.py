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

def file_number_sorter(liste, time_format, yday0):
    # files: 'RDAVFING183650.LST', 'RDAVFLNG180010.LST'
    # sort by number
    # liste = [el1, el2, el3] <-> [ind_1, ind_2, ind_3]
    if (time_format == "yday"):
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

def duo_file_reader(path_2, dt, corr):
    # path_i = foldername_i/file_name_i
    # path_1 -> txt file with bias and scale; filename with mjd
    # path_2 -> acc file with data; filename with yday
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
    
    for i in range(1, 3 + 1):
        data[:, i] *= 1000 # mm/s^2 -> m/s^2
    
    return(data)

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

def duo_file_generator(foldername, subfoldername_2,
                       yday0, dt, corr):
    # foldername_1 -> txt files with bias and scale; filenames with mjd
    # foldername_2 -> acc files with data; filenames with yday
    # yday0: 0 -> it has no zero, 1 -> it has a zero at the end of date
    #       183240: 18 -> year 2018, 324 -> day number, 0 -> daily file (bernese add on)
    # dt: if 10 -> 10s sampling
    # corr: if 0 -> no correction, if 1 -> correction
    entries_2 = file_list_generator(foldername + "/" + subfoldername_2,
                                    ".ACC", "yday", yday0) # data
    
    data_year = np.array([[0, 0, 0, 0]])
    for i in range(0, len(entries_2)):
        file_2 = entries_2[i]
        path_2 = foldername + "/" + subfoldername_2 + "/" + file_2
        data_i = duo_file_reader(path_2, dt, corr)
        data_year = np.vstack((data_year, data_i))
    data_year = data_year[1:]
    data_year = array_normalize(data_year, 0)
    
    all_name = "/all_uncorr.txt"
    name_year = foldername + all_name
    np.savetxt(name_year, data_year, delimiter = ' ')
    print("DDDDOOOONNNNEEEE")

duo_file_generator("ultimatedata/GF-C", "GF-C_new", 1, 5, 1)
# %%