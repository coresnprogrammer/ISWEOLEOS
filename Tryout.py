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
import os
import random
import matplotlib.image as mpimg
from scipy.integrate import solve_ivp as ode_solve
from scipy.optimize import curve_fit
plt.style.use('classic')
mat.rcParams['figure.facecolor'] = 'white'
print(np.random.randint(1,9))
# print(u'\u03B1') # alpha
# print(u'\u03C3') # sigma
print(os.getcwd())
os.chdir('/Users/levin/Documents/Uni/Bachelorthesis new')
print(os.getcwd())
#%%
from functions import spectrum, smoother, fitter
from functions import plot_func_6, fft_logplot_spec_3, decrease_plot_adv
from functions import array_denormalize, array_normalize, array_modifier
from functions import array_columns, decrease_unit_day
from functions import arg_finder, fft, step_data_generation
from functions import decrease_plot_hyperadv
from functions import file_extreme_day_new, ele_gen_txt
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
from functions import pre_lst_sqrs_adjstmt, fitter, mjd_to_mmdd, L_constr_all
#%%
def file_number_sorter(liste):
    # files: 'RDAVFING183650.LST', 'RDAVFLNG180010.LST'
    # sort by number
    # liste = [el1, el2, el3] <-> [ind_1, ind_2, ind_3]
    number_list = np.array([])
    for i in range(0, len(liste)):
        name = liste[i]
        number = int(name[-8 : -5])
        number_list = np.append(number_list, number)
    sorting_list = np.argsort(number_list) # [ind_2, ind_3, ind_1]
    new_liste = np.array([])
    for i in sorting_list:
        new_liste = np.append(new_liste, liste[i])
    return(new_liste)

def convert_str_double_to_float(string):
    if (string[0] == ' '):
        new_string = string[1:]
    else:
        new_string = string
    return(float(new_string.replace("D", "E")))

def ele_file_read(foldername, filename, ng):
    # foldername: string like
    # filename: string like
    # ng: if ng file -> ng = 1, if normal -> ng = 0
    file = open(foldername + '/' + filename, 'r')
    lines = file.readlines()
    file.close()
    
    c = 0
    while (c < 101):
        row = lines[c].strip()
        if (row[:2] == '11'):
            break
        if (c == 100):
            print("!!! ERROR 1 !!!")
            break
        c += 1
    
    length = int((len(lines) - c) / 3)
    
    array_MJD = np.array([])
    array_11 = np.array([])
    array_12 = np.array([])
    array_13 = np.array([])
    
    for i in range(0, length):
        j = c + 3 * i
        
        MJD_i = float(lines[j].strip()[9 : 20])
        array_MJD = np.append(array_MJD, MJD_i)
        
        acc_11_i_string = lines[j + 0].strip()[37 : 54]
        acc_11_i = convert_str_double_to_float(acc_11_i_string)
        array_11 = np.append(array_11, acc_11_i)
        
        acc_12_i_string = lines[j + 1].strip()[37 : 54]
        acc_12_i = convert_str_double_to_float(acc_12_i_string)
        array_12 = np.append(array_12, acc_12_i)
        
        acc_13_i_string = lines[j + 2].strip()[37 : 54]
        acc_13_i = convert_str_double_to_float(acc_13_i_string)
        array_13 = np.append(array_13, acc_13_i)
    
    if (ng == 0):
        c0 = 0
        while (c0 < 101):
            row = lines[c0].strip()
            if (row[:3] == 'L30'):
                L30_string = lines[c0].strip()[37 : 54]
                L30 = convert_str_double_to_float(L30_string)
                L20_string = lines[c0 + 1].strip()[37 : 54]
                L20 = convert_str_double_to_float(L20_string)
                L10_string = lines[c0 + 2].strip()[37 : 54]
                L10 = convert_str_double_to_float(L10_string)
                
                array_11 += L10
                array_12 += L20
                array_13 += L30
                break
            if (c0 == 100):
                print("!!! ERROR 2 !!!")
                break
            c0 += 1
    
    data = np.vstack((array_MJD, array_11, array_12, array_13)).T
    
    array_14 = np.array([])
    for i in range(0, len(data)):
        abs = 0
        for j in range(1, 4):
            abs += (data[i, j])**2
        array_14 = np.append(array_14, np.sqrt(abs))
    data = np.hstack((data, np.array([array_14]).T))
    
    return(data)

def ele_gen_txt(foldername, file_type):
    # file_type = 1 for normal, file_type = 2 for nongra
    # file_type = 3 for normal but no offset considered
    file_str = 0
    ng = 0
    if (file_type == 1):
        file_str = 'normal'
        ng = 0
    elif (file_type == 2):
        file_str = 'nongra'
        ng = 1
    elif (file_type == 3):
        file_str = 'normal'
        ng = 1
    
    entries_old = os.listdir(str(foldername))
    
    entries = np.array([])
    for i in range(0, len(entries_old)):
        file_i = entries_old[i]
        if (file_i[-4:] == '.ELE'):
            entries = np.append(entries, file_i)
    
    entries = file_number_sorter(entries)
    
    entries_new = np.array([])
    for i in range(0, len(entries)): # get rid of '.DS_Store'
        file_i = entries[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 10000): # avoid empty files
            if (file_type == 1): # want normal files
                if (file_i[6:8] != 'NG'):
                    entries_new = np.append(entries_new, file_i)
            elif (file_type == 2): # want nongra files
                if (file_i[6:8] == 'NG'):
                    entries_new = np.append(entries_new, file_i)
            elif (file_type == 3): # want normal files but no offset
                if (file_i[6:8] != 'NG'):
                    entries_new = np.append(entries_new, file_i)
    
    if (all(np.array(entries_old) != 'year_' + file_str)):
        os.mkdir(foldername + "/year_" + file_str)
    
    ########### year ###########
    
    data_year = np.array([[0, 0, 0, 0, 0]])
    
    for i in range(0, len(entries_new)):
        file_i = entries_new[i]
        data_i = ele_file_read(foldername, file_i, ng)
        data_year = np.vstack((data_year, data_i))
    
    data_year = data_year[1:]
    
    #data_year = array_normalize(data_year, 0)
    if (file_type != 3):
        name_year = foldername + "/year_" + file_str + "/year_" + file_str + ".txt"
    else:
        name_year = foldername + "/year_" + file_str + "/year_" + file_str + '_withoutoffset' + ".txt"
    
    np.savetxt(name_year, data_year, delimiter = ' ')
    
    print("DDDDOOOONNNNEEEE")

#ele_gen_txt('All Data new/elegrace', 2)
#%%
def ele_get_errors(foldername, filename):
    # get errors of A, E, I, NODE, PERIGEE, ARG OF LAT
    # foldername: string like
    # filename: string like
    file = open(foldername + '/' + filename, 'r')
    lines = file.readlines()
    file.close()
    
    c = 0
    while (c < 101):
        row = lines[c].strip()
        if (row[:10] == 'ARC-NUMBER'):
            break
        if (c == 100):
            print("!!! ERROR 1 !!!")
            break
        c += 1
    
    mjd = float(row[-18:])
    a_error = float(lines[c+2].strip()[-24:-12])
    e_error = float(lines[c+3].strip()[-24:-12])
    i_error = float(lines[c+4].strip()[-24:-12])
    Ω_error = float(lines[c+5].strip()[-24:-12])
    ω_error = float(lines[c+6].strip()[-24:-12])
    u_error = float(lines[c+7].strip()[-24:-12])
    
    data = np.array([mjd, a_error, e_error, i_error,
                     Ω_error, ω_error, u_error])
    
    return(data)

def ele_gen_txt_error(foldername, file_type):
    # file_type = 1 for normal, file_type = 2 for nongra
    file_str = 0
    if (file_type == 1):
        file_str = 'normal'
    elif (file_type == 2):
        file_str = 'nongra'
    
    entries_old = os.listdir(str(foldername))
    
    entries = np.array([])
    for i in range(0, len(entries_old)):
        file_i = entries_old[i]
        if (file_i[-4:] == '.ELE'):
            entries = np.append(entries, file_i)
    
    entries = file_number_sorter(entries)
    
    entries_new = np.array([])
    for i in range(0, len(entries)): # get rid of '.DS_Store'
        file_i = entries[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 10000): # avoid empty files
            if (file_type == 1): # want normal files
                if (file_i[6:8] != 'NG'):
                    entries_new = np.append(entries_new, file_i)
            elif (file_type == 2): # want nongra files
                if (file_i[6:8] == 'NG'):
                    entries_new = np.append(entries_new, file_i)
    
    if (all(np.array(entries_old) != 'year_' + file_str)):
        os.mkdir(foldername + "/year_" + file_str)
    
    data_year = np.array([[0, 0, 0, 0, 0, 0, 0]])
    
    for i in range(0, len(entries_new)):
        file_i = entries_new[i]
        data_i = ele_get_errors(foldername, file_i)
        data_year = np.vstack((data_year, data_i))
    
    data_year = data_year[1:]
    
    #data_year = array_normalize(data_year, 0)
    
    name_year = foldername + "/year_" + file_str + "/year_" + file_str + "_errors.txt"
    
    np.savetxt(name_year, data_year, delimiter = ' ')
    
    print("DDDDOOOONNNNEEEE")


def ele_σ_for_rk45(path_ele, mjd_interval):
    data = np.loadtxt(path_ele)
    mjd_start = mjd_interval[0]
    n_days = mjd_interval[1] - mjd_start
    data_trimmed = array_modifier(data, mjd_start, n_days)
    σ_a = np.mean(data_trimmed[:, 1])
    σ_e = np.mean(data_trimmed[:, 2])
    σ_ω = np.mean(data_trimmed[:, 5])
    return(σ_a, σ_e, σ_ω)




folder = "All Data new/eleswarm"
ele_gen_txt(folder, 3)
#ele_gen_txt_error(folder, 2)
#path = folder + "/year_normal/year_normal_errors.txt"
#MJD_interval = [60431, 60445]
# %%
for i in range(0, 1):
    print(i)
# %%
def plot_examples(colormaps):
    """
    Helper function to plot data with associated colormap.
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            layout='constrained', squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()
cmap = ListedColormap([(1,0,0), "blue", "green"])
plot_examples([cmap])
# %%
red1 = np.linspace(1,0,128)
green1 = np.zeros(128)

blue1 = np.linspace(0,1,128)

red2 = np.zeros(128)
green2 = np.linspace(0,1,128)
blue2 = np.linspace(1,0,128)

clist1 = np.vstack((red1, green1, blue1)).T
clist2 = np.vstack((red2, green2, blue2)).T
clist3 = np.vstack((clist1, clist2))

cols = []
for i in clist3:
    cols.append(tuple(i))
my_cmap = ListedColormap(cols)
plot_examples([cmap])
# %%
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

fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.scatter(np.linspace(0,1,256),np.linspace(0,1,256), color = cl_lin(np.linspace(0,1,256), ListedColormap(cols)))
plt.show(fig)
plt.close(fig)
# %%
def riffle(list1, list2):
    newlist = []
    for i in range(0, len(list1)):
        newlist.append(list1[i])
        newlist.append(list2[i])
    return(newlist)

l1 = [1,3,5,7]
l2 = [2,4,6,8]
print(riffle(l1,l2))
# %%
a1 = np.array([[0.1,0.1],[0.9,0.1],[0.1,0.9],[0.9,0.9]])
s = int(1e2)
az = np.linspace(0,1,s)
a2 = np.vstack((az,az)).T

fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.plot(a1[:,0], a1[:, 1], 'r.')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('Test/im1.png')
plt.show(fig)
plt.close(fig)

fig = plt.figure(figsize = (10, 5), dpi = 300)
plt.plot(a2[:,0], a2[:, 1], 'r.')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('Test/im2.png')
plt.show(fig)
plt.close(fig)
# %%
print(np.linspace(0,1,5))

# %%
from astropy.time import Time
def yyyymmddhhmm_to_mjd(string):
    t_obj = Time(string, format = 'isot', scale = 'utc')
    t_mjd = t_obj.mjd
    mjd_float = float(t_mjd)
    return(mjd_float)


mylist = ['2024-05-05T11:30', '2024-05-10T16:36', '2024-05-10T16:36',
 '2024-05-10T16:36', '2024-05-10T16:36', '2024-05-10T16:36',
 '2024-05-11T09:30', '2024-05-11T09:30', '2024-05-11T20:30',
 '2024-05-12T08:55', '2024-05-15T18:13']

l1 = [1,2,3]
yyyymmddhhmm_to_mjd(mylist[0])
for i in range(0, len(mylist)):
    l1.append(yyyymmddhhmm_to_mjd(mylist[i]))
print(l1)
# %%
fig=plt.figure(figsize=(10,5),dpi=300)
plt.plot([0,0,1,1],[0,1,0,1],'r.')
plt.axvline(0.5,color='blue',lw=1,ls=(0,(4,12)))
plt.show(fig)
plt.close(fig)
# %%
print(u'\u03B1') # alpha
print(u'\u03C3') # sigma
print(u'\u03b0')
print(u'\u03b1')
print(u'\u03b2')
print(u'\u03b3')
print(u'\u03b4')
print(u'\u03b5')
print(u'\u03b6')
print(u'\u03b7')
print(u'\u03b8')
print(u'\u03b9')
print(u'\u03c0')
print(u'\u03c1')
print(u'\u03c2')
print(u'\u03c3')
print(u'\u03c4')
print(u'\u03c5')
print(u'\u03c6')
print(u'\u03c7')
print(u'\u03c8')
print(u'\u03c9')
print(u'\u03d0')
print(u'\u03d1')
print(u'\u03d2')
print(u'\u03d3')
print(u'\u03d4')
# %%
print(u'\u0390')
print(u'\u03a0')
print(u'\u03a1')
print(u'\u03a2')
print(u'\u03a3')
print(u'\u03a4')
print(u'\u03a5')
print(u'\u03a6')
print(u'\u03a7')
print(u'\u03a8')
print(u'\u03a9')
# %%
print('[s]'[1:-1])
# %%
from astropy.time import Time
def yyyymmddhhmm_to_mjd(string):
    t_obj = Time(string, format = 'isot', scale = 'utc')
    t_mjd = t_obj.mjd
    mjd_float = float(t_mjd)
    return(mjd_float)

print(yyyymmddhhmm_to_mjd('2018-07-10T11:25'))
# %%
myarr = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
print(myarr)
print(np.vstack((myarr[:,:2].T, myarr[:,2])).T)
# %%
fig, ax = plt.subplots(figsize = (10, 5), dpi = 300)
ax.plot([0,0,1,1],[0,1,0,1],'r.')
ax.axvline(0.497, color = 'b', ls = (0,(4,12)),
           lw = 3,label = 'v1')
ax.axvline(0.509, color = 'g', ls = (-6,(2,4,2,8)),
           lw = 3,label = 'v2')
#ax.axvline(0.503, color = 'k', ls = (0,(2,2)),
#           lw = 3,label = 'v3')
#ax.set_xlim(0.4,0.6)
plt.figlegend(fontsize = 15, markerscale = 1, loc = 1,
                  bbox_to_anchor = (1,1), bbox_transform = ax.transAxes,
                  labelspacing = 0.5, ncols = 1, columnspacing = 1)
plt.show(fig)
plt.close(fig)
ls = (0,(4,12))
ls = (-6,(2,4,2,8))
# %%
print(Time('2023-2-25T12:00:00.000', format = 'isot', scale = 'utc').yday)
print(Time('2023:100:00:00:00.000', format = 'yday', scale = 'utc').isot)
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
# %%
def file_number_sorter(liste):
    # files: 'RDAVFING183650.LST', 'RDAVFLNG180010.LST'
    # sort by number
    # liste = [el1, el2, el3] <-> [ind_1, ind_2, ind_3]
    number_list = np.array([])
    for i in range(0, len(liste)):
        name = liste[i]
        number = int(name[-8 : -5])
        number_list = np.append(number_list, number)
    sorting_list = np.argsort(number_list) # [ind_2, ind_3, ind_1]
    new_liste = np.array([])
    for i in sorting_list:
        new_liste = np.append(new_liste, liste[i])
    return(new_liste)


def att_file_read(path):
    # foldername: string like
    # filename: string like
    # ng: if ng file -> ng = 1, if normal -> ng = 0
    file = np.loadtxt(path, skiprows = 11)
    t_array = file[:, : 6]
    Q_array = file[:, 6 :]
    
    t_list = np.array([])
    for i in range(0, len(t_array)):
        string = '%d-%d-%d %d:%d:%f'
        t_obj = Time(string, format = 'iso', scale = 'utc')
        t_list = np.append(t_list, float(t_obj.mjd))
    
    data = np.vstack((t_list, Q_array.T)).T
    return(data)

def att_gen_txt(foldername):
    # file_type = 1 for normal, file_type = 2 for nongra
    # file_type = 3 for normal but no offset considered
    
    entries_old = os.listdir(str(foldername))
    
    entries = np.array([])
    for i in range(0, len(entries_old)):
        file_i = entries_old[i]
        if (file_i[-4:] == '.ATT'):
            entries = np.append(entries, file_i)
    
    entries = file_number_sorter(entries)
    
    entries_new = np.array([])
    for i in range(0, len(entries)): # get rid of '.DS_Store'
        file_i = entries[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 10000): # avoid empty files
            entries_new = np.append(entries_new, file_i)
    
    if (all(np.array(entries_old) != 'year')):
        os.mkdir(foldername + "/year")
    
    ########### year ###########
    
    data_year = np.array([[0, 0, 0, 0, 0]])
    
    for i in range(0, len(entries_new)):
        file_i = entries_new[i]
        data_i = att_file_read(foldername + '/' + file_i)
        data_year = np.vstack((data_year, data_i))
    
    data_year = data_year[1:]
    
    #data_year = array_normalize(data_year, 0)
    name_year = foldername + "/year/year_ATT.txt"
    
    np.savetxt(name_year, data_year, delimiter = ' ')
    
    print("DDDDOOOONNNNEEEE")

#ele_gen_txt('All Data new/elegrace', 2)
att_gen_txt('se1a_att')
#%%
print(Time('2023-2-25T12:00:00.000', format = 'isot', scale = 'utc').mjd)
print(Time('2023-02-25T12:00:00.000', format = 'isot', scale = 'utc').mjd)
print(Time('2023:100:00:00:00.000', format = 'yday', scale = 'utc').isot)
# %%
print("pi is %f" % (np.pi))
# %%
