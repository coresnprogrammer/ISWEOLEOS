
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
from astropy.time import Time
print(os.getcwd())
os.chdir('/Users/levin/Documents/Uni/Bachelorthesis new')
print(os.getcwd())
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


def att_file_read(path):
    # foldername: string like
    # filename: string like
    # ng: if ng file -> ng = 1, if normal -> ng = 0
    file = np.loadtxt(path, skiprows = 12)
    t_array = file[:, : 6]
    Q_array = file[:, 6 :]
    
    t_list = np.array([])
    q_array = np.array([0, 0, 0, 0])
    i = 0
    while (i < len(t_array)/10):
        t_i = t_array[i*10]
        #print(t_i)
        year, month, day = t_i[0], t_i[1], t_i[2]
        hour, minute, second = t_i[3], t_i[4], t_i[5]
        #print("----------")
        #print(year, month, day, hour, minute, second)
        #print("----------")
        string = '%.0f-%.0f-%.0f %.0f:%.0f:%f' % (year, month, day, hour, minute, second)
        #print(string)
        #break
        t_obj = Time(string, format = 'iso', scale = 'utc')
        t_list = np.append(t_list, float(t_obj.mjd))
        q_array = np.vstack((q_array, Q_array[i*10]))
        i += 1
    q_array = q_array[1:]
    data = np.vstack((t_list, q_array.T)).T
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
        print("done: ", file_i)
    
    data_year = data_year[1:]
    
    #data_year = array_normalize(data_year, 0)
    name_year = foldername + "/year/year_ATT.txt"
    
    np.savetxt(name_year, data_year, delimiter = ' ')
    
    print("DDDDOOOONNNNEEEE")

#ele_gen_txt('All Data new/elegrace', 2)
#att_gen_txt('se1a_att')

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

data = np.loadtxt('se1a_att/year/year_ATT.txt')
MJD_0 = data[0,0]

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
ax.set_xlim(MJD_0 + 8, MJD_0 + 15)
ax.grid()
ax.legend()
plt.show()
plt.close()


"""
fig = plt.figure(figsize=(10,5),dpi=300)
plt.plot(data[:,0],data[:,2], 'b.', label = 'Qy')
plt.grid()
plt.legend()
plt.show()
plt.close()

fig = plt.figure(figsize=(10,5),dpi=300)
plt.plot(data[:,0],data[:,3], 'g.', label = 'Qz')
plt.grid()
plt.legend()
plt.show()
plt.close()

fig = plt.figure(figsize=(10,5),dpi=300)
plt.plot(data[:,0],data[:,4], 'm.', label = 'Qw')
plt.grid()
plt.legend()
plt.show()
plt.close()
"""
print("done")





print(Time('2023-02-25 00:00:00', format = 'iso', scale = 'utc').mjd)






