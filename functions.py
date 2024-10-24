#%%
import time as tt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from mpl_toolkits import mplot3d as d3
from scipy.optimize import curve_fit
from scipy import special
from astropy.time import Time
import scipy
import os
import matplotlib.image as mpimg
from scipy.integrate import solve_ivp as ode_solve
from scipy.optimize import curve_fit
# print(u'\u03B1') # alpha
#%%
t1 = tt.time()
s = 0
for i in range(0, 100000):
    s += i
t2 = tt.time()
print(t2 - t1)
#%%
def fft(array):
    t_array = array[:,0]
    y_array = array[:,1]
    size = np.shape(y_array)[-1]
    
    # original data orig
    fourier_orig = np.fft.fft(y_array) / size # preparing the coefficients
    new_fourier_orig = fourier_orig[1 : int(len(fourier_orig) / 2 + 1)] # only want a1, ..., an
    freq_orig = np.fft.fftfreq(size, d = t_array[1] - t_array[0]) # preparing the frequencies
    new_freq_orig = freq_orig[1 : int(len(freq_orig) / 2 + 1)] # only want frequencies to a1, ..., an
    period_orig = 1 / new_freq_orig # period
    amp_orig = 2 * np.abs(new_fourier_orig) # amplitude
    amp_tild_orig = amp_orig**2 / sum(amp_orig) # amplitude for the power spectrum
    
    orig_list = [period_orig, amp_orig]
    return(orig_list)
#%%
spring = mat.colormaps['spring']
rainbow = mat.colormaps['gist_rainbow']
plasma = mat.colormaps['plasma']
jet = mat.colormaps['jet']
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
#%%
def zero_setter(x_list, y_list, position, fac_low, fac_upp):
    # want that the amplitudes of all periods
    # in [p * fac_low, p * fac_upp] are set to 0
    newylist = y_list.copy()
    newylist = np.array(newylist)
    
    x_p = x_list[position] # dominant period p
    
    x_low = x_p * fac_low
    x_upp = x_p * fac_upp
    
    end = np.argmin(np.abs(x_list - x_low))
    start = np.argmin(np.abs(x_list - x_upp))
    
    newylist[start : end + 1] = 0
    return(newylist)
#%%
def peak_finder_adv(xlist, ylist, threshold_quotient, j, x_max, interval_list):
    # interval: if there is a small peak in this interval that should be detected
    # interval = [p1, p2], p1 < p2
    # detetect only peaks that have x < x_max
    print("peak_finder_adv version 9")
    fac = 0.25
    fac_upp = 1 + fac
    fac_low  = 1 - fac
    newxlist = xlist.copy()
    newylist = ylist.copy()
    newnewxlist = xlist.copy()
    newnewylist = ylist.copy()
    
    c = 0
    while (newxlist[c] > x_max):
        newylist[c] = 0
        c += 1
    print("c = ", c)
    if (j == 6):
        pos = np.argmax(newylist)
        m = max(newylist)
        return([pos], [m])
    # finding strongest period
    # WARNING: periods go from big to small -> in plot it starts from the right
    # pay attention to which words in comments are set in ""
    print("argmax = ", np.argmax(newylist))
    if (np.argmax(newylist) == c):
        print("ARGMAX")
        # if true --> the mean value increases linearily
        # start from the "right": find an instance where the
        # value on the neighbouring "left" is greater
        # every values to the "right" should be set to 0
        # to kill the linear increase
        count_0 = 0 # counter
        while (count_0 < 10000): # some value
            if (newylist[c + count_0] < newylist[c + count_0 + 1]):
                # beginning of ascent found
                newylist[0 : c + count_0] = 0 # set everything before to 0
                print(count_0)
                print(newxlist[count_0 + c])
                break
            count_0 += 1
    
    pos = np.argmax(newylist)
    m = max(newylist)
    poslist = [pos]
    maxlist = [m]
    # bsp: threshold_quotient = 0.5 --> threshold = 0.5 * amp_max
    threshold = threshold_quotient * m
    counter = 1
    while (m >= threshold) and (counter < 100):
        newylist = zero_setter(newxlist, newylist, poslist[-1], fac_low, fac_upp)
        if (pos == np.argmax(newylist)):
            break # endlosschlaufe an rand verhindern
        pos = np.argmax(newylist)
        m = max(newylist)
        if (m >= threshold):
            poslist.append(pos)
            maxlist.append(m)
        counter += 1
    
    # search interval
    print("intervallist")
    print(interval_list)
    if (interval_list[0][0] != interval_list[0][1]):
        for i in range(0, len(interval_list)):
            newnewxlist = xlist.copy()
            newnewylist = ylist.copy()
            
            p_low = interval_list[i][1]
            p_upp = interval_list[i][0]
            
            p_low_arg = np.argmin(np.abs(newnewxlist - p_low))
            p_upp_arg = np.argmin(np.abs(newnewxlist - p_upp))
            
            if (p_low_arg == p_upp_arg):
                p_low_arg -= 1
                p_upp_arg += 1
            
            newnewylist[:p_low_arg] = 0
            newnewylist[p_upp_arg:] = 0
            
            interval_y_max = max(newnewylist)
            interval_x_max = np.argmax(newnewylist)
            
            poslist.append(interval_x_max)
            maxlist.append(interval_y_max)
    print("poslist = ", poslist)
    return(poslist, maxlist)
#%%
def array_normalize(array, normalize_level):
    #                 / MJD_0,       , el_1_0,        el_k_0            \
    # should return: |  MJD_1 - MJD_0, el_1_1 - el_1_0, el_k_1 - el_k_0  |
    #                 \ MJD_i - MJD_0, el_1_i - el_1_0, el_k_i - el_k_0 /
    #          / MJD_1, el_1_1, el_k_1 \
    # array = |  MJD_2, el_1_2, el_k_2  |
    #          \ MJD_i, el_1_i, el_k_i /
    # normalize_level: i -> normalize up to column i, el_i_0 != 0, then el_j_0 = 0,
    #                  to only normalize MJD -> i = 0
    n_cols = len(array.T)
    row_0 = np.zeros((1, n_cols))
    for i in range(0, normalize_level + 1):
        el_k_0 = array[0, i]
        row_0[0, i] = el_k_0
        array[:, i] -= el_k_0
    new_array = np.vstack((row_0, array))
    return(new_array)
#%%
def array_denormalize(array):
    #                 / MJD_1, el_1_1, el_k_1 \
    # should return: |  MJD_2, el_1_2, el_k_2  |
    #                 \ MJD_i, el_1_i, el_k_i /
    #          / MJD_0,       , el_1_0,        el_k_0            \
    # array = |  MJD_1 - MJD_0, el_1_1 - el_1_0, el_k_1 - el_k_0  |
    #          \ MJD_i - MJD_0, el_1_i - el_1_0, el_k_i - el_k_0 /
    n_cols = len(array.T)
    row_0 = array[0]
    new_array = array[1:]
    for i in range(0, n_cols):
        new_array[:, i] += row_0[i]
    return(new_array)
#%%
def array_modifier(array, MJD_start, n_days):
    # array should be denormalized
    # MJD_start and n_days do not have to be integers
    a = 0 # find start
    while (array[a + 1, 0] <= MJD_start):
        a += 1
    array = array[a :] # crop
    b = 0 # find end
    if (array[-1, 0] <  MJD_start + n_days):
        return(array)
    else:
        while (array[b, 0] < MJD_start + n_days):
            b += 1
        array = array[: b] # crop
        return(array)
#%%
def array_columns(array, col_number_list):
    # array = [col_0, col_1, col_2, ...]
    # col_number_list = [a, b, c, ...]
    # gives: [col_a, col_b, col_c, ...]
    n_cols = len(array.T)
    n_rows = len(array)
    cols = np.hsplit(array, n_cols) # all cols
    new_array = np.ones((n_rows, 1)) # wanted cols
    for i in range(0, len(col_number_list)):
        new_array = np.hstack((new_array, cols[col_number_list[i]]))
    new_array = new_array[:, 1:]
    return(new_array)
#%%
def selector(array, pos_list): # use pos_list to select elements in array
    newlist = np.array([])
    for p in pos_list:
        newlist = np.append(newlist, array[p])
    return(newlist)
#%%
def selector_spec(array, pos_list): # use pos_list to select elements in array
    newlist = np.array([0, 0, 0])
    for p in pos_list:
        sub_new_list = np.array([array[p - 1], array[p], array[p + 1]])
        newlist = np.vstack((newlist, sub_new_list))
    newlist = newlist[1:]
    return(newlist)
#%%
def sorter(array, n, direction):
    # n: column used for sorting input
    # direction = 1 -> from lowest to highest
    # direction = - 1 -> from highest to lowest
    if (len(array) == 1):
        return(array)
    # for (nx2) arrays; sorting rows based on elements in first column from highest (first) to lowest (last)
    length = len(array)
    k = 0
    counter = 0
    new_array = np.array([[0,0]])
    while (k < length) and (counter < 50):
        col_n = array[:, n]
        pos = np.argmax(col_n) # position of row to switch
        row = array[pos]
        new_array = np.block([[new_array], [row]])
        array = np.delete(array, pos, 0)
        k += 1
        counter += 1
    new_array = np.delete(new_array, 0, 0)
    if (direction == 1):
        new_array = np.flip(new_array, 0)
    return(new_array)
#%%
def round_dot5(x):
    # round 0.5 to 1
    if (x - int(x) == 0.5):
        return(round(x + 0.5, 0))
    else:
        return(round(x, 0))
#%%
def round_uneven(x):
    # round x to the nearest uneven number
    # if x is an even number, then it should return x + 1
    x_int = 0
    x_round = int(round_dot5(x)) # round to nearest integer
    if (x_round % 2 == 1): # nearest integer is uneven
        x_int = x_round
    elif (x % 2 == 0): # x is even
        x_int = int(x + 1)
    else: # nearest integer is even -> find out if rounded up or down
        diff = x - x_round
        if (diff < 0): # it was rounded up
            x_int = int(x)
        else: # it was rounded down
            x_int = int(x + 1)
    return(x_int)
#%%
def ro(array, n):
    return(np.round(array, n))
#%%
def mod_abs(u, v):
    return(min(u % v, v - (u % v)))
#%%
def lcm_status_1(k, lcm_copy, n, tilde_p_i, hut_ε_i):
    str1 = "k = %3d | " % k
    str2 = "lcm = %12.3f | " % lcm_copy
    str3 = "p_%1d = %9.3f | " % (n, tilde_p_i)
    str4 = "ε_%1d = %6.1f" % (n, hut_ε_i)
    print(str1 + str2 + str3 + str4)
#%%
def lcm_status_2(n, hut_ε_i, hut_ε):
    str1 = "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    str2 = "----------------- "
    str3 = "ε_%1d = %6.1f < %6.1f = ε" % (n, hut_ε_i, hut_ε)
    str4 = " -----------------"
    print(str1)
    print(str2 + str3 + str4)
    print(str1)
#%%
def lcm_as_Δt(ε, N, limit, dt, per_list):
    tilde_per_list = per_list / dt
    tilde_limit = limit / dt
    hut_ε = ε / N
    tilde_lcm = tilde_per_list[0]
    n = 1
    n_tot = len(per_list)
    
    print("TO ALIGN: ", per_list)
    print("eff periods: ", tilde_per_list)
    print("hut_ε = %9.3f | tilde_limit = %9.3f" % (hut_ε, tilde_limit))
    
    k = 1
    while (k * N * tilde_lcm < tilde_limit and n < n_tot):
        lcm_copy = k * tilde_lcm
        
        tilde_p_i = tilde_per_list[n]
        hut_ε_i = mod_abs(lcm_copy, tilde_p_i)
        lcm_status_1(k, lcm_copy, n, tilde_p_i, hut_ε_i)
        if (hut_ε_i < hut_ε):
            lcm_status_2(n, hut_ε_i, hut_ε)
            tilde_lcm *= k
            n += 1
            k = 0
        
        k += 1
    return(N * tilde_lcm * dt, per_list[:n])
#%%
def jumps_array(array, n_points):
    # array = / / y_0_-n, y_0_-1 \ / y_1_-n, y_1_-1 \      \
    #         \ \ y_1_0 , y_1_n  /,\ y_2_0 , y_2_n  /, ... /
    # returns:    / ave_jmp_bef_1, jmp_btw_1, ave_jmp_aft_1 \
    # new_array = | ave_jmp_bef_2, jmp_btw_2, ave_jmp_aft_2 |
    #             \ ...          , ...      , ...           /
    ave_jmp_bef_list = np.array([]) # average jump before
    ave_jmp_aft_list = np.array([]) # average jump after
    jmp_btw_list = np.array([]) # jump between
    for i in range(0, len(array)):
        ave_jmp_bef_i = (array[i, 0, -1] - array[i, 0, 0]) / (n_points - 1)
        ave_jmp_aft_i = (array[i, 1, -1] - array[i, 1, 0]) / (n_points - 1)
        jmp_btw_i = array[i, 1, 0] - array[i, 0, -1]
        
        ave_jmp_bef_list = np.append(ave_jmp_bef_list, ave_jmp_bef_i)
        ave_jmp_aft_list = np.append(ave_jmp_aft_list, ave_jmp_aft_i)
        jmp_btw_list = np.append(jmp_btw_list, jmp_btw_i)
    new_array = np.vstack((ave_jmp_bef_list, jmp_btw_list, ave_jmp_aft_list)).T
    return(new_array)
#%%
def check_jumps_adv_new(data_list, n_days, n_points):
    # data_list = [data_1, data_2, ...]
    # n_days: number of days
    # n_points: indicates over how many datapoints before and after should be averaged
    # create  / / y_0_-n, y_0_-1 \ / y_1_-n, y_1_-1 \      \
    # array = \ \ y_1_0 , y_1_n  /,\ y_2_0 , y_2_n  /, ... /
    n_data = len(data_list)
    mjd_data = []
    jumps_data = []
    for i in range(0, n_data):
        data_i = data_list[i]
        mjd = np.array([])
        array = np.zeros((1, 2, 2)) # to initialize (will be deleted after)
        for j in range(1, n_days):
            # array_modifier goes t next good number -> [0.998, 0.999, 1.000] -> take [-n_points - 1:]
            data_bef = array_modifier(data_i, j - 1, 1)[-n_points - 1 : -1]
            data_aft = array_modifier(data_i, j, 1)[:n_points]
            
            mjd_bef = data_bef[-1, 0]
            mjd_aft = data_aft[0, 0]
            mjd_bet = (mjd_aft + mjd_bef) / 2
            mjd = np.append(mjd, mjd_bet)
            
            y_bef = data_bef[:, 1]
            y_aft = data_aft[:, 1]
            # only want y_i_-n and y_i_-1 resp. y_i_0 and y_i_n
            y_bef = np.array([y_bef[0], y_bef[-1]])
            y_aft = np.array([y_aft[0], y_aft[-1]])
            # stack data
            y_both = np.vstack((y_bef, y_aft))
            y_both = np.array([y_both]) # reshape to (1, 2, n_points)
            array = np.vstack((array, y_both))
        array = array[1:]
        mjd_data.append(mjd)
        jumps = jumps_array(array, n_points)
        jumps_data.append(jumps)
    return(mjd_data, jumps_data)
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
#%%
def file_extreme_day_new(foldername, file_type):
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
        if (file_i[-4:] == '.LST'):
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
    
    os.mkdir(foldername + "/year_" + file_str)
    
    string_list = np.array(["X_IN", "Y_IN", "Z_IN", "X_EF", "Y_EF", "Z_EF", "VX_IN", "VY_IN", "VZ_IN",
                            "r", "lat_sph", "lon_sph", "lat_ell", "lon_ell", "h_ell",
                            "u_sat", "beta_sun", "u_sun",
                            "a", "e", "i", "Omega_upp", "omega_low", "T0",
                            "rho", "air_x", "air_y", "air_z",
                            "VX_EF", "VY_EF", "VZ_EF", "air_xyz", "VXYZ_EF"])
    
    # find skiprows
    file = open(foldername + '/' + entries_new[0], 'r')
    lines = file.readlines()
    file.close()
    c = 0
    while (c < 101):
        row = lines[c].strip()
        if (row[:3] == 'SVN'):
            break
        if (c == 100):
            print("!!! ERROR 1 !!!")
            break
        c += 1
    
    c += 1
    
    data_year = np.loadtxt(foldername + '/' + entries_new[0], skiprows = c)
    for k in range(1, len(entries_new)):
        data_k = np.loadtxt(foldername + '/' + entries_new[k], skiprows = c)
        data_year = np.vstack((data_year, data_k))
    
    data_name = "year_" + file_str
    
    MJD_0 = data_year[0, 1] # [d]
    MJD = np.insert(data_year[:, 1] - MJD_0, 0, MJD_0) # [d]
    
    X_IN = data_year[:, 2] # [m]
    Y_IN = data_year[:, 3] # [m]
    Z_IN = data_year[:, 4] # [m]
    
    X_EF = data_year[:, 8] # [m]
    Y_EF = data_year[:, 9] # [m]
    Z_EF = data_year[:, 10] # [m]
    
    VX_IN = data_year[:, 5] # [m/s]
    VY_IN = data_year[:, 6] # [m/s]
    VZ_IN = data_year[:, 7] # [m/s]
    
    r = data_year[:, 11] # [m]
    lat_sph = data_year[:, 12] # [deg]
    lon_sph = data_year[:, 13] # [deg]
    
    lat_ell  = data_year[:, 14] # [deg]
    lon_ell = data_year[:, 15] # [deg]
    h_ell = data_year[:, 16] # [m]
    
    u_sat = data_year[:, 17] # [deg]
    beta_sun = data_year[:, 18] # [deg]
    u_sun = data_year[:, 19] # [deg]
    
    a_list = data_year[:, 20] # a [m]
    e_list = data_year[:, 21] # e [-]
    i_list = data_year[:, 22] # i [deg]
    Ω_list = data_year[:, 23] # Ω [deg]
    ω_list = data_year[:, 24] # ω [deg]
    T0_list = data_year[:, 25] # T0 [s]
    
    rho = data_year[:, 26] # air density [kg/m^3]
    air_x = data_year[:, 27] # air Aerodynamic acceleration in x [nm/s^2]
    air_y = data_year[:, 28] # air Aerodynamic acceleration in y [nm/s^2]
    air_z = data_year[:, 29] # air Aerodynamic acceleration in z [nm/s^2]
    air_xyz = data_year[:, 27 : 27 + 3] # air Aerodynamic acceleration in xyz [nm/s^2]
    
    VX_EF = data_year[:, 30] # [m/s]
    VY_EF = data_year[:, 31] # [m/s]
    VZ_EF = data_year[:, 32] # [m/s]
    VXYZ_EF = data_year[:, 30 : 30 + 3] # [m/s]
    
    tot_list = [X_IN, Y_IN, Z_IN, X_EF, Y_EF, Z_EF, VX_IN, VY_IN, VZ_IN,
                r, lat_sph, lon_sph, lat_ell, lon_ell, h_ell,
                u_sat, beta_sun, u_sun,
                a_list, e_list, i_list, Ω_list, ω_list, T0_list,
                rho, air_x, air_y, air_z, VX_EF, VY_EF, VZ_EF,
                air_xyz, VXYZ_EF]
    
    for l in range(0, len(string_list)):
        name = foldername + "/" + data_name + "/" + data_name + "_" + string_list[l] + ".txt"
        head = "MJD[0] = MJD_0 | " + string_list[l]
        array = 0
        if (len(tot_list[l].T) == 3):
            cols = np.vstack((np.zeros(3), tot_list[l]))
            array = np.vstack((MJD, cols.T)).T
        else:
            array = np.vstack((MJD, np.insert(tot_list[l].T, 0, 0))).T
        np.savetxt(name, array, delimiter = ' ', header = head, comments = '')
        print("generated file: " + name)
    print("DDDDOOOONNNNEEEE")

#file_extreme_day_new('hyperdata/hyperdata3', 2)
#%%
def convert_str_double_to_float(string):
    if (string[0] == ' '):
        new_string = string[1:]
    else:
        new_string = string
    return(float(new_string.replace("D", "E")))
#%%
def step_data_generation(array, fac):
    print("step_data_generation version 2")
    if (fac == 1):
        return(array)
    MJD = array[:, 0]
    acc = array[:, 1:]
    
    step = MJD[1] - MJD[0]
    h = step / fac
    
    MJD_new = np.array([])
    acc_new = acc[0]
    
    for i in range(0, len(array)):
        MJD_i = MJD[i]
        MJD_i_array = np.linspace(MJD_i, MJD_i + step, fac + 1)[: -1]
        acc_i_array = np.tile(acc[i], (fac, 1))
        
        MJD_new = np.append(MJD_new, MJD_i_array)
        acc_new = np.vstack((acc_new, acc_i_array))
    array_new = np.hstack((np.array([MJD_new]).T, acc_new[1:]))
    return(array_new)
#%%
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
#%%
def ele_gen_txt(foldername, file_type):
    # file_type = 1 for normal, file_type = 2 for nongra
    file_str = 0
    ng = 0
    if (file_type == 1):
        file_str = 'normal'
        ng = 0
    elif (file_type == 2):
        file_str = 'nongra'
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
    
    name_year = foldername + "/year_" + file_str + "/year_" + file_str + ".txt"
    
    np.savetxt(name_year, data_year, delimiter = ' ')
    
    print("DDDDOOOONNNNEEEE")

#ele_gen_txt('All Data new/elegrace', 2)
#%%
def yyyymmdd_to_mjd(date_list):
    # date_list = [yyyy, mm, dd]
    yyyy = '%.0f' % date_list[0]
    mm = '%.0f' % date_list[1]
    dd = '%.0f' % date_list[2]
    date_str = yyyy + '-' + mm + '-' + dd
    t_obj = Time(date_str)
    return(t_obj.mjd)

def flx_file(foldername, filename, MJD_start, MJD_end):
    # foldername: string like
    # filename: string like
    data = np.loadtxt(foldername + "/" + filename, skiprows = 6)
    
    start = 0
    for i in range(0, len(data)):
        if (data[i, 0] == MJD_start):
            start = i
            break
    end = 0
    for i in range(start, len(data)):
        if (data[i, 0] == MJD_end):
            end = i
            break
    
    data = data[start : end]
    
    MJD = data[:, 0]
    MJD_0 = MJD[0]
    MJD -= MJD_0
    
    FLX0 = data[:, 4]
    FLXM = data[:, 5]
    ap_day = data[:, 6]
    ap_i = data[:, 7:]
    
    fig = plt.figure(figsize = (10, 5), dpi = 300)
    plt.title(r'Solar flux', fontsize = 20)
    plt.plot(MJD, ap_day, 'k-', lw = 1, label = r'$ap_{day}$')
    plt.xlabel(r'MJD ' + str(MJD_0) + ' + t [d]', fontsize = 15)
    plt.ylabel(r'$ap_{day}$', fontsize = 15)
    if (MJD_0 <= 60002 and MJD_end >= 60002):
        plt.axvline(60002 - MJD_0, color = 'gold',
                    ls = (0, (4, 4)), lw = 2.5, zorder = 1)
    if (MJD_0 <= 60027 and MJD_end >= 60027):
        plt.axvline(60027 - MJD_0, color = 'gold',
                    ls = (0, (4, 4)), lw = 2.5, zorder = 1)
    if (MJD_0 <= 60058 and MJD_end >= 60058):
        plt.axvline(60058 - MJD_0, color = 'gold',
                    ls = (0, (4, 4)), lw = 2.5, zorder = 1)
    if (MJD_0 <= 58356 and MJD_end >= 58356):
        plt.axvline(58356 - MJD_0, color = 'gold',
                    ls = (0, (4, 4)), lw = 2.5, zorder = 1)
    plt.legend()
    plt.grid()
    plt.show(fig)
    plt.close(fig)

#flx_file('All Data new', 'FLXAP_P_MJD.FLX', 59945, 60309)

def flx_get_data(path, MJD_0, n_days_tot):
    data = np.loadtxt(path, skiprows = 6)
    
    data = array_modifier(data, MJD_0, n_days_tot)
    data = array_normalize(data, 0)
    data = data[1:]
    data_mjd = data[:, 0]
    
    data_flx_apday = data[:, 3 : 6]
    data_flx_apday = np.vstack((data_mjd, data_flx_apday.T)).T
    
    data_ap = data[:, 6:]
    data_ap = np.vstack((data_mjd, data_ap.T)).T
    
    return(data_flx_apday, data_ap)

def flx_get_data_ymd(path):
    # get data from flx file that is in yyyy mm dd format
    data = np.loadtxt(path, skiprows = 6)
    
    date_list = data[:, :3]
    mjd_list = np.array([])
    for i in range(0, len(date_list)):
        date = date_list[i]
        mjd = yyyymmdd_to_mjd(date)
        mjd_list = np.append(mjd_list, mjd)
    mjd_list = np.array([mjd_list]).T
    
    data_flx_apday = data[:, 3 : 6]
    data_ap = data[:, 6:]
    
    data_flx_apday = np.hstack((mjd_list, data_flx_apday))
    data_ap = np.hstack((mjd_list, data_ap))
    return(data_flx_apday, data_ap)
#%%
#########################################################################
########################## fit function start ###########################
#########################################################################

######################### fit func 1 ####################################
def fit_func_1(t, para, para_fit):
    # t: epoch
    # t0: epoch where trend "beginns"
    # h: offset at t = t0
    # s: trend, linear slope
    # p_list = [p_1, ..., p_N]
    # a_list = [a_1, ..., a_N]
    # ϕ_list = [ϕ_1, ..., ϕ_N]
    # p_n: a period in the set of dominant periods
    # a_n: amplitude to the corresponding period p_n
    # ϕ_n: phase shift of the corresponding period p_n
    # N: number of dominant periods
    t0, p_list = para[0], para[1]
    h, s = para_fit[0],  para_fit[1]
    a_list, ϕ_list = para_fit[2], para_fit[3]
    wave_sum = 0
    for n in range(0, len(p_list)):
        p_n = p_list[n]
        a_n = a_list[n]
        ϕ_n = ϕ_list[n]
        wave_sum += a_n * np.sin(2 * np.pi * (t - t0) / p_n + ϕ_n)
    ele = h + s * (t - t0) + wave_sum
    return(ele)

def fit_func_1_pd_h(t, para, para_fit):
    t0, p_list = para[0], para[1]
    h, s = para_fit[0],  para_fit[1]
    a_list, ϕ_list = para_fit[2], para_fit[3]
    return(np.array([1]))

def fit_func_1_pd_s(t, para, para_fit):
    t0, p_list = para[0], para[1]
    h, s = para_fit[0],  para_fit[1]
    a_list, ϕ_list = para_fit[2], para_fit[3]
    return(np.array([t - t0]))

def fit_func_1_pd_a(t, para, para_fit):
    t0, p_list = para[0], para[1]
    h, s = para_fit[0],  para_fit[1]
    a_list, ϕ_list = para_fit[2], para_fit[3]
    pd_amp_list = np.array([])
    for n in range(0, len(p_list)):
        p_n = p_list[n]
        ϕ_n = ϕ_list[n]
        pd_a_n = np.sin(2 * np.pi * (t - t0) / p_n + ϕ_n)
        pd_amp_list = np.append(pd_amp_list, pd_a_n)
    return(pd_amp_list)

def fit_func_1_pd_ϕ(t, para, para_fit):
    t0, p_list = para[0], para[1]
    h, s = para_fit[0],  para_fit[1]
    a_list, ϕ_list = para_fit[2], para_fit[3]
    pd_phase_list = np.array([])
    for n in range(0, len(p_list)):
        p_n = p_list[n]
        a_n = a_list[n]
        ϕ_n = ϕ_list[n]
        pd_ϕ_n = a_n * np.cos(2 * np.pi * (t - t0) / p_n + ϕ_n)
        pd_phase_list = np.append(pd_phase_list, pd_ϕ_n)
    return(pd_phase_list)

fit_func_1_derivs = [fit_func_1_pd_h, fit_func_1_pd_s, fit_func_1_pd_a, fit_func_1_pd_ϕ]

######################### fit func 2 ####################################
def fit_func_2(t, para, para_fit):
    # t: epoch
    # t0: epoch where trend "beginns"
    # h: offset at t = t0
    # s: trend, linear slope
    # p_list = [p_1, ..., p_N]
    # a_list = [a_1, ..., a_N]
    # ϕ_list = [ϕ_1, ..., ϕ_N]
    # p_n: a period in the set of dominant periods
    # a_n: amplitude to the corresponding period p_n
    # ϕ_n: phase shift of the corresponding period p_n
    # N: number of dominant periods
    t0 = para[0]
    h, s, p_list = para_fit[0], para_fit[1], para_fit[2]
    a_list, ϕ_list = para_fit[3], para_fit[4]
    wave_sum = 0
    for n in range(0, len(p_list)):
        p_n = p_list[n]
        a_n = a_list[n]
        ϕ_n = ϕ_list[n]
        wave_sum += a_n * np.sin(2 * np.pi * (t - t0) / p_n + ϕ_n)
    ele = h + s * (t - t0) + wave_sum
    return(ele)

def fit_func_2_pd_h(t, para, para_fit):
    t0 = para[0]
    h, s, p_list = para_fit[0], para_fit[1], para_fit[2]
    a_list, ϕ_list = para_fit[3], para_fit[4]
    return(np.array([1]))

def fit_func_2_pd_s(t, para, para_fit):
    t0 = para[0]
    h, s, p_list = para_fit[0], para_fit[1], para_fit[2]
    a_list, ϕ_list = para_fit[3], para_fit[4]
    return(np.array([t - t0]))

def fit_func_2_pd_p(t, para, para_fit):
    t0 = para[0]
    h, s, p_list = para_fit[0], para_fit[1], para_fit[2]
    a_list, ϕ_list = para_fit[3], para_fit[4]
    pd_period_list = np.array([])
    for n in range(0, len(p_list)):
        p_n = p_list[n]
        a_n = a_list[n]
        ϕ_n = ϕ_list[n]
        arg = 2 * np.pi * (t - t0) / p_n
        pd_p_n = - arg / p_n * a_n * np.cos(arg + ϕ_n)
        pd_period_list = np.append(pd_period_list, pd_p_n)
    return(pd_period_list)

def fit_func_2_pd_a(t, para, para_fit):
    t0 = para[0]
    h, s, p_list = para_fit[0], para_fit[1], para_fit[2]
    a_list, ϕ_list = para_fit[3], para_fit[4]
    pd_amp_list = np.array([])
    for n in range(0, len(p_list)):
        p_n = p_list[n]
        ϕ_n = ϕ_list[n]
        pd_a_n = np.sin(2 * np.pi * (t - t0) / p_n + ϕ_n)
        pd_amp_list = np.append(pd_amp_list, pd_a_n)
    return(pd_amp_list)

def fit_func_2_pd_ϕ(t, para, para_fit):
    t0 = para[0]
    h, s, p_list = para_fit[0], para_fit[1], para_fit[2]
    a_list, ϕ_list = para_fit[3], para_fit[4]
    pd_phase_list = np.array([])
    for n in range(0, len(p_list)):
        p_n = p_list[n]
        a_n = a_list[n]
        ϕ_n = ϕ_list[n]
        pd_ϕ_n = a_n * np.cos(2 * np.pi * (t - t0) / p_n + ϕ_n)
        pd_phase_list = np.append(pd_phase_list, pd_ϕ_n)
    return(pd_phase_list)

fit_func_2_derivs = [fit_func_2_pd_h, fit_func_2_pd_s, fit_func_2_pd_p,
                     fit_func_2_pd_a, fit_func_2_pd_ϕ]

######################### fit func 3 ####################################
def fit_func_3(t, para, para_fit):
    # t: epoch
    # t0: epoch where trend "beginns"
    # h: offset at t = t0
    # s: trend, linear slope
    t0 = para[0]
    h, s = para_fit[0], para_fit[1]
    ele = h + s * (t - t0)
    return(ele)

def fit_func_3_pd_h(t, para, para_fit):
    t0 = para[0]
    h, s = para_fit[0], para_fit[1]
    return(np.array([1]))

def fit_func_3_pd_s(t, para, para_fit):
    t0 = para[0]
    h, s = para_fit[0], para_fit[1]
    return(np.array([t - t0]))

fit_func_3_derivs = [fit_func_3_pd_h, fit_func_3_pd_s]

########################### lsa fit ######################################
def n_objects(liste):
    # liste = [1,2,[3,4],[5,6]] -> how many parameters?
    n = 0
    for i in range(0, len(liste)):
        obj_i = liste[i]
        if (type(obj_i) == list or type(obj_i) == np.ndarray):
            n += len(obj_i)
        else: # here we do not consider a deeper structure
            n += 1
    return(n)

def A(t_el_array, func_derivs, para, para_fit):
    #                    / t1, el(t1) \
    # t_el_array (data): | .., el(..) |
    #                    \ tK, el(tK) /
    # p_list = [p_1, ..., p_N]
    # a_list = [a_1, ..., a_N]
    # ϕ_list = [ϕ_1, ..., ϕ_N]
    #     / pd_h(t=t1), pd_s(t=t1), pd_a1(t=t1),...,pd_aN(t=t1), pd_ϕ1(t=t1),...,pd_ϕN(t=t1) \
    # A = | pd_h(t=..), pd_s(t=..), pd_a1(t=..),...,pd_aN(t=..), pd_ϕ1(t=..),...,pd_ϕN(t=..) |
    #     \ pd_h(t=tK), pd_s(t=tK), pd_a1(t=tK),...,pd_aN(t=tK), pd_ϕ1(t=tK),...,pd_ϕN(t=tK) /
    n_cols = n_objects(para_fit)
    n_derivs = len(func_derivs)
    A_mat = np.zeros(n_cols) # pd_h + pd_s + pd_a + pd_ϕ
    for k in range(0, len(t_el_array)):
        t_k = t_el_array[k, 0]
        row_k = np.array([])
        for n in range(0, n_derivs):
            row_k = np.hstack((row_k, func_derivs[n](t_k, para, para_fit)))
        A_mat = np.vstack((A_mat, row_k))
    A_mat = A_mat[1:]
    return(A_mat)

def P(σ0, σ_list): # Weight matrix P
    diag_list = np.array([])
    for i in range(0, len(σ_list)):
        diag_list = np.append(diag_list, (σ0**2) / (σ_list[i]**2))
    P_mat = np.diag(diag_list)
    return(P_mat)

def O_minus_C(t_el_array, func, para, para_fit):
    el_vec_observed = t_el_array[:, 1]
    el_vec_computed = np.array([])
    for k in range(0, len(t_el_array)):
        t_k = t_el_array[k, 0]
        el_k = func(t_k, para, para_fit)
        el_vec_computed = np.append(el_vec_computed, el_k)
    O_minus_C_vec = np.array([el_vec_observed - el_vec_computed]).T
    return(O_minus_C_vec)

def Δx(P_mat, A_mat, Δl_vec): # x = x0 + Δx
    inv = A_mat.T @ P_mat @ A_mat
    det = np.linalg.det(inv)
    if (det == 0):
        print("WARNING: DET = 0")
        print("A_mat:")
        print(A_mat)
        print("P_mat:")
        print(P_mat)
    term1 = np.linalg.inv(inv)
    term2 = A_mat.T @ P_mat @ Δl_vec
    Δx_vec = term1 @ term2
    return(Δx_vec)

def quotient(x_vec, Δx_vec): # calculation of relative improvement
    # x_vec = [h, s, a_1, .., a_N, ϕ_1, ..., ϕ_N].T
    # Δx_vec = [Δh, Δs, Δa_1, .., Δa_N, Δϕ_1, ..., Δϕ_N].T
    quotient_list = np.array([])
    for j in range(0, len(x_vec)):
        if (x_vec[j] != 0):
            quotient = abs(Δx_vec[j] / x_vec[j])
        else:
            quotient = 1
        quotient_list = np.append(quotient_list, quotient)
    return(quotient_list)

def m0(A_mat, P_mat, Δx_vec, Δl_vec): # calculation of m0
    n = len(A_mat)
    u = len(A_mat.T)
    v = A_mat @ Δx_vec - Δl_vec
    m0 = np.sqrt((v.T @ P_mat @ v) / (n - u))
    return(m0)

def list_code(liste):
    # [a, b, [c, d], [e,f,g]] -> [0, 0, 2, 3]
    code_list = []
    for i in range(0, len(liste)):
        obj_i = liste[i]
        if (type(obj_i) == list or type(obj_i) == np.ndarray):
            code_list.append(len(obj_i))
        else:
            code_list.append(0)
    return(code_list)

def encode_list(liste, code_list):
    # liste = [a, b, c, d, e, f, g]
    # code_list = [0, 0, 2, 3]
    # wanted: [a, b, [c, d], [e,f,g]]
    new_list = []
    s = 0
    for i in range(0, len(code_list)):
        n_objects = code_list[i]
        if (n_objects == 0):
            new_list.append(liste[s])
            s += 1
        else:
            new_list.append(liste[s : s + n_objects])
            s += n_objects
    return(new_list)

def ravel(list_liray):
    # list_liray = [1, 2, np.array([3,4]), np.array([5,6,7])]
    # returns: liste = [1, 2, 3, 4, 5, 6, 7]
    liste = []
    for i in range(0, len(list_liray)):
        obj_i = list_liray[i]
        if (type(obj_i) == list or type(obj_i) == np.ndarray):
            liste.extend(obj_i) # we do not consider a deeper structure
        else:
            liste.append(obj_i)
    return(liste)

def round_adv(x, n):
    # x: float
    # n: number of significant digits to round
    x_str = str(x)
    new_x = 0
    exp = 0
    if (len(x_str) >= 4):
        if (x_str[-4] == 'e'):
            exp = int(x_str[-3:])
            new_x = x * 10**(-exp)
            new_x = round(new_x, n - 1)
            new_x = float(str(new_x) + 'e' + str(exp))
        else:
            new_x = round(x, n-1)
    else:
        new_x = round(x, n-1)
    return(new_x)

def round_list(liste, n):
    code_list = list_code(liste)
    new_list = np.array([])
    for i in range(0, len(liste)):
        obj_i = liste[i]
        type_i = type(obj_i)
        if (type_i == list or type_i == np.ndarray):
            for j in range(0, len(obj_i)):
                sub_obj_i_j = round_adv(obj_i[j], n)
                new_list = np.append(new_list, sub_obj_i_j)
        else:
            new_list = np.append(new_list, round_adv(obj_i, n))
    new_list = encode_list(new_list, code_list)
    return(new_list)

def lst_sqrs_adjstmnt(ε, t_el_array, func, func_derivs, para, para_fit_0): # least squares adjustment
    # ε: precision
    # t_el_array: [t_k, el_k]
    # func: functional model
    # func_derivs: derivatives of functional model
    # para: given parameter (will not be fitted)
    # para_fit_0: a priori values for parameters that should be fitted
    para_fit = para_fit_0
    n = len(para_fit)
    para_fit_code = list_code(para_fit)
    σ0 = 1
    σ_list = np.ones(len(t_el_array))
    P_mat = P(σ0, σ_list)
    counter = 0 # to count and to avoid accidental infinite loop
    quotient_list = np.array([1])
    m_0 = 10
    while (max(quotient_list) > ε) and (counter < 101):
        A_mat = A(t_el_array, func_derivs, para, para_fit)
        
        x_vec = []
        for i in range(0, n):
            obj_i = para_fit[i]
            typ = type(obj_i)
            if (typ == int or typ == float or typ == np.float64):
                x_vec.append(obj_i)
            elif (typ == list or typ == np.ndarray):
                x_vec.extend(obj_i)
            else:
                print("!!!ERROR!!!\n!!!ERROR!!!\n!!!ERROR!!!\n!!!ERROR!!!\n!!!ERROR!!!\n")
        
        x_vec = np.array([x_vec]).T
        Δl_vec = O_minus_C(t_el_array, func, para, para_fit)
        Δx_vec = Δx(P_mat, A_mat, Δl_vec)
        m_0 = m0(A_mat, P_mat, Δx_vec, Δl_vec)
        
        Kxx = m_0**2 * np.linalg.inv(A_mat.T @ P_mat @ A_mat)
        e_para_fit = np.array([])
        for i in range(0, len(Kxx)):
            e_para_fit = np.append(e_para_fit, np.sqrt(Kxx[i, i]))
        
        quotient_list = quotient(x_vec, Δx_vec)
        x_vec = x_vec + Δx_vec
        para_fit = x_vec.T[0]
        
        para_fit = encode_list(para_fit, para_fit_code)
        e_para_fit = encode_list(e_para_fit, para_fit_code)
        
        counter += 1
        """
        print("t = %3d | Iteration step: %3d | m0 = " % (t_el_array[0, 0], counter), m_0)
        print("para_fit")
        print(round_list(para_fit, 3))
        print("e_para_fit")
        print(round_list(e_para_fit, 3))
        print("quotient_list")
        print(round_list(quotient_list, 3))
        print("------------------------------------------------------------")
        """
    print("t = %3d | Iteration step: %3d | m0 = " % (t_el_array[0, 0], counter), m_0)
    print("para_fit")
    print(round_list(para_fit, 6))
    print("e_para_fit")
    print(round_list(e_para_fit, 6))
    print("quotient_list")
    print(round_list(quotient_list, 3))
    print("------------------------------------------------------------")
    return([para_fit, e_para_fit, m_0])

#########################################################################
############################ fit function end ###########################
#########################################################################
#%%
#########################################################################
############################ filtering start ############################
#########################################################################

def A_filter(n, q):
    # n: window width
    n_2 = int((n - 1) / 2)
    n_list = list(range(-n_2, n_2 + 1))
    n_array = np.array([n_list]).T
    A_mat = np.array([[1] * len(n_list)]).T
    for i in range(1, q + 1):
        A_mat = np.block([A_mat, n_array**i])
    return(A_mat)

def B_filter(n, q):
    A_mat = A_filter(n, q)
    B_mat = np.linalg.inv(A_mat.T @ A_mat) @ A_mat.T
    return(B_mat)

def low_pass_filter(array, n, q):
    t_list = array[:, 0]
    y_list = array[:, 1]
    n_2 = int((n - 1) / 2)
    B_mat = B_filter(n, q)
    new_y_list = []
    for i in range(0, len(t_list[n_2 : len(t_list) - n_2])):
        y_array = np.array([y_list[n_2 + i - n_2 : n_2 + i + n_2 + 1]]).T
        x_hut = B_mat @ y_array
        new_y_list.append(x_hut[0][0])
    new_y_middle = np.array([new_y_list])
    # Boundary points: points should be located on the polynomial of order 0
    # used on the last possible point (see summary for closer details).
    B_mat_border, A_mat_border = B_filter(n, 0), A_filter(n, 0)
    y_border_1 = y_list[: 2 * n_2 + 1] # boundary points at the start
    y_border_2 = y_list[- 2 * n_2 - 1 :] # boundary points at the end
    x_hut_1 = B_mat_border @ y_border_1
    x_hut_2 = B_mat_border @ y_border_2
    new_y_border_1 = np.array([(A_mat_border @ x_hut_1)[:n_2]]) # take first half
    new_y_border_2 = np.array([(A_mat_border @ x_hut_2)[-n_2:]]) # take second half
    # put everything together
    new_y_array = np.block([new_y_border_1, new_y_middle, new_y_border_2]).T
    new_array = np.block([np.array([t_list]).T, new_y_array])
    return(new_array)

#########################################################################
############################## filtering end ############################
#########################################################################
#%%
#########################################################################
######################## fit function adv start #########################
#########################################################################

def fit_func_adv(t, para_lists_list, t_list, n):
    # para_lists_list = [s_list]
    # t_list = [t_n_1, t_n, Δt]
    t_n_1, t_n, Δt = t_list[0], t_list[1], t_list[2]
    
    s_list = para_lists_list[0]
    
    s_n_1, s_n = s_list[n - 1], s_list[n]
    
    fac1 = t_n - t
    fac2 = t - t_n_1
    
    obj = (fac1 * s_n_1 + fac2 * s_n) / Δt
    
    return(obj)

def A_adv(t_array, para_lists_list, Δt, n_s):
    # not dependend on para_array
    # t_array normalized
    # (n_s + 1) * 6 parameters
    
    s_pd_s_mat = np.zeros((len(t_array), n_s + 1))
    
    i = 1
    t_i_1 = 0
    t_i = Δt
    print("n_s = ", n_s)
    print("")
    for k in range(0, len(t_array)):
        t = t_array[k]
        if (t >= i * Δt):
            i += 1
            t_i = i * Δt
            t_i_1 = (i - 1) * Δt
        
        fac_i_1 = (t_i - t) / Δt
        fac_i = (t - t_i_1) / Δt
        print("len(s_pd_s_mat)", len(s_pd_s_mat))
        print("k = ", k)
        s_pd_s_mat[k][i - 1] = fac_i_1
        s_pd_s_mat[k][i] = fac_i
    
    A_mat = s_pd_s_mat
    return(A_mat)

def O_minus_C_adv(obs_array, para_lists_list, Δt, n_s):
    # obs_vec = [l_0, l_1, ..., l_ns, m_0, m_1, ..., m_ns]
    t_array = obs_array[:, 0]
    z_obs_list = obs_array[:, 1]
    obs_vec = np.array([z_obs_list]).T
    obj_list = np.array([])
    i = 1
    for k in range(0, len(t_array)):
        t = t_array[k]
        if (t >= i * Δt):
            i += 1
        t_list = [(i - 1) * Δt, i * Δt, Δt]
        obj = fit_func_adv(t, para_lists_list, t_list, i)
        obj_list = np.append(obj_list, obj)
    
    comp_vec = np.array([obj_list]).T
    
    O_minus_C_vec = obs_vec - comp_vec
    return(O_minus_C_vec)

def N_inv_co(P_mat, A_mat):
    inv = A_mat.T @ P_mat @ A_mat
    det = np.linalg.det(inv)
    if (det == 0):
        print("WARNING: DET = 0")
        print("A_mat:")
        print(A_mat)
        print("P_mat:")
        print(P_mat)
        print("A_mat.T @ P_mat @ A_mat:")
        print(inv)
    term1 = np.linalg.inv(inv)
    term2 = A_mat.T @ P_mat
    return(term1 @ term2)

def new_Δx(term1, Δl_vec):
    return(term1 @ Δl_vec)

def lst_sqrs_adjstmnt_adv(ε, obs_array, para_lists_list_0, Δt, n_s): # least squares adjustment
    # ε: precision
    # obs_array = [[t_i, x_i, y_i]]
    t_list = obs_array[:, 0]
    
    para_lists_list = para_lists_list_0
    length = len(para_lists_list_0)
    para_fit_code = list_code(para_lists_list)
    σ0 = 1
    #σ_list = np.ones(len(obs_array))
    #P_mat = P(σ0, σ_list)
    P_mat = np.diag(np.ones(len(obs_array)))
    counter = 0 # to count and to avoid accidental infinite loop
    quotient_list = np.array([1])
    m_0 = 10
    A_mat = A_adv(t_list, para_lists_list_0, Δt, n_s)
    half_Δx = N_inv_co(P_mat, A_mat)
    while (max(quotient_list) > ε) and (counter < 100):
        
        x_vec = []
        for i in range(0, length):
            obj_i = para_lists_list[i]
            typ = type(obj_i)
            if (typ == int or typ == float or typ == np.float64):
                x_vec.append(obj_i)
            elif (typ == list or typ == np.ndarray):
                x_vec.extend(obj_i)
            else:
                print("!!!ERROR!!!\n!!!ERROR!!!\n!!!ERROR!!!\n!!!ERROR!!!\n!!!ERROR!!!\n")
        
        x_vec = np.array([x_vec]).T
        Δl_vec = O_minus_C_adv(obs_array, para_lists_list, Δt, n_s)
        Δx_vec = new_Δx(half_Δx, Δl_vec)
        m_0 = m0(A_mat, P_mat, Δx_vec, Δl_vec)
        
        quotient_list = quotient(x_vec, Δx_vec)
        x_vec = x_vec + Δx_vec
        para_lists_list = x_vec.T[0]
        
        para_lists_list = encode_list(para_lists_list, para_fit_code)
        #e_para_fit = encode_list(e_para_fit, para_fit_code)
        
        counter += 1
        
        print("Iteration step: %3d | m0 = " % (counter), m_0)
        print("para_fit")
        print(round_list(para_lists_list, 3))
        #print("e_para_fit")
        #print(round_list(e_para_fit, 3))
        print("quotient_list")
        print(round_list(quotient_list, 3))
        print("------------------------------------------------------------")
    e_para_fit = encode_list(np.zeros(len(para_lists_list[0])), para_fit_code)
    return([para_lists_list, e_para_fit, m_0])

def get_n_s(obs_array, Δt):
    t_array = obs_array[:, 0]
    t_tot = t_array[-1] - t_array[0]
    n_s = int(np.ceil(t_tot / Δt))
    #para_lists_list_0 = 1 * [(n_s + 1) * [1]]
    return(n_s)

#########################################################################
######################### fit function adv end ##########################
#########################################################################
#%%
def arg_finder(array, el):
    pos = 0
    for i in range(0, len(array)):
        if (array[i] == el):
            pos = i
            return(pos)
    print("error, element not found!!!")
    return(0)
#%%
def dollar_remove(string):
    if (string[0] == '$'):
        return(string[1 : -1])
    else:
        return(string)
#%%
def fft_logplot_spec_3(p_o_list, a_o_list, data_spec_array,
                       per_list_list, amp_list_list, marker_spec_array,
                       Δt_list, Δt_spec_array, tit, unit, log,
                       log_fac_list, location,
                       vlines_list, vlines_specs, xlimits, ylimits,
                       save_specs):
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
    # unit: '[m]' or r'[$frac{\text{m}}{\text{d}}$]'
    # log: if 0 --> semilogx, if 1 --> loglog
    # log_fac_list -> to shift spectrum of p_o_2, p_o_3, etc
    # location: location of legend
    # vlines_list: helpful to determine interval_list in peak_finder
    # vlines_specs = [col, alpha, lw]
    # xlimits: for zooming
    # ylimits: for zooming
    # save_specs = [1, path] # 0,1 for saving
    print("fft_logplot_spec_3 version 7")
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
            plt.semilogx(p_o_list[i], new_a_o_list[i],
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
            plt.loglog(p_o_list[i], new_a_o_list[i],
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
        plt.scatter(per_list_list[i], new_amp_list_list[i],
                    color = col, s = size,
                    marker = mark,
                    fc = 'None', lw = 0.5,
                    alpha = 1, label = lab)
        for j in range(0, len(per_list_list[i])):
            str_lab = dollar_remove(text)
            string = r'$t_{xyz}^{(0)}$'.replace('0', str(j + 1)).replace('xyz', str_lab)
            plt.annotate(string, (per_list_list[i][j], new_amp_list_list[i][j]),
                         coords, textcoords = 'offset points',
                         fontsize = 10)
            string_add = ' = %.2e d' % per_list_list[i][j]
            plt.figtext(0.92, 0.875 - k, string + string_add,
                        fontsize = 10, ha = 'left')
            k += 0.05
        k += 0.03
    
    for i in range(0, n_Δt):
        lab1 = r'$Δt_{xyz}$ = '.replace('xyz', dollar_remove(Δt_spec_array[i][2]))
        lab2 = str(np.round(Δt_list[i], 3)) + ' d'
        plt.axvline(Δt_list[i], color = Δt_spec_array[i][1], alpha = 0.5,
                    ls = Δt_spec_array[i][0], lw = 5,
                    label = lab1 + lab2)
    
    for i in range(0, len(vlines_list)):
        vline = vlines_list[i]
        plt.axvline(vline, color = vlines_specs[0],
                    alpha = vlines_specs[1], lw = vlines_specs[2],
                    ls = 'solid')
    
    if (xlimits[0] != xlimits[1]):
        plt.xlim(xlimits[0], xlimits[1])
    
    if (ylimits[0] != ylimits[1]):
        plt.ylim(ylimits[0], ylimits[1])
    
    plt.xlabel(r'period [d]', fontsize = 15)
    plt.ylabel(r'amplitude ' + unit, fontsize = 15)
    plt.legend(fontsize = 12.5, loc = location)
    plt.grid()
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)
#%%
def insert(liste, liste_array):
    # liste = list that should be extended by liste_array
    # liste_array = list or array
    liste.insert(len(liste), liste_array)
    return(liste)
#%%
def spectrum(data, el_num, lcm_list, interval_list):
    # INPUT
    #    data: [t_i, el_i]
    #    el_num: # of element that gets smoothed (Omega_upp is problematic)
    #    lcm_list: contains parameters that specify the lcm
    #    interval_list: list of intervals where additional peaks should be detected
    # OUTPUT
    #    spectrum_list = [p_o_list, a_o_list, per_list, amp_list]
    #        p_o_list = periods of fft
    #        a_o_list = amplitudes of fft
    #        per_list = periods that have a peak amplitude
    #        amp_list = peak amplitudes sorted from highest to lowest
    p_spec, prec, N = lcm_list[0], lcm_list[1], lcm_list[2]
    p_max, thresh_quot, limit = lcm_list[3], lcm_list[4], lcm_list[5]
    
    orig_list = fft(data) # spectrum
    p_o_list, a_o_list = orig_list[0], orig_list[1] # period and amplitude
    # pos_list = list containing indices (wrt p_o_list) of periods in p_o_list that are peaks
    # max_list = amplitudes of peaks
    pos_list, max_list = peak_finder_adv(p_o_list, a_o_list, thresh_quot,
                                         el_num, p_max, interval_list)
    # p_o_pos_list = list containing periods of peaks
    per_list = selector(p_o_list, pos_list) # for marking peaks in spectrum
    if (len(per_list) == 1): # [[period, amplitude]]
        per_amp_array = np.array([per_list, max_list]).T
    else: # sort from highest amplitude to lowest amplitude
        per_amp_array = sorter(np.vstack((per_list, max_list)).T, 1, -1)
    per_list = per_amp_array[:, 0]
    amp_list = per_amp_array[:, 1]
    spectrum_list = [p_o_list, a_o_list, per_list, amp_list]
    return(spectrum_list)
#%%
def smoother(data, per_list, amp_list, lcm_list, el_num, q):
    # INPUT
    #    data: [t_i, el_i]
    #    per_list: periods sorted from lowest to highest
    #    amp_list: corresponding amplitudes
    #    lcm_list: contains parameters that specify the lcm
    #    el_num: # of element that gets smoothed (Omega_upp is problematic)
    #    q: polynomial order for filtering
    # OUTPUT
    #    smooth_list = [data_smoothed, per0_list, amp0_list, n_Δt, Δt_n, Δt_dt]
    #        data_smoothed: data smoothed with Savizky-Golay filter
    #        per0_list: periods regarded when smoothed
    #        amp0_list: corresponding amplitudes
    #        n_Δt: width of smoothing window (uneven)
    #        Δt_n: smoothing period
    #        Δt_dt: Δt % dt ("error")
    p_spec, prec, N = lcm_list[0], lcm_list[1], lcm_list[2]
    p_max, thresh_quot, limit = lcm_list[3], lcm_list[4], lcm_list[5]
    # sort from strongest to weakest period:
    per_amp_array = sorter(np.vstack((per_list, amp_list)).T, 1, -1)
    per_list = per_amp_array[:, 0]
    amp_list = per_amp_array[:, 1]
    # evaluate optimal window width
    dt = data[1, 0] - data[0, 0]
    print("p_spec", p_spec)
    if (p_spec != 0):
        Δt = p_spec
        per0_list , amp0_list = [], []
    elif (el_num == 6):
        Δt = 0.1
        per0_list , amp0_list = [], []
    else:
        Δt, per0_list = lcm_as_Δt(prec, N, limit, dt, per_list)
        amp0_list = amp_list[:len(per0_list)]
    print("Δt", Δt)
    n_Δt = round_uneven(Δt / dt) # window width for filtering
    Δt_n = n_Δt * dt # corresponding time interval for filtering
    Δt_dt = Δt_n % dt # "error"
    print("Δt_n = %.3f | n_Δt = %d | len = %d" % (Δt_n, n_Δt, len(data)))
    data_smoothed = low_pass_filter(data, n_Δt, q)
    print("SMOOTHING FINISHED!!!")
    smooth_list = [data_smoothed, per0_list, amp0_list, n_Δt, Δt_n, Δt_dt]
    return(smooth_list)
#%%
def fitter(data, per0_list, amp0_list, n_partition, lsa_method, ε):
    # INPUT
    #    data: [t_i, el_i] to be fitted
    #    per0_list: periods that will be used in fit (fixed or as guesses)
    #    amp0_list: corresponding amplitudes (as guesses)
    #    n_partition: duration of intervals for creating fits
    #    lsa_method: [1 -> t0, per0 fixed, 2 -> t0 fixed]
    #    ε: accuracy of fit
    # OUTPUT
    #    fit_list = [data_fitted, para_fit_array, para_e_fit_array, m0_list]
    #        data_fitted: fitted data (stacked)
    #        para_fit_array: fitted parameters stacked
    #        para_e_fit_array: error of fitted parameters stacked
    #        m0_list: a posteriori errors for each fit
    #        τ_fit_list: τ for each fitted parameter
    # create subdata
    start, end = data[0, 0], data[-1, 0]
    subdata_list = []
    while (start <= end):
        subdata = array_modifier(data, start, n_partition)
        # normalizing ???
        subdata_list.append(subdata)
        start += n_partition
    
    if (lsa_method == 1): # give t0 and periods; fit h, s, amplitudes and phases
        para = insert([0], per0_list) # given parameters
        para_fit_0 = insert([0, 0], amp0_list) # first guesses for parmeters
        para_fit_0 = insert(para_fit_0, np.zeros(len(per0_list))) # to be fitted
        fit_func = fit_func_1
        fit_func_derivs = fit_func_1_derivs
    elif (lsa_method == 2): # give t0; fit h, s, periods, amplitudes and phases
        para = [0] # given parameters
        para_fit_0 = insert([0, 0], per0_list[:, 0]) # first guesses for parmeters
        para_fit_0 = insert(para_fit_0, amp0_list[:, 1]) # to be fitted
        para_fit_0 = insert(para_fit_0, np.zeros(len(per0_list)))
        fit_func = fit_func_2
        fit_func_derivs = fit_func_2_derivs
    elif (lsa_method == 3): # give t0; fit h and s (linear regression)
        para = [0] # given parameters
        para_fit_0 = [0, 0] # first guesses for parmeters to be fitted
        fit_func = fit_func_3
        fit_func_derivs = fit_func_3_derivs
    elif (lsa_method == 4):
        t_array = data[:, 0]
        dt = t_array[1] - t_array[0]
        Δt = n_partition * dt
        n_s = get_n_s(data, n_partition)
        para_lists_list_0 = 1 * [(n_s + 1) * [1]]
        lsa_list = lst_sqrs_adjstmnt_adv(ε, data, para_lists_list_0, Δt, n_s)
        para_fit, e_para_fit, m_0 = lsa_list[0], lsa_list[1], lsa_list[2]
        τ_fit_list = np.append(Δt * np.arange(n_s - 1), data[-1, 0])
        obj_list = np.array([])
        n = 1
        for i in range(0, len(t_array)):
            t_i = t_array[i]
            if (t_i >= n * Δt):
                n += 1
            t_list = [(n - 1) * Δt, n * Δt, Δt]
            obj = fit_func_adv(t_i, para_fit, t_list, n)
            obj_list = np.append(obj_list, obj)
        data_fitted = np.vstack((t_array, obj_list))
        #fit_list = [data_fitted, para_fit, e_para_fit, m0_list, τ_fit_list]
        fit_list = [data_fitted, para_fit, e_para_fit, τ_fit_list]
        return(fit_list)
    else:
        print("ERROR!!! SPECIFY LSA-METHOD!!!")
        return(1)
    
    data_fitted = np.zeros((1, 2)) # to be deleted
    n_para = n_objects(para_fit_0)
    para_fit_array = np.zeros((1, n_para)) # to be deleted
    para_e_fit_array = np.zeros((1, n_para)) # to be deleted
    m0_list = []
    τ_fit_list = np.array([])
    for i in range(0, len(subdata_list)):
        subdata = subdata_list[i]
        lsa_list = lst_sqrs_adjstmnt(ε, subdata, fit_func,
                                     fit_func_derivs, para, para_fit_0)
        para_fit, para_e_fit, m0 = lsa_list[0], lsa_list[1], lsa_list[2]
        
        τ_i = subdata[0, 0] + n_partition / 2
        
        subdata_fitted = np.zeros((1, 2)) # to be deleted
        for j in range(0, len(subdata)):
            t_j = subdata[j, 0]
            row_j = np.array([t_j, fit_func(t_j, para, para_fit)])
            subdata_fitted = np.vstack((subdata_fitted, row_j))
        subdata_fitted = subdata_fitted[1:]
        
        para_fit_list = ravel(para_fit)
        para_e_fit_list = ravel(para_e_fit)
        m0 = ravel(m0)
        data_fitted = np.vstack((data_fitted, subdata_fitted))
        para_fit_array = np.vstack((para_fit_array, para_fit_list))
        para_e_fit_array = np.vstack((para_e_fit_array, para_e_fit_list))
        m0_list.append(m0)
        τ_fit_list = np.append(τ_fit_list, τ_i)
    data_fitted = data_fitted[1:]
    para_fit_array = para_fit_array[1:]
    para_e_fit_array = para_e_fit_array[1:]
    
    fit_list = [data_fitted, para_fit_array, para_e_fit_array, m0_list, τ_fit_list]
    return(fit_list)
#%%
def get_min_max_list_list(list_of_lists):
    min_list, max_list = [], []
    for liste in list_of_lists:
        min_list.append(min(liste))
        max_list.append(max(liste))
    minimum = min(min_list)
    maximum = max(max_list)
    return(minimum, maximum)
#%%
def averager(liste):
    array = np.array(liste)
    return(np.sum(array) / len(liste))
#%%
def column_averager(array, col_list):
    # col_list -> which columns should be averaged
    summe_list = []
    for i in col_list:
        col = array[:, i]
        summe_list.append(averager(col))
    return(summe_list)
#%%
def column_abs_averager(array, col_list):
    # col_list -> which columns should be averaged
    summe_list = []
    for i in col_list:
        col = array[:, i]
        abs_col = np.abs(col)
        summe_list.append(averager(abs_col))
    return(summe_list)
#%%
def decrease_unit_day(str_unit):
    # str_unit: for example r'[$kg m^{-3}$]'
    inner_string = str_unit[2:-2]
    inner_string = r'\frac{' + inner_string + r'}{d}'
    new_string = str_unit[:2] + inner_string + str_unit[-2:]
    return(new_string)
#%%
def time_deriv_abs_str(string):
    str_1 = string[0]
    str_2 = r'|\frac{d}{dt}'
    str_3 = string[1 : -1]
    str_4 = r'|'
    str_5 = string[-1]
    new_str = str_1 + str_2 + str_3 + str_4 + str_5
    return(new_str)
#%%
def decrease_plot_adv(para_fit, para_e_fit, el_y_label_list, para_specs,
                      ref_data, ref_specs, ref_lab_list,
                      n_partition, el_Δt_n, MJD_0, MJD_end, title, grid,
                      ):
    # para_fit = [h_i, s_i, a_i, ϕ_i]
    # para_e_fit = [σ_h_i, σ_s_i, σ_a_i, σ_ϕ_i]
    # el_y_label_list = [el, unit]
    # para_specs = [e_fac, ms, mew, capsize, mfc, mec, ecolor]
    # ref_data = reference
    # ref_specs = [α, col, lab, width]
    # ref_lab_list = [ref_lab, ref_y_axis_lab]
    # n_partition: how many fits per day
    # el_Δt_n: smoothing period
    # MJD_0: start of data
    # MJD_end: end of data
    # title: title of plot
    # grid: if 1 --> grid for data
    lab = 'slope of ' + el_y_label_list[0]
    unit_day = decrease_unit_day(el_y_label_list[1])
    slope_list = np.array([])
    slope_error_list = np.array([])
    t_list = []
    c = 0
    counter = 0
    for i in range(0, len(para_fit)):
        if (c > el_Δt_n and c + n_partition < MJD_end - MJD_0 - el_Δt_n):
            t_list.append(c + n_partition / 2)
            slope_list = np.append(slope_list, para_fit[counter, 1])
            slope_error_list = np.append(slope_error_list, para_e_fit[counter, 1])
        c += n_partition
        counter += 1
    fig, ax1 = plt.subplots(figsize = (10, 5), dpi = 500)
    fig.suptitle(title, fontsize = 20)
    
    e_fac = para_specs[0]
    ax1.errorbar(t_list, slope_list, yerr = e_fac * slope_error_list,
                 ls = 'none', marker = 'o',
                 ms = para_specs[1], mew = para_specs[2], capsize = para_specs[3],
                 mfc = para_specs[4], mec = para_specs[5], ecolor = para_specs[6],
                 label = 'slope (' + str(e_fac) + r' $\cdot$ error)', zorder = 5)
    ax1.plot(t_list, slope_list, 'b-', lw = 0.5, alpha = 1, zorder = 4)
    ax1.set_xlabel("MJD " + str(MJD_0) + " + t [d]", fontsize = 15)
    ax1.set_ylabel(lab + ' ' + unit_day, fontsize = 15)
    if (grid == 1):
        ax1.grid()
    
    ax2 = ax1.twinx()
    x_ref = ref_data[:, 0]
    y_ref = ref_data[:, 1]
    α = ref_specs[0]
    col = ref_specs[1]
    lab = ref_specs[2]
    width = ref_specs[3]
    ax2.plot(x_ref, y_ref,
            ls = '-', lw = width, color = col, alpha = α,
            label = lab)
    ax2.set_ylabel(ref_lab_list[0] + " " + ref_lab_list[1],
                   color = col, fontsize = 15)
    ax2.tick_params(axis = 'y', labelcolor = col)

    if (MJD_0 <= 60002 and MJD_end >= 60002):
        plt.axvline(60002 - MJD_0, color = 'k',
                    ls = (0, (4, 4)), lw = 1, zorder = 3)
    if (MJD_0 <= 60027 and MJD_end >= 60027):
        plt.axvline(60027 - MJD_0, color = 'k',
                    ls = (0, (4, 4)), lw = 1, zorder = 3)
    if (MJD_0 <= 60058 and MJD_end >= 60058):
        plt.axvline(60058 - MJD_0, color = 'k',
                    ls = (0, (4, 4)), lw = 1, zorder = 3)
    if (MJD_0 <= 58356 and MJD_end >= 58356):
        plt.axvline(58356 - MJD_0, color = 'k',
                    ls = (0, (4, 4)), lw = 1, zorder = 3)
    
    plt.figlegend(fontsize = 12.5, markerscale = 2.5, loc = 2,
                  bbox_to_anchor = (0, 1), bbox_transform = ax1.transAxes)
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.set_frame_on(False)
    plt.show(fig)
    plt.close(fig)
#%%
def decrease_plot_hyperadv(slope_data_list, slope_specs_list,
                           add_data_list, add_data_specs_list,
                           el_label, el_unit,
                           ref_data_list, ref_specs_list, ref_lab_list,
                           MJD_0, MJD_end, xlimits, ylimits, title, grid,
                           fac, location, anchor,
                           vline_list1, vline_list_specs1,
                           vline_list2, vline_list_specs2,
                           save_specs):
    # slope_data_list = [[τ_i, s_i, e_s_i]]
    # slope_specs_list = [[e_fac, ms, mew, capsize, mfc, mec, ecolor]]
    # add_data_list = [[t_k_i, z_k_i], (k+1), ...]
    # add_data_specs_list = [[α_k, col_k, lab_k, width_k, ms_k], (k+1), ...]
    # el_label = stringlike
    # el_unit = el_unit
    # ref_data_list = reference
    # ref_specs_list = [α, col, lab, width]
    # ref_lab_list = [ref_lab, ref_y_axis_lab]
    # n_partition: how many fits per day
    # el_Δt_n: smoothing period
    # MJD_0: start of data
    # MJD_end: end of data
    # xlimits = [xstart, xend]
    # ylimits = [ylow, yupp]
    # title: title of plot
    # grid: if 1 --> grid for data
    
    xstart = xlimits[0] + MJD_0
    xend = MJD_end - xlimits[1]
    n_days = xend - xstart
    new_slope_data_list = 0
    new_ref_data_list = 0
    new_add_data_list = 0
    if (xlimits[0] != xlimits[1]):
        MJD_start = xstart - MJD_0
        n_days = xend - xstart
        new_slope_data_list = []
        new_add_data_list = []
        new_ref_data_list = []
        for i in range(0, len(slope_data_list)):
            data_i = slope_data_list[i]
            data_i = array_modifier(data_i, xstart - MJD_0, n_days)
            new_slope_data_list.append(data_i)
        for i in range(0, len(add_data_list)):
            data_i = add_data_list[i]
            data_i = array_modifier(data_i, xstart - MJD_0, n_days)
            new_add_data_list.append(data_i)
        for i in range(0, len(ref_data_list)):
            ref_i = ref_data_list[i]
            ref_i = array_modifier(ref_i, xstart - MJD_0, n_days)
            new_ref_data_list.append(ref_i)
    else:
        new_slope_data_list = slope_data_list
        new_add_data_list = add_data_list
        new_ref_data_list = ref_data_list
    
    y_label = 'slope of ' + el_label
    unit_day = decrease_unit_day(el_unit)
    
    fig, ax1 = plt.subplots(figsize = (10, 5), dpi = 500)
    fig.suptitle(title, fontsize = 17.5)
    
    for i in range(0, len(new_slope_data_list)):
        slope_data = new_slope_data_list[i]
        slope_specs = slope_specs_list[i]
        e_fac = slope_specs[0]
        ax1.errorbar(slope_data[:, 0], slope_data[:, 1],
                     yerr = e_fac * slope_data[:, 2],
                     ls = 'none', marker = 'o',
                     ms = slope_specs[1], mew = slope_specs[2],
                     capsize = slope_specs[3], mfc = slope_specs[4],
                     mec = slope_specs[5], ecolor = slope_specs[6],
                     label = 'slope (' + str(e_fac) + r' $\cdot$ error)',
                     zorder = 5)
        ax1.plot(slope_data[:, 0], slope_data[:, 1],
                 'b-', lw = 0.5, alpha = 1, zorder = 4)
    
    for i in range(0, len(new_add_data_list)):
        add_data = new_add_data_list[i]
        add_data_specs = add_data_specs_list[i]
        α = add_data_specs[0]
        col = add_data_specs[1]
        lab = add_data_specs[2]
        width = add_data_specs[3]
        size = add_data_specs[4]
        ax1.plot(add_data[:, 0], fac * add_data[:, 1],
                 color = col, alpha = α, ls = 'solid', lw = width,
                 marker = 'o', ms = size,
                 label = lab, zorder = 6)
    
    ax1.set_xlabel("MJD " + str(MJD_0) + " + t [d]", fontsize = 15)
    ax1.set_ylabel(y_label + ' ' + unit_day, fontsize = 15)
    if (grid == 1):
        ax1.grid()
    if (xstart != xend):
        ax1.set_xlim(xstart - MJD_0, xend - MJD_0)
    y_low, y_upp = ylimits[0], ylimits[1]
    if (y_low != y_upp):
        ax1.set_ylim(y_low, y_upp)
    
    ax2 = ax1.twinx()
    for i in range(0, len(new_ref_data_list)):
        ref_data = new_ref_data_list[i]
        ref_specs = ref_specs_list[i]
        α = ref_specs[0]
        col = ref_specs[1]
        lab = ref_specs[2]
        width = ref_specs[3]
        ax2.plot(ref_data[:, 0], ref_data[:, 1],
                 ls = '-', lw = width, color = col, alpha = α,
                 label = lab)
    
    ax2.set_ylabel(ref_lab_list[0] + " " + ref_lab_list[1],
                   color = col, fontsize = 15)
    ax2.tick_params(axis = 'y', labelcolor = col)

    #if (xstart <= 60002 and xend >= 60002):
    #    plt.axvline(60002 - MJD_0, color = 'k',
    #                ls = (0, (4, 4)), lw = 1.5, zorder = 3)
    #if (xstart <= 60027 and xend >= 60027):
    #    plt.axvline(60027 - MJD_0, color = 'k',
    #                ls = (0, (4, 4)), lw = 1.5, zorder = 3)
    #if (xstart <= 60058 and xend >= 60058):
    #    plt.axvline(60058 - MJD_0, color = 'k',
    #                ls = (0, (4, 4)), lw = 1.5, zorder = 3)
    #if (xstart <= 58356 and xend >= 58356):
    #    plt.axvline(58356 - MJD_0, color = 'k',
    #                ls = (0, (4, 4)), lw = 1.5, zorder = 3)
    c = 0
    for i in range(0, len(vline_list1)):
        vline = vline_list1[i]
        col = vline_list_specs1[0]
        α = vline_list_specs1[1]
        width = vline_list_specs1[2]
        lab = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                lab = vline_list_specs1[3]
                c += 1
            plt.axvline(vline - MJD_0, color = col, alpha = α,
                        ls = (0, (4, 4)), lw = width,
                        label = lab, zorder = 3)
    c = 0
    for i in range(0, len(vline_list2)):
        vline = vline_list2[i]
        col = vline_list_specs2[0]
        α = vline_list_specs2[1]
        width = vline_list_specs2[2]
        lab = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                lab = vline_list_specs2[3]
                c += 1
            plt.axvline(vline - MJD_0, color = col, alpha = α,
                        ls = (0, (4, 4)), lw = width,
                        label = lab, zorder = 3)
    
    plt.figlegend(fontsize = 12.5, markerscale = 2.5, loc = location,
                  bbox_to_anchor = anchor, bbox_transform = ax1.transAxes)
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.set_frame_on(False)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    
    plt.show(fig)
    plt.close(fig)
#%%
def acc_file_read(foldername, filename):
    # foldername: string like
    # filename: string like
    file = open(foldername + '/' + filename, 'r')
    lines = file.readlines()
    file.close()
    
    c = 0
    while (c < 201):
        row = lines[c].strip()
        if (row[:20] == "# End of YAML header"):
            break
        if (c == 200):
            print("!!! ERROR 1 !!!")
            break
        c += 1
    c += 1
    
    length = int((len(lines) - c) / 10)
    big_array = np.array([0, 0, 0, 0])
    
    for i in range(0, length):
        j = c + 10 * i
        gps_i = float(lines[j].strip()[:9])
        
        line_i = lines[j].strip()[12:]
        liste = line_i.split(' ')
        liste = liste[:3]
        array = np.array([gps_i])
        for i in liste:
            el = float(i)
            array = np.append(array, el)
        big_array = np.vstack((big_array, array))
    big_array = big_array[1:]
    return(big_array)

#acc_file_read('newoscele/Akzelerometer', 'GFOC_ACT_181530.ACC')
#%%
def acc_gen_file(foldername):
    entries_old = os.listdir(str(foldername))
    entries = np.array([])
    for i in range(0, len(entries_old)):
        file_i = entries_old[i]
        if (file_i[-4:] == '.ACC'):
            entries = np.append(entries, file_i)
    entries = file_number_sorter(entries)
    entries_new = np.array([])
    for i in range(0, len(entries)): # get rid of '.DS_Store'
        file_i = entries[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 10000): # avoid empty files
            entries_new = np.append(entries_new, file_i)
    
    os.mkdir(foldername + "/year")
    data_year = np.array([[0, 0, 0, 0]])
    for i in range(0, len(entries_new)):
        file_i = entries_new[i]
        data_i = acc_file_read(foldername, file_i)
        data_year = np.vstack((data_year, data_i))
    data_year = data_year[1:]
    data_year = array_normalize(data_year, 0)
    
    name_year = foldername + "/year/" + "all.txt"
    np.savetxt(name_year, data_year, delimiter = ' ')
    print("DDDDOOOONNNNEEEE")

#acc_gen_file('newoscele/Akzelerometer')
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

def acc_file_read_hyper(foldername, filename):
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
    
    # sampling from 1s to 5s
    length_data = int(len(acc_data) / 5)
    new_acc_data = np.array([0, 0, 0, 0]) # to be deleted
    for i in range(0, length_data):
        j = 1 * i # sampling of 
        new_acc_data = np.vstack((new_acc_data, acc_data[j]))
    new_acc_data = new_acc_data[1:]
    
    cor_acc_data = new_acc_data[:, 0] # add corrected accelerations
    for i in range(0, 3):
        acc_col = new_acc_data[:, i + 1]
        cor_acc_col = bias_list[i] + scale_list[i] * acc_col
        cor_acc_data = np.vstack((cor_acc_data, cor_acc_col))
    cor_acc_data = cor_acc_data.T
    
    return(cor_acc_data)

def acc_gen_file_hyper(foldername):
    entries_old = os.listdir(str(foldername))
    entries = np.array([])
    for i in range(0, len(entries_old)):
        file_i = entries_old[i]
        if (file_i[-4:] == '.txt'):
            entries = np.append(entries, file_i)
    entries = file_number_sorter(entries)
    entries_new = np.array([])
    for i in range(0, len(entries)): # get rid of '.DS_Store'
        file_i = entries[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 10000): # avoid empty files
            entries_new = np.append(entries_new, file_i)
    
    os.mkdir(foldername + "/year")
    data_year = np.array([[0, 0, 0, 0]])
    for i in range(0, len(entries_new)):
        file_i = entries_new[i]
        data_i = acc_file_read_hyper(foldername, file_i)
        data_year = np.vstack((data_year, data_i))
    data_year = data_year[1:]
    data_year = array_normalize(data_year, 0)
    
    name_year = foldername + "/year/" + "all.txt"
    np.savetxt(name_year, data_year, delimiter = ' ')
    print("DDDDOOOONNNNEEEE")

#acc_gen_file_hyper("hyperdata/hyperdata1/GF-2")
#%%
#def acc_file_bias_scale_mean(foldername_hyper, filename_hyper):
#    # make file with [MJD, Bias (R,S,W), Scale (R,S,W), Mean (R,S,W)]


def acc_file_read_ultimate(foldername_ultimate, filename_ultimate,
                           foldername_hyper, filename_hyper):
    # ultimate: correct data
    # hyper: bias, scale and mean
    file = open(foldername_ultimate + '/' + filename_ultimate, 'r')
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
    acc_data = np.loadtxt(foldername_ultimate + '/' + filename_ultimate,
                          skiprows = c, usecols = (0, 4, 5, 6))
    
    # sampling from 1s to 30s
    length_data = int(len(acc_data) / 30)
    new_acc_data = np.array([0, 0, 0, 0]) # to be deleted
    for i in range(0, length_data):
        j = 1 * i # sampling of 
        new_acc_data = np.vstack((new_acc_data, acc_data[j]))
    new_acc_data = new_acc_data[1:]
    
    cor_acc_data = new_acc_data[:, 0] # add corrected accelerations
    for i in range(0, 3):
        acc_col = new_acc_data[:, i + 1]
        cor_acc_col = bias_list[i] + scale_list[i] * acc_col
        cor_acc_data = np.vstack((cor_acc_data, cor_acc_col))
    cor_acc_data = cor_acc_data.T
    
    return(cor_acc_data)


#%%
def man_file_read(foldername, filename):
    # only get start and end time
    data = np.loadtxt(foldername + "/" + filename, skiprows = 1, usecols = (1, 2))
    data += 18 / (24 * 60 * 60) # UTC -> GPS
    return(data)

def man_gen_file(foldername):
    entries_old = os.listdir(str(foldername))
    entries = np.array([])
    for i in range(0, len(entries_old)):
        file_i = entries_old[i]
        if (file_i[-4:] == '.MAN'):
            entries = np.append(entries, file_i)
    entries = file_number_sorter(entries)
    entries_new = np.array([])
    for i in range(0, len(entries)): # get rid of '.DS_Store'
        file_i = entries[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 100): # avoid empty files (86 bytes)
            entries_new = np.append(entries_new, file_i)
    
    os.mkdir(foldername + "/year")
    data_year = np.array([[0, 0]])
    for i in range(0, len(entries_new)):
        file_i = entries_new[i]
        data_i = man_file_read(foldername, file_i)
        data_year = np.vstack((data_year, data_i))
    data_year = data_year[1:]
    data_year = array_normalize(data_year, 0)
    
    name_year = foldername + "/year/" + "all.txt"
    np.savetxt(name_year, data_year, delimiter = ' ')
    print("DDDDOOOONNNNEEEE")
#man_gen_file("hyperdata/hyperdata2")
#%%
def sec_to_day(array):
    new_array = array[:, 1:]
    secs = array[:, 0]
    days = secs / 24 / 60 / 60
    new_array = np.vstack((days, new_array.T)).T
    return(new_array)
#%%
def Δa_day(n_rev_day, fac, a, ρ):
    return(- 2 * np.pi * n_rev_day * fac * a**2 * ρ)
#%%
def Δa_day_spec(n_rev_day, a, ρ):
    return(- 2 * np.pi * n_rev_day * a**2 * ρ)
#%%
def data_mean(τ_fit_list, data, n_partition):
    t_list = τ_fit_list - n_partition
    mean_array = np.array([0, 0])
    for i in range(0, len(t_list)):
        τ_i = τ_fit_list[i]
        sub_data = array_modifier(data, t_list[i], n_partition)
        mean = np.mean(sub_data[:, 1])
        mean_array = np.vstack((mean_array, np.array([τ_i, mean])))
    mean_array = mean_array[1:]
    return(mean_array)
#%%
#%%
def C_n_q_tilde(n, q_tilde, τ_n_list, t_n_list, # depend on n and q_tilde
                Δτ, L, n_s, K, # depend on data
                p_list): # depend on modelling
    # t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #     with τ_n_1 <= t_n_l < τ_n
    # τ_n_list = [τ_n_1, τ_n]
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    R = len(p_list) # 2 * R + 1 = Q
    τ_n_1, τ_n = τ_n_list[0], τ_n_list[1]
    t_n_list = np.array(t_n_list)
    
    #if (n == n_s):
    #    L_tilde = K - (n_s - 1) * L
    #    if (L_tilde != 0):
    #        L = L_tilde
            
    mat_t = np.vstack((-t_n_list, t_n_list)).T
    mat_τ = np.vstack((τ_n * np.ones(L), - τ_n_1 * np.ones(L))).T
    mat = (mat_t + mat_τ) / Δτ
    
    if (q_tilde >= 1 and q_tilde <= R): # [pd_μ_r_(n-1), pd_μ_r_n]_t=t_k
        r = q_tilde
        p_r = p_list[r - 1] # r = 1, ..., R
        mat_t_symmetric = np.vstack((t_n_list, t_n_list)).T
        mat_μ_r = np.sin(2 * np.pi / p_r * mat_t_symmetric)
        mat = mat * mat_μ_r
    elif (q_tilde >= R + 1 and q_tilde <= 2 * R): # [pd_η_r_(n-1), pd_η_r_n]_t=t_k
        r = q_tilde - R
        p_r = p_list[r - 1] # r = 1, ..., R
        mat_t_symmetric = np.vstack((t_n_list, t_n_list)).T
        mat_η_r = np.cos(2 * np.pi / p_r * mat_t_symmetric)
        mat = mat * mat_η_r
    # else (q_tilde == 0): mat = mat
    return(mat)

def E_n_q_q_prime(n, q_tilde_list, τ_n_list, t_n_list, # depend on n and q_tilde
                  Δτ, L, n_s, K, # depend on data
                  p_list, P_n): # depend on modelling
    # is actually E_(n,(q_tilde),(q_tilde_prime))
    # q_tilde_list = [q_tilde, q_tilde_prime]
    # t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #     with τ_n_1 <= t_n_l < τ_n
    # τ_n_list = [τ_n_1, τ_n]
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    # P_n: block matrix in weighting matrix P
    q_tilde, q_tilde_prime = q_tilde_list[0], q_tilde_list[1]
    C_n_q = C_n_q_tilde(n, q_tilde, τ_n_list, t_n_list,
                        Δτ, L, n_s, K, p_list)
    C_n_q_prime = C_n_q_tilde(n, q_tilde_prime, τ_n_list, t_n_list,
                              Δτ, L, n_s, K, p_list)
    #print("shape C_n_q", np.shape(C_n_q))
    #print("shape P_n", np.shape(P_n))
    mat_center_center = C_n_q.T @ P_n @ C_n_q_prime
    mat_center_left = np.zeros((2, n - 1))
    mat_center_right = np.zeros((2, n_s - n))
    
    mat_top = np.zeros((n - 1, n_s + 1))
    mat_center = np.hstack((mat_center_left, mat_center_center, mat_center_right))
    mat_bottom = np.zeros((n_s - n, n_s + 1))
    
    mat = np.vstack((mat_top, mat_center, mat_bottom))
    return(mat)

def F_n(n, τ_n_list, t_n_list, # depend on n
        Δτ, L, n_s, K, # depend on data
        p_list, P_n): # depend on modelling
    # F_n = (A_n)^T P_n A_n
    # q_tilde_list = [q_tilde, q_tilde_prime]
    # t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #     with τ_n_1 <= t_n_l < τ_n
    # τ_n_list = [τ_n_1, τ_n]
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    # P_n: block matrix in weighting matrix P
    R = len(p_list)
    Q = 2 * R + 1
    Q_tilde = Q - 1
    
    mat = np.zeros((1, Q * (n_s + 1))) # to be deleted
    for q_tilde in range(0, Q_tilde + 1): # stack row for row
        mat_help = np.zeros((n_s + 1, 1)) # to be deleted
        for q_tilde_prime in range(0, Q_tilde + 1): # stack column for column
            q_tilde_list = [q_tilde, q_tilde_prime]
            E_mat = E_n_q_q_prime(n, q_tilde_list, τ_n_list, t_n_list,
                                  Δτ, L, n_s, K, p_list, P_n)
            mat_help = np.hstack((mat_help, E_mat))
        mat_help = mat_help[:, 1:] # delete start
        mat = np.vstack((mat, mat_help))
    mat = mat[1:] # delete start
    return(mat)

def N_eff(τ_n_list_list, t_n_list_list, # depend on n
          Δτ, L, n_s, K, # depend on data
          p_list, P_n_list): # depend on modelling
    # N = A^T P A, eff -> should be efficient
    # τ_n_list_list = [τ_1_list, ..., τ_n_list, ..., τ_n_s_list]
    #     τ_n_list = [τ_n_1, τ_n]
    # t_n_list_list = [t_1_list, ..., t_n_list, ..., t_n_s_list]
    #     t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #         τ_n_1 <= t_n_l < τ_n
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    # P_n_list = [P_1, ..., P_n, ..., P_n_s]; P_n block matrix in weighting matrix P
    R = len(p_list)
    Q = 2 * R + 1
    
    mat = np.zeros((Q * (n_s + 1), Q * (n_s + 1))) # start
    for n in range(1, n_s + 1):
        τ_n_list = τ_n_list_list[n - 1] # index starts at 0
        t_n_list = t_n_list_list[n - 1] # index starts at 0
        P_n = P_n_list[n - 1] # index starts at 0
        F_n_mat = F_n(n, τ_n_list, t_n_list,
                      Δτ, L, n_s, K,p_list, P_n)
        mat = mat + F_n_mat
    return(mat)

def G_n_q_tilde(n, q_tilde, τ_n_list, t_n_list, # depend on n and q_tilde
                Δτ, L, n_s, K, # depend on data
                p_list, P_n): # depend on modelling
    # t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #     with τ_n_1 <= t_n_l < τ_n
    # τ_n_list = [τ_n_1, τ_n]
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    C_n_q = C_n_q_tilde(n, q_tilde, τ_n_list, t_n_list,
                        Δτ, L, n_s, K, p_list)
    mat_center = C_n_q.T @ P_n
    
    #if (n == n_s):
    #    L_tilde = K - (n_s - 1) * L
    #    if (L_tilde != 0):
    #        L = L_tilde
    
    mat_top = np.zeros((n - 1, L))
    mat_bottom = np.zeros((n_s - n, L))
    
    mat = np.vstack((mat_top, mat_center, mat_bottom))
    return(mat)

def H_n(n, τ_n_list, t_n_list, # depend on n and q_tilde
        Δτ, L, n_s, K, # depend on data
        p_list, P_n): # depend on modelling
    # t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #     with τ_n_1 <= t_n_l < τ_n
    # τ_n_list = [τ_n_1, τ_n]
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    R = len(p_list)
    Q = 2 * R + 1
    Q_tilde = Q - 1
    
    #if (n == n_s):
    #    L_tilde = K - (n_s - 1) * L
    #    if (L_tilde != 0):
    #        mat = np.zeros((1, L_tilde))
    #else:
    #    mat = np.zeros((1, L)) # to be deleted
    mat = np.zeros((1, L))
    for q_tilde in range(0, Q_tilde + 1):
        G_mat = G_n_q_tilde(n, q_tilde, τ_n_list, t_n_list,
                            Δτ, L, n_s, K, p_list, P_n)
        mat = np.vstack((mat, G_mat))
    mat = mat[1:] # delete start
    return(mat)

def AT_P(τ_n_list_list, t_n_list_list, # depend on n
         Δτ, L, n_s, K, # depend on data
         p_list, P_n_list): # depend on modelling
    # τ_n_list_list = [τ_1_list, ..., τ_n_list, ..., τ_n_s_list]
    #     τ_n_list = [τ_n_1, τ_n]
    # t_n_list_list = [t_1_list, ..., t_n_list, ..., t_n_s_list]
    #     t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #         τ_n_1 <= t_n_l < τ_n
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    # P_n_list = [P_1, ..., P_n, ..., P_n_s]; P_n block matrix in weighting matrix P
    R = len(p_list)
    Q = 2 * R + 1
    
    mat = np.zeros((Q * (n_s + 1), 1)) # to be deleted
    for n in range(1, n_s + 1):
        τ_n_list = τ_n_list_list[n - 1] # index starts at 0
        t_n_list = t_n_list_list[n - 1] # index starts at 0
        P_n = P_n_list[n - 1] # index starts at 0
        H_mat = H_n(n, τ_n_list, t_n_list,
                    Δτ, L, n_s, K, p_list, P_n)
        mat = np.hstack((mat, H_mat))
    mat = mat[:, 1:] # delete start
    return(mat)

def x_q_n_vec(x_vec, n, q_tilde, n_s):
    # x_vec = [a_0, ..., a_n, ..., a_(n_s),
    #          μ_(1,0), ..., μ_(1,n), ..., μ_(1,n_s), ...,
    #          μ_(r,0), ..., μ_(r,n), ..., μ_(r,n_s), ...,
    #          μ_(R,0), ..., μ_(R,n), ..., μ_(R,n_s),
    #          η_(1,0), ..., η_(1,n), ..., η_(1,n_s), ...,
    #          η_(r,0), ..., η_(r,n), ..., η_(r,n_s), ...,
    #          η_(R,0), ..., η_(R,n), ..., η_(R,n_s)]
    vec = x_vec[(n_s + 1) * q_tilde + n - 1 : (n_s + 1) * q_tilde + n + 1]
    return(vec)

def I_n(n, τ_n_list, t_n_list, # depend on n and q_tilde
        Δτ, L, n_s, K, # depend on data
        p_list, x_vec): # depend on modelling
    # t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #     with τ_n_1 <= t_n_l < τ_n
    # τ_n_list = [τ_n_1, τ_n]
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    R = len(p_list) # 2 * R + 1 = Q
    Q = 2 * R + 1
    Q_tilde = Q - 1
    
    #if (n == n_s):
    #    L_tilde = K - (n_s - 1) * L
    #    if (L_tilde != 0):
    #        vec = np.zeros((L_tilde, 1))
    #else:
    #    vec = np.zeros((L, 1))
    vec = np.zeros((L, 1))
    
    for q_tilde in range(0, Q_tilde + 1):
        C_mat = C_n_q_tilde(n, q_tilde, τ_n_list, t_n_list,
                            Δτ, L, n_s, K, p_list)
        x_q_n = x_q_n_vec(x_vec, n, q_tilde, n_s)
        vec = vec + C_mat @ x_q_n
    return(vec)

def A_x(τ_n_list_list, t_n_list_list, # depend on n and q_tilde
        Δτ, L, n_s, K, # depend on data
        p_list, x_vec):
    # τ_n_list_list = [τ_1_list, ..., τ_n_list, ..., τ_n_s_list]
    #     τ_n_list = [τ_n_1, τ_n]
    # t_n_list_list = [t_1_list, ..., t_n_list, ..., t_n_s_list]
    #     t_n_list = [t_n_1, ..., t_n_l, ..., t_n_L]
    #         τ_n_1 <= t_n_l < τ_n
    # Δτ: interval duration
    # L: number of observations in interval (L = int(K / n_s))
    #    if n = n_s -> last interval may not have length L but L_tilde = K % n_s
    # n_s: number of intervals
    # K: total number of observations
    # p_list = [p_1, ..., p_r, ..., p_R]; periods in model; R: number of modelled periods
    # x_vec = [a_n, μ_(r,n), η_(r,n)] (see def of x_q_n_vec)
    vec = np.zeros((1, 1)) # to be deleted
    for n in range(1, n_s + 1):
        τ_n_list = τ_n_list_list[n - 1] # index starts at 0
        t_n_list = t_n_list_list[n - 1] # index starts at 0
        I_vec = I_n(n, τ_n_list, t_n_list,
                    Δτ, L, n_s, K, p_list, x_vec)
        vec = np.vstack((vec, I_vec))
    vec = vec[1:] # delete start
    return(vec)

def compute_data(τ_n_list_list, t_n_list_list,
                 Δτ, L, n_s, K, p_list, x_vec):
    R = len(p_list)
    Q = 2 * R + 1
    Q_tilde = Q - 1
    comp_list = np.array([])
    for n in range(1, n_s + 1):
        τ_n_list = τ_n_list_list[n - 1]
        t_n_list = t_n_list_list[n - 1]
        #if (n == n_s):
        #    L_tilde = K - (n_s - 1) * L
        #    if (L_tilde != 0):
        #        L = L_tilde
        τ_n_1 = τ_n_list[0]
        τ_n = τ_n_list[1]
        for l in range(0, L):
            t = t_n_list[l]
            t_covec = np.array([[τ_n - t, t - τ_n_1]]) / Δτ
            tot = 0
            for q_tilde in range(0, Q_tilde + 1):
                vec = x_q_n_vec(x_vec, n, q_tilde, n_s)
                if (q_tilde >= 1 and q_tilde <= R): # μ
                    r = q_tilde
                    p_r = p_list[r - 1]
                    vec = vec * np.sin(2 * np.pi / p_r * t)
                elif (q_tilde >= R + 1 and q_tilde <= 2 * R): # η
                    r = q_tilde - R
                    p_r = p_list[r - 1]
                    vec = vec * np.cos(2 * np.pi / p_r * t)
                # else: vec = vec # a_bar
                tot = tot + t_covec @ vec
            comp_list = np.append(comp_list, tot)
    t_list = np.array(ravel(t_n_list_list))
    comp_data = np.vstack((t_list, comp_list)).T
    return(comp_data)

def gen_a_bar_array(τ_n_list_list, t_n_list_list,
                    Δτ, L, n_s, K, a_bar_list, σ2_a_bar_list):
    bar_data = np.array([0, 0, 0])
    for n in range(1, n_s + 1):
        τ_n_list = τ_n_list_list[n - 1]
        t_n_list = t_n_list_list[n - 1]
        #if (n == n_s):
        #    L_tilde = K - (n_s - 1) * L
        #    if (L_tilde != 0):
        #        L = L_tilde
        τ_n_1 = τ_n_list[0]
        τ_n = τ_n_list[1]
        for l in range(0, L):
            t = t_n_list[l]
            t_covec = np.array([[τ_n - t, t - τ_n_1]]) / Δτ
            t_covec_error = np.array([[(τ_n - t)**2, (t - τ_n_1)**2]]) / (Δτ**2)
            
            vec = x_q_n_vec(a_bar_list, n, 0, n_s)
            vec_error = x_q_n_vec(σ2_a_bar_list, n, 0, n_s)
            
            tot = t_covec @ vec
            tot_error = t_covec_error @ vec_error
            
            entry = np.array([t, tot[0], tot_error[0]]) # achtung ist im quadrat
            bar_data = np.vstack((bar_data, entry))
    bar_data = bar_data[1:]
    return(bar_data)

def v(τ_n_list_list, t_n_list_list,
      Δτ, L, n_s, K, p_list, Δx_vec, O_C_vec):
    # Ax - l
    A_x_vec = A_x(τ_n_list_list, t_n_list_list,
                  Δτ, L, n_s, K, p_list, Δx_vec)
    return(A_x_vec - O_C_vec)

def encode_list(liste, code_list):
    # liste = [a, b, c, d, e, f, g]
    # code_list = [0, 0, 2, 3]
    # wanted: [a, b, [c, d], [e,f,g]]
    new_list = []
    s = 0
    for i in range(0, len(code_list)):
        n_objects = code_list[i]
        if (n_objects == 0):
            new_list.append(liste[s])
            s = s + 1
        else:
            new_list.append(liste[s : s + n_objects])
            s = s + n_objects
    return(new_list)

def splitter(liste, L, n_s, K):
    # with help of encode_list split liste in several lists
    #L_tilde = K - (n_s - 1) * L
    code = (n_s - 1) * [L]
    #if (L_tilde != 0):
    #    L = L_tilde
    code.append(L)
    list_list = encode_list(liste, code)
    return(list_list)

def m0_eff(τ_n_list_list, t_n_list_list,
           Δτ, L, n_s, K, p_list, P_n_list, Δx_vec, O_C_vec):
    R = len(p_list)
    Q = 2 * R + 1
    v_vec = v(τ_n_list_list, t_n_list_list,
              Δτ, L, n_s, K, p_list, Δx_vec, O_C_vec)
    v_list = ravel(v_vec)
    v_splitted = splitter(v_list, L, n_s, K)
    tot = 0
    for n in range(1, n_s + 1):
        v_n_list = v_splitted[n - 1]
        P_n = P_n_list[n - 1]
        
        v_n_vec = np.array([v_n_list]).T
        tot += v_n_vec.T @ P_n @ v_n_vec
    tot = tot / (K - Q * (n_s + 1)) # n = K, u = Q * (n_s + 1)
    return(np.sqrt(tot))

def L_constr(n_s, Q):
    # shape(L_mat) = (Q * (n_s + 1), Q * (n_s - 1))
    diag_upp = np.ones(n_s - 1)
    diag_cen = - 2 * np.ones(n_s)
    diag_low = np.ones(n_s + 1)
    a_bar_mat = np.diag(diag_low) + np.diag(diag_cen, 1) + np.diag(diag_upp, 2)
    a_bar_mat = a_bar_mat[: -2]
    zero_mat_1 = np.zeros(((Q - 1) * (n_s - 1), 1 * (n_s + 1)))
    zero_mat_2 = np.zeros((Q * (n_s - 1), (Q - 1) * (n_s + 1)))
    L_mat = np.hstack((np.vstack((a_bar_mat, zero_mat_1)), zero_mat_2))
    return(L_mat)

def L_constr_all(n_s, Q):
    # shape(L_mat) = (Q * (n_s + 1), Q * (n_s - 1))
    diag_upp = np.ones(n_s - 1)
    diag_cen = - 2 * np.ones(n_s)
    diag_low = np.ones(n_s + 1)
    constr_mat = np.diag(diag_low) + np.diag(diag_cen, 1) + np.diag(diag_upp, 2)
    constr_mat = constr_mat[: -2]
    
    shape = np.shape(constr_mat)
    hlist = []
    for i in range(0, Q):
        before = i * [np.zeros(shape)]
        after = (Q - i - 1) * [np.zeros(shape)]
        before.append(constr_mat)
        before.extend(after)
        obj = np.block([before])
        hlist.append(obj)
    L_mat = np.vstack(tuple(hlist))
    return(L_mat)


def lsa_cond(ε_I, quotient_list, counter):
    # ε_I = {ε if ε_I < 1, I if ε_I is integer}
    # quotient_list: relative improvements of each parameter
    # counter: counter for iterations
    # if ε_I is < 1 -> iterate until max(quotient_list) < ε_I
    #     ex: ε_I = 1e-3 -> iterate until max(quotient_list) < 1e-3
    # if ε_I is int -> iterate until counter = ε_I
    #     ex: ε_I = 5 -> iterate until counter = 5 (5 iterations)
    if (type(ε_I) == int):
        return(counter <= ε_I)
    else:
        return(max(quotient_list) > ε_I)

def lst_sqrs_adjstmnt_eff_adv(ε_I, obs_data, τ_n_list_list, t_n_list_list,
                              Δτ, L, n_s, K, p_list, P_n_list, x_vec, constr):
    print("lst_sqrs_adjstmnt_eff_adv version 2")
    print("compute N")
    N = N_eff(τ_n_list_list, t_n_list_list,
              Δτ, L, n_s, K, p_list, P_n_list)
    print("shape N: ", np.shape(N))
    print("compute L_mat")
    if (constr == 0):
        L_mat = L_constr(n_s, 1 + 2 * len(p_list))
    elif (constr == 1):
        L_mat = L_constr_all(n_s, 1 + 2 * len(p_list))
    else:
        L_mat = np.zeros(np.shape(N))
    print("compute new N")
    N = N + ε_I * L_mat.T @ L_mat
    print("shape new N: ", np.shape(N))
    print("compute N_inv")
    N_inv = np.linalg.inv(N)
    print("shape N_inv: ", np.shape(N_inv))
    print("compute ATP")
    ATP = AT_P(τ_n_list_list, t_n_list_list,
               Δτ, L, n_s, K, p_list, P_n_list)
    print("shape ATP: ", np.shape(ATP))
    print("compute N_inv_ATP")
    N_inv_ATP = N_inv @ ATP
    print("shape N_inv_ATP: ", np.shape(N_inv_ATP))
    print("generate l_vec")
    l_vec = np.array([obs_data[:, 1]]).T
    print("shape l_vec: ", np.shape(l_vec))
    print("compute x_vec")
    x_vec = N_inv_ATP @ l_vec
    print("shape x_vec: ", np.shape(x_vec))
    print("compute m0")
    m0 = m0_eff(τ_n_list_list, t_n_list_list,
                Δτ, L, n_s, K, p_list, P_n_list, x_vec, l_vec)
    print("m0 = %.3e" % m0[0][0])
    x_list = ravel(x_vec)
    print("compute Kxx")
    Kxx = m0**2 * N_inv
    σ2_list = np.diag(Kxx) # σ^2
    print("compute fitted data")
    fitted_data = compute_data(τ_n_list_list, t_n_list_list,
                               Δτ, L, n_s, K, p_list, x_vec)
    print("compute data free of oscillations")
    bar_data = gen_a_bar_array(τ_n_list_list, t_n_list_list,
                               Δτ, L, n_s, K, x_list[: n_s + 1], σ2_list[: n_s + 1])
    fit_list = [fitted_data, x_list, σ2_list, m0[0][0], bar_data, Kxx]
    return(fit_list)

def pre_lst_sqrs_adjstmt(obs_data, n_s, K, p_list, P, apriori_list):
    # generate τ_n_list_list, t_n_list_list, L, P_n_list, x_vec, Δτ
    # apriori_list = [a_(apriori), μ_(apriori)_list, η_(apriori)_list]
    t_tot = obs_data[-1, 0] - obs_data[0, 0]
    t0 = obs_data[0, 0]
    #Δτ = np.ceil(t_tot) / n_s
    Δτ = t_tot / n_s
    L = int(K / n_s)
    print("L = ", L)
    #L_tilde = K - (n_s - 1) * L
    #print("L = ", L, " | L_tilde = ", L_tilde)
    #if (L_tilde == 0):
    #    n_s = n_s - 1
    
    t_n_list_list = splitter(obs_data[:, 0], L, n_s, K)
    τ_n_list_list = []
    for n in range(1, n_s + 1):
        τ_n_1 = t0 + (n - 1) * Δτ
        τ_n = t0 + n * Δτ
        τ_n_list_list.append([τ_n_1, τ_n])
    
    P_n_list = []
    for n in range(1, n_s + 1):
        P_n = P[(n - 1) * L : n * L, (n - 1) * L : n * L]
        P_n_list.append(P_n)
    
    R = len(p_list)
    a_apriori = apriori_list[0]
    μ_apriori_list = apriori_list[1]
    η_apriori_list = apriori_list[2]
    
    x_a_list = np.array((n_s + 1) * [a_apriori])
    x_μ_list, x_η_list = np.array([]), np.array([])
    for r in range(0, R):
        x_μ_r = np.array([(n_s + 1) * [μ_apriori_list[r]]])
        x_η_r = np.array([(n_s + 1) * [η_apriori_list[r]]])
        x_μ_list = np.append(x_μ_list, x_μ_r)
        x_η_list = np.append(x_η_list, x_η_r)
    x_vec = np.array([np.hstack((x_a_list, x_μ_list, x_η_list))]).T
    #x_vec = np.ones(((n_s + 1) * (2 * R + 1), 1)) # would also work
    
    big_list = [τ_n_list_list, t_n_list_list, L, P_n_list, x_vec, Δτ]
    return(big_list)

def gen_a_dot_array(n_t, slope_gen_list):
    # n_t: number of t points
    τ_list = slope_gen_list[0]
    Δτ = slope_gen_list[1]
    a_bar_list = slope_gen_list[2]
    σ2_a_bar_list = slope_gen_list[3]
    # t_list should be array
    # warning min(t_list) >= min(τ_list)
    #     and max(t_list) < max(τ_list)
    # τ_lis, a_bar_list and σ2_a_bar_list should all
    # be sorted from lowest to highest chronologically
    τ_0 = τ_list[0]
    τ_n_s = τ_list[-1]
    t_list = np.linspace(τ_0, τ_n_s, n_t + 1)[: -1]
    
    a_dot_list = np.array([])
    s_a_dot_list = np.array([])
    
    for j in range(1, len(τ_list)):
        a_bar_n_1 = a_bar_list[j - 1]
        a_bar_n = a_bar_list[j]
        σ2_a_bar_n_1 = σ2_a_bar_list[j - 1]
        σ2_a_bar_n = σ2_a_bar_list[j]
        
        a_dot_j = (a_bar_n - a_bar_n_1) / Δτ
        s_a_dot_j = np.sqrt((σ2_a_bar_n + σ2_a_bar_n_1) / (Δτ**2))
        
        a_dot_list = np.append(a_dot_list, a_dot_j)
        s_a_dot_list = np.append(s_a_dot_list, s_a_dot_j)
    
    array = np.zeros(3)
    for i in range(0, len(t_list)):
        t_i = t_list[i]
        for j in range(1, len(τ_list)):
            if (j * Δτ + τ_0 > t_i):
                a_dot_i = a_dot_list[j - 1]
                s_a_dot_i = s_a_dot_list[j - 1]
                
                row_i = np.array([t_i, a_dot_i, s_a_dot_i])
                array = np.vstack((array, row_i))
                break
    array = array[1:]
    return(array)


def fitter(data, per0_list, amp0_list, n_partition, lsa_method, ε_I, constr):
    print("fitter version 2")
    # INPUT
    #    data: [t_i, el_i] to be fitted
    #    per0_list: periods that will be used in fit (fixed or as guesses)
    #    amp0_list: corresponding amplitudes (as guesses)
    #    n_partition: duration of intervals for creating fits
    #    lsa_method: [1 -> t0, per0 fixed, 2 -> t0 fixed]
    #    ε: accuracy of fit
    # OUTPUT
    #    fit_list = [data_fitted, para_fit_array, para_e_fit_array, m0_list]
    #        data_fitted: fitted data (stacked)
    #        para_fit_array: fitted parameters stacked
    #        para_e_fit_array: error of fitted parameters stacked
    #        m0_list: a posteriori errors for each fit
    #        τ_fit_list: τ for each fitted parameter
    if (lsa_method != 5):
        n_partition = 1 / n_partition
        ε = ε_I
    
    if (lsa_method == 1): # give t0 and periods; fit h, s, amplitudes and phases
        para = insert([0], per0_list) # given parameters
        para_fit_0 = insert([0, 0], amp0_list) # first guesses for parmeters
        para_fit_0 = insert(para_fit_0, np.zeros(len(per0_list))) # to be fitted
        fit_func = fit_func_1
        fit_func_derivs = fit_func_1_derivs
    elif (lsa_method == 2): # give t0; fit h, s, periods, amplitudes and phases
        para = [0] # given parameters
        para_fit_0 = insert([0, 0], per0_list[:, 0]) # first guesses for parmeters
        para_fit_0 = insert(para_fit_0, amp0_list[:, 1]) # to be fitted
        para_fit_0 = insert(para_fit_0, np.zeros(len(per0_list)))
        fit_func = fit_func_2
        fit_func_derivs = fit_func_2_derivs
    elif (lsa_method == 3): # give t0; fit h and s (linear regression)
        para = [0] # given parameters
        para_fit_0 = [0, 0] # first guesses for parmeters to be fitted
        fit_func = fit_func_3
        fit_func_derivs = fit_func_3_derivs
    elif (lsa_method == 4):
        print("lsa_method == 4")
        t_array = data[:, 0]
        Δt = n_partition
        print("n_partition = ", n_partition)
        n_s = get_n_s(data, Δt)
        para_lists_list_0 = 1 * [(n_s + 1) * [6*10**6]]
        print("fitting")
        lsa_list = lst_sqrs_adjstmnt_adv(ε, data, para_lists_list_0, Δt, n_s)
        print("fitted")
        para_fit, e_para_fit, m_0 = lsa_list[0], lsa_list[1], lsa_list[2]
        τ_fit_list = np.append(Δt * np.arange(n_s - 1), data[-1, 0])
        obj_list = np.array([])
        n = 1
        for i in range(0, len(t_array)):
            t_i = t_array[i]
            if (t_i >= n * Δt):
                n += 1
            t_list = [(n - 1) * Δt, n * Δt, Δt]
            obj = fit_func_adv(t_i, para_fit, t_list, n)
            obj_list = np.append(obj_list, obj)
        data_fitted = np.vstack((t_array, obj_list)).T
        fit_list = [data_fitted, para_fit, e_para_fit, m_0, τ_fit_list]
        print("shape of data: ", np.shape(data))
        print("shape of fitted data:", np.shape(data_fitted))
        print("fitter end")
        return(fit_list)
    elif (lsa_method == 5):
        print("lsa_method = 5")
        n_s = n_partition
        K = len(data)
        P = np.diag(np.ones(K))
        p_list = per0_list
        apriori_list = [data[0, 1], amp0_list, amp0_list]
        print("prepare for fit")
        big_list = pre_lst_sqrs_adjstmt(data, n_s, K, p_list,
                                        P, apriori_list)
        τ_n_list_list, t_n_list_list = big_list[0], big_list[1]
        L, P_n_list = big_list[2], big_list[3]
        x_vec, Δτ = big_list[4], big_list[5]
        print("preparations finished")
        print("n_s = %4d | Δτ = %3.6f" % (n_s, Δτ))
        print("execute lsa")
        fit_list = lst_sqrs_adjstmnt_eff_adv(ε_I, data, τ_n_list_list, t_n_list_list,
                                             Δτ, L, n_s, K, p_list, P_n_list, x_vec, constr)
        fitted_data = fit_list[0]
        x_list = fit_list[1]
        σ2_list = fit_list[2]
        m0 = fit_list[3]
        bar_data = fit_list[4]
        Kxx = fit_list[5]
        
        τ_fit_list = np.array(τ_n_list_list)[:, 0]
        τ_fit_list = np.append(τ_fit_list, τ_n_list_list[-1][1])
        R = len(p_list)
        Q = 2 * R + 1
        Q_tilde = Q - 1
        para_list = []
        s2_para_list = []
        for q_tilde in range(0, Q_tilde + 1):
            sub_list = []
            s2_sub_list = []
            for n in range(0, n_s + 1):
                a_q_n = x_list[(n_s + 1) * q_tilde + n]
                σ2_q_n = σ2_list[(n_s + 1) * q_tilde + n]
                para = 0
                s2_para = 0
                if (q_tilde == 0): # a_bar
                    para = a_q_n
                    s2_para = σ2_q_n
                elif (q_tilde >= 1 and q_tilde <= R): # μ
                    τ_n = τ_fit_list[n]
                    r = q_tilde
                    p_r = p_list[r - 1]
                    para = a_q_n * np.sin(2 * np.pi / p_r * τ_n)
                    s2_para = σ2_q_n * (np.sin(2 * np.pi / p_r * τ_n))**2
                else: # q_tilde >= R + 1 and q_tilde <= 2 * R # η
                    τ_n = τ_fit_list[n]
                    r = q_tilde - R
                    p_r = p_list[r - 1]
                    para = a_q_n * np.cos(2 * np.pi / p_r * τ_n)
                    s2_para = σ2_q_n * (np.cos(2 * np.pi / p_r * τ_n))**2
                sub_list.append(para)
                s2_sub_list.append(s2_para)
            para_list.append(sub_list)
            s2_para_list.append(s2_sub_list)
        
        a_bar_list = x_list[: n_s + 1]
        σ2_a_bar_list = σ2_list[: n_s + 1]
        slope_gen_list = [τ_fit_list, Δτ, a_bar_list, σ2_a_bar_list]
        
        fit_list = [fitted_data, para_list, s2_para_list, m0,
                    τ_fit_list, slope_gen_list, bar_data, x_list, Kxx]
        return(fit_list)
    else:
        print("ERROR!!! SPECIFY LSA-METHOD!!!")
        return(1)
    
    # create subdata
    start, end = data[0, 0], data[-1, 0]
    subdata_list = []
    while (start <= end):
        subdata = array_modifier(data, start, n_partition)
        # normalizing ???
        subdata_list.append(subdata)
        start += n_partition
    
    data_fitted = np.zeros((1, 2)) # to be deleted
    n_para = n_objects(para_fit_0)
    para_fit_array = np.zeros((1, n_para)) # to be deleted
    para_e_fit_array = np.zeros((1, n_para)) # to be deleted
    m0_list = []
    τ_fit_list = np.array([])
    for i in range(0, len(subdata_list)):
        subdata = subdata_list[i]
        lsa_list = lst_sqrs_adjstmnt(ε, subdata, fit_func,
                                     fit_func_derivs, para, para_fit_0)
        para_fit, para_e_fit, m0 = lsa_list[0], lsa_list[1], lsa_list[2]
        
        τ_i = subdata[0, 0] + n_partition / 2
        
        subdata_fitted = np.zeros((1, 2)) # to be deleted
        for j in range(0, len(subdata)):
            t_j = subdata[j, 0]
            row_j = np.array([t_j, fit_func(t_j, para, para_fit)])
            subdata_fitted = np.vstack((subdata_fitted, row_j))
        subdata_fitted = subdata_fitted[1:]
        
        para_fit_list = ravel(para_fit)
        para_e_fit_list = ravel(para_e_fit)
        m0 = ravel(m0)
        data_fitted = np.vstack((data_fitted, subdata_fitted))
        para_fit_array = np.vstack((para_fit_array, para_fit_list))
        para_e_fit_array = np.vstack((para_e_fit_array, para_e_fit_list))
        m0_list.append(m0)
        τ_fit_list = np.append(τ_fit_list, τ_i)
    data_fitted = data_fitted[1:]
    para_fit_array = para_fit_array[1:]
    para_e_fit_array = para_e_fit_array[1:]
    
    fit_list = [data_fitted, para_fit_array, para_e_fit_array, m0_list, τ_fit_list]
    return(fit_list)
#%%
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

def lab_gen(x, configuration):
    #configuration_list = [[n_fac, R, ω, constr]]
    n_fac = configuration[0]
    R = configuration[1]
    ω = configuration[2]
    constr = configuration[3]
    lab = r'$_{n_fac}^{(R)}\tilde{x}_{ω}^{(constr)}$'
    lab = lab.replace("x", x)
    lab = lab.replace("n_fac", str(n_fac))
    lab = lab.replace("R", str(R))
    lab = lab.replace('ω', '%.0e' % ω)
    lab = lab.replace('constr', str(constr))
    return(lab)
#%%
def plot_bar(data_array_list, data_spec_array, y_label,
             flx_data, flx_spec_list, flx_y_spec_list,
             tit, MJD_0, MJD_end, xlimits, ylimits,
             vline_specs, z_order, nongra_fac, grid,
             vline_list1, vline_list_specs1,
             vline_list2, vline_list_specs2,
             save_specs):
    # data_array_list = [data_1, data_2, ...]
    # data_spec_array = [alpha_i, col_i, lab_i, lw_i]
    # y_label: label for y axis
    # ref_array_list: [ref_1, ref_2, ...]
    # ref_spec_array: [alpha_i, ]
    # ref_y_spec_list: [y_label, y_axis_col]
    # tit: title, string like
    # MJD_0: data start
    # MJD_end: data end
    # xlimits = [x_start, x_end] (to deactivate: x_start = x_end)
    # ylimits = [y_low, y_upp]: (to deactivate: y_low = y_upp)
    # vline_specs = [col, alpha, lw]
    # z_order: if 1 --> data in front of ref
    # nongra_fac: for scaling nongra data
    # grid: if 1 --> grid to data
    # vline_list = [vline_i]
    # vline_list_specs = [col_i, alpha_i, lw_i, label]
    # save_specs = [1, path] # 0,1 for saving
    xstart = xlimits[0] + MJD_0
    xend = MJD_end - xlimits[1]
    n_days = xend - xstart
    new_data_array_list = 0
    if (xlimits[0] != xlimits[1]):
        new_data_array_list = []
        for i in range(0, len(data_array_list)):
            data_i = data_array_list[i]
            data_i = array_modifier(data_i, xstart - MJD_0, n_days)
            new_data_array_list.append(data_i)
        if (len(flx_data) != 0):
            flx_data = array_modifier(flx_data, xstart - MJD_0, n_days)
    else:
        new_data_array_list = data_array_list
    
    fig, ax1 = plt.subplots(figsize = (10, 5), dpi = 300)
    fig.suptitle(tit, fontsize = 17.5)
    
    for i in range(0, len(new_data_array_list)):
        x_data = new_data_array_list[i][:, 0]
        y_data = new_data_array_list[i][:, 1]
        α = data_spec_array[i][0]
        col = data_spec_array[i][1]
        lab = data_spec_array[i][2]
        width = data_spec_array[i][3]
        if (i == 1):
            if (nongra_fac != 1):
                lab = lab + ' (scaled by ' + str(nongra_fac) + ')'
                y_data *= nongra_fac
        ax1.plot(x_data + MJD_0, y_data, color = col, ls = '-', lw = width,
                    alpha = α, zorder = 5, label = lab,)
    ax1.xaxis.set_major_formatter(mjd_to_mmdd)
    ax1.xaxis.set_minor_locator(mat.ticker.MultipleLocator(1))
    ax1.tick_params(axis = 'x', which = 'major',
                    width = 2, length = 8,
                    direction = 'out', top = False)
    ax1.tick_params(axis = 'x', which = 'minor',
                    width = 1, length = 4,
                    direction = 'out', top = False)
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    ax1.set_ylabel(y_label, fontsize = 15)
    y_low, y_upp = ylimits[0], ylimits[1]
    if (xstart != xend):
        ax1.set_xlim(xstart, xend)
    if (y_low != y_upp):
        ax1.set_ylim(y_low, y_upp)
    if (grid == 1):
        ax1.grid()
    
    if (len(flx_data) != 0):
        len_flx_data = len(flx_data.T) - 1
        col_map = flx_spec_list[2]
        α = flx_spec_list[0]
        lab_stat = flx_spec_list[1]
        ax2 = ax1.twinx()
        bar_width = 1 / (len_flx_data)
        
        colmap = mat.colormaps[col_map]
        col_list = cl_lin(np.linspace(0, 1, len(flx_data.T) - 1), colmap)
        norming = colors.BoundaryNorm(np.arange(1 / 2, len(flx_data.T) - 1, 1), colmap.N) # discrete bins
        
        α = flx_spec_list[0]
        lab_stat = flx_spec_list[1]
        if (lab_stat == 1):
            lab = r'$\text{ap}_i$'.replace('i', str(i))
        else:
            lab = None
        for i in range(1, len_flx_data):
            t_ap = flx_data[:, 0]
            ap = flx_data[:, i]
            ax2.bar(t_ap + (i + 1 / 2) * bar_width + MJD_0, ap,
                    width = bar_width, alpha = α, color = col_list[i - 1],
                    label = lab)
        
        sm = ScalarMappable(cmap = colmap, norm = norming)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, cax = ax2.inset_axes([1.1, 0, 0.02, 1]),
                            ticks = np.arange(1, len(flx_data.T) - 1, 1),
                            alpha = α)
        cbar.set_label(r'$i$', rotation = 0, labelpad = 10)
        
        ax2.set_ylabel(r'$\text{ap}_i$', color = 'k', fontsize = 15, rotation = 0)
        ax2.tick_params(axis = 'y', labelcolor = 'k')
        
        if (z_order == 1):
            ax1.set_zorder(ax2.get_zorder()+1)
            ax1.set_frame_on(False)
    
    c = 0 # for giving only one label to vlines from vline_list1
    for i in range(0, len(vline_list1)):
        vline = vline_list1[i]
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs1[3]
                c += 1
            plt.axvline(vline, color = vline_list_specs1[0],
                        alpha = vline_list_specs1[1], lw = vline_list_specs1[2],
                        ls = (0, (4, 4)), zorder = 3,
                        label = c_label)
    
    c = 0 # for giving only one label to vlines from vline_list2
    for i in range(0, len(vline_list2)):
        vline = vline_list2[i]
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs2[3]
                c += 1
            plt.axvline(vline, color = vline_list_specs2[0],
                        alpha = vline_list_specs2[1], lw = vline_list_specs2[2],
                        ls = (0, (4, 4)), zorder = 3,
                        label = c_label)
    
    plt.figlegend(fontsize = 12.5, markerscale = 5, loc = 1,
                  bbox_to_anchor = (1, 1), bbox_transform = ax1.transAxes,
                  labelspacing = 0)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    
    plt.show(fig)
    plt.close(fig)
#%%
def plot_func_6(data_array_list, data_spec_array, y_label,
                ref_array_list, ref_spec_array, ref_y_spec_list,
                tit, MJD_0, MJD_end, xlimits, ylimits,
                vline_specs, z_order, nongra_fac, grid,
                vline_list1, vline_list_specs1,
                vline_list2, vline_list_specs2,
                save_specs):
    # data_array_list = [data_1, data_2, ...]
    # data_spec_array = [alpha_i, col_i, lab_i, lw_i]
    # y_label: label for y axis
    # ref_array_list: [ref_1, ref_2, ...]
    # ref_spec_array: [alpha_i, col_i, lab_i, lw_i]
    # ref_y_spec_list: [y_label, y_axis_col]
    # tit: title, string like
    # MJD_0: data start
    # MJD_end: data end
    # xlimits = [x_start, x_end] (to deactivate: x_start = x_end)
    # ylimits = [y_low, y_upp]: (to deactivate: y_low = y_upp)
    # vline_specs = [col, alpha, lw]
    # z_order: if 1 --> data in front of ref
    # nongra_fac: for scaling nongra data
    # grid: if 1 --> grid to data
    # vline_list = [vline_i]
    # vline_list_specs = [col_i, alpha_i, lw_i, label]
    # save_specs = [1, path] # 0,1 for saving
    xstart = xlimits[0] + MJD_0
    xend = MJD_end - xlimits[1]
    n_days = xend - xstart
    new_data_array_list = 0
    new_ref_array_list = 0
    if (xlimits[0] != xlimits[1]):
        new_data_array_list = []
        new_ref_array_list = []
        for i in range(0, len(data_array_list)):
            data_i = data_array_list[i]
            data_i = array_modifier(data_i, xstart - MJD_0, n_days)
            new_data_array_list.append(data_i)
        for i in range(0, len(ref_array_list)):
            ref_i = ref_array_list[i]
            ref_i = array_modifier(ref_i, xstart - MJD_0, n_days)
            new_ref_array_list.append(ref_i)
    else:
        new_data_array_list = data_array_list
        new_ref_array_list = ref_array_list
    
    fig, ax1 = plt.subplots(figsize = (10, 5), dpi = 300)
    fig.suptitle(tit, fontsize = 17.5)
    
    for i in range(0, len(new_data_array_list)):
        x_data = new_data_array_list[i][:, 0]
        y_data = new_data_array_list[i][:, 1]
        α = data_spec_array[i][0]
        col = data_spec_array[i][1]
        lab = data_spec_array[i][2]
        width = data_spec_array[i][3]
        if (i == 1):
            if (nongra_fac != 1):
                lab = lab + ' (scaled by ' + str(nongra_fac) + ')'
                y_data *= nongra_fac
        ax1.plot(x_data + MJD_0, y_data, color = col, ls = '-', lw = width,
                    alpha = α, zorder = 5, label = lab)
    ax1.xaxis.set_major_formatter(mjd_to_mmdd)
    ax1.set_xlabel(xaxis_year([MJD_0], 'mm.dd'), fontsize = 15)
    ax1.set_ylabel(y_label, fontsize = 15)
    y_low, y_upp = ylimits[0], ylimits[1]
    if (xstart != xend):
        ax1.set_xlim(xstart, xend)
    if (y_low != y_upp):
        ax1.set_ylim(y_low, y_upp)
    if (grid == 1):
        ax1.grid()
    
    if (len(ref_array_list) != 0):
        ax2 = ax1.twinx()
        for i in range(0, len(new_ref_array_list)):
            x_ref = new_ref_array_list[i][:, 0]
            y_ref = new_ref_array_list[i][:, 1]
            α = ref_spec_array[i][0]
            col = ref_spec_array[i][1]
            lab = ref_spec_array[i][2]
            width = ref_spec_array[i][3]
            ax2.plot(x_ref + MJD_0, y_ref,
                     ls = '-', lw = width, color = col, alpha = α,
                     label = lab)
        ax2.set_ylabel(ref_y_spec_list[0], color = ref_y_spec_list[1], fontsize = 15)
        ax2.tick_params(axis = 'y', labelcolor = ref_y_spec_list[1])
        
        if (z_order == 1):
            ax1.set_zorder(ax2.get_zorder()+1)
            ax1.set_frame_on(False)
    
    c = 0 # for giving only one label to vlines from vline_list1
    for i in range(0, len(vline_list1)):
        vline = vline_list1[i]
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs1[3]
                c += 1
            plt.axvline(vline, color = vline_list_specs1[0],
                        alpha = vline_list_specs1[1], lw = vline_list_specs1[2],
                        ls = (0, (4, 4)), zorder = 3,
                        label = c_label)
    
    c = 0 # for giving only one label to vlines from vline_list2
    for i in range(0, len(vline_list2)):
        vline = vline_list2[i]
        c_label = None
        if (xstart <= vline and xend >= vline):
            if (c == 0):
                c_label = vline_list_specs2[3]
                c += 1
            plt.axvline(vline, color = vline_list_specs2[0],
                        alpha = vline_list_specs2[1], lw = vline_list_specs2[2],
                        ls = (0, (4, 4)), zorder = 3,
                        label = c_label)
    
    plt.figlegend(fontsize = 12.5, markerscale = 5, loc = 1,
                  bbox_to_anchor = (1, 1), bbox_transform = ax1.transAxes,
                  labelspacing = 0)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    
    plt.show(fig)
    plt.close(fig)
#%%
k_gauss = 0.01720209895
m_erde = 5.9722 * 10**24
mu = k_gauss**2 * m_erde
a_dot_fac = 24 * 60 * 60 * 2 / mu
def a_dot(a, v, acc):
    return(a_dot_fac * a**2 * v * acc)

