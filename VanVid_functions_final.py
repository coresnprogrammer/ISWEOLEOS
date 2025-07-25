# %%
# STANDARD MODULES
import numpy as np
import math
import os

# general constants
day_sec = 24 * 60 * 60
μ = 398600441500000 * 86400**2 
# %% epoch trimming
def array_modifier(array, MJD_start, n_days):
    """ trim array
    array = [[t1, x1, y1, ...], [t2, x2, y2, ...], ...]
    get interval [MJD_start, MJD_start + n_days)
    The pair at MJD_start + n_days is not included
    MJD_start and n_days do not have to be integers
    """
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
# %% processing LST files
def master_lst(foldername, col_mjd, parameters, pathsave):
    """ generate txt files for parameters
    foldername: string-like
    col_mjd is column number of mjd in lst files
    parameters: [['x', col('x')], ...]
        give name of element (x) and the column number (col('x')) in the lst file
        avoid "unnatural" letters like ä and α and only use lower case letters
    pathsave: path for saving masterfile
    """
    entries_old = os.listdir(str(foldername))
    entries = np.array([])
    for i in range(0, len(entries_old)):
        file_i = entries_old[i]
        if (file_i[-4:] == '.LST'):
            entries = np.append(entries, file_i)
    entries.sort()
    entries_new = np.array([])
    for i in range(0, len(entries)): # get rid of '.DS_Store'
        file_i = entries[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 10000): # avoid empty files
            entries_new = np.append(entries_new, file_i)
    
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
            print("!!! ERROR !!!")
            break
        c += 1
    c += 1
    
    data_year = np.loadtxt(foldername + '/' + entries_new[0], skiprows = c)
    for k in range(1, len(entries_new)):
        data_k = np.loadtxt(foldername + '/' + entries_new[k], skiprows = c)
        data_year = np.vstack((data_year, data_k))
    
    MJD = data_year[:, col_mjd]
    for i in range(0, len(parameters)):
        string = parameters[i][0]
        col = parameters[i][1]
        element = data_year[:, col]
        data = np.vstack((MJD, element)).T
        
        name = pathsave + string + ".txt"
        np.savetxt(name, data)
        print("generated file: " + name)
    print("DDDDOOOONNNNEEEE")
# %% processing ELE files
def convert_str_double_to_float(string):
    """ convert double-string to float
    string = '0.429874776D-08'
    where D means it is a number with double precision and it replaces the exponential
    """
    if (string[0] == ' '):
        new_string = string[1:]
    else:
        new_string = string
    return(float(new_string.replace("D", "E")))

def ele_file_read(foldername, filename):
    """ extract PCAs from ELE file
    foldername: string-like
    filename: string-like
    """
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
            
            array_11 += L30
            array_12 += L20
            array_13 += L10
            break
        if (c0 == 100):
            print("!!! ERROR 2 !!!")
            break
        c0 += 1
    
    data = np.vstack((array_MJD, array_11, array_12, array_13)).T
    
    array_14 = np.array([]) # absolute acceleration
    for i in range(0, len(data)):
        abs = 0
        for j in range(1, 4):
            abs += (data[i, j])**2
        array_14 = np.append(array_14, np.sqrt(abs))
    data = np.hstack((data, np.array([array_14]).T))
    
    return(data)

def pca_gentxt(foldername, pathsave):
    """ generate .txt file with PCAs
    foldername: string-like
    pathsave: string-like
    """
    entries_old = os.listdir(str(foldername))
    entries = np.array([])
    for i in range(0, len(entries_old)):
        file_i = entries_old[i]
        # get rid of '.DS_Store' and txt files
        if (file_i[-4:] == '.ELE'):
            entries = np.append(entries, file_i)
    entries.sort()
    
    entries_new = np.array([])
    for i in range(0, len(entries)):
        file_i = entries[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 1000): # avoid empty files
            entries_new = np.append(entries_new, file_i)
    
    data_year = np.array([[0, 0, 0, 0, 0]])
    for i in range(0, len(entries_new)):
        file_i = entries_new[i]
        data_i = ele_file_read(foldername, file_i)
        data_year = np.vstack((data_year, data_i))
    data_year = data_year[1:]
    
    name_year = pathsave + "PCA.txt"
    np.savetxt(name_year, data_year, delimiter = ' ')
    print("DDDDOOOONNNNEEEE")

def ele_get_osc(foldername, filename):
    """ extract day values of osculating elements from ELE file
    get values of A, E, I, NODE, PERIGEE, ARG OF LAT
    foldername: string-like
    filename: string-like
    """
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
    a_day = float(lines[c + 2][37 : -27].strip())
    e_day = float(lines[c + 3][37 : -27].strip())
    i_day = float(lines[c + 4][37 : -27].strip())
    Ω_day = float(lines[c + 5][37 : -27].strip())
    ω_day = float(lines[c + 6][37 : -27].strip())
    u_day = float(lines[c + 7][37 : -27].strip())
    
    data = np.array([mjd, a_day, e_day, i_day,
                     Ω_day, ω_day, u_day])
    
    return(data)

def ele_gen_txt_osc(foldername, pathsave):
    """ generate .txt file with daily values of osculating elements
    foldername: string-like
    pathsave: string-like
    """
    entries_old = os.listdir(str(foldername))
    
    entries = np.array([])
    for i in range(0, len(entries_old)):
        file_i = entries_old[i]
        # get rid of '.DS_Store' and .txt files
        if (file_i[-4:] == '.ELE'):
            entries = np.append(entries, file_i)
    entries.sort()
    
    entries_new = np.array([])
    for i in range(0, len(entries)):
        file_i = entries[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 10000): # avoid empty files
            entries_new = np.append(entries_new, file_i)
    
    data_year = np.array([[0, 0, 0, 0, 0, 0, 0]]) # mjd, a, e, i, Ω, ω, u
    for i in range(0, len(entries_new)):
        file_i = entries_new[i]
        data_i = ele_get_osc(foldername, file_i)
        data_year = np.vstack((data_year, data_i))
    data_year = data_year[1:]
    
    name_year = pathsave + "ele_osc.txt"
    np.savetxt(name_year, data_year, delimiter = ' ')
    print("DDDDOOOONNNNEEEE")

def master_ele(foldername, pathsave):
    """ creates master ELE files
    Input:
        foldername: string-like
        file_type: string-like
    Output:
        creates master ele file and master osc ele file
    """
    pca_gentxt(foldername, pathsave)
    ele_gen_txt_osc(foldername, pathsave)
    print("DDDDOOOONNNNEEEE")
# %% resampling of data
def step_data_generation(array, fac):
    """ make step data for PCAs
    array = [[mjd_1, RSW_1], ...]
    fac -> new time step = time step / fac
        example: time step = 6 min, fac = 12
            -> new time step = 6 min / 12 = 30 sec
    """
    if (fac == 1):
        return(array)
    MJD = array[:, 0]
    acc = array[:, 1:]
    
    step = MJD[1] - MJD[0] # assuming equidistant dataset
    
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

def down_sample(array, fac):
    """ down sampling: only take every fac-th entry of array """
    new_array = np.zeros((1, len(array.T)))
    c = 0
    l = len(array)
    while (c < l):
        row = array[c]
        new_array = np.vstack((new_array, row))
        c += fac
    new_array = new_array[1:]
    return(new_array)
# %% functions for numerical integration with Runge-Kutta fourth order method
def a_dot_RSW0(a, e, ω, u, R, S): # old method
    """ Gauss's perturbation equation for the semi-major axis
    Input:
        osculating elements and RSW accelerations (only R and S)
    Output:
        slope of semi-major axis
    """
    ν = (u - ω) * np.pi / 180
    fac = 2 * math.sqrt(a**3 / (μ * (1 - e*e)))
    term1 = e * math.sin(ν) * R
    term2 = (1 + e * math.cos(ν)) * S
    
    a_dot = fac * (term1 + term2)
    
    return(a_dot)

def a_dot_RSW1(a, r, e, ω, u, R, S): # advanced method
    """ Gauss's perturbation equation for the semi-major axis
    Input:
        osculating elements and RSW accelerations (only R and S)
    Output:
        slope of semi-major axis
    """
    ν = (u - ω) * np.pi / 180
    fac = 2 * math.sqrt(a**3 / (μ * (1 - e*e)))
    term1 = e * math.sin(ν) * R
    term2 = (1 + e * math.cos(ν)) * S
    
    a_dot = fac * (term1 + term2)
    
    return(a_dot)

def a_dot_RSW2(a, r, e, ω, u, R, S): # method of Ivo
    """ Gauss's perturbation equation for the semi-major axis
    Input:
        osculating elements and RSW accelerations (only R and S)
    Output:
        slope of semi-major axis
    """
    ν = (u - ω) * np.pi / 180
    fac = 2 * math.sqrt(a**3 / (μ * (1 - e*e)))
    term1 = e * math.sin(ν) * R
    term2 = a * (1 - e*e) / r * S
    
    a_dot = fac * (term1 + term2)
    
    return(a_dot)

def propagator(dt, a_n, r_list, e_list, ω_list, u_list, R_list, S_list):
    """ propagates one step = 2 * dt or from a_n to a_np2, np2 = n + 2
    Input:
        dt: sampling
        a_n: semi-major axis at time t_n
        r_list = [r_n, r_(n+1), r_(n+2)]: radial distance at times t_n, t_(n+1) and t_(n+2)
        e_list = [e_n, e_(n+1), e_(n+2)]: eccentricity at times t_n, t_(n+1) and t_(n+2)
        ω_list = [ω_n, ω_(n+1), ω_(n+2)]: argument of perigee at times t_n, t_(n+1) and t_(n+2)
        u_list = [u_n, u_(n+1), u_(n+2)]: argument of latitude at times t_n, t_(n+1) and t_(n+2)
        R_list = [R_n, R_(n+1), R_(n+2)]: radial acceleration at times t_n, t_(n+1) and t_(n+2)
        S_list = [S_n, S_(n+1), S_(n+2)]: along-track acceleration at times t_n, t_(n+1) and t_(n+2)
    Output:
        a_np2: semi-major axis at time t_(n+2)
    """
    r_n, r_np1, r_np2 = r_list[0], r_list[1], r_list[2] # r(t, t + dt, t + 2dt)
    e_n, e_np1, e_np2 = e_list[0], e_list[1], e_list[2] # e(t, t + dt, t + 2dt)
    ω_n, ω_np1, ω_np2 = ω_list[0], ω_list[1], ω_list[2] # ω(t, t + dt, t + 2dt)
    u_n, u_np1, u_np2 = u_list[0], u_list[1], u_list[2] # u(t, t + dt, t + 2dt)
    R_n, R_np1, R_np2 = R_list[0], R_list[1], R_list[2] # R(t, t + dt, t + 2dt)
    S_n, S_np1, S_np2 = S_list[0], S_list[1], S_list[2] # S(t, t + dt, t + 2dt)
    
    k1 = a_dot_RSW2(a_n, r_n, e_n, ω_n, u_n, R_n, S_n)
    k1 = 2 * dt * k1
    
    k2 = a_dot_RSW2(a_n + k1 / 2, r_np1, e_np1, ω_np1, u_np1, R_np1, S_np1)
    k2 = 2 * dt * k2
    
    k3 = a_dot_RSW2(a_n + k2 / 2, r_np1, e_np1, ω_np1, u_np1, R_np1, S_np1)
    k3 = 2 * dt * k3
    
    k4 = a_dot_RSW2(a_n + k3, r_np2, e_np2, ω_np2, u_np2, R_np2, S_np2)
    k4 = 2 * dt * k4
    
    a_np2 = a_n + (k1 + k4) / 6 + (k2 + k3) / 3
    return(a_np2)

def integrator(a_0, r_data, e_data, ω_data, u_data, acc_R_data, acc_S_data):
    """ integrates time interval with Runge-Kutta fourth order method
    Input:
        a_data = [[t_1, a_1], [t_2, a_2], ...]
        r_data = [[t_1, r_1], [t_2, r_2], ...]
        e_data = [[t_1, e_1], [t_2, e_2], ...]
        ω_data = [[t_1, ω_1], [t_2, ω_2], ...]
        u_data = [[t_1, u_1], [t_2, u_2], ...]
        acc_R_data = [[t_1, R_1], [t_2, R_2], ...]
        acc_S_data = [[t_1, S_1], [t_2, S_2], ...]
    Output:
        a_int_data = [[t_1, a_1], [t_2, a_2], ...]
            integrated semi-major axis
        a_dot_data = [[t_1, a_dot_1], [t_2, a_dot_2], ...]
            slope of semi-major axis
    """
    r_list_list = r_data[:, 1]
    e_list_list = e_data[:, 1]
    ω_list_list = ω_data[:, 1]
    u_list_list = u_data[:, 1]
    
    R_list_list = acc_R_data[:, 1] * day_sec * day_sec
    S_list_list = acc_S_data[:, 1] * day_sec * day_sec
    
    # starting point
    t_0 = u_data[0, 0] 
    a_dot_0 = a_dot_RSW2(a_0, r_list_list[0], e_list_list[0], ω_list_list[0],
                         u_list_list[0], R_list_list[0], S_list_list[0])
    
    a_int_data = np.array([[t_0, a_0]]) # t, a_int
    a_dot_data = np.array([[t_0, a_dot_0]]) # t, a_dot
    
    dt = u_data[1, 0] - u_data[0, 0] # time step
    for i in range(0, (len(u_data) - 1) // 2):
        # _np2 for_(n+2)
        n = 2 * i
        t_np2 = u_data[n + 2, 0]
        a_n = a_int_data[-1, 1]
        
        r_list = r_list_list[n : n + 2 + 1]
        e_list = e_list_list[n : n + 2 + 1]
        ω_list = ω_list_list[n : n + 2 + 1]
        u_list = u_list_list[n : n + 2 + 1]
        R_list = R_list_list[n : n + 2 + 1]
        S_list = S_list_list[n : n + 2 + 1]
        
        a_np2 = propagator(dt, a_n, r_list, e_list, ω_list,
                                               u_list, R_list, S_list)
        
        a_dot_np2 = a_dot_RSW2(a_np2, r_list[-1], e_list[-1], ω_list[-1],
                               u_list[-1], R_list[-1], S_list[-1])
        
        a_int_row = np.array([t_np2, a_np2])
        a_dot_row = np.array([t_np2, a_dot_np2])
        
        a_int_data = np.vstack((a_int_data, a_int_row))
        a_dot_data = np.vstack((a_dot_data, a_dot_row))
    
    return(a_int_data, a_dot_data)

def master_integrator(path, mjd_interval, typus):
    """INPUT:
            path_ele_osc: data from ELE-Files (file generated with master_ele function)
            path_PCA: data from ELE-Files (file generated with master_ele function)
            mjd_interval: time interval for integration
    OUTPUT:
        a_int_data: integrated semi-major axis
        a_dot_data: slope of integrated semi-major axis
    """
    MJD_start, MJD_end = mjd_interval[0], mjd_interval[1]
    n_days = MJD_end - MJD_start
    
    r_data = np.loadtxt(path + "r.txt")
    e_data = np.loadtxt(path + "e.txt")
    ω_data = np.loadtxt(path + "omega_small.txt")
    u_data = np.loadtxt(path + "u.txt")
    
    r_data = array_modifier(r_data, MJD_start, n_days) # crop data
    e_data = array_modifier(e_data, MJD_start, n_days) # crop data
    ω_data = array_modifier(ω_data, MJD_start, n_days) # crop data
    u_data = array_modifier(u_data, MJD_start, n_days) # crop data
    
    #e_0, ω_0 = e_data[0, 1], ω_data[0, 1]
    
    ele_osc_data = np.loadtxt(path + "ele_osc.txt")
    ele_osc_data = array_modifier(ele_osc_data, MJD_start, n_days) # crop data
    a_0 = ele_osc_data[0, 1] # WARNING: IT MAY BE BETTER TO USE AN AVERAGED VALUE !!!

    acc_R_data = np.loadtxt(path + typus + ".txt", usecols = (0, 1))
    acc_S_data = np.loadtxt(path + typus + ".txt", usecols = (0, 2))
    
    acc_R_data = array_modifier(acc_R_data, MJD_start, n_days) # crop data
    acc_S_data = array_modifier(acc_S_data, MJD_start, n_days) # crop data
    
    # resample data for error estimation
    dt_u = (u_data[-1, 0] - u_data[0, 0]) / (len(u_data) - 1)
    dt_acc = (acc_R_data[-1, 0] - acc_R_data[0, 0]) / (len(acc_R_data) - 1)
    
    r_data_high = r_data # high resolution data
    e_data_high = e_data
    ω_data_high = ω_data
    u_data_high = u_data
    
    r_data_low = down_sample(r_data, 2) # low resolution data (resolution half of high resolution data)
    e_data_low = down_sample(e_data, 2)
    ω_data_low = down_sample(ω_data, 2)
    u_data_low = down_sample(u_data, 2)
    
    if (typus == 'PCA'):
        fac = int(np.round(dt_acc / dt_u))
        acc_R_data_high = step_data_generation(acc_R_data, fac)
        acc_S_data_high = step_data_generation(acc_S_data, fac)
        
        fac_low = int(np.round(dt_acc / (2 * dt_u)))
        acc_R_data_low = step_data_generation(acc_R_data, fac_low)
        acc_S_data_low = step_data_generation(acc_S_data, fac_low)
    if (typus == 'ACC'):
        fac = int(np.round(dt_u / dt_acc))
        acc_R_data_high = down_sample(acc_R_data, fac)
        acc_S_data_high = down_sample(acc_S_data, fac)
        
        fac_low = int(np.round((2 * dt_u) / dt_acc))
        acc_R_data_low = down_sample(acc_R_data, fac_low)
        acc_S_data_low = down_sample(acc_S_data, fac_low)    
    
    a_int_data_high, a_dot_data_high = integrator(a_0, r_data_high, e_data_high, ω_data_high, u_data_high,
                                                  acc_R_data_high, acc_S_data_high)
    
    a_int_data_low, a_dot_data_low = integrator(a_0, r_data_low, e_data_low, ω_data_low,
                                                u_data_low, acc_R_data_low, acc_S_data_low)
    
    a_int_data_high = down_sample(a_int_data_high, 2)
    a_dot_data_high = down_sample(a_dot_data_high, 2)
    
    σ_a_int_data = np.abs(a_int_data_high[:, 1 :] - a_int_data_low[:, 1 : ]) / 15 # or 30?
    σ_a_dot_data = np.abs(a_dot_data_high[:, 1 :] - a_dot_data_low[:, 1 : ]) / 15 # or 30?
    
    a_int_data = np.hstack((a_int_data_high, σ_a_int_data))
    a_dot_data = np.hstack((a_dot_data_high, σ_a_dot_data))
    
    return(a_int_data, a_dot_data)
# %%