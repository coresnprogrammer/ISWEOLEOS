# %%
# STANDARD MODULES
import time as tt
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mat
import os
from astropy.time import Time

plt.style.use('classic')
mat.rcParams['figure.facecolor'] = 'white'

print(np.random.randint(1,9))
print(u'\u03B1') # alpha

# general constants
day_sec = 24 * 60 * 60
sec_to_day = 1 / day_sec
G  = 6.674e-11 / (sec_to_day**2)
Me = 5.972e24
μ = G * Me
# %% basic array cropping/trimming
def array_columns(array, col_number_list):
    """ extract columns from an array
    array = [col_0, col_1, col_2, ...]
    col_number_list = [a, b, c, ...]
    gives: [col_a, col_b, col_c, ...]
    """
    n_cols = len(array.T)
    n_rows = len(array)
    cols = np.hsplit(array, n_cols) # all cols
    new_array = np.ones((n_rows, 1)) # wanted cols
    for i in range(0, len(col_number_list)):
        new_array = np.hstack((new_array, cols[col_number_list[i]]))
    new_array = new_array[:, 1:]
    return(new_array)

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
# %% process LST data
def file_number_sorter(liste):
    """ sort entries in a list by date
    liste = ['RDAMFING233420.LST', 'RDAMFING233425.LST', ...]
    """
    # liste = [el1, el2, el3] <-> [ind_1, ind_2, ind_3]
    number_list = np.array([])
    for i in range(0, len(liste)):
        name = liste[i]
        number = int(name[-8 : -5])
        number_list = np.append(number_list, number)
    sorting_list = np.argsort(number_list) # [ind_2, ind_3, ind_1]
    new_list = np.array([])
    for i in sorting_list:
        new_list = np.append(new_list, liste[i])
    return(new_list)

def master_lstold(foldername, file_type, col_mjd, parameters):
    """ generate txt files for parameters
    foldername: string-like
    file_type: 1 for NL (normal, nominal), 2 for NG (nongra, non-gravitational)
    col_mjd is column number of mjd in lst files
    parameters: [['x', col('x')], ...]
        give name of element (x) and the column number (col('x')) in the lst file
        avoid "unnatural" letters like ä and α and only use lower case letters
    """
    file_str = 0
    if (file_type == 1):
        file_str = 'NL'
    elif (file_type == 2):
        file_str = 'NG'
    
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
    
    data_name = "year_" + file_str
    MJD = data_year[:, col_mjd]
    for i in range(0, len(parameters)):
        string = parameters[i][0]
        col = parameters[i][1]
        element = data_year[:, col]
        data = np.vstack((MJD, element)).T
        
        name = foldername + "/" + data_name + "_" + string + ".txt"
        np.savetxt(name, data)
        print("generated file: " + name)
    print("DDDDOOOONNNNEEEE")

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
# %% process ELE data
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

def ele_file_readold(foldername, filename, ng):
    """ extract PCAs from ELE file
    foldername: string like
    filename: string like
    ng: if NG file -> ng = 1, if NL -> ng = 0
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
    
    if (ng == 0): # need to add offset
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
    
    array_14 = np.array([]) # absolute acceleration
    for i in range(0, len(data)):
        abs = 0
        for j in range(1, 4):
            abs += (data[i, j])**2
        array_14 = np.append(array_14, np.sqrt(abs))
    data = np.hstack((data, np.array([array_14]).T))
    
    return(data)

def ele_gen_txtold(foldername, file_type):
    """ generate .txt file with PCAs
    foldername: string like
    file_type: 1 for normal (nominal), 2 for nongra (non-gravitational)
    """
    file_str = 0
    ng = 0
    if (file_type == 1):
        file_str = 'NL'
        ng = 0
    elif (file_type == 2):
        file_str = 'NG'
        ng = 1
    
    entries_old = os.listdir(str(foldername))
    entries = np.array([])
    for i in range(0, len(entries_old)):
        file_i = entries_old[i]
        # get rid of '.DS_Store' and txt files
        if (file_i[-4:] == '.ELE'):
            entries = np.append(entries, file_i)
    entries = file_number_sorter(entries)
    
    entries_new = np.array([])
    for i in range(0, len(entries)):
        file_i = entries[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 10000): # avoid empty files
            if (file_type == 1): # want normal files
                if (file_i[6:8] != 'NG'):
                    entries_new = np.append(entries_new, file_i)
            elif (file_type == 2): # want nongra files
                if (file_i[6:8] == 'NG'):
                    entries_new = np.append(entries_new, file_i)
    
    data_year = np.array([[0, 0, 0, 0, 0]])
    for i in range(0, len(entries_new)):
        file_i = entries_new[i]
        data_i = ele_file_readold(foldername, file_i, ng)
        data_year = np.vstack((data_year, data_i))
    data_year = data_year[1:]
    
    name_year = foldername + "/year_" + file_str + ".txt"
    np.savetxt(name_year, data_year, delimiter = ' ')
    print("DDDDOOOONNNNEEEE")

def ele_get_errorsold(foldername, filename):
    """ extract day errors of osculating elements from ELE file
    get errors of A, E, I, NODE, PERIGEE, ARG OF LAT
    foldername: string like
    filename: string like
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
    a_error = float(lines[c+2].strip()[-24:-12])
    e_error = float(lines[c+3].strip()[-24:-12])
    i_error = float(lines[c+4].strip()[-24:-12])
    Ω_error = float(lines[c+5].strip()[-24:-12])
    ω_error = float(lines[c+6].strip()[-24:-12])
    u_error = float(lines[c+7].strip()[-24:-12])
    
    data = np.array([mjd, a_error, e_error, i_error,
                     Ω_error, ω_error, u_error])
    return(data)

def ele_gen_txt_errorold(foldername, file_type):
    """ generate .txt file with daily errors of osculating elements
    foldername: string like
    filename: string like
    file_type: if NG file -> file_type = 2, if NL -> file_type = 1
    """
    # file_type = 1 for normal, file_type = 2 for nongra
    file_str = 0
    if (file_type == 1):
        file_str = 'NL'
    elif (file_type == 2):
        file_str = 'NG'
    
    entries_old = os.listdir(str(foldername))
    
    entries = np.array([])
    for i in range(0, len(entries_old)):
        file_i = entries_old[i]
        # get rid of '.DS_Store' and .txt files
        if (file_i[-4:] == '.ELE'):
            entries = np.append(entries, file_i)
    entries = file_number_sorter(entries)
    
    entries_new = np.array([])
    for i in range(0, len(entries)):
        file_i = entries[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 10000): # avoid empty files
            if (file_type == 1): # want normal files
                if (file_i[6:8] != 'NG'):
                    entries_new = np.append(entries_new, file_i)
            elif (file_type == 2): # want nongra files
                if (file_i[6:8] == 'NG'):
                    entries_new = np.append(entries_new, file_i)
    
    data_year = np.array([[0, 0, 0, 0, 0, 0, 0]])
    for i in range(0, len(entries_new)):
        file_i = entries_new[i]
        data_i = ele_get_errorsold(foldername, file_i)
        data_year = np.vstack((data_year, data_i))
    data_year = data_year[1:]
    
    name_year = foldername + "/year_" + file_str + "_errors.txt"
    np.savetxt(name_year, data_year, delimiter = ' ')
    print("DDDDOOOONNNNEEEE")

def master_eleold(foldername, file_type):
    """ creates master ELE files
    Input:
        foldername: string-like
        file_type: string-like
    Output:
        creates master ele file and master error ele file
    """
    ele_gen_txtold(foldername, file_type)
    ele_gen_txt_errorold(foldername, file_type)
    print("DDDOOOONNNNEEEE")

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
            
            array_11 += L10
            array_12 += L20
            array_13 += L30
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

def ele_get_errors(foldername, filename):
    """ extract day errors of osculating elements from ELE file
    get errors of A, E, I, NODE, PERIGEE, ARG OF LAT
    foldername: string like
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
    a_error = float(lines[c+2].strip()[-24:-12])
    e_error = float(lines[c+3].strip()[-24:-12])
    i_error = float(lines[c+4].strip()[-24:-12])
    Ω_error = float(lines[c+5].strip()[-24:-12])
    ω_error = float(lines[c+6].strip()[-24:-12])
    u_error = float(lines[c+7].strip()[-24:-12])
    
    data = np.array([mjd, a_error, e_error, i_error,
                     Ω_error, ω_error, u_error])
    return(data)

def ele_gen_txt_error(foldername, pathsave):
    """ generate .txt file with daily errors of osculating elements
    foldername: string like
    filename: string like
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
    
    data_year = np.array([[0, 0, 0, 0, 0, 0, 0]])
    for i in range(0, len(entries_new)):
        file_i = entries_new[i]
        data_i = ele_get_errors(foldername, file_i)
        data_year = np.vstack((data_year, data_i))
    data_year = data_year[1:]
    
    name = pathsave + "errors.txt"
    np.savetxt(name, data_year, delimiter = ' ')
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
    ele_gen_txt_error(foldername, pathsave)
    print("DDDDOOOONNNNEEEE")
# %% process ACC data
def find_string_length(string, target_string):
    """ gives length of whole string until end of target string
    ex: string = "123456789", target_string = "456"
        --> length = 6
    Input:
        string: string-like
        target_string: string-like
    Output:
        length: integer
    """
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
    """ read bias or scale from strings
    something is happening ...
    """
    string = line[find_string_length(line, "(R,S,W)"):]
    string_list = string.split(' ')
    string_list = [obj for obj in string_list if obj != ''] # remove ''
    new_string_list = [float(obj) for obj in string_list]
    return(new_string_list)

def file_number_sorter_acc(liste, time_format, yday0):
    """ sort entries by date
    Input:
        liste: contains entries
        time_format: "mjd" or "yday"
        yday0: see description of master_acc
    Output:
        new_list: sorted by date
    """
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

def duo_file_reader(path_1, path_2, dt, calib):
    """ read file containing bias and scale and
    file containing uncalibrated data
    Input:
        (path_i = foldername_i/file_name_i)
        path_1 -> txt file with bias and scale; filename with mjd
        path_2 -> acc file with data; filename with yday
        dt: sampling in seconds
        calib: 0 for no calibration, 1 for calibration
    Output:
        data: calibrated data
        m_s_b_list = [mean_array, scale_array, bias_array]
    """
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
    
    # subtract mean
    for i in range(1, 3 + 1):
        mean = np.mean(data[:, i])
        data[:, i] -= mean
        mean_array = np.append(mean_array, mean)
    
    # apply scale and bias and pay attention to units
    for i in range(1, 3 + 1):
        scale = scale_list[i - 1]
        bias = bias_list[i - 1]
        
        if (calib == 1):
            data[:, i] = data[:, i] * scale + bias * 1000 # m/s^2 -> mm/s^2
        
        data[:, i] /= 1000 # mm/s^2 -> m/s^2
        
        scale_array = np.append(scale_array, scale)
        bias_array = np.append(bias_array, bias)
    
    # calculate absolute acceleration
    a_acc_list = np.array([])
    for i in range(0, len(data)):
        abs = 0
        for j in range(1, 4):
            abs += (data[i, j])**2
        a_acc_list = np.append(a_acc_list, math.sqrt(abs))
    data = np.hstack((data, np.array([a_acc_list]).T))
    
    m_s_b_list = [mean_array, scale_array, bias_array]
    return(data, m_s_b_list)

def file_list_generator(foldername, file_format, time_format, yday0):
    """ create list with file names
    Input:
        foldername: string like
        file_format: ".txt" or ".ACC"
        time_format: "mjd" or "yday"
        yday0: see description of master_acc
    Output:
        entries_c: entries
    """
    entries_a = os.listdir(str(foldername))
    entries_b = np.array([])
    for i in range(0, len(entries_a)):
        file_i = entries_a[i]
        if (file_i[-4:] == file_format):
            entries_b = np.append(entries_b, file_i)
    entries_b = file_number_sorter_acc(entries_b, time_format, yday0)
    entries_c = np.array([])
    for i in range(0, len(entries_b)): # get rid of '.DS_Store'
        file_i = entries_b[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 10000): # avoid empty files
            entries_c = np.append(entries_c, file_i)
    return(entries_c)

def master_acc(foldername, subfoldername_1, subfoldername_2,
               yday0, dt, calib):
    """ create master files for ACC data
    Input:
        foldername: folder containing subfolder_1 and subfolder_2
        subfoldername_1: subfolder_1 with bias and scales to extract
                         in my case they are .txt files
        subfoldername_2: subfolder_2 with uncalibrated ACC data
                         in my case they are .ACC files
        yday0: 0 if data files have not an additional zero at ending e.g. 58270
               1 if data files have an additional zero at ending e.g. 183240
        dt: sampling in seconds
        calib: 0 to create uncalibrated ACC data
               1 to calibrate ACC data with mean, bias and scale
    remark: how the calibrated ACC data is generated heavily
            depends on the structure of files containing the
            uncalibrated data and the files containing the
            biases and scales.
            important is that for each day and for each direction
            to each value Γ the following formula is applied:
            Γ_calibrated = (Γ_uncalibrated - Γ_mean) * Γ_scale + Γ_scale.
            also important: pay attention to units, the calibrated values
            should be in m/s^2.
    """
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
    
    data_year = np.zeros((1, 5)) # [t, R, S, W, A (absolute)]
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
            data_i, m_s_b_i_list = duo_file_reader(path_1, path_2, dt, calib)
            
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
    
    path = foldername + "/ALL/"
    os.mkdir(path)
    
    acc_name = "acc.txt"
    if (calib == 0):
        acc_name = "acc_uncalibrated.txt"
    else:
        acc_name = "acc_calibrated.txt"
    name_year = path + acc_name
    name_mean = path + "mean.txt"
    name_scale = path + "scale.txt"
    name_bias = path + "bias.txt"
    
    np.savetxt(name_year, data_year, delimiter = ' ')
    np.savetxt(name_mean, mean_year, delimiter = ' ')
    np.savetxt(name_scale, scale_year, delimiter = ' ')
    np.savetxt(name_bias, bias_year, delimiter = ' ')
    print("DDDDOOOONNNNEEEE")
# %% process MAN data
def man_file_read(foldername, filename):
    """ read MAN file and extract start and end time of manoeuvres
    needs conversion from UTC to GPS time
    Input:
        foldername: string-like
        filename: string-like
    Output:
        [[t1_start, t1_end], [t2_start, t2_end], ...]
    """
    # only get start and end time
    data = np.loadtxt(foldername + "/" + filename, skiprows = 1, usecols = (1, 2))
    data += 18 / (24 * 60 * 60) # GPS = UTC + 18s
    return(data)

def master_man(foldername, pathsave):
    """ generate a master file containing all MANs (= manoeuvres)
    contains start and end time of the MANs
    Input:
        foldername: string-like
    Output:
        [[t1_start, t1_end], [t2_start, t2_end], ...]
    """
    entries_old = os.listdir(str(foldername)) # get entries
    entries = np.array([])
    for i in range(0, len(entries_old)):
        file_i = entries_old[i]
        if (file_i[-4:] == '.MAN'): # only want MAN files
            entries = np.append(entries, file_i)
    entries.sort() # sort by day
    
    entries_new = np.array([])
    for i in range(0, len(entries)): # get rid of '.DS_Store'
        file_i = entries[i]
        file_size = os.path.getsize(foldername + '/' + file_i)
        if (file_size > 100): # avoid empty files (= 86 bytes)
            entries_new = np.append(entries_new, file_i)
    
    data_year = np.array([[0, 0]]) # delete later
    for i in range(0, len(entries_new)):
        file_i = entries_new[i]
        data_i = man_file_read(foldername, file_i)
        data_year = np.vstack((data_year, data_i))
    data_year = data_year[1:]
    
    name_year = pathsave + "MAN.txt" # path/name of master file
    np.savetxt(name_year, data_year, delimiter = ' ')
    print("DDDDOOOONNNNEEEE")
# %% process ap data
def yyyymmdd_to_mjd(date_list):
    """ conversion of time format
    date_list = [yyyy, mm, dd]
    """
    yyyy = '%.0f' % date_list[0]
    mm = '%.0f' % date_list[1]
    dd = '%.0f' % date_list[2]
    date_str = yyyy + '-' + mm + '-' + dd
    t_obj = Time(date_str)
    return(t_obj.mjd)

def flx_get_data_ymd(path):
    """ get ap index data
    path = 'folder/FLXAP_P.FLX'
    """
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
# %% process CME data
def yyyymmddhhmm_to_mjd(string):
    """ used for CME arrival times conversion
    ex. converting 2018-07-10T11:25Z to MJD
    """
    t_obj = Time(string, format = 'isot', scale = 'utc')
    t_mjd = t_obj.mjd
    mjd_float = float(t_mjd)
    return(mjd_float)
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
# %% spectral analysis
def fft(array):
    """ perform Fast Fourier Transformation
    array = [[t_1, x_1], [t_2, x_2], ...]
    gives: orig_list = [period_orig, amp_orig] (periods, amplitudes)
    """
    t_array = array[:,0]
    y_array = array[:,1]
    size = np.shape(y_array)[-1]
    N = (size + 1) // 2
    
    # original data orig
    fourier_orig = np.fft.fft(y_array) / size # preparing the coefficients
    new_fourier_orig = fourier_orig[1 : N] # only want a1, ..., an
    freq_orig = np.fft.fftfreq(size, d = t_array[1] - t_array[0]) # preparing the frequencies
    new_freq_orig = freq_orig[1 : N] # only want frequencies to a1, ..., an
    period_orig = 1 / new_freq_orig # period
    amp_orig = 2 * np.abs(new_fourier_orig) # amplitude
    amp_tild_orig = amp_orig**2 / sum(amp_orig) # amplitude for the power spectrum
    
    orig_list = [period_orig, amp_orig]
    return(orig_list)

def zero_setter(x_list, y_list, position, fac_low, fac_upp):
    """ set elements in a sublist of a list to zero
    xlist - periods
    ylist - amplitudes
    position - index in xlist of period with max amplitude
    fac_low, fac_upp - to define interval size
    """
    # want that the amplitudes of all periods
    # in interval [p * fac_low, p * fac_upp] are set to 0
    newylist = y_list.copy()
    newylist = np.array(newylist)
    
    x_p = x_list[position] # dominant period p
    
    x_low = x_p * fac_low
    x_upp = x_p * fac_upp
    
    end = np.argmin(np.abs(x_list - x_low))
    start = np.argmin(np.abs(x_list - x_upp))
    
    newylist[start : end + 1] = 0
    return(newylist)

def selector(array, pos_list):
    """ select elements in an array based on index list
    array - list
    pos_list - list with indices of array
    """
    newlist = np.array([])
    for p in pos_list:
        newlist = np.append(newlist, array[p])
    return(newlist)

def peak_finder_adv(array, lcm_list, interval_n_list):
    """ detect peaks in spectrum
    array = [[p_1, a_1], [p_2, a_2], ...]
    lcm_list = [p_spec, ε, N, p_max, thresh_fac, limit]
        thresh_fac: detect amplitudes > thresh_fac * max_amp
        p_max: search periods shorter than p_max
    interval_n_list = [[p_1a, p_1b, n1], [p_2a, p_2b, n2], ...]
        interval is from p_ia to p_ib (p_ia < p_ib)
        Detect the ni largest amplitudes in that interval.
        For no interval search: interval_n_list = [[0,0]].
        
    gives:
        periods: periods of detected peaks
        maxlist: amplitudes of detected peaks
        threshold: everything above this amplitude got detected
        xlist[c + count_0]: don't ask me why
    
    remark: do not use this algorithm on variables with a spectrum
        which is just increasing (for example: omega_upp is bad)
    """
    thresh_fac, p_max = lcm_list[4], lcm_list[3]
    
    xlist = array[:,0]
    ylist = array[:,1]
    
    fac = 0.1
    fac_upp = 1 + fac
    fac_low  = 1 - fac
    newxlist = xlist + 0 # copy list
    newylist = ylist + 0 # copy list
    
    c = 0
    while (newxlist[c] > p_max):
        newylist[c] = 0
        c += 1
    
    # find period with largest amplitude
    # WARNING: periods go from big to small -> in plot it starts from the right
    # pay attention to which words in comments are set in ""
    #print("argmax = ", np.argmax(newylist))
    count_0 = 0 # counter
    if (np.argmax(newylist) == c):
        #print("ARGMAX")
        # if true -> linear increase
        # start from the "right": find an instance where the
        # value on the neighbouring "left" is greater
        # every values to the "right" should be set to 0
        # to kill the linear increase
        while (count_0 < 10000): # some value to avoid accidental infinity
            if (newylist[c + count_0] < newylist[c + count_0 + 1]):
                # beginning of ascent found
                newylist[0 : c + count_0] = 0 # set everything before to 0
                #print(count_0)
                #print(newxlist[count_0 + c])
                break
            count_0 += 1
    
    pos = np.argmax(newylist)
    m = max(newylist)
    poslist = [pos]
    maxlist = [m]
    # example: thresh_fac = 0.5 --> threshold = 0.5 * amp_max
    threshold = thresh_fac * m
    
    counter = 1
    while (m >= threshold) and (counter < 100):
        newylist = zero_setter(newxlist, newylist, poslist[-1], fac_low, fac_upp)
        if (pos == np.argmax(newylist)):
            break # avoid infinity loop at border
        pos = np.argmax(newylist)
        m = max(newylist)
        if (m >= threshold):
            poslist.append(pos)
            maxlist.append(m)
        counter += 1
    
    # search interval
    if (interval_n_list[0][0] != interval_n_list[0][1]):
        for i in range(0, len(interval_n_list)):
            newnewxlist = xlist + 0 # copy list
            newnewylist = ylist + 0 # copy list
            
            p_low = interval_n_list[i][1]
            p_upp = interval_n_list[i][0]
            ni = interval_n_list[i][2]
            
            p_low_arg = np.argmin(np.abs(newnewxlist - p_low))
            p_upp_arg = np.argmin(np.abs(newnewxlist - p_upp))
            
            if (p_low_arg == p_upp_arg):
                p_low_arg -= 1
                p_upp_arg += 1
                if (p_low_arg == -1):
                    p_low_arg = 0
            
            newnewylist[:p_low_arg] = 0
            newnewylist[p_upp_arg:] = 0
            
            count = 0
            while (count < ni):
                interval_y_max = max(newnewylist)
                interval_x_max = np.argmax(newnewylist)

                poslist.append(interval_x_max)
                maxlist.append(interval_y_max)
                newnewylist = zero_setter(newnewxlist, newnewylist, poslist[-1], fac_low, fac_upp)
                count += 1
    
    periods = selector(xlist, poslist)
    maxlist = np.array(maxlist)
    return(periods, maxlist, threshold, xlist[c + count_0])

def spectrum(data, lcm_list, interval_n_list):
    """ performs spectral analysis
    
    data: [t_i, el_i]
    lcm_list = [p_spec, ε, N, p_max, thresh_fac, limit]
    interval_n_list = [[p_1a, p_1b, n1], [p_2a, p_2b, n2], ...]
        interval is from p_ia to p_ib
        detect the ni largest amplitudes in that interval.
        for no interval search: interval_n_list = [[0,0]]
    
    gives:
    spectrum_list = [p_o_list, a_o_list, per_list, amp_list]
        p_o_list = periods of fft
        a_o_list = amplitudes of fft
        per_list = periods that have a peak amplitude
        amp_list = peak amplitudes sorted from highest to lowest
    """
    orig_list = fft(data) # spectrum
    p_o_list, a_o_list = orig_list[0], orig_list[1] # period and amplitude
    p_a_array = np.vstack((p_o_list, a_o_list)).T
    
    # per_list = periods of peaks
    # amp_list = amplitudes of peaks
    per_list, amp_list, trash1, trash2 = peak_finder_adv(p_a_array, lcm_list, interval_n_list)
    spectrum_list = [p_o_list, a_o_list, per_list, amp_list]
    return(spectrum_list)
# %% smoothing
def sorter(array, n, direction):
    """ sort array
    only for arrays with 2 columns!
    n: column used for sorting input
    direction = 1 -> from lowest to highest
    direction = - 1 -> from highest to lowest
    """
    if (len(array) == 1):
        return(array)
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

def round_dot5(x):
    """ proper rounding
    round 0.5 to 1.0
    13.5 -> 14.0, 13.49 -> 13.0
    """
    if (x - int(x) == 0.5):
        return(round(x + 0.5, 0))
    else:
        return(round(x, 0))

def round_uneven(x):
    """ round to nearest uneven number
    if x is an even number, then it should return x + 1
    """
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

def mod_abs(u, v):
    """ find absolute (smallest) remainder
    idea illustrated with example:
    u = 26
    v = 7
    u % v = 5 = u - 3 * v
    v % u = v = v - 0 * u
    but: mod_abs = 2 = | u - 4 * v |
    """
    mod1 = u % v
    fac1 = (u - mod1) / v # corresponding factor
    mod2 = v - (u % v)
    fac2 = (u + mod2) / v # corresponding factor
    if (mod1 < mod2):
        return(mod1, fac1)
    if (mod2 < mod1):
        return(mod2, fac2)

def lcm_status_1(k, lcm_copy, n, tilde_p_i, hut_ε_i, fac):
    """ To give update """
    str1 = "k = %3d: " % k
    str2 = "|lcm - f * p_%1d| = " % n
    str3 = "|%12.3f - %3d * %9.3f|" % (lcm_copy, fac, tilde_p_i)
    str4 = " = %6.2f = ε_%1d"% (hut_ε_i, n)
    print(str1 + str2 + str3 + str4)

def lcm_status_2(n, hut_ε_i, hut_ε):
    """ To give update """
    str1 = "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    str2 = "----------------------- "
    str3 = "ε_%1d = %6.2f < %6.2f = ε" % (n, hut_ε_i, hut_ε)
    str4 = " ------------------------"
    print(str2 + str3 + str4)
    print(str1)

def lcm_as_Δt(lcm_list, dt, per_list):
    """ lcm is least common multiple lcm of periods in per_list times N
    lcm = N * (ε_1 + f_1 * p_1) = N * (ε_2 + f_2 * p_2) = ...
    where ε_i < ε / N
    give lcm and all periods that could be "aligned"
    
    lcm_list: [p_spec, ε, N, p_max, thresh_fac, limit]
        ε: radius in dt units. the multiple of the aligned
            periods should be within this radius around the lcm
        N: if a N-times the lcm should be evaluated
        limit: upper boundary of the lcm
    
    This function gives only an "approximate" lcm.
    For my needs it was good enough, but it can be improved.
    """
    # do everything in units of the timestep
    ε, N, limit = lcm_list[1], lcm_list[2], lcm_list[5]
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
        hut_ε_i, fac = mod_abs(lcm_copy, tilde_p_i)
        lcm_status_1(k, lcm_copy, n, tilde_p_i, hut_ε_i, fac) # message
        if (hut_ε_i < hut_ε):
            lcm_status_2(n, hut_ε_i, hut_ε) # message
            tilde_lcm *= k
            n += 1
            k = 0
        k += 1
    return(N * tilde_lcm * dt, per_list[:n])

def A_filter(n, q):
    """ Preparation for Savitzky-Golay Filter
    n: window width
    q: filter degree
    """
    n_2 = int((n - 1) / 2)
    n_list = list(range(-n_2, n_2 + 1))
    n_array = np.array([n_list]).T
    A_mat = np.array([[1] * len(n_list)]).T
    for i in range(1, q + 1):
        A_mat = np.block([A_mat, n_array**i])
    return(A_mat)

def B_filter(n, q):
    """ Preparation for Savitzky-Golay Filter
    n: window width
    q: filter degree
    """
    A_mat = A_filter(n, q)
    B_mat = np.linalg.inv(A_mat.T @ A_mat) @ A_mat.T
    return(B_mat)

def low_pass_filter(array, n, q):
    """ executes Savitzky-Golay filter
    array = [[x1, y1], [x2, y2], ...]
    n: window width
    q: filter degree
    Boundary points: points should be located on the polynomial of order 0
        used on the last possible point.
    """
    t_list = array[:, 0]
    y_list = array[:, 1]
    rad = int((n - 1) / 2) # radius
    B_mat = B_filter(n, q)
    new_y_list = []
    for i in range(0, len(t_list[rad : len(t_list) - rad])):
        y_array = np.array([y_list[rad + i - rad : rad + i + rad + 1]]).T
        x_hut = B_mat @ y_array
        new_y_list.append(x_hut[0][0])
    new_y_middle = np.array([new_y_list])
    # Boundary points
    B_mat_border, A_mat_border = B_filter(n, 0), A_filter(n, 0)
    y_border_1 = y_list[: 2 * rad + 1] # boundary points at the start
    y_border_2 = y_list[- 2 * rad - 1 :] # boundary points at the end
    x_hut_1 = B_mat_border @ y_border_1
    x_hut_2 = B_mat_border @ y_border_2
    new_y_border_1 = np.array([(A_mat_border @ x_hut_1)[:rad]]) # take first half
    new_y_border_2 = np.array([(A_mat_border @ x_hut_2)[-rad:]]) # take second half
    # put everything together
    new_y_array = np.block([new_y_border_1, new_y_middle, new_y_border_2]).T
    new_array = np.block([np.array([t_list]).T, new_y_array])
    return(new_array)

def smoother(data, per_list, amp_list, lcm_list, q):
    """ applies Savitzky-Golay filter to data
    
    data: [t_i, el_i]
    per_list: peak periods sorted from lowest to highest
    amp_list: corresponding amplitudes
    lcm_list = [p_spec, ε, N, p_max, thresh_fac, limit]
    q: polynomial order for filtering
    
    gives:
    smooth_list = [data_smoothed, per0_list, amp0_list, n_Δt, Δt_n, Δt_dt]
        data_smoothed: data smoothed with Savizky-Golay filter
        per0_list: periods regarded when smoothed
        amp0_list: corresponding amplitudes
        n_Δt: width of smoothing window (uneven)
        Δt_n: smoothing period
        Δt_dt: Δt % dt ("error")
    """
    p_spec = lcm_list[0] # manually chosen smoothing period
    
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
    else:
        Δt, per0_list = lcm_as_Δt(lcm_list, dt, per_list)
        amp0_list = amp_list[:len(per0_list)]
    print("Δt", Δt)
    
    n_Δt = round_uneven(Δt / dt) # window width for filtering
    Δt_n = n_Δt * dt # corresponding time interval for filtering
    Δt_dt = Δt_n % dt # "error"
    print("Δt_n = %.3f | n_Δt = %d | len = %d" % (Δt_n, n_Δt, len(data)))
    
    # smooth data
    data_smoothed = low_pass_filter(data, n_Δt, q)
    print("SMOOTHING FINISHED!!!")
    smooth_list = [data_smoothed, per0_list, amp0_list, n_Δt, Δt_n, Δt_dt]
    return(smooth_list)

def smoother_give_n(data, per_list, amp_list, lcm_list, q):
    """ applies Savitzky-Golay filter to data
    
    data: [t_i, el_i]
    per_list: peak periods sorted from lowest to highest
    amp_list: corresponding amplitudes
    lcm_list = [p_spec, ε, N, p_max, thresh_fac, limit]
    q: polynomial order for filtering
    
    gives:
    smooth_list = [data_smoothed, per0_list, amp0_list, n_Δt, Δt_n, Δt_dt]
        data_smoothed: data smoothed with Savizky-Golay filter
        per0_list: periods regarded when smoothed
        amp0_list: corresponding amplitudes
        n_Δt: width of smoothing window (uneven)
        Δt_n: smoothing period
        Δt_dt: Δt % dt ("error")
    """
    p_spec = lcm_list[0] # manually chosen smoothing period
    
    # sort from strongest to weakest period:
    per_amp_array = sorter(np.vstack((per_list, amp_list)).T, 1, -1)
    per_list = per_amp_array[:, 0]
    amp_list = per_amp_array[:, 1]
    
    # evaluate optimal window width
    dt = (data[-1, 0] - data[0, 0])/(len(data) - 1)
    print("p_spec", p_spec)
    if (p_spec != 0):
        Δt = p_spec
        per0_list , amp0_list = [], []
    else:
        Δt, per0_list = lcm_as_Δt(lcm_list, dt, per_list)
        amp0_list = amp_list[:len(per0_list)]
    print("Δt", Δt)
    
    n_Δt = round_uneven(Δt / dt) # window width for filtering
    Δt_n = n_Δt * dt # corresponding time interval for filtering
    Δt_dt = (Δt_n - Δt) / dt # "error"
    print("Δt_n = %.3f | n_Δt = %d | len = %d" % (Δt_n, n_Δt, len(data)))
    
    return(n_Δt, Δt_n, Δt_dt)
# %% functions for lsa (least squares adjustment)
def B_n_θ(n, θ_small, τ_n_list, t_n_list, Δτ, κ, n_s, K, p_list):
    """ compute matrix B_n_θ
    Input:
        n: current subinterval
        θ_small: current fit parameter
        τ_n_list = [τ_(n-1), τ_n]: vertices of n-th subinterval
        t_n_list: epochs in n-th subinterval
        Δτ: duration of subinterval
        κ: number of epochs per subinterval (this is a small kappa)
        n_s: number of subintervals
        K: number of data points
        p_list: periods to be fitted
    Output:
        mat = B_n_θ
    """
    R = len(p_list) # 2 * R + 1 = θ_tot
    τ_n_1, τ_n = τ_n_list[0], τ_n_list[1]
    t_n_list = np.array(t_n_list)
            
    mat_t = np.vstack((-t_n_list, t_n_list)).T
    mat_τ = np.vstack((τ_n * np.ones(κ), - τ_n_1 * np.ones(κ))).T
    mat = (mat_t + mat_τ) / Δτ
    
    if (θ_small >= 1 and θ_small <= R): # [pd_μ_r_(n-1), pd_μ_r_n]_t=t_k
        r = θ_small
        p_r = p_list[r - 1] # r = 1, ..., R
        mat_t_symmetric = np.vstack((t_n_list, t_n_list)).T
        mat_μ_r = np.sin(2 * np.pi / p_r * mat_t_symmetric)
        mat = mat * mat_μ_r
    elif (θ_small >= R + 1 and θ_small <= 2 * R): # [pd_η_r_(n-1), pd_η_r_n]_t=t_k
        r = θ_small - R
        p_r = p_list[r - 1] # r = 1, ..., R
        mat_t_symmetric = np.vstack((t_n_list, t_n_list)).T
        mat_η_r = np.cos(2 * np.pi / p_r * mat_t_symmetric)
        mat = mat * mat_η_r
    # else (θ_small == 0): mat = mat
    return(mat)

def H_n_θ1_θ2(n, θ_small_list, τ_n_list, t_n_list,
              Δτ, κ, n_s, K, p_list, P_n):
    """ compute matrix H_n_θ1_θ2
    Input:
        n: current subinterval
        θ_small: current fit parameter
        τ_n_list = [τ_(n-1), τ_n]: vertices of n-th subinterval
        t_n_list: epochs in n-th subinterval
        Δτ: duration of subinterval
        κ: number of epochs per subinterval (this is a small kappa)
        n_s: number of subintervals
        K: number of data points
        p_list: periods to be fitted
        P_n: weight matrix for n-th subinterval
    Output:
        mat = H_n_θ1_θ2
    """
    # is actually E_(n,(q_tilde),(q_tilde_prime))
    θ_small_1, θ_small_2 = θ_small_list[0], θ_small_list[1]
    B_n_θ1_mat = B_n_θ(n, θ_small_1, τ_n_list, t_n_list,
                       Δτ, κ, n_s, K, p_list)
    B_n_θ2_mat = B_n_θ(n, θ_small_2, τ_n_list, t_n_list,
                       Δτ, κ, n_s, K, p_list)
    
    mat_center_center = B_n_θ1_mat.T @ P_n @ B_n_θ2_mat
    mat_center_left = np.zeros((2, n - 1))
    mat_center_right = np.zeros((2, n_s - n))
    
    mat_top = np.zeros((n - 1, n_s + 1))
    mat_center = np.hstack((mat_center_left, mat_center_center, mat_center_right))
    mat_bottom = np.zeros((n_s - n, n_s + 1))
    
    mat = np.vstack((mat_top, mat_center, mat_bottom))
    return(mat)

def H_n(n, τ_n_list, t_n_list, Δτ, κ, n_s, K, p_list, P_n):
    """ compute matrix H_n = (D_n)^T * D_n * D_n
    Input:
        n: current subinterval
        τ_n_list = [τ_(n-1), τ_n]: vertices of n-th subinterval
        t_n_list: epochs in n-th subinterval
        Δτ: duration of subinterval
        κ: number of epochs per subinterval (this is a small kappa)
        n_s: number of subintervals
        K: number of data points
        p_list: periods to be fitted
        P_n: weight matrix for n-th subinterval
    Output:
        mat = H_n
    """
    R = len(p_list)
    θ_tot = 2 * R + 1
    θ_big = θ_tot - 1
    
    mat = np.zeros((1, θ_tot * (n_s + 1))) # to be deleted
    for θ_small_1 in range(0, θ_big + 1): # stack row for row
        mat_help = np.zeros((n_s + 1, 1)) # to be deleted
        for θ_small_2 in range(0, θ_big + 1): # stack column for column
            θ_small_list = [θ_small_1, θ_small_2]
            H_n_θ1_θ2_mat = H_n_θ1_θ2(n, θ_small_list, τ_n_list, t_n_list,
                                      Δτ, κ, n_s, K, p_list, P_n)
            mat_help = np.hstack((mat_help, H_n_θ1_θ2_mat))
        mat_help = mat_help[:, 1:] # delete start
        mat = np.vstack((mat, mat_help))
    mat = mat[1:] # delete start
    return(mat)

def N_eff(τ_n_list_list, t_n_list_list, Δτ,
          κ, n_s, K, p_list, P_n_list):
    """ compute matrix N = D^T * P * D efficiently (this is why there is a 'eff')
    Input:
        τ_n_list_list = [[τ_(n-1), τ_n]]: pair wise epochs of vertices
        t_n_list_list: epochs split into subintervals
        Δτ: duration of subinterval
        κ: number of epochs per subinterval (this is a small kappa)
        n_s: number of subintervals
        K: number of data points
        p_list: periods to be fitted
        P_n_list: list with weight matrices for each subinterval
    Output:
        mat = N
    """
    R = len(p_list)
    θ_tot = 2 * R + 1
    
    mat = np.zeros((θ_tot * (n_s + 1), θ_tot * (n_s + 1))) # start
    for n in range(1, n_s + 1):
        τ_n_list = τ_n_list_list[n - 1] # index starts at 0
        t_n_list = t_n_list_list[n - 1] # index starts at 0
        P_n = P_n_list[n - 1] # index starts at 0
        H_n_mat = H_n(n, τ_n_list, t_n_list,
                      Δτ, κ, n_s, K,p_list, P_n)
        mat = mat + H_n_mat
    return(mat)

def I_n_θ(n, θ_small, τ_n_list, t_n_list,
                Δτ, κ, n_s, K, p_list, P_n):
    """ compute matrix I_n_θ
    Input:
        n: current subinterval
        θ_small: current fit parameter
        τ_n_list = [τ_(n-1), τ_n]: vertices of n-th subinterval
        t_n_list: epochs in n-th subinterval
        Δτ: duration of subinterval
        κ: number of epochs per subinterval (this is a small kappa)
        n_s: number of subintervals
        K: number of data points
        p_list: periods to be fitted
        P_n: weight matrix for n-th subinterval
    Output:
        mat = I_n_θ
    """
    B_n_θ_mat = B_n_θ(n, θ_small, τ_n_list, t_n_list,
                      Δτ, κ, n_s, K, p_list)
    mat_center = B_n_θ_mat.T @ P_n
    
    mat_top = np.zeros((n - 1, κ))
    mat_bottom = np.zeros((n_s - n, κ))
    
    mat = np.vstack((mat_top, mat_center, mat_bottom))
    return(mat)

def I_n(n, τ_n_list, t_n_list, Δτ, κ, n_s, K, p_list, P_n):
    """ compute matrix I_n
    Input:
        n: current subinterval
        τ_n_list = [τ_(n-1), τ_n]: vertices of n-th subinterval
        t_n_list: epochs in n-th subinterval
        Δτ: duration of subinterval
        κ: number of epochs per subinterval (this is a small kappa)
        n_s: number of subintervals
        K: number of data points
        p_list: periods to be fitted
        P_n: weight matrix for n-th subinterval
    Output:
        mat = I_n
    """
    R = len(p_list)
    θ_tot = 2 * R + 1
    θ_big = θ_tot - 1
    
    mat = np.zeros((1, κ))
    for θ_small in range(0, θ_big + 1):
        I_n_θ_mat = I_n_θ(n, θ_small, τ_n_list, t_n_list,
                          Δτ, κ, n_s, K, p_list, P_n)
        mat = np.vstack((mat, I_n_θ_mat))
    mat = mat[1:] # delete start
    return(mat)

def DT_P(τ_n_list_list, t_n_list_list, Δτ,
         κ, n_s, K, p_list, P_n_list):
    """ compute matrix D^T * P
    Input:
        τ_n_list_list = [[τ_(n-1), τ_n]]: pair wise epochs of vertices
        t_n_list_list: epochs split into subintervals
        Δτ: duration of subinterval
        κ: number of epochs per subinterval (this is a small kappa)
        n_s: number of subintervals
        K: number of data points
        p_list: periods to be fitted
        P_n_list: list with weight matrices for each subinterval
    Output:
        mat = D^T * P
    """
    R = len(p_list)
    θ_tot = 2 * R + 1
    
    mat = np.zeros((θ_tot * (n_s + 1), 1)) # to be deleted
    for n in range(1, n_s + 1):
        τ_n_list = τ_n_list_list[n - 1] # index starts at 0
        t_n_list = t_n_list_list[n - 1] # index starts at 0
        P_n = P_n_list[n - 1] # index starts at 0
        I_n_mat = I_n(n, τ_n_list, t_n_list,
                      Δτ, κ, n_s, K, p_list, P_n)
        mat = np.hstack((mat, I_n_mat))
    mat = mat[:, 1:] # delete start
    return(mat)

def x_θ_n(x_vec, n, θ_small, n_s):
    """ compute vector x_θ_n = [[x_θ_(n-1)], [x_θ_n]]
    Input:
        x_vec = [a_n, μ_(r,n), η_(r,n)]: vector of fit parameters
        n: current subinterval
        θ_small: current fit parameter
        n_s: number of subintervals
    Output:
        vec = x_θ_n
    """
    vec = x_vec[(n_s + 1) * θ_small + n - 1 : (n_s + 1) * θ_small + n + 1]
    return(vec)

def J_n(n, τ_n_list, t_n_list, Δτ, κ, n_s, K, p_list, x_vec):
    """ compute vector J_n
    Input:
        n: current subinterval
        τ_n_list = [τ_(n-1), τ_n]: vertices of n-th subinterval
        t_n_list: epochs in n-th subinterval
        Δτ: duration of subinterval
        κ: number of epochs per subinterval (this is a small kappa)
        n_s: number of subintervals
        K: number of data points
        p_list: periods to be fitted
        x_vec = [a_n, μ_(r,n), η_(r,n)]: vector of fit parameters
    Output:
        vec = J_n
    """
    R = len(p_list)
    θ_tot = 2 * R + 1
    θ_big = θ_tot - 1
    
    vec = np.zeros((κ, 1))
    for θ_small in range(0, θ_big + 1):
        B_n_θ_mat = B_n_θ(n, θ_small, τ_n_list, t_n_list,
                          Δτ, κ, n_s, K, p_list)
        x_θ_n_vec = x_θ_n(x_vec, n, θ_small, n_s)
        vec = vec + B_n_θ_mat @ x_θ_n_vec
    return(vec)

def D_x(τ_n_list_list, t_n_list_list,
        Δτ, κ, n_s, K, p_list, x_vec):
    """ compute vector D * x
    Input:
        τ_n_list_list = [[τ_(n-1), τ_n]]: pair wise epochs of vertices
        t_n_list_list: epochs split into subintervals
        Δτ: duration of subinterval
        κ: number of epochs per subinterval (this is a small kappa)
        n_s: number of subintervals
        K: number of data points
        p_list: periods to be fitted
        x_vec = [a_n, μ_(r,n), η_(r,n)]: vector of fit parameters
    Output:
        vec = D * x
    """
    vec = np.zeros((1, 1)) # to be deleted
    for n in range(1, n_s + 1):
        τ_n_list = τ_n_list_list[n - 1] # index starts at 0
        t_n_list = t_n_list_list[n - 1] # index starts at 0
        J_n_vec = J_n(n, τ_n_list, t_n_list,
                      Δτ, κ, n_s, K, p_list, x_vec)
        vec = np.vstack((vec, J_n_vec))
    vec = vec[1:] # delete start
    return(vec)

def ravel(list_liray):
    """ flatten list once
    Input:
        list_liray = [1, 2, np.array([3,4]), np.array([5,6,7])]
    Output:
        liste = [1, 2, 3, 4, 5, 6, 7]
    """
    liste = []
    for i in range(0, len(list_liray)):
        obj_i = list_liray[i]
        if (type(obj_i) == list or type(obj_i) == np.ndarray):
            liste.extend(obj_i) # do not consider a deeper structure
        else:
            liste.append(obj_i)
    return(liste)

def compute_data(τ_n_list_list, t_n_list_list,
                 Δτ, κ, n_s, K, p_list, x_vec):
    """ reproduce data with fit parameters
    Input:
        τ_n_list_list = [[τ_(n-1), τ_n]]: pair wise epochs of vertices
        t_n_list_list: epochs split into subintervals
        Δτ: duration of subinterval
        κ: number of epochs per subinterval (this is a small kappa)
        n_s: number of subintervals
        K: number of data points
        p_list: periods to be fitted
        x_vec = [a_n, μ_(r,n), η_(r,n)]: vector of fit parameters
    Output:
        comp_data = [[t_i, a_i]]: data reproduced with fit parameters
    """
    R = len(p_list)
    θ_tot = 2 * R + 1
    θ_big = θ_tot - 1
    comp_list = np.array([])
    for n in range(1, n_s + 1):
        τ_n_list = τ_n_list_list[n - 1]
        t_n_list = t_n_list_list[n - 1]
        τ_n_1 = τ_n_list[0] # τ_(n-1)
        τ_n = τ_n_list[1]
        for ϰvar in range(0, κ): # ϰvar because python can't distinguish between ϰ and κ
            t = t_n_list[ϰvar]
            t_covec = np.array([[τ_n - t, t - τ_n_1]]) / Δτ
            tot = 0
            for θ_small in range(0, θ_big + 1):
                vec = x_θ_n(x_vec, n, θ_small, n_s) # x_θ_n function can be used here as well
                if (θ_small >= 1 and θ_small <= R): # μ
                    r = θ_small
                    p_r = p_list[r - 1]
                    vec = vec * math.sin(2 * np.pi / p_r * t)
                elif (θ_small >= R + 1 and θ_small <= 2 * R): # η
                    r = θ_small - R
                    p_r = p_list[r - 1]
                    vec = vec * math.cos(2 * np.pi / p_r * t)
                # else (θ_small == 0): vec = vec
                tot = tot + t_covec @ vec
            comp_list = np.append(comp_list, tot)
    t_list = np.array(ravel(t_n_list_list))
    comp_data = np.vstack((t_list, comp_list)).T
    return(comp_data)

def gen_a_bar_array(τ_n_list_list, t_n_list_list, Δτ,
                    κ, n_s, K, a_bar_list, σ2_a_bar_list):
    """ reproduce data with only a_bar
    Input:
        τ_n_list_list = [[τ_(n-1), τ_n]]: pair wise epochs of vertices
        t_n_list_list: epochs split into subintervals
        Δτ: duration of subinterval
        κ: number of epochs per subinterval (this is a small kappa)
        n_s: number of subintervals
        K: number of data points
        a_bar_list: a_bar parameters
        σ2_a_bar_list: squared errors of a_bar parameters
    Output:
        bar_data = [[t_i, a_i, σ_a_i]] data reproduced with only a_bar
    """
    bar_data = np.array([0, 0, 0])
    for n in range(1, n_s + 1):
        τ_n_list = τ_n_list_list[n - 1]
        t_n_list = t_n_list_list[n - 1]
        τ_n_1 = τ_n_list[0] # τ_(n-1)
        τ_n = τ_n_list[1]
        for ϰvar in range(0, κ):
            t = t_n_list[ϰvar]
            t_covec = np.array([[τ_n - t, t - τ_n_1]]) / Δτ
            t_covec_error = np.array([[(τ_n - t)**2, (t - τ_n_1)**2]]) / (Δτ * Δτ)
            
            vec = x_θ_n(a_bar_list, n, 0, n_s) # x_θ_n function can be used here as well
            vec_error = x_θ_n(σ2_a_bar_list, n, 0, n_s) # x_θ_n function can be used here as well
            
            tot = t_covec @ vec
            tot_error = np.sqrt(t_covec_error @ vec_error)
            
            entry = np.array([t, tot[0], tot_error[0]])
            bar_data = np.vstack((bar_data, entry))
    bar_data = bar_data[1:]
    return(bar_data)

def v(τ_n_list_list, t_n_list_list, Δτ,
      κ, n_s, K, p_list, x_vec, l_vec):
    """ calculate corrections v (is a vector)
    Input:
        τ_n_list_list = [[τ_(n-1), τ_n]]: pair wise epochs of vertices
        t_n_list_list: epochs split into subintervals
        Δτ: duration of subinterval
        κ: number of epochs per subinterval (this is a small kappa)
        n_s: number of subintervals
        K: number of data points
        p_list: periods to be fitted
        x_vec = [a_n, μ_(r,n), η_(r,n)]: vector of fit parameters
        l_vec: vector of observations
    Output:
        v = D * x - l: corrections
    """
    D_x_vec = D_x(τ_n_list_list, t_n_list_list,
                  Δτ, κ, n_s, K, p_list, x_vec)
    return(D_x_vec - l_vec)

def encode_list(liste, code_list):
    """ encoding list into several sublists
    Input:
        liste = [a, b, c, d, e, f, g]
        code_list = [0, 0, 2, 3]
    Output:
        new_list = [a, b, [c, d], [e, f, g]]
    """
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

def splitter(liste, κ, n_s, K):
    """ with the help of encode_list split liste in several lists
    Input:
        liste: list to be splitted into several lists
        κ: number of epochs per subinterval (this is a small kappa)
        n_s: number of subintervals
        K: number of data points
    Output:
        list_list: list with lists
    """
    code = (n_s - 1) * [κ]
    code.append(κ)
    list_list = encode_list(liste, code)
    return(list_list)

def m0_eff(τ_n_list_list, t_n_list_list, Δτ,
           κ, n_s, K, p_list, P_n_list, x_vec, l_vec):
    """ calculate estimated a posteriori standard deviation of unit weight m0
    Input:
        τ_n_list_list = [[τ_(n-1), τ_n]]: pair wise epochs of vertices
        t_n_list_list: epochs split into subintervals
        Δτ: duration of subinterval
        κ: number of epochs per subinterval (this is a small kappa)
        n_s: number of subintervals
        K: number of data points
        p_list: periods to be fitted
        P_n_list: list with weight matrices for each subinterval
        x_vec = [a_n, μ_(r,n), η_(r,n)]: vector of fit parameters
        l_vec: vector of observations
    Output:
        [[m0]]: estimated a posteriori standard deviation of unit weight
    """
    R = len(p_list)
    θ_tot = 2 * R + 1
    v_vec = v(τ_n_list_list, t_n_list_list,
              Δτ, κ, n_s, K, p_list, x_vec, l_vec)
    v_list = ravel(v_vec)
    v_splitted = splitter(v_list, κ, n_s, K)
    tot = 0
    for n in range(1, n_s + 1):
        v_n_list = v_splitted[n - 1]
        P_n = P_n_list[n - 1]
        
        v_n_vec = np.array([v_n_list]).T
        tot += v_n_vec.T @ P_n @ v_n_vec
    tot = tot / (K - θ_tot * (n_s + 1)) # n = K, u = θ_tot * (n_s + 1)
    return(np.sqrt(tot))

def C_constr(n_s, θ_tot):
    """ for constraining only a_bar
    Input:
        n_s: number of subintervals
        θ_tot: number of fit parameters
    Output:
        C_mat: shape(C_mat) = (θ_tot * (n_s + 1), θ_tot * (n_s - 1))
    """
    diag_upp = np.ones(n_s - 1)
    diag_cen = - 2 * np.ones(n_s)
    diag_low = np.ones(n_s + 1)
    a_bar_mat = np.diag(diag_low) + np.diag(diag_cen, 1) + np.diag(diag_upp, 2)
    a_bar_mat = a_bar_mat[: -2]
    zero_mat_1 = np.zeros(((θ_tot - 1) * (n_s - 1), 1 * (n_s + 1)))
    zero_mat_2 = np.zeros((θ_tot * (n_s - 1), (θ_tot - 1) * (n_s + 1)))
    C_mat = np.hstack((np.vstack((a_bar_mat, zero_mat_1)), zero_mat_2))
    return(C_mat)

def C_constr_all(n_s, θ_tot):
    """ for constraining a_bar, μ_r and η_r
    Input:
        n_s: number of subintervals
        θ_tot: number of fit parameters
    Output:
        C_mat: shape(C_mat) = (θ_tot * (n_s + 1), θ_tot * (n_s - 1))
    """
    diag_upp = np.ones(n_s - 1)
    diag_cen = - 2 * np.ones(n_s)
    diag_low = np.ones(n_s + 1)
    constr_mat = np.diag(diag_low) + np.diag(diag_cen, 1) + np.diag(diag_upp, 2)
    constr_mat = constr_mat[: -2]
    
    shape = np.shape(constr_mat)
    hlist = []
    for i in range(0, θ_tot):
        before = i * [np.zeros(shape)]
        after = (θ_tot - i - 1) * [np.zeros(shape)]
        before.append(constr_mat)
        before.extend(after)
        obj = np.block([before])
        hlist.append(obj)
    C_mat = np.vstack(tuple(hlist))
    return(C_mat)

def lst_sqrs_adjstmnt_eff_adv(ψ, obs_data, τ_n_list_list, t_n_list_list, Δτ,
                              κ, n_s, K, p_list, P_n_list, x_vec, constr):
    """ performs the lsa (least squares adjustment)
    Input:
        ψ: constraining factor
        obs_data: [[t_i, a_i]] to be fitted
        τ_n_list_list = [[τ_(n-1), τ_n]]: pair wise epochs of vertices
        t_n_list_list: epochs split into subintervals
        Δτ: duration of subinterval
        κ: number of epochs per subinterval (this is a small kappa)
        n_s: number of subintervals
        K: number of data points
        p_list: periods to be fitted
        P_n_list: list with weight matrices for each subinterval
        x_vec = [a_n, μ_(r,n), η_(r,n)]: apriori fit parameters
              = [a_0, ..., a_n, ..., a_(n_s),
                 μ_(1,0), ..., μ_(1,n), ..., μ_(1,n_s), ...,
                 μ_(r,0), ..., μ_(r,n), ..., μ_(r,n_s), ...,
                 μ_(R,0), ..., μ_(R,n), ..., μ_(R,n_s),
                 η_(1,0), ..., η_(1,n), ..., η_(1,n_s), ...,
                 η_(r,0), ..., η_(r,n), ..., η_(r,n_s), ...,
                 η_(R,0), ..., η_(R,n), ..., η_(R,n_s)]
        constr: 0 for only constraining a_bar, 1 for also constraining μ_r and η_r
    Output:
        fit_list = [fitted_data, x_list, σ2_list, m0[0][0], bar_data, Kxx]
            fitted_data: reproduced data
            x_vec = [a_n, μ_(r,n), η_(r,n)]: fit parameters
                  = [a_0, ..., a_n, ..., a_(n_s),
                     μ_(1,0), ..., μ_(1,n), ..., μ_(1,n_s), ...,
                     μ_(r,0), ..., μ_(r,n), ..., μ_(r,n_s), ...,
                     μ_(R,0), ..., μ_(R,n), ..., μ_(R,n_s),
                     η_(1,0), ..., η_(1,n), ..., η_(1,n_s), ...,
                     η_(r,0), ..., η_(r,n), ..., η_(r,n_s), ...,
                     η_(R,0), ..., η_(R,n), ..., η_(R,n_s)]
            σ2_list: squared errors of fit parameters (covariances ignored)
            m0: estimated a posteriori standard deviation of unit weight
            bar_data: data produced with only a_bar
            Kxx: covariance matrix
    remark: print messages enable easy monitoring of the process.
    """
    print("compute N") # normal equation matrix
    N = N_eff(τ_n_list_list, t_n_list_list,
              Δτ, κ, n_s, K, p_list, P_n_list)
    print("shape N: ", np.shape(N))
    
    print("compute L_mat") # pseudo-observations
    θ_tot = 1 + 2 * len(p_list)
    if (constr == 0):
        C_mat = C_constr(n_s, θ_tot)
    elif (constr == 1):
        C_mat = C_constr_all(n_s, θ_tot)
    else:
        C_mat = np.zeros(np.shape(N))
    
    print("compute new N")
    N = N + ψ * C_mat.T @ C_mat # N -> N + ψ * N_constr
    print("shape new N: ", np.shape(N))
    
    print("compute N_inv")
    N_inv = np.linalg.inv(N) # N^(-1)
    print("shape N_inv: ", np.shape(N_inv))
    
    print("compute DTP") # D^T * P
    DTP = DT_P(τ_n_list_list, t_n_list_list,
               Δτ, κ, n_s, K, p_list, P_n_list)
    print("shape DTP: ", np.shape(DTP))
    
    print("compute N_inv_DTP")
    N_inv_DTP = N_inv @ DTP # N^(-1) * D^T * P
    print("shape N_inv_DTP: ", np.shape(N_inv_DTP))
    
    print("generate l_vec") # observations
    l_vec = np.array([obs_data[:, 1]]).T
    print("shape l_vec: ", np.shape(l_vec))
    
    print("compute x_vec")
    x_vec = N_inv_DTP @ l_vec # fit parameters
    print("shape x_vec: ", np.shape(x_vec))
    
    print("compute m0") # estimated a posteriori standard deviation of unit weight
    m0 = m0_eff(τ_n_list_list, t_n_list_list,
                Δτ, κ, n_s, K, p_list, P_n_list, x_vec, l_vec)
    print("m0 = %.3e" % m0[0][0])
    
    x_list = ravel(x_vec) # fit parameters as list
    
    print("compute Kxx")
    Kxx = m0**2 * N_inv # covariance matrix
    σ2_list = np.diag(Kxx) # σ^2
    
    print("compute fitted data") # reproduced data
    fitted_data = compute_data(τ_n_list_list, t_n_list_list,
                               Δτ, κ, n_s, K, p_list, x_vec)
    
    print("compute data free of oscillations") # data reproduced only with a_bar
    bar_data = gen_a_bar_array(τ_n_list_list, t_n_list_list,
                               Δτ, κ, n_s, K, x_list[: n_s + 1], σ2_list[: n_s + 1])
    
    fit_list = [fitted_data, x_list, σ2_list, m0[0][0], bar_data, Kxx]
    return(fit_list)

def pre_lst_sqrs_adjstmt(obs_data, n_s, K, p_list, P, apriori_list):
    """ preparations for lsa
    Input:
        obs_data: [[t_i, a_i]] to be fitted
        n_s: number of subintervals
        K: number of data points
        p_list: periods to be fitted
        P: weight matrix
        apriori_list: apriori parameters for a_bar, μ_r and η_r
    Output:
        big_list = [τ_n_list_list, t_n_list_list, L, P_n_list, x_vec, Δτ]
            τ_n_list_list = [[τ_(n-1), τ_n]]: pair wise epochs of vertices
            t_n_list_list: epochs split into subintervals
            κ: number of epochs per subinterval (this is a small kappa)
            P_n_list: list with weight matrices for each subinterval
            x_vec = [a_n, μ_(r,n), η_(r,n)]: apriori fit parameters
            Δτ: duration of subinterval
    """
    t_tot = obs_data[-1, 0] - obs_data[0, 0]
    t0 = obs_data[0, 0]
    Δτ = t_tot / n_s
    κ = int(K / n_s)
    print("κ = ", κ)
    
    t_n_list_list = splitter(obs_data[:, 0], κ, n_s, K)
    τ_n_list_list = []
    for n in range(1, n_s + 1):
        τ_n_1 = t0 + (n - 1) * Δτ # n_1 = "n-1"
        τ_n = t0 + n * Δτ
        τ_n_list_list.append([τ_n_1, τ_n])
    
    P_n_list = []
    for n in range(1, n_s + 1):
        P_n = P[(n - 1) * κ : n * κ, (n - 1) * κ : n * κ]
        P_n_list.append(P_n)
    
    R = len(p_list) # number of periods to be fitted
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
    
    big_list = [τ_n_list_list, t_n_list_list, κ, P_n_list, x_vec, Δτ]
    return(big_list)

def gen_a_dot_array(n_t, slope_gen_list):
    """ generates a_dot data from fit results
    Input:
        n_t: number of data points
        slope_gen_list = [τ_fit_list, Δτ, a_bar_list, σ2_a_bar_list]
            τ_fit_list: epochs of vertices
            Δτ: duration of subinterval
            a_bar_list: a_bar parameters
            σ2_a_bar_list: squared errors of a_bar parameters
    Output:
        array = [[t_i, a_dot_i, σ_a_dot_i]]
    """
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
        s_a_dot_j = math.sqrt((σ2_a_bar_n + σ2_a_bar_n_1) / (Δτ * Δτ))
        
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

def fitter(data, per0_list, amp0_list, n_s, ψ, constr):
    """ master function for executing fit/lsa (least squares adjustment)
    Input:
       data: [[t_i, a_i]] to be fitted
       per0_list: periods that will be used in fit
       amp0_list: corresponding amplitudes (as guesses)
       n_s: number of subintervals
       ψ: constraining factor
       constr: 0 for only constraining a_bar, 1 for also constraining μ_r and η_r
    Output: fit_list = [fitted_data, para_list, s2_para_list, m0,
                        τ_fit_list, slope_gen_list, bar_data, x_list, Kxx]
        fitted_data: data produced with fit model
        para_list: terms in fit model (a_bar, μ_r * sin and η_r * cos)
        s2_para_list: squared errors of terms in fit model
        m0: estimated a posteriori standard deviation of unit weight
        τ_fit_list: time vertices
        slope_gen_list: ingredients to generate slope data
        bar_data: data produced with only a_bar
        x_list = [a_n, μ_(r,n), η_(r,n)]: fit parameters
        Kxx: covariance matrix
    remark: this is the fifth lsa_method that i used in my work. if you are
            interested in knowing what else i already tried just contact me.
    remark: this lsa is simplified. the weight matrix has to be diagonal and
            the number of data points has to be an integer multiple of n_s.
            in other words: each subinterval contains the same number of
            data points and the number of data points is an integer.
            reason: code was easier to implement this way.
    """
    K = len(data)
    P = np.diag(np.ones(K)) # weight matrix
    apriori_list = [data[0, 1], amp0_list, amp0_list] # apriori parameters for a_bar, μ_r and η_r
    print("prepare for fit")
    big_list = pre_lst_sqrs_adjstmt(data, n_s, K, per0_list,
                                    P, apriori_list)
    τ_n_list_list, t_n_list_list = big_list[0], big_list[1]
    κ, P_n_list = big_list[2], big_list[3]
    x_vec, Δτ = big_list[4], big_list[5]
    print("preparations finished")
    print("n_s = %4d | Δτ = %3.6f" % (n_s, Δτ))
    
    print("execute lsa")
    fit_list = lst_sqrs_adjstmnt_eff_adv(ψ, data, τ_n_list_list, t_n_list_list,
                                         Δτ, κ, n_s, K, per0_list, P_n_list,
                                         x_vec, constr)
    fitted_data = fit_list[0] # reproduced data
    x_list = fit_list[1] # fit parameters
    σ2_list = fit_list[2] # squared errors of fit parameters (covariances ignored)
    m0 = fit_list[3] # estimated a posteriori standard deviation of unit weight
    bar_data = fit_list[4] # data produced with only a_bar
    Kxx = fit_list[5] # covariance matrix
    
    # time vertices
    τ_fit_list = np.array(τ_n_list_list)[:, 0]
    τ_fit_list = np.append(τ_fit_list, τ_n_list_list[-1][1])
    
    # I can't remember anymore why I did para_list and s2_para_list but whatever
    R = len(per0_list)
    θ_tot = 2 * R + 1
    θ_big = θ_tot - 1 # I wanted to use an upper case θ but it looks almost identical to the lower case one
    para_list = []
    s2_para_list = []
    for θ_small in range(0, θ_big + 1):
        sub_list = []
        s2_sub_list = []
        for n in range(0, n_s + 1):
            a_q_n = x_list[(n_s + 1) * θ_small + n]
            σ2_q_n = σ2_list[(n_s + 1) * θ_small + n]
            para = 0
            s2_para = 0
            if (θ_small == 0): # a_bar
                para = a_q_n
                s2_para = σ2_q_n
            elif (θ_small >= 1 and θ_small <= R): # μ
                τ_n = τ_fit_list[n]
                r = θ_small
                p_r = per0_list[r - 1]
                para = a_q_n * math.sin(2 * np.pi / p_r * τ_n)
                s2_para = σ2_q_n * (math.sin(2 * np.pi / p_r * τ_n))**2
            else: # (θ_small >= R + 1 and θ_small <= 2 * R) η
                τ_n = τ_fit_list[n]
                r = θ_small - R
                p_r = per0_list[r - 1]
                para = a_q_n * math.cos(2 * np.pi / p_r * τ_n)
                s2_para = σ2_q_n * (math.cos(2 * np.pi / p_r * τ_n))**2
            sub_list.append(para)
            s2_sub_list.append(s2_para)
        para_list.append(sub_list)
        s2_para_list.append(s2_sub_list)
    
    # collect ingredients for slope_gen_list (ingredients to generate slope data)
    a_bar_list = x_list[: n_s + 1]
    σ2_a_bar_list = σ2_list[: n_s + 1]
    slope_gen_list = [τ_fit_list, Δτ, a_bar_list, σ2_a_bar_list]
    
    fit_list = [fitted_data, para_list, s2_para_list, m0,
                τ_fit_list, slope_gen_list, bar_data, x_list, Kxx]
    return(fit_list)
# %% functions for numerical integration with Runge-Kutta fourth order method
def a_dot_RSW(a, e, ω, u, R, S, σ_a, σ_e, σ_ω):
    """ Gauss's perturbation equation for the semi-major axis
    Input:
        osculating elements (+ some errors) and RSW accelerations (only R and S)
    Output:
        slope of semi-major axis + error
    """
    ν = (u - ω) * np.pi / 180
    fac = 2 * math.sqrt(a**3 / (μ * (1 - e*e)))
    term1 = e * math.sin(ν) * R
    term2 = (1 + e * math.cos(ν)) * S
    
    a_dot = fac * (term1 + term2)
    
    term_σ_a = 3 / 2 * (term1 + term2) / a
    term_σ_e = (math.sin(ν) * R + (e + math.cos(ν)) * S) / (1 - e*e)
    term_σ_ω = e * (math.cos(ν) * R - math.sin(ν) * S)
    
    term_σ_a_2 = (term_σ_a * σ_a)**2
    term_σ_e_2 = (term_σ_e * σ_e)**2
    term_σ_ω_2 = (term_σ_ω * σ_ω)**2
    
    a_dot_error = fac * math.sqrt(term_σ_a_2 + term_σ_e_2 + term_σ_ω_2)
    
    return(a_dot, a_dot_error)

def propagator(dt, a_n, u_list, R_list, S_list, e, ω, σ_a, σ_e, σ_ω):
    """ propagates one step = 2 * dt or from a_n to a_np2, np2 = n + 2
    Input:
        dt: sampling
        a_n: semi-major axis at time t_n
        u_list = [u_n, u_(n+1), u_(n+2)]: argument of latitude at times t_n, t_(n+1) and t_(n+2)
        R_list = [R_n, R_(n+1), R_(n+2)]: radial acceleration at times t_n, t_(n+1) and t_(n+2)
        S_list = [S_n, S_(n+1), S_(n+2)]: along-track acceleration at times t_n, t_(n+1) and t_(n+2)
        e: eccentricity (constant)
        ω: argument of perigee (constant)
        σ_a: error of semi-major axis at time t_n
        σ_e: error of eccentricity (constant)
        σ_ω: error of argument of perigee (constant)
    Output:
        a_np2: semi-major axis at time t_(n+2)
        a_np2_error: error of semi-major axis at time t_(n+2)
    """
    u_n, u_np1, u_np2 = u_list[0], u_list[1], u_list[2] # u(t, t + dt, t + 2dt)
    R_n, R_np1, R_np2 = R_list[0], R_list[1], R_list[2] # R(t, t + dt, t + 2dt)
    S_n, S_np1, S_np2 = S_list[0], S_list[1], S_list[2] # S(t, t + dt, t + 2dt)
    
    k1, σ_k1 = a_dot_RSW(a_n, e, ω, u_n, R_n, S_n, σ_a, σ_e, σ_ω)
    k1, σ_k1 = 2 * dt * k1, 2 * dt * σ_k1
    k2, σ_k2 = a_dot_RSW(a_n + k1 / 2, e, ω, u_np1, R_np1, S_np1, math.sqrt(σ_a*σ_a +  σ_k1*σ_k1 / 4), σ_e, σ_ω)
    k2, σ_k2 = 2 * dt * k2, 2 * dt * σ_k2
    k3, σ_k3 = a_dot_RSW(a_n + k2 / 2, e, ω, u_np1, R_np1, S_np1, math.sqrt(σ_a*σ_a +  σ_k2*σ_k2 / 4), σ_e, σ_ω)
    k3, σ_k3 = 2 * dt * k3, 2 * dt * σ_k3
    k4, σ_k4 = a_dot_RSW(a_n + k3, e, ω, u_np2, R_np2, S_np2, math.sqrt(σ_a*σ_a +  σ_k3*σ_k3), σ_e, σ_ω)
    k4, σ_k4 = 2 * dt * k4, 2 * dt * σ_k4
    
    a_np2 = a_n + (k1 + k4) / 6 + (k2 + k3) / 3
    a_np2_error = np.sqrt(σ_a*σ_a + (σ_k1*σ_k1 + σ_k4*σ_k4) / 36 + (σ_k2*σ_k2 + σ_k3*σ_k3) / 9)
    return(a_np2, a_np2_error)

def ele_σ_for_rk45(path_ele_σ, mjd_interval):
    """ for getting errors of osculating elements
    Input:
        path_ele_σ: path to master ele error file
        mjd_interval: interval for integration
    Output:
        σ_a: error of semi-major axis at start of integration
        σ_e: error of eccentricity (constant)
        σ_ω: error of argument of perigee (constant)
    """
    data = np.loadtxt(path_ele_σ)
    mjd_start = mjd_interval[0]
    n_days = mjd_interval[1] - mjd_start
    data_trimmed = array_modifier(data, mjd_start, n_days)
    σ_a = data_trimmed[0, 1]
    σ_e = data_trimmed[0, 2]
    σ_ω = data_trimmed[0, 5]
    return(σ_a, σ_e, σ_ω)

def integrator(a_0, e, ω, u_data, acc_R_data, acc_S_data, σ_a_0, σ_e, σ_ω):
    """ integrates time interval with Runge-Kutta fourth order method
    Input:
        a_data = [[t_1, a_1], [t_2, a_2], ...]
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
    u_list_list = u_data[:, 1]
    R_list_list = acc_R_data[:, 1] * day_sec * day_sec
    S_list_list = acc_S_data[:, 1] * day_sec * day_sec
    
    # starting point
    t_0 = u_data[0, 0] 
    a_dot_0, σ_a_dot_0 = a_dot_RSW(a_0, e, ω, u_list_list[0], R_list_list[0],
                                   S_list_list[0], σ_a_0, σ_e, σ_ω)
    
    a_int_data = np.array([[t_0, a_0, σ_a_0]])
    a_dot_data = np.array([[t_0, a_dot_0, σ_a_dot_0]])
    
    dt = u_data[1, 0] - u_data[0, 0] # time step
    for i in range(0, (len(u_data) - 1)// 2):
        # _np2 for_(n+2)
        n = 2 * i
        t_np2 = u_data[n + 2, 0]
        a_n = a_int_data[-1, 1]
        σ_a = a_int_data[-1, 2]
        
        u_list = u_list_list[n : n + 2 + 1]
        R_list = R_list_list[n : n + 2 + 1]
        S_list = S_list_list[n : n + 2 + 1]
        
        a_np2, σ_a_np2 = propagator(dt, a_n, u_list, R_list, S_list,
                                    e, ω, σ_a, σ_e, σ_ω)
        
        a_dot_np2, σ_a_dot_np2 = a_dot_RSW(a_np2, e, ω, u_list[-1],
                                           R_list[-1], S_list[-1],
                                           σ_a_np2, σ_e, σ_ω)
        
        a_int_row = np.array([t_np2, a_np2, σ_a_np2])
        a_dot_row = np.array([t_np2, a_dot_np2, σ_a_dot_np2])
        
        a_int_data = np.vstack((a_int_data, a_int_row))
        a_dot_data = np.vstack((a_dot_data, a_dot_row))
    
    return(a_int_data, a_dot_data)

def master_integrator(u_data, path, mjd_interval, typus):
    """INPUT:
            u_data: data from LST-Files [[mjd1, u1], [mjd2, u2], ...]
            path_ele_osc: data from ELE-Files (file generated with master_ele function)
            path_PCA: data from ELE-Files (file generated with master_ele function)
            mjd_interval: time interval for integration
    OUTPUT:
        a_int_data: integrated semi-major axis
        a_dot_data: slope of integrated semi-major axis
    """
    MJD_start, MJD_end = mjd_interval[0], mjd_interval[1]
    n_days = MJD_end - MJD_start
    
    u_data = array_modifier(u_data, MJD_start, n_days) # crop data
    
    ele_osc_data = np.loadtxt(path + "ele_osc.txt")
    ele_osc_data = array_modifier(ele_osc_data, MJD_start, n_days) # crop data
    a_0 = ele_osc_data[0, 1]
    e, ω = ele_osc_data[0, 2], ele_osc_data[0, 5]
    
    
    acc_R_data = np.loadtxt(path + typus + ".txt", usecols = (0, 1))
    acc_S_data = np.loadtxt(path + typus + ".txt", usecols = (0, 2))
    
    acc_R_data = array_modifier(acc_R_data, MJD_start, n_days) # crop data
    acc_S_data = array_modifier(acc_S_data, MJD_start, n_days) # crop data
    
    σ_a_0, σ_e, σ_ω = ele_σ_for_rk45(path + "errors.txt", mjd_interval)
    
    # resample PCA data
    dt_u = (u_data[-1, 0] - u_data[0, 0]) / (len(u_data) - 1)
    dt_acc = (acc_R_data[-1, 0] - acc_R_data[0, 0]) / (len(acc_R_data) - 1)
    if (typus == 'PCA'):
        fac = int(np.round(dt_acc / dt_u))
        acc_R_data = step_data_generation(acc_R_data, fac)
        acc_S_data = step_data_generation(acc_S_data, fac)
    if (typus == 'ACC'):
        fac = int(np.round(dt_u / dt_acc))
        acc_R_data = down_sample(acc_R_data, fac)
        acc_S_data = down_sample(acc_S_data, fac)
    
    a_int_data, a_dot_data = integrator(a_0, e, ω, u_data, acc_R_data, acc_S_data,
                                        σ_a_0, σ_e, σ_ω)
    return(a_int_data, a_dot_data)
# %% for plotting
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
                 ylims, save_specs):
    # ultimative plot: slope of semi-major axis, ap index, CMEs, MANs
    t_min, t_max = t_min_t_max(slope_data_list)
    xstart = t_min
    xend = t_max
    n_days = xend - xstart
    new_slope_data_list = slope_data_list
    flx_data = array_modifier(flx_data, xstart, n_days)
    
    fig, ax1 = plt.subplots(figsize = (12,5), dpi = 300)
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
        ax1.fill_between(x_slope, y_slope - s_y_slope, y_slope + s_y_slope,
                         color = ecol, alpha = e_α, zorder = 9,
                         label = e_lab)
    
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
    ax1.tick_params(axis = 'y', labelcolor = 'k', labelsize = 15)
    ax1.set_xlabel(xaxis_year([xstart], 'mm.dd'), fontsize = 15)
    y_lab = r'$\dot{\bar{a}}$ [md$^{-1}$]'
    ax1.set_ylabel(y_lab, fontsize = 15)
    
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
        flx_ylabel =  r'$ap$'
        ax2.set_ylabel(flx_ylabel, color = col, fontsize = 15, rotation = 0)
        ax2.tick_params(axis = 'y', labelcolor = col, color = col, labelsize = 15)
    
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
    
    plt.figlegend(fontsize = 15, markerscale = 1, loc = 9,
                  bbox_to_anchor = (0.5, 1.25), bbox_transform = ax1.transAxes,
                  labelspacing = 1, ncols = 4, columnspacing = 1)
    
    if (save_specs[0] == 1):
        fig.savefig(save_specs[1], transparent = None, dpi = 'figure',
                    format = None, metadata = None, bbox_inches = 'tight',
                    facecolor = 'auto', pad_inches = 'layout',
                    edgecolor = 'auto', backend = None)
    plt.show(fig)
    plt.close(fig)
# %%