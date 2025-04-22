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
#%%
