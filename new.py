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