#%%
import time as tt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
from matplotlib import colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits import mplot3d as d3
import os
import scipy
from scipy import signal
import matplotlib.image as mpimg
# print(u'\u03B1') # alpha
#%%
from functions import fft
#%%
freq1 = 60  # Hz
freq2 = 90  # Hz
sampling_rate = 1000  # samples per second
time = 1  # seconds
t = np.linspace(0, time, time * sampling_rate)
#%%
y = np.sin(2*np.pi*freq1*t) + np.sin(2*np.pi*freq2*t)
# y + random noise
yn = y * 0.3 * np.random.rand(len(t))
dt = 250  # observation
#%%
# plot
plt.figure(tight_layout=True)

plt.subplot(211)
plt.plot(t[:dt], y[:dt])
plt.title('y(t)')
plt.grid()
plt.xlabel('t(s)')

plt.subplot(212)
plt.plot(t[:dt], yn[:dt])
plt.title('yn(t)')
plt.grid()
plt.xlabel('t(s)')
#%%
# fft
Y = np.abs(np.fft.fft(y))
Yn = np.abs(np.fft.fft(yn))
f = np.arange(0, len(Y)) * sampling_rate/len(Y)
df = int(200 * len(y) / sampling_rate)
#%%
#fft plots
plt.figure(tight_layout=True)

plt.subplot(211)
plt.plot(f[:df], Y[:df])
plt.title('|Y(f)|')
plt.grid()
plt.xlabel('f(Hz)')

plt.subplot(212)
plt.plot(f[:df], Yn[:df])
plt.title('|Yn(f)|')
plt.grid()
plt.xlabel('f(Hz)')
# filter bandstop - reject 60 Hztop)
lowcut = 57
highcut = 63
order = 4
nyq = 0.5 * sampling_rate
low = lowcut / nyq
high = highcut / nyq
b, a = signal.butter(order, [low, high], btype='bandstop')

y_filt = signal.filtfilt(b, a, y)
yn_filt = signal.filtfilt(b, a, yn)
#%%
# filtered signal
plt.figure(tight_layout=True)

plt.subplot(211)
plt.plot(t[:dt], y_filt[:dt])
plt.title('y_filt(t)')
plt.grid()
plt.xlabel('t(s)')

plt.subplot(212)
plt.plot(t[:dt], yn_filt[:dt])
plt.title('yn_filt(t)')
plt.grid()
plt.xlabel('t(s)')
#%%
# fft
Y = np.abs(np.fft.fft(y_filt))
Yn = np.abs(np.fft.fft(yn_filt))
f = np.arange(0, len(Y)) * sampling_rate/len(Y)
df = int(200 * len(y_filt) / sampling_rate)
#fft plots
plt.figure(tight_layout=True)

plt.subplot(211)
plt.plot(f[:df], Y[:df])
plt.title('|Y(f) filered|')
plt.grid()
plt.xlabel('f(Hz)')

plt.subplot(212)
plt.plot(f[:df], Yn[:df])
plt.title('|Yn(f)| filered')
plt.grid()
plt.xlabel('f(Hz)')
# %%
white = np.random.rand(len(t))
white_data = np.vstack((t, white)).T
plt.figure(figsize = (10, 5), dpi = 300)
plt.title("white noise")
plt.plot(white_data[:, 0], white_data[:,1], 'r-')
plt.show()
plt.close()
#%%
fft_white = fft(white_data)
plt.figure(figsize = (10, 5), dpi = 300)
plt.title("white noise spectrum")
plt.semilogx(fft_white[0], fft_white[1], 'r-')
plt.show()
plt.close()
#%%
white_filt = signal.filtfilt(b, a, white)
plt.figure(figsize = (10, 5), dpi = 300)
plt.title("white noise filtered")
plt.plot(t, white_filt, 'r-')
plt.show()
plt.close()
# %%
fft_white_filtered = fft(np.vstack((t, white_filt)).T)
plt.figure(figsize = (10, 5), dpi = 300)
plt.title("white noise filtered spectrum")
plt.semilogx(fft_white_filtered[0], fft_white_filtered[1], 'r-')
plt.show()
plt.close()
# %%
