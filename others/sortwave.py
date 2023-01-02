import numpy as np
import os
from numpy.core.arrayprint import format_float_positional
import ot
import matplotlib.pyplot as plt 
import pywt
from tqdm import tqdm

def sort_wave(wave,wavelet) :

    ntim = len(wave)

    # Wavelet transform (Multi resolution analysis) #
    coeffs = pywt.wavedec(wave,wavelet)
    nlevel = pywt.dwt_max_level(ntim,wavelet) + 1
    ncoeffs = [len(c) for c in coeffs]
    zero_coeffs = [np.zeros_like(c) for c in coeffs]
    wave_levels = []
    for i in range(nlevel):
        coeffs_level = zero_coeffs.copy()
        #nl = int(len(coeffs_level[1])/2)
        #coeffs_level[1][nl] = 1.0
        coeffs_level[i] = coeffs[i].copy()
        wave_level = pywt.waverec(coeffs_level,wavelet)
        wave_levels += [wave_level]

    return wave_levels

time = np.arange(0,360,0.01)

n1 = 3
n2 = 26

n1 = str(n1)
n2 = str(n2)

os.chdir("./data/furukawa_data/201308041229/20130804")

wave1 = np.load(f"201308041229_F{n1}_verticle_modified.npy")
wave2 = np.load(f"201308041229_F{n2}_verticle_modified.npy")

wavelet = pywt.Wavelet('dmey')

sort_wave1 = sort_wave(wave1,wavelet)
sort_wave2 = sort_wave(wave2,wavelet)

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.tick_params(direction = "inout", length = 5, colors = "blue",labelsize=10)
ax1.plot(time[0:12000], wave1[0:12000], c='red',linewidth = 0.5)
ax1.set_xlim(0,100)
ax1.set_ylim(-30,30)
ax1.set_xlabel('time[s]',fontsize=10)
ax1.set_ylabel(r'acc[cm/s$^2$]',fontsize=10)
ax1.grid(True)

ax2 = fig.add_subplot(2,1,2)
ax2.tick_params(direction = "inout", length = 5, colors = "blue",labelsize=10)
ax2.plot(time[0:12000], wave2[0:12000], c='red',linewidth = 0.5)
ax2.set_xlim(0,100)
ax2.set_ylim(-30,30)
ax2.set_xlabel('time[s]',fontsize=10)
ax2.set_ylabel(r'acc[cm/s$^2$]',fontsize=10)
ax2.grid(True)

fig.tight_layout()
fig.savefig(f"F{n1}_F{n2}_wave.eps")
fig.savefig(f"F{n1}_F{n2}_wave.png")
plt.clf()
plt.close()

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.tick_params(direction = "inout", length = 5, colors = "blue",labelsize=10)
ax1.plot(time[0:12000], sort_wave1[3][0:12000], c='red',linewidth = 0.5)
ax1.set_xlim(0,100)
ax1.set_ylim(-30,30)
ax1.set_xlabel('time[s]',fontsize=10)
ax1.set_ylabel(r'acc[cm/s$^2$]',fontsize=10)
ax1.grid(True)

ax2 = fig.add_subplot(2,1,2)
ax2.tick_params(direction = "inout", length = 5, colors = "blue",labelsize=10)
ax2.plot(time[0:12000], sort_wave2[3][0:12000], c='red',linewidth = 0.5)
ax2.set_xlim(0,100)
ax2.set_ylim(-30,30)
ax2.set_xlabel('time[s]',fontsize=10)
ax2.set_ylabel(r'acc[cm/s$^2$]',fontsize=10)
ax2.grid(True)

fig.tight_layout()
fig.savefig(f"F{n1}_F{n2}_wave_sorted.eps")
fig.savefig(f"F{n1}_F{n2}_wave_sorted.png")
plt.clf()
plt.close()


