import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.ticker
import pywt

wave_file = "data/1995JR_Takatori.acc"
tim,wave = np.loadtxt(wave_file,usecols=(0,2),unpack=True)
dt = tim[1]-tim[0]
ntim = len(wave)

# Wavelet transform (Multi resolution analysis) #
w = pywt.Wavelet('dmey')   # Discrete Meyer wavelet
coeffs = pywt.wavedec(wave,w)
nlevel = pywt.dwt_max_level(ntim,w) + 1

ncoeffs = [len(c) for c in coeffs]
zero_coeffs = [np.zeros_like(c) for c in coeffs]

wave_levels = []
for i in range(nlevel):
    coeffs_level = zero_coeffs.copy()
    coeffs_level[i] = coeffs[i].copy()
    wave_level = pywt.waverec(coeffs_level,w)
    wave_levels += [wave_level]

# Power spectrum
freq,Pwave = scipy.signal.periodogram(wave,1./dt)
Pwave_levels = [scipy.signal.periodogram(wl,1./dt)[1] for wl in wave_levels]


# Output
output_file = "result/wave_multiresolution.dat"
output_line = np.vstack([tim,wave_levels])
np.savetxt(output_file,output_line.T)

output_file = "result/power_multiresolution.dat"
output_line = np.vstack([freq,Pwave_levels])
np.savetxt(output_file,output_line.T)


# Plots
fig,ax = plt.subplots(nrows=nlevel,ncols=2,figsize=(12,8))

for i in range(nlevel):
    ax[i,0].plot(tim,wave,color="gray")
    ax[i,0].plot(tim,wave_levels[i],color="red",label=i)
    ax[i,0].grid()
    ax[i,0].legend()

    ax[i,1].set_xscale("log")
    ax[i,1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax[i,1].set_yscale("log")
    ax[i,1].plot(freq,Pwave,color="gray")
    ax[i,1].plot(freq,Pwave_levels[i],color="red")
    ax[i,1].grid()

ax[-1,0].set_xlabel("tim (s)")
ax[-1,1].set_xlabel("frequency (Hz)")

fig.tight_layout()
plt.show()
