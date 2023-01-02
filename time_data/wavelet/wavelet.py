import numpy as np
import matplotlib.pyplot as plt
import dtcwt

def wave_to_scalogram(wave):
    ntim = len(wave)
    wave_t = dtcwt.Transform1d().forward(wave)

    nlevel = len(wave_t.highpasses) + 1
    scalogram = np.empty((ntim,nlevel),dtype=complex)

    nlow = len(wave_t.lowpass)
    scalogram[:,0] = np.repeat(wave_t.lowpass,int(ntim/nlow))

    for index,wthp in enumerate(wave_t.highpasses):
        n = len(wthp)
        scalogram[:,index+1] = np.repeat(wthp,int(ntim/n))

    return scalogram

def scalogram_to_wave(scalogram):
    ntim = len(scalogram[:,0])
    nlevel = len(scalogram[0,:])

    dummy = np.zeros(ntim)
    wave_t = dtcwt.Transform1d().forward(dummy)

    nlow = len(wave_t.lowpass)
    wave_t.lowpass[:,0] = np.real(np.reshape(scalogram[:,0],(-1,int(ntim/nlow))).mean(axis=1))

    for index in range(0,nlevel-1):
        n = len(wave_t.highpasses[index])
        wave_t.highpasses[index][:,0] = np.reshape(scalogram[:,index+1],(-1,int(ntim/n))).mean(axis=1)

    wave = dtcwt.Transform1d().inverse(wave_t)

    return wave

def plot_scalogram(tim,scalogram):
    nlevel = len(scalogram[0,:])
    level = np.linspace(0,nlevel,nlevel,endpoint=False)
    T,L = np.meshgrid(tim,level)

    plt.figure()
    plt.pcolormesh(T,L,np.abs(scalogram).T)
    plt.xlabel('time [s]')
    plt.ylabel('level')
    plt.show()


input_file = "input/1995JR_Takatori.acc"

tim,wave = np.loadtxt(input_file,usecols=(0,2),unpack=True)

scalogram = wave_to_scalogram(wave)
#plot_scalogram(tim,scalogram)

wave_rep = scalogram_to_wave(scalogram)

plt.figure()
plt.plot(tim,wave)
plt.plot(tim,wave_rep)
plt.show()
