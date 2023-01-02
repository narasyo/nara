import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.ticker
import pywt
import os

def calc_devided(numer,demon) :

    return numer/demon

#################
#　線形正規化法 #
#################

def normalized_linear(wave1,wave2,sum_time) :

    xmin = 0
    xmax = 0.01*sum_time
    h = (xmax-xmin)/sum_time
    c1 = np.abs(min(wave1))
    c2 = np.abs(min(wave2))
    const = max(c1,c2)
    pos_amp1_list = []
    for uni_amp1 in wave1 :
        pos_amp1 = uni_amp1 + const
        pos_amp1_list += [pos_amp1]
        
    pos_amp2_list = []
    for uni_amp2 in wave2 :
        pos_amp2 = uni_amp2 + const
        pos_amp2_list += [pos_amp2]

    s1 = h*(np.sum(pos_amp1_list)-pos_amp1_list[0]/2-pos_amp1_list[sum_time-1]/2)

    s2 = h*(np.sum(pos_amp2_list)-pos_amp2_list[0]/2-pos_amp2_list[sum_time-1]/2)

    amp1_norm_list = []
    for pos1 in pos_amp1_list:
        amp1_norm_list.append(calc_devided(pos1,s1))

    amp2_norm_list =[]
    for pos2 in pos_amp2_list:
        amp2_norm_list.append(calc_devided(pos2,s2))

    amp1_norm = np.array(amp1_norm_list)
    amp2_norm = np.array(amp2_norm_list)

    return pos_amp1_list,pos_amp2_list,amp1_norm,amp2_norm

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

    return nlevel,wave_levels

os.chdir("./data/furukawa_data/201212071719/20121207_modified/")

wave1 = np.load("201212071719_F1_EW_modified.npy")
wave2 = np.load("201212071719_F3_EW_modified.npy")
w = pywt.Wavelet('dmey')

tim_list = np.arange(0,480,0.01)
t = 48000
tim2_list = np.arange(-20,500,0.01)
t2 = 52000

const_wave1,const_wave2,normalized_wave1,normalized_wave2 = normalized_linear(wave1,wave2,t)

nlevel1,wave_levels1 = sort_wave(wave1,w)
nlevel2,wave_levels2 = sort_wave(wave2,w)

freq_list = ["~0.1","0.1~0.2","0.2~0.4","0.4~0.8","0.8~1.6","1.6~3.2","3.2~6.4","6.4~12.8","12.8~25.6","25.6~51.2"]
number = np.arange(0,10,1)
# # Plots
# fig,ax = plt.subplots(nrows=nlevel1,ncols=2,figsize=(12,8))

for i,freq in zip(number,freq_list):   
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('time[s]')
    ax.set_ylabel('acceleration[gal]')
    ax.set_xlim(-20,500)
    ax.set_ylim(-150,150)
    ax.plot(tim_list,wave1,color="gray")
    ax.plot(tim_list,wave_levels1[i],color="red",label=i)
    # ax[i,0].plot(tim_list,wave1,color="gray")
    # ax[i,0].plot(tim_list,wave_levels1[i],color="red",label=i)
    # ax[i,0].grid()
    # ax[i,0].legend()

    # #ax[i,1].set_xscale("log")
    # ax[i,1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # #ax[i,1].set_yscale("log")
    # #ax[i,1].plot(freq,Pwave,color="gray")
    # ax[i,1].plot(freq,Pwave_levels[i],color="red")
    # ax[i,1].grid()

    # ax[-1,0].set_xlabel("tim (s)")
    # ax[-1,1].set_xlabel("frequency (Hz)")

    fig.tight_layout()
    fig.savefig(f"20121207_F1_EW_sortwave_{freq}[Hz].png")

const_wave1,const_wave2,normalized_wave1,normalized_wave2 = normalized_linear(wave_levels1[5],wave_levels2[5],t)

zero = np.zeros(t2)
fig1 = plt.figure()

ax11 = fig1.add_subplot(2,1,1)
ax11.set_xlabel('time[s]')
ax11.set_ylabel('acceleration[gal]')
ax11.set_xlim(-20,500)
ax11.set_ylim(-150,250)
ax11.tick_params(direction = "inout", length = 5, colors = "blue")
ax11.plot(tim_list, wave1, color = "red")
ax11.plot(tim2_list,zero,color="gray")

ax12 = fig1.add_subplot(2,1,2)
ax12.set_xlabel('time[s]')
ax12.set_ylabel('acceleration[gal]')
ax12.set_xlim(-20,500)
ax12.set_ylim(-150,250)
ax12.tick_params(direction = "inout", length = 5, colors = "blue")
ax12.plot(tim_list, wave2, color = "red")
ax12.plot(tim2_list,zero,color="gray")

fig1.tight_layout()
fig1.savefig('20121207_F1_F3.png')

plt.clf()
plt.close()

fig2 = plt.figure()

ax21 = fig2.add_subplot(2,1,1)
ax21.set_xlabel('time[s]')
ax21.set_ylabel('acceleration[gal]')
ax21.set_xlim(-20,500)
ax21.set_ylim(-150,250)
ax21.tick_params(direction = "inout", length = 5, colors = "blue")
ax21.plot(tim_list, const_wave1, color = "red")
ax21.plot(tim2_list,zero,color="gray")

ax22 = fig2.add_subplot(2,1,2)
ax22.set_xlabel('time[s]')
ax22.set_ylabel('acceleration[gal]')
ax22.set_xlim(-20,500)
ax22.set_ylim(-150,250)
ax22.tick_params(direction = "inout", length = 5, colors = "blue")
ax22.plot(tim_list, const_wave2, color = "red")
ax22.plot(tim2_list,zero,color="gray")

fig2.tight_layout()
fig2.savefig('20121207_F1_F3_const.png')

plt.clf()
plt.close()

fig3 = plt.figure()

ax31 = fig3.add_subplot(2,1,1)
ax31.set_xlabel('time[s]')
ax31.set_ylabel('probability')
ax31.set_ylim(0,0.01)
ax31.tick_params(direction = "inout", length = 5, colors = "blue")
ax31.plot(tim_list, normalized_wave1, color = "red")

ax32 = fig3.add_subplot(2,1,2)
ax32.set_xlabel('time[s]')
ax32.set_ylabel('probability')
ax32.set_ylim(0,0.01)
ax32.tick_params(direction = "inout", length = 5, colors = "blue")
ax32.plot(tim_list, normalized_wave2, color = "red")

fig3.tight_layout()
fig3.savefig('20121207_F1_F3_normalized.png')

plt.clf()
plt.close()