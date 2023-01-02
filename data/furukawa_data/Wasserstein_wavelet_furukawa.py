import numpy as np
import os
import ot
import matplotlib.pyplot as plt 
import pywt
from tqdm import tqdm
import similarity
import sys
import pathlib
sys.path.append(pathlib.Path(__file__).resolve().parents[2].as_posix())
from others.plot import plot


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

def data_select(date,time) :

    os.chdir(f"./{date}{time}/{date}")
     
    if date == "20120327" :
        number1_list = [1,2,3,4,5,7,10,11,13,14,15,16,17,19,21,23]
        tim_list = np.arange(0,360,0.01)
        t = 36000
        Number = np.arange(1,16,1)
    elif date == "20120830" :
        number1_list = [1,2,4,5,6,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,28,29,30]
        tim_list = np.arange(0,300,0.01)
        t = 30000
        Number = np.arange(1,23,1)
    elif date == "20121207" :
        number1_list = [1,3,5,6,11,13,14,15,16,17,19,21,24,25,26,27,28,31,33]
        tim_list = np.arange(0,480,0.01)
        t = 48000
        Number = np.arange(1,19,1)
    elif date == "20130804" :
        number1_list = [1,2,3,4,5,9,11,12,14,15,16,19,20,22,23,24,25,26,27,28,29,30,31,36]
        tim_list = np.arange(0,360,0.01)
        t = 36000
        Number = np.arange(1,24,1)

    return number1_list,tim_list,t,Number

def load_wave(date,time,n1,direction) :

    if len(n1) == 1 :
        wave = np.load(f"{date}{time}_F{n1}_verticle_modified.npy")
    
    elif len(n1) == 2 :
        wave = np.load(f"{date}{time}_F{n1}_verticle_modified.npy")

    return wave

def freq_select(date) :

    if date == "20120327" or "20121207" or "20130804" :

        number = np.arange(1,10,1)
        freq1_list = [0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6]
        freq2_list = [0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2]
        freq_initial = 0.1

    elif date == "20120830" :

        number = np.arange(1,9,1)
        freq1_list = [0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6]
        freq2_list = [0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2]
        freq_initial = 0.2

    return number,freq1_list,freq2_list,freq_initial

def plot_figure(date,direction,distance,was) :

    number,freq1_list,freq2_list,freq_initial = freq_select(date)
    coef1 = np.corrcoef(distance, was[:,0])
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.scatter(distance,was[:,0], c='red', marker='.')
    ax1.tick_params(direction = "inout", length = 5, colors = "blue",labelsize=10)
    ax1.set_xlabel('distance[km]',fontsize=10)
    ax1.set_ylabel('2d wasserstein metric',fontsize=10)
    y1_max = max(was[:,0])
    ax1.set_xlim(0,)
    ax1.set_ylim(y1_max,0)
    ax1.grid(True)
    fig1.tight_layout()
    fig1.savefig(f"{date}_{direction}_~{freq_initial}Hz_new.png")
    fig1.savefig(f"{date}_{direction}_~{freq_initial}Hz_new.eps")
    plt.clf()
    plt.close()

    for i,f1,f2 in zip(number,freq1_list,freq2_list) :
        coef2 = np.corrcoef(distance,was[:,i])
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        ax2.scatter(distance,was[:,i], c='red', marker='.')
        ax2.tick_params(direction = "inout", length = 5, colors = "blue",labelsize=10)
        ax2.set_xlabel('distance[km]',fontsize=10)
        ax2.set_ylabel('2d wasserstein metric',fontsize=10)
        y2_max = max(was[:,i])
        ax2.set_xlim(0,)
        ax2.set_ylim(y2_max,0)
        ax2.grid(True)
        fig2.tight_layout()
        fig2.savefig(f"{date}_{direction}_{f1}Hz~{f2}Hz_new.png")
        fig2.savefig(f"{date}_{direction}_{f1}Hz~{f2}Hz_new.eps")
        plt.clf()
        plt.close()


date_input = "20130804"
time_input = "1229"
direction_input = "UD"

os.chdir('./data/furukawa_data')

number1_list,tim_list,t,Number = data_select(date_input,time_input)
tim_list = np.arange(0,150,0.01)

was_list_list,mse_list_list = [],[]
for N,n1 in tqdm(zip(Number,number1_list)) :
    str_n1 = str(n1)
    wave1 = load_wave(date_input,time_input,str_n1,direction_input)
    wave1 = wave1[0:15000]
    wavelet1 = pywt.Wavelet('dmey')
    wave_levels1 = sort_wave(wave1,wavelet1)

    number2_list = number1_list[N:]
    for n2 in number2_list :
        str_n2 = str(n2)
        wave2 = load_wave(date_input,time_input,str_n2,direction_input)
        wave2 = wave2[0:15000]
        wavelet2 = pywt.Wavelet('dmey')
        wave_levels2 = sort_wave(wave2,wavelet2)
        
        was_list = []
        for w1,w2 in zip(wave_levels1,wave_levels2) :
            sim = similarity.similarity(tim_list,w1,w2)
            was = sim.wasserstein_softplus_normalizing()
            was_list += [was]
        was_list_list += [was_list]

        mse_list = []
        for w1,w2 in zip(wave_levels1,wave_levels2) :
            sim = similarity.similarity(tim_list,w1,w2)
            mse = sim.mse()
            mse_list += [mse]
        mse_list_list += [mse_list]

os.chdir("../")
dis = np.loadtxt(f"distance_{date_input}.txt",usecols=(2),unpack=True)
was_list_list=np.array(was_list_list)
mse_list_list = np.array(mse_list_list)
array, file_freq1, file_freq2, file_init = freq_select(date_input)
os.chdir(f"./{date_input}_Wasserstein_softplus")
np.save(f"{date_input}_{direction_input}_wasserstein.npy",was_list_list)
os.chdir("../")
os.chdir(f"./{date_input}_mse")
np.save(f"{date_input}_{direction_input}_mse.npy",mse_list_list)


# freq_list_list,coh_list_list = [],[]
# for N,n1 in tqdm(zip(Number,number1_list)) :
#     str_n1 = str(n1)
#     wave1 = load_wave(date_input,time_input,str_n1,direction_input)
#     wave1 = wave1[0:15000]
#     number2_list = number1_list[N:]
#     for n2 in number2_list :
#         str_n2 = str(n2)
#         wave2 = load_wave(date_input,time_input,str_n2,direction_input)
#         wave2 = wave2[0:15000]
#         sim = similarity.similarity(tim_list,wave1,wave2)
#         freq_list,coh_list = sim.coherence(0.5)
#         print(freq_list)
#         coh_list_list += [coh_list]

# number_list = [98,196,392,784]
# os.chdir("../")
# dis = np.loadtxt(f"distance_{date_input}.txt",usecols=(2),unpack=True)
# coh_list_list = np.array(coh_list_list)
# os.chdir(f"./{date_input}_coherence")
# np.save(f"{date_input}_{direction_input}_coh_window=0.5[Hz].npy",coh_list_list)
# np.save(f"{date_input}_{direction_input}_freq.npy",freq_list)