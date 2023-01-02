import numpy as np
import os
import ot
import matplotlib.pyplot as plt 
import pywt
from tqdm import tqdm

def calc_devided(numer,demon) :

    return numer/demon

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

def normalized(wave,sum_time) :

    xmin = 0
    xmax = 0.01*sum_time
    h = (xmax-xmin)/sum_time

    s = h*(np.sum(wave)-wave[0]/2-wave[sum_time-1]/2)

    amp_norm_list = []
    for pos in wave:
        amp_norm_list.append(calc_devided(pos,s))

    amp_norm = np.array(amp_norm_list)

    return amp_norm

def data_select(date) :

    
    os.chdir("./data/furukawa_data/{a}_modified".format(a=date))

    if date == "20120327" :
        number1_list = [1,2,3,4,5,7,10,11,13,14,15,16,17,19,21,23]
        tim_list = np.arange(0,360,0.01)
        t = 36000
        Number = np.arange(1,16,1)
    elif date == "20120830" :
        number1_list = [1,2,4,5,6,7,8,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]
        tim_list = np.arange(0,300,0.01)
        t = 30000
        Number = np.arange(1,26,1)
    elif date == "20121207" :
        number1_list = [1,3,4,5,6,8,10,11,13,14,15,16,17,19,20,21,23,24,25,26,27,28,30,31,32,33,34]
        tim_list = np.arange(0,480,0.01)
        t = 48000
        Number = np.arange(1,27,1)
    elif date == "20130804" :
        number1_list = [1,2,3,4,5,8,9,11,12,14,15,16,17,19,20,22,23,24,25,26,27,28,29,30,31,32,34,35,36]
        tim_list = np.arange(0,360,0.01)
        t = 36000
        Number = np.arange(1,29,1)

    return number1_list,tim_list,t,Number

date_input = input('日付を入力>>')
time_input = input('時刻を入力>>')

number1_list,tim_list,t,Number = data_select(date_input)

was_list_list = []
for N,n1 in tqdm(zip(Number,number1_list)) :
    n1 = str(n1)
    wave1_EW = np.load("{a}{b}_F{c}_EW_modified.npy".format(a=date_input,b=time_input,c=n1))
    wave1_NS = np.load("{a}{b}_F{c}_NS_modified.npy".format(a=date_input,b=time_input,c=n1))
    wavelet1 = pywt.Wavelet('dmey')
    wave_EW_levels1 = sort_wave(wave1_EW,wavelet1)
    wave_NS_levels1 = sort_wave(wave1_NS,wavelet1)
    
    w1_norm_list = []
    for w1_EW,w1_NS in zip(wave_EW_levels1,wave_NS_levels1) :
        w1 = np.sqrt(w1_EW**2+w1_NS**2)
        w1_norm = normalized(w1,t)
        w1_norm_list += [w1_norm]

    number2_list = number1_list[N:]
    for n2 in number2_list :
        n2 = str(n2)
        wave2_EW = np.load("{a}{b}_F{c}_EW_modified.npy".format(a=date_input,b=time_input,c=n2))
        wave2_NS = np.load("{a}{b}_F{c}_NS_modified.npy".format(a=date_input,b=time_input,c=n2))
        wavelet2 = pywt.Wavelet('dmey')
        wave_EW_levels2 = sort_wave(wave2_EW,wavelet2)
        wave_NS_levels2 = sort_wave(wave2_NS,wavelet2)
        
        w2_norm_list = []
        for w2_EW,w2_NS in zip(wave_EW_levels2,wave_NS_levels2) :
            w2 = np.sqrt(w2_EW**2+w2_NS**2)
            w2_norm = normalized(w2,t)
            w2_norm_list += [w2_norm]
    
        was_list = [] 
        for w1,w2 in zip(w1_norm_list,w2_norm_list) :

            was = ot.emd2_1d(tim_list,tim_list,w1/sum(w1),w2/sum(w2),metric='minkowski',p=2)

            was_list += [was]
        was_list_list += [was_list]

os.chdir("../")
d = "distance_{a}.txt".format(a=date_input)
dis = np.loadtxt(d,usecols=(2),unpack=True)
was_list_list=np.array(was_list_list)
array = np.arange(1,10,1)
file_freq1 = [0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6]
file_freq2 = [0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2]
os.chdir("./{a}_Wasserstein_squared".format(a=date_input))

plt.figure()
print(len(dis))
print(len(was_list_list[:,0]))
coef1 = np.corrcoef(dis, was_list_list[:,0])
fig1 = plt.figure()
ax = fig1.add_subplot(1,1,1)
ax.scatter(dis,was_list_list[:,0], c='red', marker='.')
ax.set_title('{a} squared ~0.1Hz r={c}'.format(a=date_input,c=coef1[0,1]))
ax.set_xlabel('distance[km]')
ax.set_ylabel('2d wasserstein metric')
ax.grid(True)
fig1.tight_layout()
fig1.savefig("{a}_squared_freq<=0.1Hz.png".format(a=date_input))
plt.clf()
plt.close()

for i,f1,f2 in zip(array,file_freq1,file_freq2) :
    coef2 = np.corrcoef(dis,was_list_list[:,i])
    fig2 = plt.figure()
    ax = fig2.add_subplot(1,1,1)
    ax.scatter(dis,was_list_list[:,i], c='red', marker='.')
    ax.set_title('{a} squared {F1}Hz~{F2}Hz r={c}'.format(a=date_input,F1=f1,F2=f2,c=coef2[0,1]))
    ax.set_xlabel('distance[km]')
    ax.set_ylabel('2d wasserstein metric')
    ax.grid(True)
    fig2.tight_layout()
    fig2.savefig("{a}_squared_{F1}Hz<=freq<={F2}Hz.png".format(a=date_input,F1=f1,F2=f2))
    plt.clf()
    plt.close()