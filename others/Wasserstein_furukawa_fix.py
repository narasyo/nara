import numpy as np
import os
from numpy.core.arrayprint import format_float_positional
import ot
import matplotlib.pyplot as plt 
import pywt
from tqdm import tqdm

def calc_devided(numer,demon) :

    return numer/demon

def softplus_normalizing(wave1,wave2,sum_time) :

    xmin = 0
    xmax = 0.01*sum_time
    h = (xmax-xmin)/sum_time
    wave1 = np.array(wave1)
    wave2 = np.array(wave2)

    b =  3.0/(max([max(abs(wave1)),max(abs(wave2))]))
    w1_pos = np.log(np.exp(wave1*b)+1)
    w2_pos = np.log(np.exp(wave2*b)+1)
    s1 = h*(np.sum(w1_pos)-w1_pos[0]/2-w1_pos[sum_time-1]/2)
    s2 = h*(np.sum(w2_pos)-w2_pos[0]/2-w2_pos[sum_time-1]/2)
    p1 = [w1/s1 for w1 in w1_pos]
    p2 = [w2/s2 for w2 in w2_pos]

    return p1,p2

#######################
#  符号感応型正規化法 #
#######################

def normalized_sign_sensitive(wave1,wave2,sum_time,c) :

    xmin = 0
    xmax = 0.01*sum_time
    h = (xmax-xmin)/sum_time
    # c1 = np.abs(min(wave1))
    # c2 = np.abs(min(wave2))
    # const = max(c1,c2)
    pos_amp1_list = []
    for uni_amp1 in wave1 :
        if uni_amp1 >= 0 :
            pos_amp1 = uni_amp1+1/c
            pos_amp1_list += [pos_amp1]
        else :
            pos_amp1 = (np.exp(c*uni_amp1))/c
            pos_amp1_list += [pos_amp1]

    pos_amp2_list = []
    for uni_amp2 in wave2 :
        if uni_amp2 >= 0 :
            pos_amp2 = uni_amp2+1/c
            pos_amp2_list += [pos_amp2]
        else :
            pos_amp2 = np.exp(c*uni_amp2)/c
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

    return amp1_norm,amp2_norm

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

    return amp1_norm,amp2_norm

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
        Number = np.arange(0,16,1)
    elif date == "20120830" :
        number1_list = [1,2,4,5,6,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,28,29,30]
        tim_list = np.arange(0,300,0.01)
        t = 30000
        Number = np.arange(0,23,1)
    elif date == "20121207" :
        number1_list = [1,3,5,6,11,13,14,15,16,17,19,21,24,25,26,27,28,31,33]
        tim_list = np.arange(0,480,0.01)
        t = 48000
        Number = np.arange(0,19,1)
    elif date == "20130804" :
        number1_list = [1,2,3,4,5,9,11,12,14,15,16,17,19,20,22,23,24,25,26,27,28,29,30,31,36]
        tim_list = np.arange(0,360,0.01)
        t = 36000
        Number = np.arange(0,25,1)

    return number1_list, tim_list, t, Number

def load_wave(date,time,n1,direction) :

    if len(n1) == 1 :
        wave = np.load(f"{date}{time}_F{n1}_{direction}_modified.npy")
    
    elif len(n1) == 2 :
        wave = np.load(f"{date}{time}_F{n1}_{direction}_modified.npy")

    return wave

def freq_select(date) :

    if date == "20120327" or "20121207" or "20130804" :

        number = np.arange(0,9,1)
        freq1_list = [0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6]
        freq2_list = [0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2]
        freq_initial = 0.1

    elif date == "20120830" :

        number = np.arange(0,8,1)
        freq1_list = [0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6]
        freq2_list = [0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2]
        freq_initial = 0.2

    return number,freq1_list,freq2_list,freq_initial

date_input = "20130804"
time_input = "1229"
direction_input = "verticle"

os.chdir("./data/furukawa_data")

# for date_input,time_input in zip(date_list,time_list) :
#     for direction_input in direction_list :

number1_list,tim_list,t,Number = data_select(date_input,time_input)

const = np.full(t,100/t)

for N,n1 in tqdm(zip(Number,number1_list)) :
    str_n1 = str(n1)
    wave1 = load_wave(date_input,time_input,str_n1,direction_input) 
    wavelet1 = pywt.Wavelet('dmey')
    wave_levels1 = sort_wave(wave1,wavelet1)

    was_list = [] 
    number2_list = [x for x in number1_list if x != n1]
    for n2 in number2_list :
        str_n2 = str(n2)
        wave2 = load_wave(date_input,time_input,str_n2,direction_input)
        wavelet2 = pywt.Wavelet('dmey')
        wave_levels2 = sort_wave(wave2,wavelet2)
        
        was_freq_list = [] 
        for w1,w2 in zip(wave_levels1,wave_levels2) :
            w1_norm,w2_norm = softplus_normalizing(w1,w2,t)
            was_pos = ot.emd2_1d(tim_list,tim_list,w1_norm/sum(w1_norm),w2_norm/sum(w2_norm),metric='minkowski',p=2)
            w1_norm_neg,w2_norm_neg = softplus_normalizing(-w1,-w2,t)
            was_neg = ot.emd2_1d(tim_list,tim_list,w1_norm_neg/sum(w1_norm_neg),w2_norm_neg/sum(w2_norm_neg),metric='minkowski',p=2)
            was = was_pos+was_neg
            was_freq_list += [was]
            if(n2==30 and n1==2):
                print(was)
        was_list += [was_freq_list]

    os.chdir("../")

    d = f"distance_{date_input}.txt"
    obs_n1,obs_n2,distant = np.loadtxt(d,dtype = 'str',unpack=True)
    new_distant = []
    for o1,o2,dis in zip(obs_n1,obs_n2,distant) :
        if o1 == 'F0{a}'.format(a=str_n1) or o2 == 'F0{a}'.format(a=str_n1) :
            new_distant += [dis]

        elif o1 == 'F{a}'.format(a=str_n1) or o2 == 'F{a}'.format(a=str_n1) :
            new_distant += [dis]

    float_new_distant = []
    for item in new_distant :
        float_new_distant.append(float(item))

    new_distant = np.array(float_new_distant)
    was_list = np.array(was_list)
    array,file_freq1,file_freq2,freq_init = freq_select(date_input)
    os.chdir("./{a}_Wasserstein_softplus_modified_fix".format(a=date_input))

    coef1 = np.corrcoef(new_distant, was_list[:,0])
    fig1 = plt.figure()
    ax = fig1.add_subplot(1,1,1)
    ax.set_title('{a} {b} ~{c}Hz F{d} r={e}'.format(a=date_input,b=direction_input,c=freq_init,d=str_n1,e=coef1[0,1]))
    ax.set_xlabel('distance[km]',fontsize=25)
    ax.set_xlim(0, 3.5)
    ax.set_ylabel('2d wasserstein metric',fontsize=25)
    ax.grid(True)
    number = np.arange(0,len(new_distant),1) 
    for i,n2 in zip(number,number2_list):
        ax.plot(new_distant[i], was_list[i,0], marker='.')
        ax.annotate(n2, xy=(new_distant[i],was_list[i,0]))
    fig1.tight_layout()
    fig1.savefig("{a}_{b}_freq<={c}Hz_F{d}_.png".format(a=date_input,b=direction_input,c=freq_init,d=str_n1))
    plt.clf()
    plt.close()

    for i,f1,f2 in zip(array,file_freq1,file_freq2) :
        coef2 = np.corrcoef(new_distant,was_list[:,i])
        fig2 = plt.figure()
        ax = fig2.add_subplot(1,1,1)
        ax.set_title('{a} {b} {F1}Hz~{F2}Hz F={c} r={d}'.format(a=date_input,b=direction_input,F1=f1,F2=f2,c=str_n1,d=coef2[0,1]))
        ax.set_xlabel('distance[km]')
        ax.set_xlim(0, 3.5)
        ax.set_ylim(19,0)
        ax.set_ylabel('2d wasserstein metric')
        ax.grid(True)
        for j,n2 in zip(number,number2_list):
            ax.plot(new_distant[j], was_list[j,i], marker='.')
            ax.annotate(n2, xy=(new_distant[j],was_list[j,i]))
        fig2.tight_layout()
        fig2.savefig("{a}_{b}_{F1}Hz<=freq<={F2}Hz_F{c}_.png".format(a=date_input,b=direction_input,F1=f1,F2=f2,c=str_n1))
        plt.clf()
        plt.close()

    os.chdir("../")
    os.chdir("./20130804")