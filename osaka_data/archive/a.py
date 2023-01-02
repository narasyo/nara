import numpy as np
import os
import ot
import matplotlib.pyplot as plt

def calc_devided(numer,demon) :

    return numer/demon

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

os.chdir("./osaka_data/archive")

wave1 = np.load("OSA_acc_ud_modified.npy")
wave2 = np.load("OSB_acc_ud_modified.npy")

lag2_level = np.arange(100,-100,-1)
print(lag2_level)
t = np.arange(0,200,1)
time = 200
wave1_lag = wave1[4800:5000]
was_list = []
for l2 in lag2_level :
    wave2_lag = wave2[4800-l2:5000-l2]
    w1_norm,w2_norm = normalized_linear(wave1_lag,wave2_lag,time)
    was = ot.emd2_1d(t,t,w1_norm/sum(w1_norm),w2_norm/sum(w2_norm),metric='minkowski',p=2)
    was_list += [was]

plt.plot(t,was_list)
plt.show()


