import numpy as np
import os
from numpy.core.arrayprint import format_float_positional
import ot
import matplotlib.pyplot as plt 
import pywt
from tqdm import tqdm

def calc_devided(numer,demon) :

    return numer/demon

###########
#　正規化 #
###########

def normalized(wave,sum_time) :

    xmin = 0
    xmax = 0.01*sum_time
    h = (xmax-xmin)/sum_time

    s = h*(np.sum(wave)-wave[0]/2-wave[sum_time-1]/2)

    amp_norm_list = []
    for pos1 in wave:
        amp_norm_list.append(calc_devided(pos1,s))

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

    return number1_list, tim_list, t, Number

date_input = input('日付を入力>>')
time_input = input('時刻を入力>>')

number1_list,tim_list,t,Number = data_select(date_input)

was_list = []
for N,n1 in tqdm(zip(Number,number1_list)) :
    n1 = str(n1)
    wave1_EW = np.load("{a}{b}_F{c}_EW_modified.npy".format(a=date_input,b=time_input,c=n1))
    wave1_NS = np.load("{a}{b}_F{c}_NS_modified.npy".format(a=date_input,b=time_input,c=n1))
    w1_list = []
    for w1_EW,w1_NS in zip(wave1_EW,wave1_NS) :
        w1 = np.sqrt(w1_EW**2+w1_NS**2)
        w1_list += [w1]
    
    w1_norm = normalized(w1_list,t)

    number2_list = number1_list[N:]
    for n2 in number2_list :
        n2 = str(n2)
        wave2_EW = np.load("{a}{b}_F{c}_EW_modified.npy".format(a=date_input,b=time_input,c=n2))
        wave2_NS = np.load("{a}{b}_F{c}_NS_modified.npy".format(a=date_input,b=time_input,c=n2))
        w2_list = []
        for w2_EW,w2_NS in zip(wave2_EW,wave2_NS) :
            w2 = np.sqrt(w2_EW**2+w2_NS**2)
            w2_list += [w2]
        
        w2_norm = normalized(w2_list,t)

        was = ot.emd2_1d(tim_list,tim_list,w1_norm/sum(w1_norm),w2_norm/sum(w2_norm),metric='minkowski',p=2)

        was_list += [was]

os.chdir("../")
d = "distance_{a}.txt".format(a=date_input)
dis = np.loadtxt(d,usecols=(2),unpack=True)
was_list = np.array(was_list)

plt.figure()
print(len(dis))
print(len(was_list))
coef1 = np.corrcoef(dis, was_list)
fig1 = plt.figure()
ax = fig1.add_subplot(1,1,1)
ax.scatter(dis,was_list, c='red', marker='.')
ax.set_title('{a} r={c} amplitude'.format(a=date_input,c=coef1[0,1]))
ax.set_xlabel('distance[km]')
ax.set_ylabel('2d wasserstein metric')
ax.grid(True)
fig1.tight_layout()
fig1.savefig("{a}_amplitude.png".format(a=date_input))


