import numpy as np
import os
import ot
import matplotlib.pyplot as plt 

tim=np.arange(0,360,0.01)
t = 36000

os.chdir("./data/furukawa_data/20120327_modified")
number1_list = [1,2,3,4,5,7,10,11,13,14,15,16,17,19,21,23]
Number = np.arange(1,16,1)
was_list = []
for N,n1 in zip(Number,number1_list) :
    n1 = str(n1)
    amp1 = np.load("201203272001_F{a}_EW_modified.npy".format(a=n1))
    number2_list = number1_list[N:]
    for n2 in number2_list :
        n2 = str(n2)
        amp2 = np.load("201203272001_F{b}_EW_modified.npy".format(b=n2))

        xmin = 0
        xmax = 0.01*t
        h = (xmax-xmin)/t
        c1 = np.abs(min(amp1))
        c2 = np.abs(min(amp2))
        c = max(c1,c2)

        pos_amp1_list = []
        for uni_amp1 in amp1 :
            if uni_amp1 >= 0 :
                pos_amp1 = uni_amp1+1/c
                pos_amp1_list += [pos_amp1]
            else :
                pos_amp1 = (np.exp(c*uni_amp1))/c
                pos_amp1_list += [pos_amp1]

        pos_amp2_list = []
        for uni_amp2 in amp2 :
            if uni_amp2 >= 0 :
                pos_amp2 = uni_amp2+1/c
                pos_amp2_list += [pos_amp2]
            else :
                pos_amp2 = np.exp(c*uni_amp2)/c
                pos_amp2_list += [pos_amp2]
    
        s1 = h*(np.sum(pos_amp1_list)-pos_amp1_list[0]/2-pos_amp1_list[t-1]/2)

        s2 = h*(np.sum(pos_amp2_list)-pos_amp2_list[0]/2-pos_amp2_list[t-1]/2)

        def calc_devided_s1(dev1):
            return dev1/s1

        def calc_devided_s2(dev2):
            return dev2/s2

        amp1_norm_list = []
        for pos1 in pos_amp1_list:
            amp1_norm_list.append(calc_devided_s1(pos1))

        amp2_norm_list =[]
        for pos2 in pos_amp2_list:
            amp2_norm_list.append(calc_devided_s2(pos2))

        amp1_norm = np.array(amp1_norm_list)
        amp2_norm = np.array(amp2_norm_list)
        was = ot.emd2_1d(tim,tim,amp1_norm/sum(amp1_norm),amp2_norm/sum(amp2_norm),metric='minkowski',p=2)
        was_list += [was]

os.chdir("../")
d = "distance_20120327.txt"
dis = np.loadtxt(d,usecols=(2),unpack=True)
plt.scatter(dis,was_list)
plt.show()