import numpy as np 
import os
import matplotlib.pyplot as plt 
from tqdm import tqdm

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

    return amp1_norm,amp2_norm

def data_select(date,direction) :

    if direction == "EW" or "NS" :
        os.chdir("./{a}_modified".format(a=date))
    else :
        os.chdir("./{a}".format(a=date)) 

    if date == "20120327" :
        number1_list = [1,2,3,4,5,7,10,11,13,14,15,16,17,19,21,23]
        tim_list = np.arange(0,360,0.01)
        t = 36000
        Number = np.arange(1,16,1)
        value = 0.007

    elif date == "20120830" :
        number1_list = [1,2,4,5,6,7,8,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]
        tim_list = np.arange(0,300,0.01)
        t = 30000
        Number = np.arange(1,26,1)
        value = 0.007

    elif date == "20121207" :
        number1_list = [1,3,4,5,6,8,10,11,13,14,15,16,17,19,20,21,23,24,25,26,27,28,30,31,32,33,34]
        tim_list = np.arange(0,480,0.01)
        t = 48000
        Number = np.arange(1,27,1)
        value = 0.005

    elif date == "20130804" :
        number1_list = [1,2,3,4,5,8,9,11,12,14,15,16,17,19,20,22,23,24,25,26,27,28,29,30,31,32,34,35,36]
        tim_list = np.arange(0,360,0.01)
        t = 36000
        Number = np.arange(1,29,1)
        value = 0.005

    return number1_list, tim_list, t, Number, value

os.chdir("./data/furukawa_data/")

date_list = ["20120327","20120830","20121207", "20130804"]
time_list = ["2001","0405","1719","1229"]
direction_list = ["EW","NS"]

for date_input,time_input in zip(date_list,time_list) :
    for direction_input in direction_list :
        number1_list,tim_list,t,Number,value = data_select(date_input,direction_input)
        const = np.full(t,100/t)

        for N,n1 in tqdm(zip(Number,number1_list)) :
            str_n1 = str(n1)
            wave1 = np.load(f'{date_input}{time_input}_F{str_n1}_{direction_input}_modified.npy') 
            number2_list = number1_list[N:]

            for n2 in number2_list :
                str_n2 = str(n2)
                wave2 = np.load(f'{date_input}{time_input}_F{str_n2}_{direction_input}_modified.npy')
                
                os.chdir("../")
                os.chdir(f"./{date_input}_linear_plot")

                wave1_norm,wave2_norm = normalized_linear(wave1,wave2,t)

                fig = plt.figure()

                ax1 = fig.add_subplot(3,1,1)
                ax1.set_title(f'{date_input} {direction_input} constant')
                ax1.set_xlabel('time[s]')
                ax1.set_ylabel('probability')
                ax1.set_ylim(0,value)
                ax1.tick_params(direction = "inout", length = 5, colors = "blue")
                ax1.plot(tim_list, const, color = "red")

                ax2 = fig.add_subplot(3,1,2)
                ax2.set_title(f'{date_input} {direction_input} F{str_n1}')
                ax2.set_xlabel('time[s]')
                ax2.set_ylabel('probability')
                ax2.set_ylim(0,value)
                ax2.tick_params(direction = "inout", length = 5, colors = "blue")
                ax2.plot(tim_list, wave1_norm, color = "red")

                ax3 = fig.add_subplot(3,1,3)
                ax3.set_title(f'{date_input} {direction_input} F{str_n2}')
                ax3.set_xlabel('time[s]')
                ax3.set_ylabel('probability')
                ax3.set_ylim(0,value)
                ax3.tick_params(direction = "inout", length = 5, colors = "blue")
                ax3.plot(tim_list, wave2_norm, color = "red")

                fig.tight_layout()
                fig.savefig(f'{date_input}_{direction_input}_F{str_n1}&F{str_n2}.png')
                
                plt.clf()
                plt.close()

                os.chdir("../")
                os.chdir(f"./{date_input}_modified")
        os.chdir("../")
    






