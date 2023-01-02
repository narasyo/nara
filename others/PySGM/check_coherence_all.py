import numpy as np
import matplotlib.pyplot as plt
import os
import vector
from numpy.core.arrayprint import format_float_positional
import ot
import matplotlib.pyplot as plt 
from tqdm import tqdm

def data_select(date,time) :

    # if direction == "EW" or "NS" :
    os.chdir(f"./{date}{time}/{date}")
    # else :
    #     os.chdir("./data/furukawa_data/{a}".format(a=date)) 

    if date == "20120327" :
        number1_list = [1,2,3,4,5,7,10,11,13,14,15,16,17,19,21,23]
        tim_list = np.arange(0,360,0.01)
        t = 36000
        Number = np.arange(1,16,1)
        freq_number = [98,197,393]

    elif date == "20120830" :
        number1_list = [1,2,4,5,6,7,8,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]
        tim_list = np.arange(0,300,0.01)
        t = 30000
        Number = np.arange(1,26,1)
        freq_number = [49,98,197]

    elif date == "20121207" :
        number1_list = [1,3,5,6,11,13,14,15,16,17,19,21,24,25,26,27,28,31,33]
        tim_list = np.arange(0,480,0.01)
        t = 48000
        Number = np.arange(1,19,1)
        freq_number = [49,98,197,393,784,1568,3136]

    elif date == "20130804" :
        number1_list = [1,2,3,4,5,9,11,12,14,15,16,17,19,20,22,23,24,25,26,27,28,29,30,31,36]
        tim_list = np.arange(0,360,0.01)
        t = 36000
        Number = np.arange(1,25,1)
        freq_number = [49,98,197,393,784,1568,3136]

    return number1_list, tim_list, t, Number, freq_number

os.chdir("./data/furukawa_data")

date_input = "20130804"
time_input = "1229"
direction_input = "verticle"

windows = [0.1,0.2,0.4,0.8,1.6,3.2,6.4]
frequency = [0.15,0.3,0.6,1.2,2.4,4.8,9.6]

number1_list,tim_list,t,Number,freq_number = data_select(date_input,time_input)
    
for window in windows :
    coh_list = []
    for N,n1 in tqdm(zip(Number,number1_list)) :
        str_n1 = str(n1)
        wave1 = np.load(f"{date_input}{time_input}_F{str_n1}_{direction_input}_modified.npy")
        acc1 = vector.vector("",tim_list,wave1)

        number2_list = number1_list[N:]
        for n2 in number2_list :
            str_n2 = str(n2)
            wave2 = np.load(f"{date_input}{time_input}_F{str_n2}_{direction_input}_modified.npy")
            acc2 = vector.vector("",tim_list,wave2)
            freq,coh = acc1.coherence(acc2,window)
            coh_list += [coh]
    coh_array = np.array(coh_list)

    os.chdir("../")
    d = f"distance_{date_input}.txt"
    dis = np.loadtxt(d,usecols=(2),unpack=True)

    os.chdir(f"./{date_input}_modified_coherence")

    for fn,fr in zip(freq_number,frequency) :
        coef = np.corrcoef(dis,coh_array[:,fn])
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.tick_params(direction = "inout", length = 5, colors = "blue",labelsize=10)
        ax.scatter(dis,coh_array[:,fn], c='red', marker='.')
        ax.set_xlabel('distance[km]',fontsize=10)
        ax.set_ylabel('lagged coherence',fontsize=10)
        ax.set_xlim(0,)
        ax.set_ylim(0,1)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(f"{date_input}_{direction_input}_{fr}Hz_coherence_window={window}Hz.png")
        fig.savefig(f"{date_input}_{direction_input}_{fr}Hz_coherence_window={window}Hz.eps")
        plt.clf()
        plt.close()



    os.chdir("../")
    os.chdir("./20130804")

# if direction_input == "EW" or "NS" :
# os.chdir(f"./{date_input}_modified")
# else :
#     os.chdir(f"./{date_input}") 

os.chdir("../")

