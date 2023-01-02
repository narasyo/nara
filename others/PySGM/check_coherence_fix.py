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
    #     os.chdir("./{a}_modified".format(a=date))
    # else :
    os.chdir(f"./{date}{time}/{date}/") 

    if date == "20120327" :
        number1_list = [1,2,3,4,5,7,10,11,13,14,15,16,17,19,21,23]
        tim_list = np.arange(0,360,0.01)
        t = 36000
        Number = np.arange(0,16,1)
        freq_number = [98,197,393]

    elif date == "20120830" :
        number1_list = [1,2,4,5,6,7,8,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]
        tim_list = np.arange(0,300,0.01)
        t = 30000
        Number = np.arange(0,26,1)
        freq_number = [49,98,197]

    elif date == "20121207" :
        number1_list = [1,3,4,5,6,8,10,11,13,14,15,16,17,19,20,21,23,24,25,26,27,28,30,31,32,33,34]
        tim_list = np.arange(0,480,0.01)
        t = 48000
        Number = np.arange(0,27,1)
        freq_number = [98,197,393]

    elif date == "20130804" :
        number1_list = [1,2,3,4,5,9,11,12,14,15,16,17,19,20,22,23,24,25,26,27,28,29,30,31,36]
        tim_list = np.arange(0,360,0.01)
        t = 36000
        Number = np.arange(0,25,1)
        freq_number = [393]

    return number1_list, tim_list, t, Number, freq_number

os.chdir("./data/furukawa_data")

date_input = "20130804"
time_input = "1229"
direction_input = "verticle"

window = 0.8
frequency = [1.2]

# for date_input,time_input in zip(date_list,time_list) :
#     for direction_input in direction_list :
number1_list,tim_list,t,Number,freq_number = data_select(date_input,time_input)

for N,n1 in tqdm(zip(Number,number1_list)) :
    str_n1 = str(n1)
    wave1 = np.load(f"{date_input}{time_input}_F{str_n1}_{direction_input}_modified.npy")
    acc1 = vector.vector("",tim_list,wave1)
    coh_list = [] 
    number2_list = [x for x in number1_list if x != n1]
    for n2 in number2_list :
        n2 = str(n2)
        wave2 = np.load(f"{date_input}{time_input}_F{n2}_{direction_input}_modified.npy")
        acc2 = vector.vector("",tim_list,wave2)
        freq,coh = acc1.coherence(acc2,window)
        #print(max(freq))

        coh_list += [coh]
    coh_array = np.array(coh_list)

    os.chdir("../")
    d = f"distance_{date_input}.txt"
    obs_n1,obs_n2,distant = np.loadtxt(d,dtype = 'str',unpack=True)
    new_distant = []
    for o1,o2,dis in zip(obs_n1,obs_n2,distant) :
        if o1 == f'F0{str_n1}' or o2 == f'F0{str_n1}' :
            new_distant += [dis]

        elif o1 == f'F{str_n1}' or o2 == f'F{str_n1}' :
            new_distant += [dis]

    float_distant = [float(x) for x in new_distant]

    os.chdir(f"./{date_input}_modified_coherence_fix")

    for fn,fr in zip(freq_number,frequency) :     
        # coef2 = np.corrcoef(new_distant,coh_array[:,fn])
        fig1 = plt.figure()
        ax = fig1.add_subplot(1,1,1)
        ax.set_title(f'{date_input} {direction_input} {fr}Hz F={str_n1}')
        ax.set_xlabel('distance[km]')
        ax.set_xlim(0, 3.5)
        ax.set_ylim(0,1)
        ax.set_ylabel('lagged coherence')
        ax.grid(True)
        for j,n2 in zip(Number,number2_list):
            ax.plot(float_distant[j], coh_array[j,fn], marker='.')
            ax.annotate(n2, xy=(float_distant[j],coh_array[j,fn]))
        fig1.tight_layout()
        fig1.savefig(f"{date_input}_{direction_input}_{fr}Hz_F{str_n1}.png")
        plt.clf()
        plt.close()

    os.chdir("../")
    os.chdir("./20130804")

# if direction_input == "EW" or "NS" :
#     os.chdir("./{a}_modified".format(a=date_input))
# else :
#     os.chdir("./{a}".format(a=date_input)) 

# os.chdir("../")

