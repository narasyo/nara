import numpy as np 
import os

##############
#動かすな！！#
##############
def data_select(date) : 

    if date == "20120327" :
        number1_list = [1,2,3,4,5,7,10,11,13,14,15,16,17,19,21,23]
        s_list = ["21","23"]

    elif date == "20120830" :
        number1_list = [1,2,4,5,6,7,8,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]
        s_list = ["20","21","22","23"]

    elif date == "20121207" :
        number1_list = [1,3,4,5,6,8,10,11,13,14,15,16,17,19,20,21,23,24,25,26,27,28,30,31,32,33,34]
        s_list = ["20","21","23"]

    elif date == "20130804" :
        number1_list = [1,2,3,4,5,8,9,11,12,14,15,16,17,19,20,22,23,24,25,26,27,28,29,30,31,32,34,35,36]   
        s_list = ["20","22","23"]

    return number1_list,s_list

date_list = ["20120327","20120830","20121207","20130804"]
time_list = ["2001","0405","1719","1229"]
direction_list = ["EW","NS","verticle"]

os.chdir("./data/furukawa_data")
for date,time in zip(date_list,time_list) :
    n_list,s_list = data_select(date)
    os.chdir(f"./{date}{time}")

    for direction in direction_list :
        if direction == "verticle" :
            os.chdir(f"./{date}")
            for s in s_list :
                wave = np.load(f"{date}{time}_F{s}_{direction}_modified.npy")
                new_wave = -wave
                np.save(f"{date}{time}_F{s}_{direction}_modified.npy",new_wave)
            os.chdir("../")
        
        elif direction == "EW" or "NS" :
            os.chdir(f"./{date}_modified")
            for s in s_list :
                wave = np.load(f"{date}{time}_F{s}_{direction}_modified.npy")
                new_wave = -wave
                np.save(f"{date}{time}_F{s}_{direction}_modified.npy",new_wave)
            os.chdir("../")
        
    os.chdir("../")





