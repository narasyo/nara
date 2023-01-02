import numpy as np 
import os

def data_select(date) : 

    if date == "20120327" :
        number1_list = [1,2,3,4,5,7,10,11,13,14,15,16,17,19,21,23]
        tim_list = np.arange(0,360,0.01)
        t = 36000
        Number = np.arange(0,16,1)
    elif date == "20120830" :
        number1_list = [1,2,4,5,6,7,8,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]
        tim_list = np.arange(0,300,0.01)
        t = 30000
        Number = np.arange(0,26,1)
    elif date == "20121207" :
        number1_list = [1,3,4,5,6,8,10,11,13,14,15,16,17,19,20,21,23,24,25,26,27,28,30,31,32,33,34]
        tim_list = np.arange(0,480,0.01)
        t = 48000
        Number = np.arange(0,27,1)
    elif date == "20130804" :
        number1_list = [1,2,3,4,5,8,9,11,12,14,15,16,17,19,20,22,23,24,25,26,27,28,29,30,31,32,34,35,36]
        tim_list = np.arange(0,360,0.01)
        t = 36000
        Number = np.arange(0,29,1)

    return number1_list, tim_list, t, Number

date_list = ["20120327","20120830","20121207","20130804"]
time_list = ["2001","0405","1719","1229"]

os.chdir("./data/furukawa_data/")

for date,time in zip(date_list,time_list) :

    os.chdir(f"./{date}{time}/{date}_modified")
    number1_list, tim_list, t, Number = data_select(date)

    for n1 in number1_list :

        str_n1 = str(n1)
        wave1 = np.load(f"{date}{time}_F{str_n1}_EW_modified.npy")
        wave2 = np.load(f"{date}{time}_F{str_n1}_NS_modified.npy")

        wave_squared = wave1**2 + wave2**2

        max_index = np.argmax(wave_squared)
        max_time = 0.01*max_index
        print(f"{date}の地震の観測点F{str_n1}における最大振幅観測時刻は>>",max_time)

    os.chdir("../")
    os.chdir("../")




    
