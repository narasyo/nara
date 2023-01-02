from socket import AF_AX25
import numpy as np 
import os
import matplotlib.pyplot as plt
from sympy import integrate
from PySGM.vector import vector

def load_wave(date,time,n1,direction) :

    if len(n1) == 1 :
        wave = np.load(f"{date}{time}_F{n1}_{direction}_modified.npy")
    
    elif len(n1) == 2 :
        wave = np.load(f"{date}{time}_F{n1}_{direction}_modified.npy")

    return wave

def plot_wave(time,wave):

    fig = plt.figure(figsize=(25,10))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('time[s]',fontsize =25)
    ax1.set_ylabel(r'acceleration[cm/s$^2$]',fontsize=25)
    ax1.set_xlim(0,200)
    ax1.set_ylim(-100,100)
    ax1.tick_params(direction = "inout", length = 5, colors = "blue",labelsize=25)
    ax1.plot(time[0:36000], wave[0:36000], color = "red",linewidth=0.7)

    fig.tight_layout()
    fig.savefig(f"fig4-4.eps")
    fig.savefig(f"fig4-4.png")
    plt.clf()
    plt.close()

number1_list = [1,3,4,5,6,11,13,14,15,16,17,19,20,21,24,25,26,27,28,30,31,32,33,34]
tim1_list = np.arange(0,480,0.01)
tim2_list = np.arange(0,360,0.01)
t1 = 48000
t2 = 36000
dt = 0.01
low = 0.2
high = 50
Number = np.arange(1,24,1)

os.chdir("./data/furukawa_data/201308041229/20130804")
# os.mkdir("./wave")

wave1 = load_wave("20130804","1229","1","verticle")

plot_wave(tim2_list,wave1)
