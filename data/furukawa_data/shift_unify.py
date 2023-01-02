import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import similarity
import sys
import pathlib
sys.path.append(pathlib.Path(__file__).resolve().parents[2].as_posix())
from others.plot import plot

os.chdir("./Python_Code/data/furukawa_data")
os.chdir("201308041229/20130804")

number1_list = [1,3,5,6,11,13,14,15,16,17,19,21,24,25,26,27,28,31,33]
number2_list = [1,2,3,4,5,9,11,12,14,15,16,19,20,22,23,24,25,26,27,28,29,30,31,36]
time = np.arange(0,360,0.01)

wave_list = []
title_list = []
i_list = []
for n2 in number2_list :
    n2 = str(n2)
    wave = np.load(f"201308041229_F{n2}_verticle_modified.npy")
    for w,i in zip(wave,range(36000)) :
        if np.abs(w) >= 0.5 :
            i_list += [i]
            break
    wave_list += [wave]
    title = f"F{n2}"
    title_list += [title]
wave_array = np.array(wave_list)
os.chdir("../")
os.mkdir("./20130804_wavedata")
os.chdir("./20130804_wavedata")
zeros = np.zeros(500)
new_wave_list = []
for wave,i,n2 in zip(wave_array,i_list,number2_list) :
    new_wave = wave[i:]
    new_wave = np.append(zeros,new_wave)
    new_wave = new_wave[:15000]
    np.save(f"20130804_F{n2}_UD_unify.npy",new_wave)
    new_wave_list += [new_wave]

color = "black"
new_wave = np.array(new_wave_list)
wave_graph = plot(time[:15000],new_wave,24,"",figsize=(30,200),
                  fontsize=50,
                  labelsize=50,
                  linewidth=2,
                  color=color,
                  xlim=[0,150],
                  ylim=[-50,50],
                  params_length=10,
                  xlabel="time[s]",
                  ylabel=r"acceleration[cm/s$^2$]",
                  title=title_list,
                  name="wave_shift_unify")
wave_graph.plot_graph_row(True)