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
for n2 in number2_list :
    n2 = str(n2)
    wave = np.load(f"201308041229_F{n2}_verticle_modified.npy")
    wave_list += [wave]
    title = f"F{n2}"
    title_list += [title]

color = "black"
wave = np.array(wave_list)
wave = wave[0:24]
wave_graph = plot(time,wave,24,"",figsize=(30,200),
                  fontsize=50,
                  labelsize=50,
                  linewidth=2,
                  color=color,
                  xlim=[5,10],
                  ylim=[-10,10],
                  params_length=10,
                  xlabel="time[s]",
                  ylabel=r"acceleration[cm/s$^2$]",
                  title=title_list,
                  name="wave")
wave_graph.plot_graph_row(True)