import numpy as np
import os
import ot
import matplotlib.pyplot as plt 
import pywt
from tqdm import tqdm
import similarity
import sys
import pathlib
sys.path.append(pathlib.Path(__file__).resolve().parents[2].as_posix())
from others.plot import plot

def freq_select(date) :

    if date == "20120327" or "20121207" or "20130804" :

        number = np.arange(1,8,1)
        freq1_list = [0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6]
        freq2_list = [0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2]
        freq_initial = 0.1

    elif date == "20120830" :

        number = np.arange(1,9,1)
        freq1_list = [0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6]
        freq2_list = [0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2]
        freq_initial = 0.2

    return number,freq1_list,freq2_list,freq_initial

date_input = "20130804"
time_input = "1229"
direction_input = "UD"
os.chdir('./data/furukawa_data/201308041229')

dis = np.loadtxt(f"distance_{date_input}.txt",usecols=(2),unpack=True)
array, file_freq1, file_freq2, file_init = freq_select(date_input)
os.chdir("./20130804_Wasserstein_softplus")
was_list_list = np.load(f"{date_input}_{direction_input}_wasserstein.npy")
print(was_list_list.shape)

y1_max = max(was_list_list[:,0])
color = "black"
wave_graph = plot(dis,was_list_list[:,0],"","",figsize=(30,30),
                  fontsize=50,
                  labelsize=50,
                  linewidth=2,
                  color=color,
                  xlim=[0,4],
                  ylim=[y1_max,0],
                  params_length=10,
                  xlabel="distance[km]",
                  ylabel="wasserstein metric",
                  title=f"{date_input} {direction_input} ~{file_init}Hz wasserstein",
                  name=f"{date_input}_{direction_input}_~{file_init}Hz_wasserstein")
wave_graph.plot_scatter()

for i,f1,f2 in zip(array,file_freq1,file_freq2) :
    y2_max = max(was_list_list[:,i])
    color = "black"
    wave_graph = plot(dis,was_list_list[:,i],"","",figsize=(30,30),
                    fontsize=50,
                    labelsize=50,
                    linewidth=2,
                    color=color,
                    xlim=[0,4],
                    ylim=[y2_max,0],
                    params_length=10,
                    xlabel="distance[km]",
                    ylabel="wasserstein metric",
                    title=f"{date_input} {direction_input} {f1}Hz~{f2}Hz wasserstein",
                    name=f"{date_input}_{direction_input}_{f1}Hz~{f2}Hz_wasserstein")
    wave_graph.plot_scatter()


"""

dis = np.loadtxt(f"distance_{date_input}.txt",usecols=(2),unpack=True)
os.chdir("./20130804_coherence")
coh_list_list = np.load(f"{date_input}_{direction_input}_coh_window=0.5[Hz].npy")
freq_list = np.load(f"{date_input}_{direction_input}_coh_window=0.5[Hz].npy")
f_list = [0.3,0.6,1.2,2.4]
n_list = [25,49,98,196]
for n,f in zip(n_list,f_list) :
    color = "black"
    wave_graph = plot(dis,coh_list_list[:,n],"","",figsize=(30,30),
                    fontsize=50,
                    labelsize=50,
                    linewidth=2,
                    color=color,
                    xlim=[0,4],
                    ylim=[0,1],
                    params_length=10,
                    xlabel="distance[km]",
                    ylabel="lagged coherence",
                    title=f"{date_input} {direction_input} {f}Hz coherence",
                    name=f"{date_input}_{direction_input}_{f}Hz_coherence")
    wave_graph.plot_scatter()

"""