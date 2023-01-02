import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import similarity
import sys
import pathlib
sys.path.append(pathlib.Path(__file__).resolve().parents[1].as_posix())
from others.plot import plot

def plot_similarity(shift,mse,was_linear,was_exp,was_soft,name) :

    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(shift, mse, color = "green", linewidth=1.0)
    ax1.set_xlabel("shift[s]")
    ax1.set_ylabel("MSE")
    ax1.set_title("MSE")
    ax1.set_xlim(-40,40)
    ax1.set_ylim(0,)
    ax1.tick_params(direction="inout",labelsize = 10,length=5)

    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(shift,was_linear,color = "green", linewidth=1.0)
    ax2.set_xlabel("shift[s]")
    ax2.set_ylabel(r"$W_{2}^2$")
    ax2.set_title(r"$W_{2}^2$(linear normalizing)")
    ax2.set_xlim(-40,40)
    ax2.set_ylim(0,)
    ax2.tick_params(direction="inout",labelsize = 10,length=5)

    ax3 = fig.add_subplot(2,2,3)
    ax3.plot(shift,was_exp,color = "green", linewidth=1.0)
    ax3.set_xlabel("shift[s]")
    ax3.set_ylabel(r"$W_{2}^2$")
    ax3.set_title(r"$W_{2}^2$(exponential normalizing)")
    ax3.set_xlim(-40,40)
    ax3.set_ylim(0,)
    ax3.tick_params(direction="inout",labelsize = 10,length=5)

    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(shift,was_soft,color = "green", linewidth=1.0)
    ax4.set_xlabel("shift[s]")
    ax4.set_ylabel(r"$W_{2}^2$")
    ax4.set_title(r"$W_{2}^2$(softplus normalizing)")
    ax4.set_xlim(-40,40)
    ax4.set_ylim(0,)
    ax4.tick_params(direction="inout",labelsize = 10,length=5)
    # show plots
    fig.tight_layout()
    fig.savefig(f"{name}.png")
    fig.savefig(f"{name}.eps")

def plot_wave(time_list,wave1,wave2,name) :

    point = {
    "start":[10,400],
    "end":[90,400]
    }

    fig1,ax1 = plt.subplots()
    ax1.plot(time_list,wave1,color="blue",linewidth=0.6)
    ax1.plot(time_list,wave2,color="red",linewidth=0.6)
    ax1.set_xlabel(r"$t$[s]",fontsize = 10)
    ax1.set_ylabel(r"acceleration[cm/s$^2$]",fontsize = 10)
    ax1.set_xlim(0,40.96)
    # ax1.set_ylim()
    ax1.tick_params(direction="inout",labelsize = 10,length=5)
    ax1.legend([r"$f(t)$",r"$\beta(t)*f(t)$"], loc="upper right", fontsize=15)
    ax1.annotate("", xy=point["end"], xytext=point["start"],arrowprops=dict(shrink=0, width=1, headwidth=15, headlength=10, connectionstyle='arc3',facecolor='gray', edgecolor='gray'))
    fig1.tight_layout()
    fig1.savefig(f"{name}.png")
    fig1.savefig(f"{name}.eps")


######################
#  拡大・縮小(全体)  #
######################

"""

wave_file = "coherence/data/1995JR_Takatori.acc"
wave1 = np.loadtxt(wave_file,usecols=(2),unpack=True)
tim = np.arange(0,40.96,0.01)

alpha_list = np.arange(0,2,0.01)

wave2_list,mse_list,linear_list,exp_list,soft_list = [],[],[],[],[]
for alpha in tqdm(alpha_list):

    wave2 = alpha*wave1

    sim = similarity.similarity(tim,wave1,wave2)

    # mse = sim.mse()
    # linear = sim.wasserstein_linear_normalizing()
    # exp = sim.wasserstein_exponential_normalizing()
    # soft = sim.wasserstein_softplus_normalizing()

    wave2_list += [wave2]
    # mse_list += [mse]
    # linear_list += [linear]
    # exp_list += [exp]
    # soft_list += [soft]

wave2_list = np.array(wave2_list)

os.chdir("./soturon_tex/figure")
plot_wave(tim,wave1,wave2_list[199],"kobe_wave_mag")
# plot_similarity(alpha_list,mse_list,linear_list,exp_list,soft_list,"kobe_mag")

"""

######################
#  拡大・縮小(一部)  #
######################

"""

wave_file = "coherence/data/1995JR_Takatori.acc"
wave1 = np.loadtxt(wave_file,usecols=(2),unpack=True)
tim = np.arange(0,40.96,0.01)

beta_list = np.arange(0,2,0.01)

wave2_list,mse_list,linear_list,exp_list,soft_list = [],[],[],[],[]
for beta in tqdm(beta_list):

    wave2 = []
    for w1,i in zip(wave1,range(4096)) :
        if i <= 1000 :
            wave2 += [w1]
        elif 1001 <= i <= 2000 :
            wave2 += [beta*w1]
        elif 2001 <= i :
            wave2 += [w1]

    wave2 = np.array(wave2)

    # sim = similarity.similarity(tim,wave1,wave2)

    # mse = sim.mse()
    # linear = sim.wasserstein_linear_normalizing()
    # exp = sim.wasserstein_exponential_normalizing()
    # soft = sim.wasserstein_softplus_normalizing()

    wave2_list += [wave2]
    # mse_list += [mse]
    # linear_list += [linear]
    # exp_list += [exp]
    # soft_list += [soft]

wave2_list = np.array(wave2_list)

os.chdir("./soturon_tex/figure")
plot_wave(tim,wave1,wave2_list[199],"kobe_wave_partmag")
# plot_similarity(alpha_list,mse_list,linear_list,exp_list,soft_list,"kobe_partmag")

"""

###########################
#  時間シフト(ノイズあり) #
###########################

"""

dlen = 4096 #ノイズデータのデータ長
mean = 0.0  #ノイズの平均値
std  = 300  #ノイズの分散

y1 = np.random.normal(mean,std,dlen)
y2 = np.random.normal(mean,std,dlen)

noise_mag1 = np.arange(0,1,0.002)
noise_mag2 = np.arange(1,0,-0.00027809)
noise_window = np.append(noise_mag1,noise_mag2)

noise1_list,noise2_list = [],[]
for n1,n2,win in zip(y1,y2,noise_window) :
    noise1 = n1*win
    noise2 = n2*win
    noise1_list += [noise1]
    noise2_list += [noise2]

noise1_array = np.array(noise1_list)
noise2_array = np.array(noise2_list)

wave_file = "coherence/data/1995JR_Takatori.acc"
wave1 = np.loadtxt(wave_file,usecols=(2),unpack=True)

wave1_noise = wave1 + noise1_array
tim = np.arange(0,122.88,0.01)

zeros = np.zeros(4096)
wave1_array = np.append(zeros,wave1_noise)
wave1_array = np.append(wave1_array,zeros)
wave2 = wave1 + noise2_array
tau = np.arange(-40.96,40.96,0.01)

wave2_list,mse_list,linear_list,exp_list,soft_list,sem_list = [],[],[],[],[],[]
for i,s in tqdm(zip(range(8192),tau)) :

    shift_number = int(s*100)
    zeros1_shift = np.zeros(4096+shift_number)
    zeros2_shift = np.zeros(4096-shift_number)
    wave2_array = np.array(wave2)
    wave2_array = np.append(zeros1_shift,wave2_array)
    wave2_array = np.append(wave2_array,zeros2_shift)
    
    sim = similarity.similarity(tim,wave1_array,wave2_array)

    # mse = sim.mse()
    # linear = sim.wasserstein_linear_normalizing()
    # exp = sim.wasserstein_exponential_normalizing()
    soft = sim.wasserstein_softplus_normalizing()
    numer = np.sum((wave1_array+wave2_array)**2)
    demon = 2*np.sum((wave1_array**2+wave2_array**2))
    
    sem = numer/demon

    sem_list += [sem]

    wave2_list += [wave2_array]
    # mse_list += [mse]
    # linear_list += [linear]
    # exp_list += [exp]
    soft_list += [soft]

wave2_list = np.array(wave2_list)

os.chdir("./kansaishibu/value")
np.save("wave1_shift_noise.npy",wave1_array)
np.save("wave2_shift_noise.npy",wave2_list)
np.save("soft_shift_noise.npy",soft_list)
np.save("sem_shift_noise.npy",sem_list)


# os.chdir("./kansaishibu/figure")

"""

###########################
#  時間シフト(ノイズなし) #
###########################

"""

wave_file = "coherence/data/1995JR_Takatori.acc"
wave1 = np.loadtxt(wave_file,usecols=(2),unpack=True)
tim = np.arange(0,122.88,0.01)

os.chdir("./kansaishibu/value")

zeros = np.zeros(4096)
wave1_array = np.append(zeros,wave1)
wave1_array = np.append(wave1_array,zeros)
wave2 = wave1
tau = np.arange(-40.96,40.96,0.01)

wave2_list,mse_list,linear_list,exp_list,soft_list,sem_list = [],[],[],[],[],[]
for i,s in tqdm(zip(range(8192),tau)) :

    shift_number = int(s*100)
    zeros1_shift = np.zeros(4096+shift_number)
    zeros2_shift = np.zeros(4096-shift_number)
    wave2_array = np.array(wave2)
    wave2_array = np.append(zeros1_shift,wave2_array)
    wave2_array = np.append(wave2_array,zeros2_shift)
    
    sim = similarity.similarity(tim,wave1_array,wave2_array)

    wave2_list += [wave2_array]
    mse = sim.mse()
    linear = sim.wasserstein_linear_normalizing()
    exp = sim.wasserstein_exponential_normalizing()
    soft = sim.wasserstein_softplus_normalizing()

    numer = np.sum((wave1_array+wave2_array)**2)
    demon = 2*np.sum((wave1_array**2+wave2_array**2))
    
    sem = numer/demon

    sem_list += [sem]

    mse_list += [mse]
    linear_list += [linear]
    exp_list += [exp]
    soft_list += [soft]

wave2_list = np.array(wave2_list)

plot_similarity(tau,mse_list,linear_list,exp_list,soft_list,"kobe_shift")
plot_wave(tim,wave1_array,wave2_list[0],"kobe_wave_shift")

"""

#############
#  地震波形 #
#############


# wave_file = "coherence/data/1995JR_Takatori.acc"
# wave1 = np.loadtxt(wave_file,usecols=(2),unpack=True)
# tim = np.arange(0,40.96,0.01)

# os.chdir("./soturon_tex/figure")

# fig,ax = plt.subplots()
# ax.plot(tim,wave1,color="red",linewidth=0.8)
# ax.set_xlabel(r"$t$[s]",fontsize = 10)
# ax.set_ylabel(r"acceleration[cm/s$^2$]",fontsize = 10)
# ax.tick_params(direction="inout",labelsize = 10,length=10)
# fig.tight_layout()
# fig.savefig("kobe_wave.png")
# fig.savefig("kobe_wave.eps")

# wave_file = "coherence/data/1995JR_Takatori.acc"
# wave1 = np.loadtxt(wave_file,usecols=(2),unpack=True)
# tim = np.arange(0,40.96,0.01)
# zeros = np.zeros(4096)
# wave1_array = np.append(zeros,wave1)
# wave1_array = np.append(wave1_array,zeros)

os.chdir("./kansaishibu/value")

tau = np.arange(-40.96,40.96,0.01)
tim = np.arange(0,122.88,0.01)

wave1_array = np.load("wave1_shift_noise.npy")
wave2_list = np.load("wave2_shift_noise.npy")
soft_list = np.load("soft_shift.npy")
sem_list = np.load("sem_shift_noise.npy")
os.chdir("../figure")


graph_wave = plot(tau,soft_list,"","",figsize=(10,10),fontsize=40,labelsize=10,linewidth=3.0,
            color="limegreen",xlim=[-40.96,40.96],ylim=[0,3],params_length=10.0,xlabel="shift[s]",ylabel="wasserstein metric",title="",name="shift_soft")
graph_wave.plot_graph()



