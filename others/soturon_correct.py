import numpy as np
import ot
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import similarity
import PySGM
from scipy import signal

def ricker(tim,fp,tp,amp):

    t1 = ((tim-tp)*np.pi*fp)**2
    return (2*t1-1)*np.exp(-t1)*amp

def basic_wave(tim) :

    wave = []
    for t in tim :
        y1 = ricker(t,0.4,-0.5,20)
        y2 = -ricker(t,0.6,-0.25,25)
        y3 = ricker(t,0.8,0,40)
        y4 = -ricker(t,0.6,0.25,25)
        y5 = ricker(t,0.4,0.5,20)
        y = y1 + y2 + y3 + y4 + y5
        wave += [y]
    
    return wave

def make_noise() :

    mean = 0.0  #ノイズの平均値
    std1 = 20  #ノイズの分散
    std2 = 40  #ノイズの分散
    
    noise1 = np.random.normal(mean,std1,200)
    noise2 = np.random.normal(mean,std2,200)

    zeros1 = np.zeros(700)
    zeros2 = np.zeros(701)
    noise_array = np.append(zeros1,noise1)
    noise_array = np.append(noise_array,noise2)
    noise_array = np.append(noise_array,noise1)
    noise_array = np.append(noise_array,zeros2)

    tilt_pos = np.linspace(0,1,100)
    tilt_neg = np.linspace(1,0,100)
    ones = np.ones(400)

    window_array = np.append(zeros1,tilt_pos)
    window_array = np.append(window_array,ones)
    window_array = np.append(window_array,tilt_neg)
    window_array = np.append(window_array,zeros2)

    noise_array = noise_array*window_array

def plot_wave(time_list,wave1,wave2,name) :

    point = {
    "start":[0,-20],
    "end":[0,-40]
    }

    fig1,ax1 = plt.subplots(figsize=(30,10))
    ax1.plot(time_list,wave1,color="blue",linewidth=2.0)
    ax1.plot(time_list,wave2,color="red",linewidth=2.0)
    ax1.set_xlabel(r"$t$[s]",fontsize = 20)
    ax1.set_ylabel(r"$y$",fontsize = 20)
    ax1.set_xlim(-10,10)
    ax1.set_ylim(-50,50)
    ax1.tick_params(direction="inout",labelsize = 20,length=10)
    ax1.legend([r"$f(t)$",r"$\alpha*f(t)$"], loc="upper right", fontsize=30)
    ax1.annotate("", xy=point["end"], xytext=point["start"],arrowprops=dict(shrink=0, width=1, headwidth=15, headlength=10, connectionstyle='arc3',facecolor='gray', edgecolor='gray'))
    fig1.tight_layout()
    fig1.savefig(f"{name}.png")
    fig1.savefig(f"{name}.eps")

def plot_similarity(shift,mse,was_linear,was_exp,was_soft,name) :

    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(shift, mse, color = "green", linewidth=1.0)
    ax1.set_xlabel("shift[s]")
    ax1.set_ylabel("MSE")
    ax1.set_title("MSE")
    ax1.tick_params(direction="inout",labelsize = 10,length=5)

    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(shift,was_linear,color = "green", linewidth=1.0)
    ax2.set_xlabel("shift[s]")
    ax2.set_ylabel(r"$W_{2}^2$")
    ax2.set_title(r"$W_{2}^2$(linear normalizing)")
    ax2.tick_params(direction="inout",labelsize = 10,length=5)

    ax3 = fig.add_subplot(2,2,3)
    ax3.plot(shift,was_exp,color = "green", linewidth=1.0)
    ax3.set_xlabel("shift[s]")
    ax3.set_ylabel(r"$W_{2}^2$")
    ax3.set_title(r"$W_{2}^2$(exponential normalizing)")
    ax3.tick_params(direction="inout",labelsize = 10,length=5)

    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(shift,was_soft,color = "green", linewidth=1.0)
    ax4.set_xlabel("shift[s]")
    ax4.set_ylabel(r"$W_{2}^2$")
    ax4.set_title(r"$W_{2}^2$(softplus normalizing)")
    ax4.tick_params(direction="inout",labelsize = 10,length=5)
    # show plots
    fig.tight_layout()
    fig.savefig(f"{name}.png")
    fig.savefig(f"{name}.eps")

os.chdir("./soturon_tex/figure_correct")

N = 2000 # 波形のプロット数
M = 1400 # 時間シフトのコマ数
xmin = -10
xmax = 10
tim = np.linspace(xmin,xmax,N+1)
tau = np.linspace(-7,7,M+1)

wave1 = basic_wave(tim)
wave2 = wave1
wave1_array = np.array(wave1)
del wave2[0:700]
del wave2[600:1301]
wave2_list,mse_list,linear_list,exp_list,soft_list = [],[],[],[],[]
for i,s in tqdm(zip(range(2001),tau)) :

    shift_number = int(s*100)
    zeros1_shift = np.zeros(700+shift_number)
    zeros2_shift = np.zeros(701-shift_number)
    wave2_array = np.array(wave2)
    wave2_array = np.append(zeros1_shift,wave2_array)
    wave2_array = np.append(wave2_array,zeros2_shift)
    
    similarity1 = similarity.similarity(tim,wave1_array,wave2_array)
    was_linear = similarity1.wasserstein_linear_normalizing()
    was_exp = similarity1.wasserstein_exponential_normalizing()
    was_soft = similarity1.wasserstein_softplus_normalizing()
    mse = similarity1.mse()

    wave2_list += [wave2_array]
    mse_list += [mse]
    linear_list += [was_linear]
    exp_list += [was_exp]
    soft_list += [was_soft]

# plot_wave(tim,wave1_array,wave2_list[0],"shift")
# plot_similarity(tau,mse_list,linear_list,exp_list,soft_list,"shift_compare_mse_w2")



"""
wave1 = basic_wave(tim)
wave1_array = np.array(wave1)
wave2 = wave1

magnitude = np.arange(0,2,0.01)
wave2_list,mse_list,linear_list,exp_list,soft_list = [],[],[],[],[]
for mag in tqdm(magnitude) :
    wave2_array = mag * wave1_array
    similarity1 = similarity.similarity(tim,wave1_array,wave2_array)
    was_linear = similarity1.wasserstein_linear_normalizing()
    was_exp = similarity1.wasserstein_exponential_normalizing()
    was_soft = similarity1.wasserstein_softplus_normalizing()
    mse = similarity1.mse()

    wave2_list += [wave2_array]
    mse_list += [mse]
    linear_list += [was_linear]
    exp_list += [was_exp]
    soft_list += [was_soft]

plot_wave(tim,wave1_array,wave2_list[199],"magnitude")
plot_similarity(magnitude,mse_list,linear_list,exp_list,soft_list,"magnitude_compare_mse_w2")
    
"""

"""
wave1 = basic_wave(tim)
wave1_array = np.array(wave1)
wave2 = wave1

magnitude = np.arange(0,2,0.01)
wave2_list,mse_list,linear_list,exp_list,soft_list = [],[],[],[],[]
for i,mag in tqdm(zip(range(200),magnitude)) :
    if 110 <= i <= 120 :
        wave2_array = mag * wave1_array
    similarity1 = similarity.similarity(tim,wave1_array,wave2_array)
    was_linear = similarity1.wasserstein_linear_normalizing()
    was_exp = similarity1.wasserstein_exponential_normalizing()
    was_soft = similarity1.wasserstein_softplus_normalizing()
    mse = similarity1.mse()

    wave2_list += [wave2_array]
    mse_list += [mse]
    linear_list += [was_linear]
    exp_list += [was_exp]
    soft_list += [was_soft]

plot_wave(tim,wave1_array,wave2_list[199],"magnitude")
plot_similarity(magnitude,mse_list,linear_list,exp_list,soft_list,"magnitude_compare_mse_w2")

"""

# wave1 = basic_wave(tim)
# wave1 = np.array(wave1)   
# freq, P = signal.periodogram(wave1,100)

# plt.plot(freq,P)
# plt.xlim(0,5)
# plt.show()

# wave = similarity.similarity(tim,wave1)
