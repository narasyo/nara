import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import similarity


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

def plot_coherence(shift,similarity,freq,window) :

    freq = round(freq,1)
    window = round(window,1)

    fig,ax = plt.subplots()
    ax.plot(shift, similarity, color = "green", linewidth=1.0)
    ax.set_xlabel("shift")
    ax.set_ylabel("coherence")
    ax.set_title(f"coherence({freq}[Hz])_smoothing_width={window}[Hz]")
    ax.tick_params(direction="inout",labelsize = 10,length=5)
    fig.tight_layout()
    fig.savefig(f"kobe_shift_coherence_{freq}[Hz]_window={window}[Hz].png")
    fig.savefig(f"kobe_shift_coherence_{freq}[Hz]_window={window}[Hz].eps")
    plt.close()
    plt.clf()

def plot_coherence_soturon(shift,similarity1,similarity2,similarity3,similarity4,similarity5,similarity6,freq1,freq2,freq3,freq4,freq5,freq6,window,name) :

    freq1 = round(freq1,1)
    freq2 = round(freq2,1)
    freq3 = round(freq3,1)
    freq4 = round(freq4,1)
    freq5 = round(freq5,1)
    freq6 = round(freq6,1)

    window = round(window,1)

    fig = plt.figure()

    ax1 = fig.add_subplot(3,2,1)
    ax1.plot(shift, similarity1, color = "green", linewidth=1.0)
    ax1.set_xlabel("shift[s]")
    ax1.set_ylabel("coherence")
    ax1.set_title(f"coherence({freq1}[Hz])",fontsize=10)
    ax1.set_xlim(-40,40)
    ax1.set_ylim(0,1)
    ax1.tick_params(direction="inout",labelsize = 10,length=5)

    ax2 = fig.add_subplot(3,2,2)
    ax2.plot(shift,similarity2,color = "green", linewidth=1.0)
    ax2.set_xlabel("shift[s]")
    ax2.set_ylabel("coherence")
    ax2.set_title(f"coherence({freq2}[Hz])",fontsize=10)
    ax2.set_xlim(-40,40)
    ax2.set_ylim(0,1)
    ax2.tick_params(direction="inout",labelsize = 10,length=5)

    ax3 = fig.add_subplot(3,2,3)
    ax3.plot(shift,similarity3,color = "green", linewidth=1.0)
    ax3.set_xlabel("shift[s]")
    ax3.set_ylabel("coherence")
    ax3.set_title(f"coherence({freq3}[Hz])",fontsize=10)
    ax3.set_xlim(-40,40)
    ax3.set_ylim(0,1)
    ax3.tick_params(direction="inout",labelsize = 10,length=5)

    ax4 = fig.add_subplot(3,2,4)
    ax4.plot(shift,similarity4,color = "green", linewidth=1.0)
    ax4.set_xlabel("shift[s]")
    ax4.set_ylabel("coherence")
    ax4.set_title(f"coherence({freq4}[Hz])",fontsize=10)
    ax4.set_xlim(-40,40)
    ax4.set_ylim(0,1)
    ax4.tick_params(direction="inout",labelsize = 10,length=5)

    ax5 = fig.add_subplot(3,2,5)
    ax5.plot(shift,similarity5,color = "green", linewidth=1.0)
    ax5.set_xlabel("shift[s]")
    ax5.set_ylabel("coherence")
    ax5.set_title(f"coherence({freq5}[Hz])",fontsize=10)
    ax5.set_xlim(-40,40)
    ax5.set_ylim(0,1)
    ax5.tick_params(direction="inout",labelsize = 10,length=5)

    ax6 = fig.add_subplot(3,2,6)
    ax6.plot(shift,similarity6,color = "green", linewidth=1.0)
    ax6.set_xlabel("shift[s]")
    ax6.set_ylabel("coherence")
    ax6.set_title(f"coherence({freq6}[Hz])",fontsize=10)
    ax6.set_xlim(-40,40)
    ax6.set_ylim(0,1)
    ax6.tick_params(direction="inout",labelsize = 10,length=5)
    # show plots
    fig.tight_layout()
    fig.savefig(f"{name}.png")
    fig.savefig(f"{name}.eps")

def plot_coherence_smoothing(shift,similarity1,similarity2,similarity3,similarity4,similarity5,similarity6,window1,window2,window3,window4,window5,window6,name) :

    fig = plt.figure()

    ax1 = fig.add_subplot(3,2,1)
    ax1.plot(shift, similarity1, color = "green", linewidth=1.0)
    ax1.set_xlabel("shift[s]")
    ax1.set_ylabel("coherence")
    ax1.set_title(f"smoothing width {window1}[Hz]",fontsize=10)
    ax1.set_xlim(-40,40)
    ax1.set_ylim(0,1)
    ax1.tick_params(direction="inout",labelsize = 10,length=5)

    ax2 = fig.add_subplot(3,2,2)
    ax2.plot(shift,similarity2,color = "green", linewidth=1.0)
    ax2.set_xlabel("shift[s]")
    ax2.set_ylabel("coherence")
    ax2.set_title(f"smoothing width {window2}[Hz]",fontsize=10)
    ax2.set_xlim(-40,40)
    ax2.set_ylim(0,1)
    ax2.tick_params(direction="inout",labelsize = 10,length=5)

    ax3 = fig.add_subplot(3,2,3)
    ax3.plot(shift,similarity3,color = "green", linewidth=1.0)
    ax3.set_xlabel("shift[s]")
    ax3.set_ylabel("coherence")
    ax3.set_title(f"smoothing width {window3}[Hz]",fontsize=10)
    ax3.set_xlim(-40,40)
    ax3.set_ylim(0,1)
    ax3.tick_params(direction="inout",labelsize = 10,length=5)

    ax4 = fig.add_subplot(3,2,4)
    ax4.plot(shift,similarity4,color = "green", linewidth=1.0)
    ax4.set_xlabel("shift[s]")
    ax4.set_ylabel("coherence")
    ax4.set_title(f"smoothing width {window4}[Hz]",fontsize=10)
    ax4.set_xlim(-40,40)
    ax4.set_ylim(0,1)
    ax4.tick_params(direction="inout",labelsize = 10,length=5)

    ax5 = fig.add_subplot(3,2,5)
    ax5.plot(shift,similarity5,color = "green", linewidth=1.0)
    ax5.set_xlabel("shift[s]")
    ax5.set_ylabel("coherence")
    ax5.set_title(f"smoothing width {window5}[Hz]",fontsize=10)
    ax5.set_xlim(-40,40)
    ax5.set_ylim(0,1)
    ax5.tick_params(direction="inout",labelsize = 10,length=5)

    ax6 = fig.add_subplot(3,2,6)
    ax6.plot(shift,similarity6,color = "green", linewidth=1.0)
    ax6.set_xlabel("shift[s]")
    ax6.set_ylabel("coherence")
    ax6.set_title(f"smoothing width {window6}[Hz]",fontsize=10)
    ax6.set_xlim(-40,40)
    ax6.set_ylim(0,1)
    ax6.tick_params(direction="inout",labelsize = 10,length=5)
    # show plots
    fig.tight_layout()
    fig.savefig(f"{name}.png")
    fig.savefig(f"{name}.eps")

######################
#  拡大・縮小(全体)  #
######################

"""

wave_file = "coherence/data/1995JR_Takatori.acc"
wave1 = np.loadtxt(wave_file,usecols=(2),unpack=True)
tim = np.arange(0,40.96,0.01)

alpha_list = np.arange(0,2,0.01)

os.chdir("./soturon_tex/figure")
window = [0.5]
c_list,f_list = [],[]
for w in tqdm(window) :
    freq_list,coh_list = [],[]
    for alpha in tqdm(alpha_list):

        wave2 = alpha*wave1

        sim = similarity.similarity(tim,wave1,wave2)
        freq,coh = sim.coherence(w)
        freq_list += [freq]
        coh_list += [coh]


    coh_list = np.array(coh_list)
    freq_list = np.array(freq_list)
    number = [10,20,30,40,51,62]
    for n in number :
        c_list += [coh_list[:,n]]
        f_list += [freq_list[1,n]]

plot_coherence_soturon(alpha_list,c_list[0],c_list[1],c_list[2],c_list[3],c_list[4],c_list[5],f_list[0],f_list[1],f_list[2],f_list[3],f_list[4],f_list[5],w,"mag_coherence")

"""

######################
#  拡大・縮小(一部)  #
######################

"""

wave_file = "coherence/data/1995JR_Takatori.acc"
wave1 = np.loadtxt(wave_file,usecols=(2),unpack=True)
tim = np.arange(0,40.96,0.01)

beta_list = np.arange(0,2,0.01)
os.chdir("./soturon_tex/figure")
window = [0.5]
c_list,f_list = [],[]
for w in tqdm(window) :
    freq_list,coh_list = [],[]
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

        sim = similarity.similarity(tim,wave1,wave2)

        freq,coh = sim.coherence(w)
        freq_list += [freq]
        coh_list += [coh]

    coh_list = np.array(coh_list)
    freq_list = np.array(freq_list)
    number = [10,20,30,40,51,62]
    for n in number :
        c_list += [coh_list[:,n]]
        f_list += [freq_list[1,n]]

plot_coherence_soturon(beta_list,c_list[0],c_list[1],c_list[2],c_list[3],c_list[4],c_list[5],f_list[0],f_list[1],f_list[2],f_list[3],f_list[4],f_list[5],w,"partmag_coherence")

"""

###########################
#  時間シフト(ノイズあり) #
###########################


dlen = 4096 #ノイズデータのデータ長
mean = 0.0  #ノイズの平均値
std  = 500  #ノイズの分散

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

os.chdir("./soturon_tex/figure")

zeros = np.zeros(4096)
wave1_array = np.append(zeros,wave1_noise)
wave1_array = np.append(wave1_array,zeros)
wave2 = wave1 + noise2_array
tau = np.arange(-40.96,40.96,0.01)

window = [0.5]
c_list,f_list = [],[]
for w in tqdm(window) :

    freq_list,coh_list = [],[]
    for i,s in zip(range(8192),tau) :

        shift_number = int(s*100)
        zeros1_shift = np.zeros(4096+shift_number)
        zeros2_shift = np.zeros(4096-shift_number)
        wave2_array = np.array(wave2)
        wave2_array = np.append(zeros1_shift,wave2_array)
        wave2_array = np.append(wave2_array,zeros2_shift)
        
        sim = similarity.similarity(tim,wave1_array,wave2_array)

        freq,coh = sim.coherence(w)
        freq_list += [freq]
        coh_list += [coh]

    coh_list = np.array(coh_list)
    freq_list = np.array(freq_list)
    number = [40,80,120,160,205,245]
    for n in number :
        c_list += [coh_list[:,n]]
        f_list += [freq_list[1,n]]

plot_coherence_soturon(tau, c_list[0],c_list[1],c_list[2],c_list[3],c_list[4],c_list[5],f_list[0],f_list[1],f_list[2],f_list[3],f_list[4],f_list[5],w,"noise_coherence")



###########################
#  時間シフト(ノイズなし) #
###########################

"""

wave_file = "coherence/data/1995JR_Takatori.acc"
wave1 = np.loadtxt(wave_file,usecols=(2),unpack=True)

tim = np.arange(0,122.88,0.01)

os.chdir("./soturon_tex/figure")

zeros = np.zeros(4096)
wave1_array = np.append(zeros,wave1)
wave1_array = np.append(wave1_array,zeros)
wave2 = wave1 
tau = np.arange(-40.96,40.96,0.01)

window = [0.1,0.2,0.3,0.4,0.5,0.6]
c_list,f_list = [],[]
for w in tqdm(window) :

    freq_list,coh_list = [],[]
    for i,s in zip(range(8192),tau) :

        shift_number = int(s*100)
        zeros1_shift = np.zeros(4096+shift_number)
        zeros2_shift = np.zeros(4096-shift_number)
        wave2_array = np.array(wave2)
        wave2_array = np.append(zeros1_shift,wave2_array)
        wave2_array = np.append(wave2_array,zeros2_shift)
        
        sim = similarity.similarity(tim,wave1_array,wave2_array)

        freq,coh = sim.coherence(w)
        freq_list += [freq]
        coh_list += [coh]

    coh_list = np.array(coh_list)
    freq_list = np.array(freq_list)
    number = [120]
    for n in number :
        c_list += [coh_list[:,n]]
        f_list += [freq_list[1,n]]
    
plot_coherence_smoothing(tau,c_list[0],c_list[1],c_list[2],c_list[3],c_list[4],c_list[5],window[0],window[1],window[2],window[3],window[4],window[5],"shift_coherence_smoothing")

print(f_list)

"""