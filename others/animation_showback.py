import matplotlib.pyplot as plt
import numpy as np
import os
import ot
import pywt
from tqdm import tqdm
import matplotlib.animation as animation

def calc_devided(numer,demon) :

    return numer/demon

def normalized_linear(wave1,wave2,sum_time) :

    xmin = 0
    xmax = 0.01*sum_time
    h = (xmax-xmin)/sum_time
    c1 = np.abs(min(wave1))
    c2 = np.abs(min(wave2))
    const = max(c1,c2)
    pos_amp1_list = []
    for uni_amp1 in wave1 :
        pos_amp1 = uni_amp1 + const
        pos_amp1_list += [pos_amp1]
        
    pos_amp2_list = []
    for uni_amp2 in wave2 :
        pos_amp2 = uni_amp2 + const
        pos_amp2_list += [pos_amp2]

    s1 = h*(np.sum(pos_amp1_list)-pos_amp1_list[0]/2-pos_amp1_list[sum_time-1]/2)

    s2 = h*(np.sum(pos_amp2_list)-pos_amp2_list[0]/2-pos_amp2_list[sum_time-1]/2)

    amp1_norm_list = []
    for pos1 in pos_amp1_list:
        amp1_norm_list.append(calc_devided(pos1,s1))

    amp2_norm_list =[]
    for pos2 in pos_amp2_list:
        amp2_norm_list.append(calc_devided(pos2,s2))

    amp1_norm = np.array(amp1_norm_list)
    amp2_norm = np.array(amp2_norm_list)

    return amp1_norm,amp2_norm

def softplus_normalizing(wave1,wave2,sum_time) :

    """[ソフトプラス正規化法]

    Args:
        wave1 ([list]): [地震波形1]
        wave2 ([list]): [地震波形2]
        sum_time ([int]): [合計時間]

    Returns:
        [float]: [確率密度関数1,確率密度関数2]
    """

    xmin = 0
    xmax = 0.01*sum_time
    h = (xmax-xmin)/sum_time
    wave1 = np.array(wave1)
    wave2 = np.array(wave2)

    max_wave1 = max(np.abs(wave1))
    max_wave2 = max(np.abs(wave2))

    b = 3.0/(max(max_wave1,max_wave2))
    w1_pos = np.log(np.exp(wave1*b)+1)
    w2_pos = np.log(np.exp(wave2*b)+1)
    s1 = h*(np.sum(w1_pos)-w1_pos[0]/2-w1_pos[sum_time-1]/2)
    s2 = h*(np.sum(w2_pos)-w2_pos[0]/2-w2_pos[sum_time-1]/2)
    p1 = [w1/s1 for w1 in w1_pos]
    p2 = [w2/s2 for w2 in w2_pos]

    return p1,p2

def ricker(tim,fp,tp,amp):

    t1 = ((tim-tp)*np.pi*fp)**2

    return (2*t1-1)*np.exp(-t1)*amp

def cdf(p) :

    a=0
    P = []
    for ele in p[:10000] :
        a = a + ele
        P += [a]
    P = np.array(P)
    P = P/100
    return(P)

def plot_wave(t,y,color) :

    fig,ax = plt.subplots()
    ax.plot(t,y,color=color)

def plot_probability(t,p,color) :

    fig2,ax2 = plt.subplots()
    ax2.plot(t,p,color=color)
    ax2.axis("off")
    ax2.set_ylim(0,0.10)
    fig2.savefig("wasserstein_wave_pf.png")
    fig2.savefig("wasserstein_wave_pf.eps")

def plot_cdf(t,P,color) :

    fig3,ax3 = plt.subplots()
    ax3.plot(t,P,color=color)
    # ax3.axis("off")
    ax3.set_ylim(0,1)
    # ax3.set_xlim(-2,2)
    fig3.savefig("wasserstein_wave_Pf.png")
    fig3.savefig("wasserstein_wave_Pf.eps")

def plot_cdf_fg(t,P,Q,color1,color2) :

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1,1,1)
    ax4 = fig3.add_subplot(1,1,1)
    ax3.plot(t,P,color=color1)
    # ax3.axis("off")
    ax3.set_ylim(0,1)
    # ax3.set_xlim(-2,2)
    ax4.plot(t,Q,color=color2)
    ax4.set_ylim(0,1)
    fig3.savefig("wasserstein_wave_all.png")
    fig3.savefig("wasserstein_wave_all.eps")

def animation_wave(tim,wave1_list,wave2_list,ylim_min,ylim_max,total_time,frame) :

    fig,ax = plt.subplots(figsize=(20,5))
    ax.set_ylim(ylim_min,ylim_max)
    # ax.axis("off")
    ims = []

    cnt = 0
    k = (len(wave2_list)-1)/frame
    k= int(k)
    for w1,w2 in tqdm(zip(wave1_list,wave2_list)) :
        if cnt%k==0:
            im = ax.plot(tim,w1,color="red",linewidth=1.5)
            im += ax.plot(tim,w2,color="blue",linewidth=1.5)
            ims.append(im)
        cnt+=1
    interval = total_time/((len(wave2_list)-1)/k)*1000
    ani = animation.ArtistAnimation(fig, ims, interval=interval)

    return ani

def animation_similarity(tau, similarity_list, total_time, frame, line_color, dot_color, dot_size) :

    fig,ax = plt.subplots()
    # ax.axis("off")
    ims = []
    k = 1400/frame
    k=int(k)
    for i,s,sim in zip(range(M+1),tau,similarity_list) :
        if i%k==0 :
            im = ax.plot(tau,similarity_list,color=line_color)
            im += ax.plot(s,sim,color=dot_color,marker='.',markersize=dot_size)
            ims.append(im)
    interval = total_time/((len(l21_list)-1)/k)*1000
    ani = animation.ArtistAnimation(fig, ims, interval=interval)

    return ani

def wasserstein_metric(tim,wave1,wave2) :

    normalized_wave1_pos,normalized_wave2_pos = softplus_normalizing(wave1,wave2,len(tim))
    was_pos = ot.emd2_1d(tim,tim,normalized_wave1_pos/sum(normalized_wave1_pos),normalized_wave2_pos/sum(normalized_wave2_pos),metric='minkowski',p=2)
    normalized_wave1_neg,normalized_wave2_neg = softplus_normalizing(-wave1,-wave2,len(tim))
    was_neg = ot.emd2_1d(tim,tim,normalized_wave1_neg/sum(normalized_wave1_neg),normalized_wave2_neg/sum(normalized_wave2_neg),metric='minkowski',p=2)
    was = was_pos + was_neg

    return was

def wasserstein_metric_linear(tim,wave1,wave2) :

    normalized_wave1_pos,normalized_wave2_pos = normalized_linear(wave1,wave2,len(tim))
    was_pos = ot.emd2_1d(tim,tim,normalized_wave1_pos/sum(normalized_wave1_pos),normalized_wave2_pos/sum(normalized_wave2_pos),metric='minkowski',p=2)

    return was_pos


os.chdir("./soturon_happyou")

N = 2000 # 波形のプロット数
M = 1400 # 時間シフトのコマ数
xmin = -10
xmax = 10
tim = np.linspace(xmin,xmax,N+1)
tau = np.linspace(-7,7,M+1)
mean = 0.0  #ノイズの平均値
std1 = 20  #ノイズの分散
std2 = 40  #ノイズの分散

noise1 = np.random.normal(mean,std1,200)
noise2 = np.random.normal(mean,std2,200)
noise3 = np.random.normal(mean,std1,200)
noise4 = np.random.normal(mean,std2,200)

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

w1_list,l21_list,f1t_list,g1t_list,f2t_list,g2t_list,w2_list,l22_list,f3t_list,l23_list,w3_list = [],[],[],[],[],[],[],[],[],[],[]
for s in tqdm(tau) :

    shift_number = int(s*100)
    zeros1_shift = np.zeros(700+shift_number)
    zeros2_shift = np.zeros(701-shift_number)
    noise_array_shift = np.append(zeros1_shift,noise3)
    noise_array_shift = np.append(noise_array_shift,noise4)
    noise_array_shift = np.append(noise_array_shift,noise3)
    noise_array_shift = np.append(noise_array_shift,zeros2_shift)

    window_array_shift = np.append(zeros1_shift,tilt_pos)
    window_array_shift = np.append(window_array_shift,ones)
    window_array_shift = np.append(window_array_shift,tilt_neg)
    window_array_shift = np.append(window_array_shift,zeros2_shift)

    noise_array_shift = noise_array_shift*window_array_shift

    f1t,g1t = [],[]
    for t in tim :
        f1 = ricker(t,0.4,-0.5,20)
        f2 = -ricker(t,0.6,-0.25,25)
        f3 = ricker(t,0.8,0,40)
        f4 = -ricker(t,0.6,0.25,25)
        f5 = ricker(t,0.4,0.5,20)
        f = f1 + f2 + f3 + f4 + f5

        g1 = ricker(t,0.4,-0.5+s,20)
        g2 = -ricker(t,0.6,-0.25+s,25)
        g3 = ricker(t,0.8,s,40)
        g4 = -ricker(t,0.6,0.25+s,25)
        g5 = ricker(t,0.4,0.5+s,20)
        g = g1 + g2 + g3 + g4 + g5

        g1t += [g]
        f1t += [f]

    f1t = np.array(f1t)
    g1t = np.array(g1t)
    f2t = f1t + noise_array
    g2t = g1t + noise_array_shift
    f3t = 5*f1t

    was1 = wasserstein_metric_linear(tim,f1t,g1t)
    was2 = wasserstein_metric_linear(tim,f2t,g2t)
    was3 = wasserstein_metric_linear(tim,f3t,g1t)

    l21 = np.mean((f1t-g1t)**2)
    l22 = np.mean((f2t-g2t)**2)
    l23 = np.mean((f3t-g1t)**2)

    w1_list += [was1]
    w2_list += [was2]
    l21_list += [l21]
    l22_list += [l22]
    f1t_list += [f1t]
    g1t_list += [g1t]
    f2t_list += [f2t]
    g2t_list += [g2t]
    f3t_list += [f3t]
    w3_list += [was3]
    l23_list += [l23]

# plt.plot(tau,w2_list)
# plt.show()

# plt.plot(tau,l22_list)
# plt.show()

ani1 = animation_wave(tim, f1t_list, g1t_list, ylim_min=-40, ylim_max=30, total_time=5,frame=70) #frameは1400の約数
ani1.save('ani1.gif', writer="imagemagick")

# ani2 = animation_wave(tim, f2t_list, g2t_list, ylim_min=-120, ylim_max=110, total_time=5,frame=70) #frameは1400の約数
# ani2.save('ani2.gif', writer="imagemagick")

# ani3 = animation_similarity(tau,l21_list, total_time=5, frame=70, line_color="black", dot_color="red", dot_size=20) #frameは1400の約数
# ani3.save('ani3.gif', writer="imagemagick")

# ani4 = animation_similarity(tau,w1_list, total_time=5, frame=70, line_color="darkorange", dot_color="red", dot_size=20) #frameは1400の約数
# ani4.save('ani4.gif', writer="imagemagick")

# ani5 = animation_similarity(tau,l22_list, total_time=5, frame=70, line_color="black", dot_color="red", dot_size=20) #frameは1400の約数
# ani5.save('ani5.gif', writer="imagemagick")

# ani6 = animation_similarity(tau,w2_list, total_time=5, frame=70, line_color="darkorange", dot_color="red", dot_size=20) #frameは1400の約数
# ani6.save('ani6.gif', writer="imagemagick")

ani7 = animation_similarity(tau,w3_list, total_time=5, frame=70, line_color="darkorange", dot_color="red", dot_size=20) #frameは1400の約数
ani7.save('ani7.gif', writer="imagemagick")

ani8 = animation_similarity(tau,l23_list, total_time=5, frame=70, line_color="black", dot_color="red", dot_size=20) #frameは1400の約数
ani8.save('ani8.gif', writer="imagemagick")

ani9 = animation_wave(tim, f3t_list, g1t_list, ylim_min=-200, ylim_max=200, total_time=5,frame=70) #frameは1400の約数
ani9.save('ani9.gif', writer="imagemagick")