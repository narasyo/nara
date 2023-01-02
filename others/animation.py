import matplotlib.pyplot as plt
import numpy as np
import os
import ot
import pywt
from tqdm import tqdm
import matplotlib.animation as animation

def softplus_normalizing(wave1,wave2,sum_time) :

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

def animation_wave(tim,wave1,wave2,ylim_min,ylim_max) :

    fig,ax = plt.subplots()
    ims = []

    for w1,w2 in tqdm(zip(wave1,wave2)) :
        im = ax.plot(tim,wave1,color="red",linewidth=0.2)
        ax.set_ylim(ylim_min,ylim_max)
        ax.axis("off")
        im = ax.plot(tim,wave2,color="blue",linewidth=0.5)
        ims.append(im)

    ani1 = animation.ArtistAnimation(fig1, ims1, interval=100)


os.chdir("./soturon_happyou")

N = 10**4
M = 10**2
xmin = -10
xmax = 10
h = (xmax-xmin)/N
tim = np.linspace(xmin,xmax,N)
tau = np.linspace(-7,7,M)


w_list,l2_list,ft_list,gt_list = [],[],[],[]
for s in tqdm(tau) :
    ft,gt = [],[]
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

        gt += [g]
        ft += [f]

    ft = np.array(ft)
    gt = np.array(gt)

    pf,pg = softplus_normalizing(ft,gt,len(tim))

    l2 = np.mean((ft-gt)**2)
    w = ot.emd2_1d(tim,tim,pf/sum(pf),pg/sum(pg),metric='minkowski',p=2)

    w_list += [w]
    l2_list += [l2]
    ft_list += [ft]
    gt_list += [gt]

fig1,ax1 = plt.subplots()
ims1 = []

for ft_,gt_ in tqdm(zip(ft_list,gt_list)) :
    im1 = ax1.plot(tim,ft_list[1],color="red",linewidth=0.2)
    ax1.set_ylim(-20,25)
    ax1.axis("off")
    im1 = ax1.plot(tim,gt_,color="blue",linewidth=0.5)
    ims1.append(im1)

ani1 = animation.ArtistAnimation(fig1, ims1, interval=100)
ani1.save('ani1.gif', writer="imagemagick")






