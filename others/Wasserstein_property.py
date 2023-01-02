import numpy as np
import ot
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from PySGM.vector import vector

def ricker(tim,fp,tp,amp):

    t1 = ((tim-tp)*np.pi*fp)**2
    return (2*t1-1)*np.exp(-t1)*amp

def linear_normalizing(w1,w2,h,N) :

    w1_min = min(w1)
    w2_min = min(w2)
    c = max(abs(w1_min),abs(w2_min))
    w1_pos = w1+c
    w2_pos = w2+c
    s1 = h*(np.sum(w1_pos)-w1_pos[0]/2-w1_pos[N]/2)
    s2 = h*(np.sum(w2_pos)-w2_pos[0]/2-w2_pos[N]/2)
    p1 = [wave1/s1 for wave1 in w1_pos]
    p2 = [wave2/s2 for wave2 in w2_pos]
    return p1,p2

def softplus_normalizing(wave1,wave2,h,N) :

    b = 3.0/(max(abs(wave1)))

    pos_amp1_list = []
    for uni_amp1 in wave1 :
        if b*uni_amp1 > 0 :
            pos_amp1 = b*uni_amp1+np.log(np.exp(-uni_amp1*b)+1)
            pos_amp1_list += [pos_amp1]
        else :
            pos_amp1 = np.log(np.exp(uni_amp1*b)+1)
            pos_amp1_list += [pos_amp1]

    pos_amp2_list = []
    for uni_amp2 in wave2 :
        if b*uni_amp2 > 0 :
            pos_amp2 = b*uni_amp2+np.log(np.exp(-uni_amp2*b)+1)
            pos_amp2_list += [pos_amp2]
        else :
            pos_amp2 = np.log(np.exp(uni_amp2*b)+1)
            pos_amp2_list += [pos_amp2]
    
    w1_pos = np.array(pos_amp1_list)
    w2_pos = np.array(pos_amp2_list)

    s1 = h*(np.sum(w1_pos)-w1_pos[0]/2-w1_pos[N-1]/2)
    s2 = h*(np.sum(w2_pos)-w2_pos[0]/2-w2_pos[N-1]/2)
    p1 = [w1/s1 for w1 in w1_pos]
    p2 = [w2/s2 for w2 in w2_pos]

    return p1,p2

def exponential_normalizing(wave1,wave2,h,N) :

    b = 3.0/(max(abs(wave1)))

    pos_amp1_list = []
    for uni_amp1 in wave1 :
        pos_amp1 = np.exp(b*uni_amp1)
        pos_amp1_list += [pos_amp1]

    pos_amp2_list = []
    for uni_amp2 in wave2 :
        pos_amp2 = np.exp(b*uni_amp2)
        pos_amp2_list += [pos_amp2]

    w1_pos = np.array(pos_amp1_list)
    w2_pos = np.array(pos_amp2_list)

    s1 = h*(np.sum(w1_pos)-w1_pos[0]/2-w1_pos[N-1]/2)
    s2 = h*(np.sum(w2_pos)-w2_pos[0]/2-w2_pos[N-1]/2)
    p1 = [w1/s1 for w1 in w1_pos]
    p2 = [w2/s2 for w2 in w2_pos]

    return p1,p2

# os.mkdir("./soturon_tex/figure_correct")
os.chdir("./soturon_tex/figure_correct")
N = (10**3)*2
xmin = -10
xmax = 10
h = (xmax-xmin)/N
time_list = np.linspace(xmin,xmax,N+1)
wave1 = -ricker(time_list,0.2,0.0,50) 
# wave2 = -2*ricker(time_list,0.2,0.0,50)

##########
# fig2-3 #
##########

"""
point = {
    "start":[-15,20],
    "end":[15,20]
}

fig1,ax1 = plt.subplots()
ax1.plot(time_list,wave1,color="blue")
ax1.plot(time_list,wave2,color="red")
ax1.set_xlabel("t[s]")
ax1.set_ylabel("y")
ax1.set_xlim(-20,20)
ax1.legend([r"$f(t)$",r"$f(t-s)$"], loc="upper right", fontsize=12)
ax1.annotate("", xy=point["end"], xytext=point["start"],arrowprops=dict(shrink=0, width=1, headwidth=8, headlength=10, connectionstyle='arc3',facecolor='gray', edgecolor='gray'))
fig1.savefig("fig2-3.png")
fig1.savefig("fig2-3.eps")

"""

##########
# fig2-5 #
##########

"""

point1 = {"start":[-2,-22],"end":[-2,-40]}
point2 = {"start":[0,50],"end":[0,90]}
point3 = {"start":[2,-22],"end":[2,-40]}

fig1,ax1 = plt.subplots()
ax1.plot(time_list,wave1,color="blue")
ax1.plot(time_list,wave2,color="red")
ax1.set_xlabel("t[s]")
ax1.set_ylabel("y")
ax1.set_xlim(-10,10)
ax1.legend([r"$f(t)$",r"$\alpha*f(t)$"], loc="upper right", fontsize=12)
ax1.text(0.5,75,r"$\times\alpha$",fontsize=12)
ax1.text(2.5,-30,r"$\times\alpha$",fontsize=12)
ax1.text(-3.5,-30,r"$\times\alpha$",fontsize=12)
ax1.annotate("", xy=point1["end"], xytext=point1["start"],arrowprops=dict(shrink=0, width=1, headwidth=8, headlength=10, connectionstyle='arc3',facecolor='gray', edgecolor='gray'))
ax1.annotate("", xy=point2["end"], xytext=point2["start"],arrowprops=dict(shrink=0, width=1, headwidth=8, headlength=10, connectionstyle='arc3',facecolor='gray', edgecolor='gray'))
ax1.annotate("", xy=point3["end"], xytext=point3["start"],arrowprops=dict(shrink=0, width=1, headwidth=8, headlength=10, connectionstyle='arc3',facecolor='gray', edgecolor='gray'))
fig1.savefig("fig2-5.png")
fig1.savefig("fig2-5.eps")

"""


##########
# fig2-4 #
##########

"""

tau_list = np.linspace(-15.0,15.0,1001)

w_linear_list,w_soft_list,w_exp_list,l2_list = [],[],[],[]
for tau in tqdm(tau_list):

    wave2 = -ricker(time_list,0.2,tau,50)

    l2 = np.mean((wave1-wave2)**2)
    l2_list += [l2]

    prob1_soft,prob2_soft = softplus_normalizing(wave1,wave2,h,N)
    w_soft_pos = ot.emd2_1d(time_list,time_list,prob1_soft/sum(prob1_soft),prob2_soft/sum(prob2_soft),metric='minkowski',p=2)
    prob1_soft,prob2_soft = softplus_normalizing(-wave1,-wave2,h,N)
    w_soft_neg = ot.emd2_1d(time_list,time_list,prob1_soft/sum(prob1_soft),prob2_soft/sum(prob2_soft),metric='minkowski',p=2)
    w_soft = w_soft_pos + w_soft_neg
    w_soft_list += [w_soft]
    
    prob1_exp,prob2_exp = exponential_normalizing(wave1,wave2,h,N)
    w_exp_pos = ot.emd2_1d(time_list,time_list,prob1_exp/sum(prob1_exp),prob2_exp/sum(prob2_exp),metric='minkowski',p=2)
    prob1_exp,prob2_exp = exponential_normalizing(-wave1,-wave2,h,N)
    w_exp_neg = ot.emd2_1d(time_list,time_list,prob1_exp/sum(prob1_exp),prob2_exp/sum(prob2_exp),metric='minkowski',p=2)
    w_exp = w_exp_pos + w_exp_neg
    w_exp_list += [w_exp]

    prob1,prob2 = linear_normalizing(wave1,wave2,h,N)
    w = ot.emd2_1d(time_list,time_list,prob1/sum(prob1),prob2/sum(prob2),metric='minkowski',p=2)
    w_linear_list += [w]

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.plot(tau_list,l2_list, color = "green", linewidth=1.0)
ax1.set_xlabel("shift[s]")
ax1.set_ylabel("MSE")
ax1.set_title("MSE")

ax2 = fig.add_subplot(2,2,2)
ax2.plot(tau_list,w_linear_list,color = "green", linewidth=1.0)
ax2.set_xlabel("shift[s]")
ax2.set_ylabel(r"$W_{2}^2$")
ax2.set_title(r"$W_{2}^2$(linear normalizing)")

ax3 = fig.add_subplot(2,2,3)
ax3.plot(tau_list,w_exp_list,color = "green", linewidth=1.0)
ax3.set_xlabel("shift[s]")
ax3.set_ylabel(r"$W_{2}^2$")
ax3.set_title(r"$W_{2}^2$(exponential normalizing)")

ax4 = fig.add_subplot(2,2,4)
ax4.plot(tau_list,w_soft_list,color = "green", linewidth=1.0)
ax4.set_xlabel("shift[s]")
ax4.set_ylabel(r"$W_{2}^2$")
ax4.set_title(r"$W_{2}^2$(softplus normalizing)")

# show plots
fig.tight_layout()
fig.savefig("fig2-4.png")
fig.savefig("fig2-4.eps")

"""

##########
# fig2-6 #
##########

"""

alpha_list = np.linspace(0,2,1001)

w_soft_list,l2_list = [],[]
for alpha in tqdm(alpha_list):

    wave2 = -alpha*ricker(time_list,0.2,0.0,50)

    l2 = np.mean((wave1-wave2)**2)
    l2_list += [l2]

    prob1_soft,prob2_soft = softplus_normalizing(wave1,wave2,h,N)
    w_soft_pos = ot.emd2_1d(time_list,time_list,prob1_soft/sum(prob1_soft),prob2_soft/sum(prob2_soft),metric='minkowski',p=2)
    prob1_soft,prob2_soft = softplus_normalizing(-wave1,-wave2,h,N)
    w_soft_neg = ot.emd2_1d(time_list,time_list,prob1_soft/sum(prob1_soft),prob2_soft/sum(prob2_soft),metric='minkowski',p=2)
    w_soft = w_soft_pos + w_soft_neg
    w_soft_list += [w_soft]

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.plot(alpha_list,l2_list, color = "green", linewidth=1.0)
ax1.set_xlabel(r"$\alpha$(magnification)")
ax1.set_ylabel("MSE")
ax1.set_title("MSE")

ax4 = fig.add_subplot(1,2,2)
ax4.plot(alpha_list,w_soft_list,color = "green", linewidth=1.0)
ax4.set_xlabel(r"$\alpha$(magnification)")
ax4.set_ylabel(r"$W_{2}^2$")
ax4.set_title(r"$W_{2}^2$(softplus normalizing)")

# show plots
fig.tight_layout()
fig.savefig("fig2-6.png")
fig.savefig("fig2-6.eps")

"""
##########
# fig2-7 #
##########

"""

wave2 = []
for w1,i in zip(wave1,range(10001)) :
    if i <= 4500 :
        wave2 += [w1]
    elif 4501 <= i <= 5500 :
        wave2 += [2*w1]
    elif 5501 <= i :
        wave2 += [w1]
    
point1 = {"start":[0,50],"end":[0,90]}

fig1,ax1 = plt.subplots()
ax1.plot(time_list,wave1,color="blue",alpha=0.7)
ax1.plot(time_list,wave2,color="red",alpha=0.7)
ax1.set_xlabel("t[s]")
ax1.set_ylabel("y")
ax1.set_xlim(-10,10)
ax1.legend([r"$f(t)$",r"$\beta*f(t)$"], loc="upper right", fontsize=12)
ax1.text(0.5,75,r"$\times\beta$",fontsize=12)
ax1.annotate("", xy=point1["end"], xytext=point1["start"],arrowprops=dict(shrink=0, width=1, headwidth=8, headlength=10, connectionstyle='arc3',facecolor='gray', edgecolor='gray'))
fig1.savefig("fig2-7.png")
fig1.savefig("fig2-7.eps")

"""

##########
# fig2-8 #
##########

"""
beta_list = np.linspace(0,2,1001)

w_soft_list,l2_list = [],[]
for beta in tqdm(beta_list):

    wave2 = []
    for w1,i in zip(wave1,range(10001)) :
        if i <= 4500 :
            wave2 += [w1]
        elif 4501 <= i <= 5500 :
            wave2 += [beta*w1]
        elif 5501 <= i :
            wave2 += [w1]

    wave2 = np.array(wave2)

    l2 = np.mean((wave1-wave2)**2)
    l2_list += [l2]

    prob1_soft,prob2_soft = softplus_normalizing(wave1,wave2,h,N)
    w_soft_pos = ot.emd2_1d(time_list,time_list,prob1_soft/sum(prob1_soft),prob2_soft/sum(prob2_soft),metric='minkowski',p=2)
    prob1_soft,prob2_soft = softplus_normalizing(-wave1,-wave2,h,N)
    w_soft_neg = ot.emd2_1d(time_list,time_list,prob1_soft/sum(prob1_soft),prob2_soft/sum(prob2_soft),metric='minkowski',p=2)
    w_soft = w_soft_pos + w_soft_neg
    w_soft_list += [w_soft]

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.plot(beta_list,l2_list, color = "green", linewidth=1.0)
ax1.set_xlabel(r"$\beta$(magnification)")
ax1.set_ylabel("MSE")
ax1.set_title("MSE")

ax4 = fig.add_subplot(1,2,2)
ax4.plot(beta_list,w_soft_list,color = "green", linewidth=1.0)
ax4.set_xlabel(r"$\beta$(magnification)")
ax4.set_ylabel(r"$W_{2}^2$")
ax4.set_title(r"$W_{2}^2$(softplus normalizing)")

# show plots
fig.tight_layout()
fig.savefig("fig2-8.png")
fig.savefig("fig2-8.eps")

"""

####################
#  fig_correct2-1  #
####################

"""

tau_list = np.linspace(-15.0,15.0,1001)
w1 = vector("",time_list,wave1)

coh_list,freq_list = [],[]
for tau in tqdm(tau_list):
    wave2 = -ricker(time_list,0.2,tau,50)
    w2 = vector("",time_list,wave2)
    freq,coh = w1.coherence(w2,0.5)
    coh_list += [coh]
    freq_list += [freq]

coh_list = np.array(coh_list)
freq_list = np.array(freq_list)

print(coh_list[200,:])
print(freq_list[200,:])
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(freq_list[200,:],coh_list[200,:],color = "green", linewidth=1.0)
ax1.set_xlabel("shift[s]")
ax1.set_ylabel("coherence")
ax1.set_title("coherence")

# ax2 = fig.add_subplot(2,2,2)
# ax2.plot(tau_list,w_linear_list,color = "green", linewidth=1.0)
# ax2.set_xlabel("shift[s]")
# ax2.set_ylabel(r"$W_{2}^2$")
# ax2.set_title(r"$W_{2}^2$(linear normalizing)")

# ax3 = fig.add_subplot(2,2,3)
# ax3.plot(tau_list,w_exp_list,color = "green", linewidth=1.0)
# ax3.set_xlabel("shift[s]")
# ax3.set_ylabel(r"$W_{2}^2$")
# ax3.set_title(r"$W_{2}^2$(exponential normalizing)")

# ax4 = fig.add_subplot(2,2,4)
# ax4.plot(tau_list,w_soft_list,color = "green", linewidth=1.0)
# ax4.set_xlabel("shift[s]")
# ax4.set_ylabel(r"$W_{2}^2$")
# ax4.set_title(r"$W_{2}^2$(softplus normalizing)")

# show plots
fig.tight_layout()
fig.savefig("fig_correct2-1.png")
fig.savefig("fig_correct2-1.eps")

"""