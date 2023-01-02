import numpy as np
import os
import matplotlib.pyplot as plt
from pyparsing import line

os.chdir("./osaka_data/archive")

vel_A = np.loadtxt("OSA.vel",usecols=(2),unpack=True)
vel_B = np.loadtxt("OSB.vel",usecols=(2),unpack=True)
vel_C = np.loadtxt("OSC.vel",usecols=(2),unpack=True)
vel_f = np.loadtxt("FKS186I7.vel",usecols=(1),unpack=True)

time = np.arange(0,240,0.01)
time_list = np.arange(30,170,20)

linewidth=1.0
figsize = (40,10)
fontsize=35
labelsize=25
os.chdir("./vel_NS")

for t in time_list :
    
    n1 = int(100*t)
    n2 = int(2000+100*t)
    n3 = int(6000+100*t)
    n4 = int(8000+100*t)
    vel_A_part = np.abs(vel_A[n1:n2])
    vel_B_part = np.abs(vel_B[n1:n2])
    vel_C_part = np.abs(vel_C[n1:n2])
    vel_f_part = np.abs(vel_f[n3:n4])

    max_A=round(np.max(vel_A_part),2)
    max_B=round(np.max(vel_B_part),2)
    max_C=round(np.max(vel_C_part),2)
    max_f=round(np.max(vel_f_part),2)

    max_value = max(max_A,max_B,max_C,max_f)

    fig, axes = plt.subplots(4,1,
                                figsize=figsize,facecolor="white",
                                linewidth=0,edgecolor="black",
                                subplot_kw=dict(facecolor="white"))

    axes[0].plot(time[n1:n2], vel_A[n1:n2], linewidth=linewidth,label="OSA_vel",color="blue")
    axes[1].plot(time[n1:n2], vel_B[n1:n2], linewidth=linewidth,label="OSB_vel",color="red")
    axes[2].plot(time[n1:n2], vel_C[n1:n2], linewidth=linewidth,label="OSC_vel",color="green")
    axes[3].plot(time[n1:n2], vel_f[n3:n4], linewidth=linewidth,label="FKS_vel",color="black")

    axes[0].set_ylim(-max_value,max_value)
    axes[0].set_xlim(t,t+20)
    axes[1].set_ylim(-max_value,max_value)
    axes[1].set_xlim(t,t+20)
    axes[2].set_ylim(-max_value,max_value)
    axes[2].set_xlim(t,t+20)
    axes[3].set_ylim(-max_value,max_value)
    axes[3].set_xlim(t,t+20)

    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['left'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['left'].set_visible(False)
    axes[2].spines['bottom'].set_visible(False)
    axes[3].spines['right'].set_visible(False)
    axes[3].spines['top'].set_visible(False)
    axes[3].spines['left'].set_visible(False)

    axes[0].axes.xaxis.set_visible(False)
    axes[0].axes.yaxis.set_visible(False)
    axes[1].axes.xaxis.set_visible(False)
    axes[1].axes.yaxis.set_visible(False)
    axes[2].axes.xaxis.set_visible(False)
    axes[2].axes.yaxis.set_visible(False)
    axes[3].axes.yaxis.set_visible(False)

    axes[3].tick_params(axis='x',direction="inout",length=15.0,labelsize=fontsize,)

    axes[3].set_xlabel("Time[s]",fontsize=fontsize)

    axes[0].text(t+5,1.2*max_value,f"the 2018 Northern Osaka Prefecture Earthquake(vel_NS)({t}[s]~{t+20}[s])",size=fontsize)
    axes[0].text(t,0.8*max_value,"OSA",size=fontsize)
    axes[1].text(t,0.8*max_value,"OSB",size=fontsize)
    axes[2].text(t,0.8*max_value,"OSC",size=fontsize)
    axes[3].text(t,0.8*max_value,"FKS",size=fontsize)


    fig.savefig(f"osaka_vel_NS_{t}[s]~{t+20}[s].png")
    fig.savefig(f"osaka_vel_NS_{t}[s]~{t+20}[s].eps")
    fig.tight_layout()
    plt.clf()
    plt.close()

"""

start = 8000
end = 11000
s = start/100
e = end/100

linewidth=0.4
figsize = (25,10)
fontsize=25
labelsize=25

os.chdir("../")
os.chdir("../")
os.chdir("./soturon_tex/figure")
fig = plt.figure()
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

ax1.plot(time[start:end], vel_ud_A[start:end], linewidth=linewidth,label="OSA_acc",color="blue")
ax2.plot(time[start:end], vel_ud_B[start:end], linewidth=linewidth,label="OSB_acc",color="red")
ax3.plot(time[start:end], vel_ud_C[start:end], linewidth=linewidth,label="OSC_acc",color="green")
ax4.plot(time[start:end], vel_ud_f[start+6000:end+6000], linewidth=linewidth,label="FKS_acc",color="black")

ax1.set_ylim(-2,2)
ax1.set_xlim(s,e)
ax2.set_ylim(-2,2)
ax2.set_xlim(s,e)
ax3.set_ylim(-2,2)
ax3.set_xlim(s,e)
ax4.set_ylim(-2,2)
ax4.set_xlim(s,e)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.spines['left'].set_visible(False)

ax1.axes.xaxis.set_visible(False)
ax1.axes.yaxis.set_visible(False)
ax2.axes.xaxis.set_visible(False)
ax2.axes.yaxis.set_visible(False)
ax3.axes.xaxis.set_visible(False)
ax3.axes.yaxis.set_visible(False)
ax4.axes.yaxis.set_visible(False)

ax4.tick_params(axis='x')

ax4.set_xlabel("Time[s]")

fig.savefig(f"osaka_vel_new_{s}s~{e}s.png")
fig.savefig(f"osaka_vel_new_{s}s~{e}s.eps")
fig.tight_layout()
plt.clf()
plt.close()

"""

"""
fig, ax2 = plt.subplots(facecolor="w",figsize=figsize)
ax2.plot(time, vel_ud_A, label="OSA_vel",linewidth=linewidth)
ax2.plot(time, vel_ud_B, label="OSB_vel",linewidth=linewidth)
ax2.plot(time, vel_ud_C, label="OSC_vel",linewidth=linewidth)
ax2.plot(time, vel_ud_f[6000:30000], label="FKS_vel",linewidth=linewidth)
ax2.set_xlabel("time[s]",fontsize=fontsize)
ax2.set_ylabel("velocity[cm/s]",fontsize=fontsize)
ax2.set_ylim(-5,5)
ax2.set_xlim(0,240)
ax2.tick_params(axis='x', labelsize=labelsize)
ax2.tick_params(axis='y', labelsize=labelsize)
ax2.legend(fontsize=20)
fig.savefig("osaka_vel.png")
fig.savefig("osaka_vel.eps")
fig.tight_layout()
plt.clf()
plt.close()

number_list = [35]
for n in number_list:
    time = np.arange(n,n+150,0.01)
    fig, ax3 = plt.subplots(facecolor="w",figsize=figsize)
    # ax3.plot(time, acc_ud_A[n*100:n*100+500], label="OSA_acc",linewidth=linewidth)
    # ax3.plot(time, acc_ud_B[n*100:n*100+500], label="OSB_acc",linewidth=linewidth)
    # ax3.plot(time, acc_ud_C[n*100:n*100+500], label="OSC_acc",linewidth=linewidth)
    ax3.plot(time, vel_ud_f[n*100:n*100+15000], label="FKS_vel",linewidth=linewidth)
    ax3.set_xlabel("time[s]",fontsize=fontsize)
    ax3.set_ylabel(r"acceleration[cm/s$^2$]",fontsize=fontsize)
    ax3.set_ylim(-10,10)
    ax3.set_xlim(n+45,n+150)
    ax3.set_position([0.1,0.1,0.8,0.8])
    ax3.tick_params(axis='x', labelsize=labelsize)
    ax3.tick_params(axis='y', labelsize=labelsize)
    ax3.legend(loc="upper left",fontsize=20)
    m=n+150
    # fig.savefig(f"osaka_acc_zoom_{n}[s]~{m}[s].png")
    # fig.savefig(f"osaka_acc_zoom_{n}[s]~{m}[s].eps")
    # fig.tight_layout()
    plt.show()
    # plt.clf()
    # plt.close()

np.save("OSA_acc_ud.npy",acc_ud_A)
np.save("OSB_acc_ud.npy",acc_ud_B)
np.save("OSC_acc_ud.npy",acc_ud_C)
np.save("FKS_acc_ud.npy",acc_ud_f[6000:30000])
np.save("OSA_vel_ud.npy",vel_ud_A)
np.save("OSB_vel_ud.npy",vel_ud_B)
np.save("OSC_vel_ud.npy",vel_ud_C)
np.save("FKS_vel_ud.npy",vel_ud_f[6000:30000])
"""
