import numpy as np
import matplotlib.pyplot as plt
from pyparsing import line
import sys
import matplotlib.cm as cm
sys.path.append('./Python_Code')
from PySGM.vector import vector
from peer_reviewed.format_shift import Format

Format.params()

def baseline_correct(wave) :

      ave = np.average(wave)
      new_wave = wave - ave

      return new_wave

def wave_acc_load(): #EW成分#

      acc_A = np.loadtxt("./Python_Code/osaka_data/archive/OSA.acc",usecols=(1),unpack=True)
      acc_B = np.loadtxt("./Python_Code/osaka_data/archive/OSB.acc",usecols=(1),unpack=True)
      acc_C = np.loadtxt("./Python_Code/osaka_data/archive/OSC.acc",usecols=(1),unpack=True)
      acc_f = np.loadtxt("./Python_Code/osaka_data/archive/FKS186I7.acc",usecols=(2),unpack=True)

      acc_A = vector.bandpass(acc_A,dt=0.01,low=0.1,high=10)
      acc_B = vector.bandpass(acc_B,dt=0.01,low=0.1,high=10)
      acc_C = vector.bandpass(acc_C,dt=0.01,low=0.1,high=10)
      acc_f = vector.bandpass(acc_f,dt=0.01,low=0.1,high=10)

      acc_A = baseline_correct(acc_A)
      acc_B = baseline_correct(acc_B)
      acc_C = baseline_correct(acc_C)
      acc_f = baseline_correct(acc_f)

      return acc_A,acc_B,acc_C,acc_f

def plot_acc():

      time = np.arange(0,100,0.01)
      acc_A,acc_B,acc_C,acc_f = wave_acc_load()

      acc_A = acc_A[0:10000]
      acc_B = acc_B[0:10000]
      acc_C = acc_C[0:10000]
      acc_f = acc_f[6000:16000]

      acc_A_part = np.abs(acc_A)
      acc_B_part = np.abs(acc_B)
      acc_C_part = np.abs(acc_C)
      acc_f_part = np.abs(acc_f)

      max_A=round(np.max(acc_A_part),1)
      max_B=round(np.max(acc_B_part),1)
      max_C=round(np.max(acc_C_part),1)
      max_f=round(np.max(acc_f_part),1)

      max_value = max(max_A,max_B,max_C,max_f)

      fig, axes = plt.subplots(4,1,facecolor="white",linewidth=0,edgecolor="black",subplot_kw=dict(facecolor="white"))

      axes[0].plot(time, acc_A,label="OSA_acc",color=cm.magma(0/4))
      axes[1].plot(time, acc_B,label="OSB_acc",color=cm.magma(1/4))
      axes[2].plot(time, acc_C,label="OSC_acc",color=cm.magma(2/4))
      axes[3].plot(time, acc_f,label="FKS_acc",color=cm.magma(3/4))

      axes[0].set_ylim(-max_value,max_value)
      axes[0].set_xlim(0,100)
      axes[1].set_ylim(-max_value,max_value)
      axes[1].set_xlim(0,100)
      axes[2].set_ylim(-max_value,max_value)
      axes[2].set_xlim(0,100)
      axes[3].set_ylim(-max_value,max_value)
      axes[3].set_xlim(0,100)

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

      axes[3].tick_params(axis='x')

      axes[3].set_xlabel("Time[s]")

      axes[0].text(0,0.8*max_value,"OSA")
      axes[1].text(0,0.8*max_value,"OSB")
      axes[2].text(0,0.8*max_value,"OSC")
      axes[3].text(0,0.8*max_value,"FKS")

      axes[0].text(80,0.8*max_value,fr"Max_amp = {max_A}[cm/s$^2$]",fontsize=50)
      axes[1].text(80,0.8*max_value,fr"Max_amp = {max_B}[cm/s$^2$]",fontsize=50)
      axes[2].text(80,0.8*max_value,fr"Max_amp = {max_C}[cm/s$^2$]",fontsize=50)
      axes[3].text(80,0.8*max_value,fr"Max_amp = {max_f}[cm/s$^2$]",fontsize=50)

      fig.savefig(f"./Python_Code/peer_reviewed/figure/osaka_wave/osaka_acc_EW.png")
      fig.tight_layout()
      plt.clf()
      plt.close()

def wave_vel_load():

      vel_A = np.loadtxt("./Python_Code/osaka_data/archive/OSA.vel",usecols=(3),unpack=True)
      vel_B = np.loadtxt("./Python_Code/osaka_data/archive/OSB.vel",usecols=(3),unpack=True)
      vel_C = np.loadtxt("./Python_Code/osaka_data/archive/OSC.vel",usecols=(3),unpack=True)
      vel_f = np.loadtxt("./Python_Code/osaka_data/archive/FKS186I7.vel",usecols=(3),unpack=True)

      vel_A = vector.bandpass(vel_A,dt=0.01,low=0.1,high=10)
      vel_B = vector.bandpass(vel_B,dt=0.01,low=0.1,high=10)
      vel_C = vector.bandpass(vel_C,dt=0.01,low=0.1,high=10)
      vel_f = vector.bandpass(vel_f,dt=0.01,low=0.1,high=10)

      vel_A = baseline_correct(vel_A)
      vel_B = baseline_correct(vel_B)
      vel_C = baseline_correct(vel_C)
      vel_f = baseline_correct(vel_f)

      return vel_A,vel_B,vel_C,vel_f

def acc_to_vel():

      acc_A_correct,acc_B,acc_C,acc_f = wave_acc_load()
      vel_A_correct = vector.integration(acc_A_correct,dt=0.01,low=0.1,high=10)
 
      acc_A = np.loadtxt("./Python_Code/osaka_data/archive/OSA.acc",usecols=(1),unpack=True)
      vel_A = vector.integration(acc_A,dt=0.01,low=0.1,high=10)

      vel_A_data = np.loadtxt("./Python_Code/osaka_data/archive/OSA.vel",usecols=(1),unpack=True)
 
      from peer_reviewed.format_wave import Format
      Format.params()

      fig,ax = plt.subplots()
      time = np.arange(0,240,0.01)
      ax.plot(time,vel_A_correct,label="hoseigo",linewidth=1)
      # ax.plot(time,vel_A,label="sonomama",linewidth=1)
      ax.plot(time,vel_A_data,label="data",linewidth=1)
      ax.set_ylim(-1,1)
      ax.legend()
      fig.tight_layout()

      fig.savefig("./Python_Code/peer_reviewed/figure/osaka_wave/compare.png")

def plot_vel():

      time = np.arange(0,200,0.01)
      vel_A,vel_B,vel_C,vel_f = wave_vel_load()

      vel_A = vel_A[0:20000]
      vel_B = vel_B[0:20000]
      vel_C = vel_C[0:20000]
      vel_f = vel_f[6000:26000]

      vel_A_part = np.abs(vel_A)
      vel_B_part = np.abs(vel_B)
      vel_C_part = np.abs(vel_C)
      vel_f_part = np.abs(vel_f)

      max_A=round(np.max(vel_A_part),1)
      max_B=round(np.max(vel_B_part),1)
      max_C=round(np.max(vel_C_part),1)
      max_f=round(np.max(vel_f_part),1)

      max_value = max(max_A,max_B,max_C,max_f)

      fig, axes = plt.subplots(4,1,facecolor="white",linewidth=0,edgecolor="black",subplot_kw=dict(facecolor="white"))

      axes[0].plot(time, vel_A,label="OSA_vel",color=cm.magma(0/4))
      axes[1].plot(time, vel_B,label="OSB_vel",color=cm.magma(1/4))
      axes[2].plot(time, vel_C,label="OSC_vel",color=cm.magma(2/4))
      axes[3].plot(time, vel_f,label="FKS_vel",color=cm.magma(3/4))

      axes[0].set_ylim(-max_value,max_value)
      axes[0].set_xlim(0,200)
      axes[1].set_ylim(-max_value,max_value)
      axes[1].set_xlim(0,200)
      axes[2].set_ylim(-max_value,max_value)
      axes[2].set_xlim(0,200)
      axes[3].set_ylim(-max_value,max_value)
      axes[3].set_xlim(0,200)

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

      axes[3].tick_params(axis='x')

      axes[3].set_xlabel("Time[s]")

      axes[0].text(0,0.8*max_value,"OSA")
      axes[1].text(0,0.8*max_value,"OSB")
      axes[2].text(0,0.8*max_value,"OSC")
      axes[3].text(0,0.8*max_value,"FKS")

      axes[0].text(160,0.8*max_value,fr"Max_amp = {max_A}[cm/s]",fontsize=70)
      axes[1].text(160,0.8*max_value,fr"Max_amp = {max_B}[cm/s]",fontsize=70)
      axes[2].text(160,0.8*max_value,fr"Max_amp = {max_C}[cm/s]",fontsize=70)
      axes[3].text(160,0.8*max_value,fr"Max_amp = {max_f}[cm/s]",fontsize=70)

      fig.savefig(f"./Python_Code/peer_reviewed/figure/osaka_wave/osaka_vel_UD.png")
      fig.tight_layout()
      plt.clf()
      plt.close()

def vel_to_dis():

      vel_A,vel_B,vel_C,vel_f = wave_vel_load()
      dis_A = vector.integration(vel_A,dt=0.01,low=0.1,high=10)
      dis_B = vector.integration(vel_B,dt=0.01,low=0.1,high=10)
      dis_C = vector.integration(vel_C,dt=0.01,low=0.1,high=10)
      dis_f = vector.integration(vel_f,dt=0.01,low=0.1,high=10)
 
      return dis_A,dis_B,dis_C,dis_f

def plot_dis():

      time = np.arange(0,100,0.01)
      dis_A,dis_B,dis_C,dis_f = vel_to_dis()

      dis_A = dis_A[0:10000]
      dis_B = dis_B[0:10000]
      dis_C = dis_C[0:10000]
      dis_f = dis_f[6000:16000]

      dis_A_part = np.abs(dis_A)
      dis_B_part = np.abs(dis_B)
      dis_C_part = np.abs(dis_C)
      dis_f_part = np.abs(dis_f)

      max_A=round(np.max(dis_A_part),1)
      max_B=round(np.max(dis_B_part),1)
      max_C=round(np.max(dis_C_part),1)
      max_f=round(np.max(dis_f_part),1)

      max_value = max(max_A,max_B,max_C,max_f)

      fig, axes = plt.subplots(4,1,facecolor="white",linewidth=0,edgecolor="black",subplot_kw=dict(facecolor="white"))

      axes[0].plot(time, dis_A,label="OSA_dis",color=cm.magma(0/4))
      axes[1].plot(time, dis_B,label="OSB_dis",color=cm.magma(1/4))
      axes[2].plot(time, dis_C,label="OSC_dis",color=cm.magma(2/4))
      axes[3].plot(time, dis_f,label="FKS_dis",color=cm.magma(3/4))

      axes[0].set_ylim(-max_value,max_value)
      axes[0].set_xlim(0,100)
      axes[1].set_ylim(-max_value,max_value)
      axes[1].set_xlim(0,100)
      axes[2].set_ylim(-max_value,max_value)
      axes[2].set_xlim(0,100)
      axes[3].set_ylim(-max_value,max_value)
      axes[3].set_xlim(0,100)

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

      axes[3].tick_params(axis='x')

      axes[3].set_xlabel("Time[s]")

      axes[0].text(0,0.8*max_value,"OSA")
      axes[1].text(0,0.8*max_value,"OSB")
      axes[2].text(0,0.8*max_value,"OSC")
      axes[3].text(0,0.8*max_value,"FKS")

      axes[0].text(80,0.8*max_value,fr"Max_amp = {max_A}[cm]",fontsize=50)
      axes[1].text(80,0.8*max_value,fr"Max_amp = {max_B}[cm]",fontsize=50)
      axes[2].text(80,0.8*max_value,fr"Max_amp = {max_C}[cm]",fontsize=50)
      axes[3].text(80,0.8*max_value,fr"Max_amp = {max_f}[cm]",fontsize=50)

      fig.savefig(f"./Python_Code/peer_reviewed/figure/osaka_wave/osaka_dis_EW.png")
      fig.tight_layout()
      plt.clf()
      plt.close()

plot_vel()

