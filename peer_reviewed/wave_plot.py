import numpy as np
from scipy import signal
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.append('./Python_Code')
from PySGM.vector import vector


def init():

      time = np.arange(0,40,0.01)
      return time

def ricker(tim,fp,tp,amp):
      
      t1 = ((tim-tp)*np.pi*fp)**2

      return (2*t1-1)*np.exp(-t1)*amp

def plot(time): #リッカー波

      fig,ax = plt.subplots(figsize=(20,8))
      ax.plot(time,-ricker(time,0.2,5,4),color="red",label=r"$f(t-s)$")
      ax.plot(time,-ricker(time,0.2,20,4),color="blue",label=r"$f(t)$")
      ax.set_xlabel(r"$t$[s]",size=20)
      ax.set_ylabel("amplitude",size=20)
      ax.legend(fontsize=20)
      point = {'start': [6, 3],'end': [30, 3]}
      ax.annotate('', xy=point['end'], xytext=point['start'],
                  arrowprops=dict(shrink=0, width=1, headwidth=15, 
                                    headlength=10, connectionstyle='arc3',
                                    facecolor='gray', edgecolor='gray'))
      ax.tick_params(direction = "in", length = 10, labelsize=20)

      fig.savefig("./Python_Code/peer_reviewed/figure/ricker.png")

def synthetic_ricker_left(tim):
      
      f1 = ricker(tim,0.6,1.5,15)
      f2 = -ricker(tim,0.8,1.75,25)
      f3 = ricker(tim,1,2,40)
      f4 = -ricker(tim,0.8,2.25,25)
      f5 = ricker(tim,0.6,2.5,15)

      fleft = f1+f2+f3+f4+f5

      f6 = -ricker(tim,0.6,4.5,50)
      f7 = ricker(tim,0.8,4.75,70)
      f8 = -ricker(tim,1,5,120)
      f9 = ricker(tim,0.8,5.25,70)
      f10 = -ricker(tim,0.6,5.5,50)

      fcenter = f6+f7+f8+f9+f10
      
      f11 = ricker(tim,0.6,7.5,15)
      f12 = -ricker(tim,0.8,7.75,25)
      f13 = ricker(tim,1,8,40)
      f14 = -ricker(tim,0.8,8.25,25)
      f15 = ricker(tim,0.6,8.5,15)

      fright = f11+f12+f13+f14+f15

      return fleft+fcenter+fright

def synthetic_ricker_right(tim):
      
      f1 = ricker(tim,0.6,16.5,15)
      f2 = -ricker(tim,0.8,16.75,25)
      f3 = ricker(tim,1,17,40)
      f4 = -ricker(tim,0.8,17.25,25)
      f5 = ricker(tim,0.6,17.5,15)

      fleft = f1+f2+f3+f4+f5

      f6 = -ricker(tim,0.6,19.5,50)
      f7 = ricker(tim,0.8,19.75,70)
      f8 = -ricker(tim,1,20,120)
      f9 = ricker(tim,0.8,20.25,70)
      f10 = -ricker(tim,0.6,20.5,50)

      fcenter = f6+f7+f8+f9+f10
      
      f11 = ricker(tim,0.6,22.5,15)
      f12 = -ricker(tim,0.8,22.75,25)
      f13 = ricker(tim,1,23,40)
      f14 = -ricker(tim,0.8,23.25,25)
      f15 = ricker(tim,0.6,23.5,15)

      fright = f11+f12+f13+f14+f15

      return fleft+fcenter+fright

def plot2(time,fleft,fright): #合成リッカー波

      fig,ax = plt.subplots(figsize=(20,8))
      fleft_env = np.abs(signal.hilbert(fleft))
      fright_env = np.abs(signal.hilbert(fright))
      ax.plot(time,fleft,color="red",label=r"$f(t-s)$")
      ax.plot(time,fleft_env,color="red",linestyle="dashed",label=r"$f(t-s)$_envelope")
      ax.plot(time,fright,color="blue",label=r"$f(t)$")
      ax.plot(time,fright_env,color="blue",linestyle="dashed",label=r"$f(t)$_envelope")
      ax.plot(time,np.zeros_like(time),color="black")
      ax.legend(fontsize=20)
      point = {'start': [7, 40],'end': [30, 40]}
      ax.annotate('', xy=point['end'], xytext=point['start'],
                  arrowprops=dict(shrink=0, width=1, headwidth=15, 
                                    headlength=10, connectionstyle='arc3',
                                    facecolor='gray', edgecolor='gray'))
      ax.set_xlabel(r"$t$[s]",size=20)
      ax.set_ylabel("amplitude",size=20)
      ax.tick_params(direction = "in", length = 10, labelsize=20)
      fig.savefig("./Python_Code/peer_reviewed/figure/synthetic_ricker.png")

def fks_acc():

      # time = np.loadtxt("./Python_Code/osaka_data/archive/FKS186I7.acc",
      #                   usecols=(0),unpack=True)
      # time = time[9000:20000]
      #wave_ns = np.loadtxt("./Python_Code/osaka_data/archive/FKS186I7.acc",usecols=(1),unpack=True) #FKS acc NS
      #wave_ud = np.loadtxt("./Python_Code/osaka_data/archive/FKS186I7.acc",usecols=(3),unpack=True)
      wave_ew = np.loadtxt("./Python_Code/osaka_data/archive/FKS186I7.acc",
                  usecols=(2),unpack=True)  
      wave_ew = wave_ew[9500:19500]
      zeros30000 = np.zeros(30000)
      zeros15000 = np.zeros(15000)
      wave_left = np.hstack([wave_ew,zeros30000])
      wave_right = np.hstack([zeros15000,wave_ew,zeros15000])

      time = np.arange(0,400,0.01)

      return time,wave_left,wave_right

def plot3(time,wave_left,wave_right) : #FKS福島の波形(東西加速度)

      fig,ax = plt.subplots(figsize=(20,8))
      ax.plot(time,wave_left,color="red",label=r"$f(t-s)$",linewidth=0.5)
      ax.plot(time,wave_right,color="blue",label=r"$f(t)$",linewidth=0.5)
      ax.plot(time,np.zeros_like(time),color="black")
      ax.legend(fontsize=30)
      point = {'start': [30, 50],'end': [300, 50]}
      ax.annotate('', xy=point['end'], xytext=point['start'],
                  arrowprops=dict(shrink=0, width=1, headwidth=15, 
                                    headlength=10, connectionstyle='arc3',
                                    facecolor='gray', edgecolor='gray'))
      ax.set_xlabel(r"$t$[s]",size=20)
      ax.set_ylabel("amplitude",size=20)
      ax.set_ylim(-75,)
      ax.tick_params(direction = "in", length = 10, labelsize=20)
      fig.savefig("./Python_Code/peer_reviewed/figure/FKS_acc_EW.png")
      
def plot4(time,wave_left,wave_right) : #FKS福島の包絡線(東西加速度)

      fig,ax = plt.subplots(figsize=(20,8))
      fleft_env = np.abs(signal.hilbert(wave_left))
      fright_env = np.abs(signal.hilbert(wave_right))

      ax.plot(time,fleft_env,color="red",label=r"$f(t-s)$_envelope",linewidth=0.5)
      ax.plot(time,fright_env,color="blue",label=r"$f(t)$_envelope",linewidth=0.5)
      ax.plot(time,np.zeros_like(time),color="black")
      ax.legend(fontsize=30)
      point = {'start': [30, 50],'end': [300, 50]}
      ax.annotate('', xy=point['end'], xytext=point['start'],
                  arrowprops=dict(shrink=0, width=1, headwidth=15, 
                                    headlength=10, connectionstyle='arc3',
                                    facecolor='gray', edgecolor='gray'))
      ax.set_xlabel(r"$t$[s]",size=20)
      ax.set_ylabel("amplitude",size=20)
      ax.set_ylim(-75,)
      ax.tick_params(direction = "in", length = 10, labelsize=20)
      fig.savefig("./Python_Code/peer_reviewed/figure/FKS_acc_EW_envelope.png")
           
def fks_dis():

      wave_ew_vel = np.loadtxt("./Python_Code/osaka_data/archive/FKS186I7.vel",usecols=(2),unpack=True)
      wave_ew_dis = vector.integration(wave_ew_vel,dt=0.01,low=0.1,high=10)
      wave_ew_dis = wave_ew_dis[9500:19500]
      zeros30000 = np.zeros(30000)
      zeros15000 = np.zeros(15000)
      wave_left = np.hstack([wave_ew_dis,zeros30000])
      wave_right = np.hstack([zeros15000,wave_ew_dis,zeros15000])

      time = np.arange(0,400,0.01)

      return time,wave_left,wave_right
      
def plot5(time,wave_left,wave_right) : #FKS福島の波形(東西変位)

      fig,ax = plt.subplots(figsize=(20,8))
      ax.plot(time,wave_left,color="red",label=r"$f(t-s)$",linewidth=0.5)
      ax.plot(time,wave_right,color="blue",label=r"$f(t)$",linewidth=0.5)
      ax.plot(time,np.zeros_like(time),color="black")
      ax.legend(fontsize=30)
      point = {'start': [30, 50],'end': [300, 50]}
      ax.annotate('', xy=point['end'], xytext=point['start'],
                  arrowprops=dict(shrink=0, width=1, headwidth=15, 
                                    headlength=10, connectionstyle='arc3',
                                    facecolor='gray', edgecolor='gray'))
      ax.set_xlabel(r"$t$[s]",size=20)
      ax.set_ylabel("amplitude",size=20)
      ax.tick_params(direction = "in", length = 10, labelsize=20)
      fig.savefig("./Python_Code/peer_reviewed/figure/FKS_dis_EW.png")

time = init()
plot(time)


