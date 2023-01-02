import matplotlib.pyplot as plt
import numpy as np
import ot
import sys
sys.path.append('./Python_Code')
from others.similarity import Similarity
from PySGM.vector import vector
from tqdm import tqdm
import matplotlib.animation as animation
import math


def ricker(tim,fp,tp,amp):

    t1 = ((tim-tp)*np.pi*fp)**2

    return (2*t1-1)*np.exp(-t1)*amp

def animation_wave(tim,wave1_list,wave2_list,ylim_min,ylim_max,total_time,frame) :

    fig,ax = plt.subplots(figsize=(20,5))
    ax.set_ylim(ylim_min,ylim_max)
    ax.axis("off")
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
    ax.axis("off")
    ims = []
    k = 1500/frame
    k=int(k)
    for i,s,sim in zip(range(1501),tau,similarity_list) :
        if i%k==0 :
            im = ax.plot(tau,similarity_list,color=line_color)
            im += ax.plot(s,sim,color=dot_color,marker='.',markersize=dot_size)
            ims.append(im)
    interval = total_time/((len(similarity_list)-1)/k)*1000
    ani = animation.ArtistAnimation(fig, ims, interval=interval)

    return ani

def synthetic_ricker_left(tim):
      
      f1 = ricker(tim,2,1.25,1)
      f2 = -ricker(tim,3,1.375,5/3)
      f3 = ricker(tim,4,1.5,8/3)
      f4 = -ricker(tim,3,1.625,5/3)
      f5 = ricker(tim,2,1.75,1)

      fleft = f1+f2+f3+f4+f5

      f6 = -ricker(tim,2,2.25,10/3)
      f7 = ricker(tim,3,2.375,14/3)
      f8 = -ricker(tim,4,2.5,8)
      f9 = ricker(tim,3,2.625,14/3)
      f10 = -ricker(tim,2,2.75,10/3)

      fcenter = f6+f7+f8+f9+f10
      
      f11 = ricker(tim,2,3.25,1)
      f12 = -ricker(tim,3,3.375,5/3)
      f13 = ricker(tim,4,3.5,8/3)
      f14 = -ricker(tim,3,3.625,5/3)
      f15 = ricker(tim,2,3.75,1)

      fright = f11+f12+f13+f14+f15

      return -fleft+fcenter-fright

def synthetic_ricker_right(tim):
      
      f1 = ricker(tim,2,8.75,1)
      f2 = -ricker(tim,3,8.875,5/3)
      f3 = ricker(tim,4,9.0,8/3)
      f4 = -ricker(tim,3,9.125,5/3)
      f5 = ricker(tim,2,9.25,1)

      fleft = f1+f2+f3+f4+f5

      f6 = -ricker(tim,2,9.75,10/3)
      f7 = ricker(tim,3,9.875,14/3)
      f8 = -ricker(tim,4,10,8)
      f9 = ricker(tim,3,10.125,14/3)
      f10 = -ricker(tim,2,10.25,10/3)

      fcenter = f6+f7+f8+f9+f10
      
      f11 = ricker(tim,2,10.75,1)
      f12 = -ricker(tim,3,10.875,5/3)
      f13 = ricker(tim,4,11,8/3)
      f14 = -ricker(tim,3,11.125,5/3)
      f15 = ricker(tim,2,11.25,1)

      fright = f11+f12+f13+f14+f15

      return -fleft+fcenter-fright

def pink_noise(std,tim_len):

      wn = np.random.normal(0,std,tim_len)

      Wn = np.fft.fft(wn)
      freq = np.fft.fftfreq(len(wn),d=0.01)
      df = freq[1] - freq[0]

      Wp = np.ones_like(Wn)
      for (i,f) in enumerate(freq):
            if f > 1.e-8:
                  # print(f,2.0*math.pi*f)
                  coef = (0.0+1.0j)*np.sqrt(2.0*math.pi*f)
                  Wp[i] = Wn[i] / coef
            elif f < -1.e-8:
                  coef = -(0.0+1.0j)*np.sqrt(2.0*math.pi*abs(f))
                  Wp[i] = Wn[i] / coef
            else:
                  Wp[i] = 0.0 + 0.0j

      wp = np.real(np.fft.ifft(Wp))
       
      return wp

def main():

    tim = np.linspace(0,20,2001)

    wave1 = synthetic_ricker_left(tim)
    wave2 = synthetic_ricker_right(tim) 

    std = 10

    noise1500 = pink_noise(std,1500)
    noise2000_1 = pink_noise(std,2001)
    noise2000_2 = pink_noise(std,2001)

    noise1500_bandpass = vector.bandpass(noise1500,dt=0.01,low=0.5,high=10)
    noise2000_1_bandpass = vector.bandpass(noise2000_1,dt=0.01,low=0.5,high=10)
    noise2000_2_bandpass = vector.bandpass(noise2000_2,dt=0.01,low=0.5,high=10)

    wave_move = wave1 + noise2000_1_bandpass
    wave_fix = wave2 + noise2000_2_bandpass

    wave_left = np.hstack([noise1500_bandpass,wave_move])

    M = 1500 # 時間シフトのコマ数

    tau = np.linspace(0,15,1501)

    w_list,l2_list,f_list,g_list = [],[],[],[]
    for i,s in tqdm(enumerate(tau)) :

        wave_move = wave_left[1500-i:3501-i]
        sim_org = Similarity(tim,wave_move,wave_fix,"")
        was = sim_org.wasserstein_softplus_normalizing(con=3)
        l2 = sim_org.mse()

        w_list += [was]
        l2_list += [l2]
        f_list += [wave_move]
        g_list += [wave_fix]

    np.save(f"./Python_Code/zenkoku/f_{std}.npy",f_list)
    np.save(f"./Python_Code/zenkoku/g_{std}.npy",g_list)
    np.save(f"./Python_Code/zenkoku/w_{std}.npy",w_list)
    np.save(f"./Python_Code/zenkoku/l2_{std}.npy",l2_list)

def plot_animation():

    tim = np.linspace(0,20,2001)
    f_list = np.load("./Python_Code/zenkoku/f_10.npy")
    g_list = np.load("./Python_Code/zenkoku/g_10.npy")
    l2_list = np.load("./Python_Code/zenkoku/l2_10.npy")
    w_list = np.load("./Python_Code/zenkoku/w_10.npy")
    tau = np.linspace(0,15,1501)

    ani1 = animation_wave(tim, f_list, g_list, ylim_min=-10, ylim_max=10, total_time=5,frame=75) #frameは1400の約数
    ani1.save('./Python_Code/zenkoku/wave_10.gif', writer="imagemagick")

    ani3 = animation_similarity(tau,l2_list, total_time=5, frame=75, line_color="black", dot_color="red", dot_size=20) #frameは1400の約数
    ani3.save('./Python_Code/zenkoku/l2_10.gif', writer="imagemagick")

    ani4 = animation_similarity(tau,w_list, total_time=5, frame=75, line_color="brown", dot_color="red", dot_size=20) #frameは1400の約数
    ani4.save('./Python_Code/zenkoku/was_10.gif', writer="imagemagick")

plot_animation()
# main()