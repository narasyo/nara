import numpy as np
from scipy import signal
import math
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import seaborn as sns
import pandas as pd
import joblib
from typing import Optional
import contextlib
from joblib import Parallel,delayed
sys.path.append('./Python_Code')
from PySGM.vector import vector
from others.similarity import Similarity
from tqdm.auto import tqdm

def ricker(tim,fp,tp,amp):
      
      t1 = ((tim-tp)*np.pi*fp)**2

      return (2*t1-1)*np.exp(-t1)*amp

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


#############

def main_ricker(): 

      tim = np.arange(0,20,0.01)

      wave1 = -ricker(tim,0.5,2.5,10)
      wave2 = -ricker(tim,0.5,10,10)

      std_list = [0,5,10,15,20,25,30]
      length = len(std_list)

      def process1(std) :

            was_s_list,l2_s_list,env_s_list = [],[],[]

            for _ in tqdm(range(100)):

                  noise1500 = pink_noise(std,1500)
                  noise2000_1 = pink_noise(std,2000)
                  noise2000_2 = pink_noise(std,2000)

                  noise1500_bandpass = vector.bandpass(noise1500,dt=0.01,low=0.1,high=1.0)
                  noise2000_1_bandpass = vector.bandpass(noise2000_1,dt=0.01,low=0.1,high=1.0)
                  noise2000_2_bandpass = vector.bandpass(noise2000_2,dt=0.01,low=0.1,high=1.0)
                  
                  wave_move = wave1 + noise2000_1_bandpass
                  wave_fix = wave2 + noise2000_2_bandpass

                  wave_move = np.hstack([noise1500_bandpass,wave_move])
                  
                  was_s,l2_s,env_s = shift_ricker(tim,wave_move,wave_fix)

                  was_s_list += [was_s]
                  l2_s_list += [l2_s]
                  env_s_list += [env_s]

            was_s_array = np.array(was_s_list)
            l2_s_array = np.array(l2_s_list)
            env_s_array = np.array(env_s_list)

            np.save(f"./Python_Code/peer_reviewed/value/optimal_time/ricker_was_{std}",was_s_array)
            np.save(f"./Python_Code/peer_reviewed/value/optimal_time/ricker_l2_{std}",l2_s_array)
            np.save(f"./Python_Code/peer_reviewed/value/optimal_time/ricker_env_{std}",env_s_array)        
      
      Parallel(n_jobs=-1)([delayed(process1)(std) for std in std_list])

def shift_ricker(tim,wave_left,wave_fix):
           
      wave_fix_env = np.abs(signal.hilbert(wave_fix))

      shift_time = np.arange(0,15,0.01)
      was_list,l2_org_list,l2_env_list = [],[],[]
      for i,s in enumerate(shift_time):

            wave_move = wave_left[1500-i:3500-i]
                  
            wave_move_env = np.abs(signal.hilbert(wave_move))
            sim_org = Similarity(tim,wave_move,wave_fix,"")
            was = sim_org.wasserstein_softplus_normalizing(con=3)
            l2_org = sim_org.mse()
            sim_env = Similarity(tim,wave_move_env,wave_fix_env,"")
            l2_env = sim_env.mse()

            was_list += [was]
            l2_org_list += [l2_org]
            l2_env_list += [l2_env]

      was_min_index = was_list.index(min(was_list))
      l2_min_index = l2_org_list.index(min(l2_org_list))
      env_min_index = l2_env_list.index(min(l2_env_list))

      was_s = was_min_index/100-7.5
      l2_s = l2_min_index/100-7.5
      print(l2_s)
      env_s = env_min_index/100-7.5

      return was_s,l2_s,env_s

def plot_different_ricker() :
      
      from peer_reviewed.format_shift import Format
      Format.params()

      shift = np.arange(-7.5,7.5,0.01)
      
      name_list = ["l2","env","was"]

      for name in name_list:

            data0 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/ricker_{name}_0.npy")
            data5 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/ricker_{name}_5.npy")
            data10 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/ricker_{name}_10.npy")
            data15 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/ricker_{name}_15.npy")
            data20 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/ricker_{name}_20.npy")
            data25 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/ricker_{name}_25.npy")
            data30 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/ricker_{name}_30.npy")
            
            fontsize = 25

            sns.set(font_scale=2.5)
            sns.set_style('whitegrid')
            sns.set_palette('winter')

            df = pd.DataFrame({'30':data30,'25':data25,'20':data20,'15':data15,'10':data10,'5':data5,'0':data0})

            df_melt = pd.melt(df)

            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(1,1,1)
            sns.boxplot(x='value', y='variable', data=df_melt, showfliers=False, ax=ax,palette="gnuplot_r")
            sns.stripplot(x='value', y='variable', data=df_melt, jitter=True, color='black', ax=ax)

            ax.set_xlim(-7.5,7.5)
            ax.set_xlabel(r"$\tau$[s]",fontsize = fontsize)
            ax.set_ylabel("standard_deviation",fontsize=fontsize)

            fig.tight_layout()

            plt.savefig(f"./Python_Code/peer_reviewed/figure/optimal_time/ricker_{name}")
            plt.clf()
            plt.close()

##############

def main_synthetic1():

      tim = np.arange(0,20,0.01)

      wave1 = synthetic_ricker_left(tim)
      wave2 = synthetic_ricker_right(tim) 

      max_wave = np.max(wave1)
      wave1 = wave1*10/max_wave
      wave2 = wave2*10/max_wave

      std_list = [0,5,10,15,20,25,30]
      length = len(std_list)

      def process2(std) :

            was_s_list,l2_s_list,env_s_list = [],[],[]

            for _ in tqdm(range(100)):

                  noise1500 = pink_noise(std,1500)
                  noise2000_1 = pink_noise(std,2000)
                  noise2000_2 = pink_noise(std,2000)

                  noise1500_bandpass = vector.bandpass(noise1500,dt=0.01,low=0.5,high=10)
                  noise2000_1_bandpass = vector.bandpass(noise2000_1,dt=0.01,low=0.5,high=10)
                  noise2000_2_bandpass = vector.bandpass(noise2000_2,dt=0.01,low=0.5,high=10)
                  
                  wave_move = wave1 + noise2000_1_bandpass
                  wave_fix = wave2 + noise2000_2_bandpass

                  wave_move = np.hstack([noise1500_bandpass,wave_move])
                  
                  was_s,l2_s,env_s = shift_ricker(tim,wave_move,wave_fix)

                  was_s_list += [was_s]
                  l2_s_list += [l2_s]
                  env_s_list += [env_s]

            was_s_array = np.array(was_s_list)
            l2_s_array = np.array(l2_s_list)
            env_s_array = np.array(env_s_list)

            np.save(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic_was_{std}",was_s_array)
            np.save(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic_l2_{std}",l2_s_array)
            np.save(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic_env_{std}",env_s_array)        
      
      Parallel(n_jobs=-1)([delayed(process2)(std) for std in std_list])

def plot_different_synthetic1():

      from peer_reviewed.format_shift import Format
      Format.params()

      shift = np.arange(-7.5,7.5,0.01)
      
      name_list = ["l2","env","was"]

      for name in name_list:

            data0 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic_{name}_0.npy")
            data5 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic_{name}_5.npy")
            data10 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic_{name}_10.npy")
            data15 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic_{name}_15.npy")
            data20 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic_{name}_20.npy")
            data25 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic_{name}_25.npy")
            data30 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic_{name}_30.npy")
            
            fontsize = 25

            sns.set(font_scale=2.5)
            sns.set_style('whitegrid')
            sns.set_palette('gnuplot_r')

            df = pd.DataFrame({'30':data30,'25':data25,'20':data20,'15':data15,'10':data10,'5':data5,'0':data0})

            df_melt = pd.melt(df)

            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(1,1,1)
            sns.boxplot(x='value', y='variable', data=df_melt, showfliers=False, ax=ax,palette="gnuplot_r")
            sns.stripplot(x='value', y='variable', data=df_melt, jitter=True, color='black', ax=ax)

            ax.set_xlim(-7.5,7.5)
            ax.set_xlabel(r"$\tau$[s]",fontsize = fontsize)
            ax.set_ylabel("standard_deviation",fontsize=fontsize)

            fig.tight_layout()

            plt.savefig(f"./Python_Code/peer_reviewed/figure/optimal_time/synthetic_{name}")
            plt.clf()
            plt.close()

###############

def main_synthetic2():

      tim = np.arange(0,20,0.01)

      wave1 = synthetic_ricker_left(tim) - ricker(tim,0.5,2.5,10)
      wave2 = synthetic_ricker_right(tim) - ricker(tim,0.5,10,10)

      max_wave = np.max(wave1)
      wave1 = wave1*10/max_wave
      wave2 = wave2*10/max_wave

      std_list = [0,5,10,15,20,25,30]
      length = len(std_list)

      def process3(std) :

            was_s_list,l2_s_list,env_s_list = [],[],[]

            for _ in tqdm(range(100)):

                  noise1500 = pink_noise(std,1500)
                  noise2000_1 = pink_noise(std,2000)
                  noise2000_2 = pink_noise(std,2000)

                  noise1500_bandpass = vector.bandpass(noise1500,dt=0.01,low=0.1,high=10)
                  noise2000_1_bandpass = vector.bandpass(noise2000_1,dt=0.01,low=0.1,high=10)
                  noise2000_2_bandpass = vector.bandpass(noise2000_2,dt=0.01,low=0.1,high=10)
                  
                  wave_move = wave1 + noise2000_1_bandpass
                  wave_fix = wave2 + noise2000_2_bandpass

                  wave_move = np.hstack([noise1500_bandpass,wave_move])
                  
                  was_s,l2_s,env_s = shift_ricker(tim,wave_move,wave_fix)

                  was_s_list += [was_s]
                  l2_s_list += [l2_s]
                  env_s_list += [env_s]

            was_s_array = np.array(was_s_list)
            l2_s_array = np.array(l2_s_list)
            env_s_array = np.array(env_s_list)

            np.save(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic2_was_{std}",was_s_array)
            np.save(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic2_l2_{std}",l2_s_array)
            np.save(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic2_env_{std}",env_s_array)        
      
      Parallel(n_jobs=-1)([delayed(process3)(std) for std in std_list])

def plot_different_synthetic2():

      from peer_reviewed.format_shift import Format
      Format.params()

      shift = np.arange(-7.5,7.5,0.01)
      
      name_list = ["l2","env","was"]

      for name in name_list:

            data0 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic2_{name}_0.npy")
            data5 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic2_{name}_5.npy")
            data10 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic2_{name}_10.npy")
            data15 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic2_{name}_15.npy")
            data20 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic2_{name}_20.npy")
            data25 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic2_{name}_25.npy")
            data30 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/synthetic2_{name}_30.npy")

            fontsize = 25

            sns.set(font_scale=2.5)
            sns.set_style('whitegrid')
            sns.set_palette('gnuplot_r')

            df = pd.DataFrame({'30':data30,'25':data25,'20':data20,'15':data15,'10':data10,'5':data5,'0':data0})

            df_melt = pd.melt(df)

            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(1,1,1)
            sns.boxplot(x='value', y='variable', data=df_melt, showfliers=False, ax=ax,palette="gnuplot_r")
            sns.stripplot(x='value', y='variable', data=df_melt, jitter=True, color='black', ax=ax)

            ax.set_xlim(-7.5,7.5)
            ax.set_xlabel(r"$\tau$[s]",fontsize = fontsize)
            ax.set_ylabel("standard_deviation",fontsize=fontsize)

            fig.tight_layout()

            plt.savefig(f"./Python_Code/peer_reviewed/figure/optimal_time/synthetic2_{name}")
            plt.clf()
            plt.close()

###############

def main_fks():

      tim = np.arange(0,400,0.01)
      wave = np.loadtxt("./Python_Code/osaka_data/archive/FKS186I7.acc",
            usecols=(2),unpack=True)
      wave = wave[9500:19500]
      zeros15000 = np.zeros(15000)
      zeros30000 = np.zeros(30000)
      wave1 = np.hstack([zeros15000,wave,zeros15000])
      wave2 = np.hstack([wave,zeros30000])

      std_list = [0,40,80,120,160,200]

      def process4(std):

            was_s_list,l2_s_list,env_s_list = [],[],[]

            for _ in tqdm(range(10)) :

                  noise30000 = pink_noise(std,30000)
                  noise40000_1 = pink_noise(std,40000)
                  noise40000_2 = pink_noise(std,40000)

                  noise30000_bandpass = vector.bandpass(noise30000,dt=0.01,low=0.1,high=10)
                  noise40000_1_bandpass = vector.bandpass(noise40000_1,dt=0.01,low=0.1,high=10)
                  noise40000_2_bandpass = vector.bandpass(noise40000_2,dt=0.01,low=0.1,high=10)
                  
                  wave_move = wave2 + noise40000_1_bandpass
                  wave_fix = wave1 + noise40000_2_bandpass

                  wave_move = np.hstack([noise30000_bandpass,wave_move])
                  
                  was_s,l2_s,env_s = shift_fks(tim,wave_move,wave_fix)

                  was_s_list += [was_s]
                  l2_s_list += [l2_s]
                  env_s_list += [env_s]

            was_s_array = np.array(was_s_list)
            l2_s_array = np.array(l2_s_list)
            env_s_array = np.array(env_s_list)

            np.save(f"./Python_Code/peer_reviewed/value/optimal_time/fks10_was_{std}",was_s_array)
            np.save(f"./Python_Code/peer_reviewed/value/optimal_time/fks10_l2_{std}",l2_s_array)
            np.save(f"./Python_Code/peer_reviewed/value/optimal_time/fks10_env_{std}",env_s_array)    

      
      Parallel(n_jobs=-1)([delayed(process4)(std) for std in std_list])
        
def shift_fks(tim,wave_left,wave_fix):
           
      wave_fix_env = np.abs(signal.hilbert(wave_fix))

      shift_time = np.arange(0,300,0.01)
      was_list,l2_org_list,l2_env_list = [],[],[]
      for i,_ in enumerate(shift_time):

            wave_move = wave_left[30000-i:70000-i]
                  
            wave_move_env = np.abs(signal.hilbert(wave_move))
            sim_org = Similarity(tim,wave_move,wave_fix,"")
            was = sim_org.wasserstein_softplus_normalizing(con=3)
            l2_org = sim_org.mse()
            sim_env = Similarity(tim,wave_move_env,wave_fix_env,"")
            l2_env = sim_env.mse()

            was_list += [was]
            l2_org_list += [l2_org]
            l2_env_list += [l2_env]

      was_min_index = was_list.index(min(was_list))
      l2_min_index = l2_org_list.index(min(l2_org_list))
      env_min_index = l2_env_list.index(min(l2_env_list))

      was_s = was_min_index/100-150
      l2_s = l2_min_index/100-150
      env_s = env_min_index/100-150

      return was_s,l2_s,env_s

def plot_different_fks():

      from peer_reviewed.format_shift import Format
      Format.params()

      shift = np.arange(-150,150,0.01)
      
      name_list = ["l2","env","was"]

      for name in name_list:

            data0 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/fks_{name}_0.npy")
            data40 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/fks_{name}_40.npy")
            data80 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/fks_{name}_80.npy")
            data120 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/fks_{name}_120.npy")
            data160 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/fks_{name}_160.npy")
            data200 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/fks_{name}_200.npy")
            
            fontsize = 25

            sns.set(font_scale=2.5)
            sns.set_style('whitegrid')
            sns.set_palette('gnuplot_r')

            df = pd.DataFrame({'200':data200,'160':data160,'120':data120,'80':data80,'40':data40,'0':data0})

            df_melt = pd.melt(df)

            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(1,1,1)
            sns.boxplot(x='value', y='variable', data=df_melt, showfliers=False, ax=ax,palette="gnuplot_r")
            sns.stripplot(x='value', y='variable', data=df_melt, jitter=True, color='black', ax=ax)

            ax.set_xlim(-150,150)
            ax.set_xlabel(r"$\tau$[s]",fontsize = fontsize)
            ax.set_ylabel("standard_deviation",fontsize=fontsize)

            fig.tight_layout()

            plt.savefig(f"./Python_Code/peer_reviewed/figure/optimal_time/fks_{name}")
            plt.clf()
            plt.close()

def load_fks():
     
     name_list = ["l2","env","was"]
     std_list = ["0","40","80","120","160","200"]

     for name in name_list:
            for std in std_list :
                  
                  value1 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/fks1_{name}_{std}.npy")
                  value2 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/fks2_{name}_{std}.npy")
                  value3 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/fks3_{name}_{std}.npy")
                  value4 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/fks4_{name}_{std}.npy")
                  value5 = np.load(f"./Python_Code/peer_reviewed/value/optimal_time/fks5_{name}_{std}.npy")

                  array = np.hstack((value1,value2,value3,value4,value5))

                  np.save(f"./Python_Code/peer_reviewed/value/optimal_time/fks_{name}_{std}.npy",array)
                  print(array.shape)

plot_different_fks()
