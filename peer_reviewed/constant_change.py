import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
mpl.use("Agg")
import sys
sys.path.append('./Python_Code')
from others.similarity import Similarity
from PySGM.vector import vector
from tqdm import tqdm


# sns.set()
# sns.set_palette("winter",3)


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

def main_synthetic1():

      tim = np.arange(0,20,0.01)

      wave1 = synthetic_ricker_left(tim)
      wave2 = synthetic_ricker_right(tim) 

      std_list = [0,2]

      for std in tqdm(std_list) :

            noise1500 = np.random.normal(0.0,std,1500)
            noise2000_1 = np.random.normal(0.0,std,2000)
            noise2000_2 = np.random.normal(0.0,std,2000)

            noise1500_bandpass = vector.bandpass(noise1500,dt=0.01,low=0.5,high=10)
            noise2000_1_bandpass = vector.bandpass(noise2000_1,dt=0.01,low=0.5,high=10)
            noise2000_2_bandpass = vector.bandpass(noise2000_2,dt=0.01,low=0.5,high=10)
            
            wave_move = wave1 + noise2000_1_bandpass
            wave_fix = wave2 + noise2000_2_bandpass

            wave_move = np.hstack([noise1500_bandpass,wave_move])

            shift_synthetic1(tim,wave_move,wave_fix,std)

def shift_synthetic1(tim,wave_left,wave_fix,std):

      shift_time = np.arange(0,15,0.01)
      con_list = [1,2,3,4,5]
      for con in con_list:

            was_list = []
            for i,s in tqdm(enumerate(shift_time)):

                  wave_move = wave_left[1500-i:3500-i]
                        
                  sim_org = Similarity(tim,wave_move,wave_fix,"")
                  was = sim_org.wasserstein_softplus_normalizing(con)

                  was_list += [was]

            was_array = np.array(was_list)
            
            np.save(f"./Python_Code/peer_reviewed/value/constant_change/{std}_{con}.npy",was_array)

def plot_shift_synthetic1() :

      from peer_reviewed.format_shift import Format
      Format.params()

      shift = np.arange(0,15,0.01)
      
      std_list = [0,2]

      for std in std_list :
      
            fig,ax = plt.subplots()

            con_list = [1,2,3,4,5]
            length = len(con_list)

            for i,con in enumerate(con_list) :
                  value = np.load(f"./Python_Code/peer_reviewed/value/constant_change/{std}_{con}.npy") 
                  ax.plot(shift,value,color=cm.plasma(i/length),label=f"constant={con}")
            ax.set_xlabel(r"$s$[s]")
            ax.set_ylabel(r"$W_{2}^2$")
            ax.set_ylim(0,)
            ax.tick_params()
            ax.legend()
            fig.savefig(f"./Python_Code/peer_reviewed/figure/constant_change/{std}.png")
            plt.clf()
            plt.close()

plot_shift_synthetic1()