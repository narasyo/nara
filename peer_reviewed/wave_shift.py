import numpy as np
from scipy import signal
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.append('./Python_Code')
from PySGM.vector import vector
from others.similarity import Similarity
from tqdm import tqdm

def ricker(tim,fp,tp,amp):
      
      t1 = ((tim-tp)*np.pi*fp)**2

      return (2*t1-1)*np.exp(-t1)*amp

def synthetic_ricker_left(tim,tp):
      
      f1 = ricker(tim,0.6,tp-3.5,15)
      f2 = -ricker(tim,0.8,tp-3.25,25)
      f3 = ricker(tim,1,tp-3,40)
      f4 = -ricker(tim,0.8,tp-2.75,25)
      f5 = ricker(tim,0.6,tp-2.5,15)

      fleft = f1+f2+f3+f4+f5

      f6 = -ricker(tim,0.6,tp-0.5,50)
      f7 = ricker(tim,0.8,tp-0.25,70)
      f8 = -ricker(tim,1,tp,120)
      f9 = ricker(tim,0.8,tp+0.25,70)
      f10 = -ricker(tim,0.6,tp+0.5,50)

      fcenter = f6+f7+f8+f9+f10
      
      f11 = ricker(tim,0.6,tp+2.5,15)
      f12 = -ricker(tim,0.8,tp+2.75,25)
      f13 = ricker(tim,1,tp+3,40)
      f14 = -ricker(tim,0.8,tp+3.25,25)
      f15 = ricker(tim,0.6,tp+3.5,15)

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

def shift_ricker():
      
      tim = np.arange(0,40,0.01)
      wave_fix = -ricker(tim,0.2,20,4)
      shift_time = np.arange(0,30,0.01)
      was_list,l2_list = [],[]
      for s in tqdm(shift_time):
            wave_move = -ricker(tim,0.2,5+s,4)
            sim = Similarity(tim,wave_move,wave_fix,"")
            was = sim.wasserstein_softplus_normalizing()
            l2 = sim.mse()
            was_list += [was]
            l2_list += [l2]
      
      was_array = np.array(was_list)
      l2_array = np.array(l2_list)
      np.save("./Python_Code/peer_reviewed/value/was_ricker.npy",was_array)
      np.save("./Python_Code/peer_reviewed/value/was_l2.npy",l2_array)

def plot_shift1(was,l2):

      shift_time = np.arange(0,30,0.01)
      fig,ax = plt.subplots(figsize=(10,8))
      ax.plot(shift_time,was,color="limegreen")
      ax.set_xlabel(r"$s$[s]",size=20)
      ax.set_ylabel("wasserstein metric",size=20)
      ax.tick_params(direction = "in", length = 10, labelsize=20)
      fig.savefig("./Python_Code/peer_reviewed/figure/was_synthetic_ricker.png")
      plt.clf()
      plt.close()

      fig,ax = plt.subplots(figsize=(10,8))
      ax.plot(shift_time,l2,color="black")
      ax.set_xlabel(r"$s$[s]",size=20)
      ax.set_ylabel("MSE",size=20)
      ax.tick_params(direction = "in", length = 10, labelsize=20)
      fig.savefig("./Python_Code/peer_reviewed/figure/l2_synthetic_ricker_env.png")
      plt.clf()
      plt.close()

def shift_synthetic_ricker():

      tim = np.arange(0,40,0.01)
      wave_fix = synthetic_ricker_right(tim)
      wave_fix_env = np.abs(signal.hilbert(wave_fix))
      shift_time = np.arange(0,30,0.01)
      was_list,l2_org_list,l2_env_list = [],[],[]
      for s in tqdm(shift_time):
            wave_move = synthetic_ricker_left(tim,5+s)
            wave_move_env = np.abs(signal.hilbert(wave_move))
            sim_org = Similarity(tim,wave_move,wave_fix,"")
            was = sim_org.wasserstein_softplus_normalizing()
            l2_org = sim_org.mse()
            sim_env = Similarity(tim,wave_move_env,wave_fix_env,"")
            l2_env = sim_env.mse()

            was_list += [was]
            l2_org_list += [l2_org]
            l2_env_list += [l2_env]

      was_array = np.array(was_list)
      l2_org_array = np.array(l2_org_list)
      l2_env_array = np.array(l2_env_list)

      np.save("./Python_Code/peer_reviewed/value/was_synthetic_ricker.npy",was_array)
      np.save("./Python_Code/peer_reviewed/value/l2_synthetic_ricker_org.npy",l2_org_array)
      np.save("./Python_Code/peer_reviewed/value/l2_synthetic_ricker_env.npy",l2_env_array)

def miyake_residual(dis1,dis2,acc_env1,acc_env2):

      dis_sum = np.sum((dis1-dis2)**2)
      dis1_norm = np.sqrt(np.sum(dis1**2))
      dis2_norm = np.sqrt(np.sum(dis2**2))

      env_sum = np.sum((acc_env1-acc_env2)**2)
      env1_norm = np.sqrt(np.sum(acc_env1**2))
      env2_norm = np.sqrt(np.sum(acc_env2**2))

      return dis_sum/(dis1_norm*dis2_norm)+env_sum/(env1_norm*env2_norm)

def shift_fks():
      
      tim = np.arange(0,400,0.01)
      wave_ew_acc = np.loadtxt("./Python_Code/osaka_data/archive/FKS186I7.acc",
            usecols=(2),unpack=True)     
      wave_ew_acc = wave_ew_acc[9500:19500]
      zeros15000 = np.zeros(15000)
      wave_fix = np.hstack([zeros15000,wave_ew_acc,zeros15000])
      wave_fix_env = np.abs(signal.hilbert(wave_fix))

      wave_ew_vel = np.loadtxt("./Python_Code/osaka_data/archive/FKS186I7.vel",usecols=(2),unpack=True)
      wave_ew_dis = vector.integration(wave_ew_vel,dt=0.01,low=0.1,high=10)
      wave_ew_dis = wave_ew_dis[9500:19500] 
      wave_dis_fix = np.hstack([zeros15000,wave_ew_dis,zeros15000])


      shift_time = np.arange(0,300,0.01)
      was_list,l2_org_list,l2_env_list,res_list = [],[],[],[]
      for i,s in tqdm(enumerate(shift_time)):
            zeros_left = np.zeros(i)
            zeros_right = np.zeros(30000-i)
            wave_move = np.hstack([zeros_left,wave_ew_acc,zeros_right])
            wave_move_env = np.abs(signal.hilbert(wave_move))
            wave_dis_move = np.hstack([zeros_left,wave_ew_dis,zeros_right])
            sim_org = Similarity(tim,wave_move,wave_fix,"")
            was = sim_org.wasserstein_softplus_normalizing()
            l2_org = sim_org.mse()
            sim_env = Similarity(tim,wave_move_env,wave_fix_env,"")
            l2_env = sim_env.mse()

            res = miyake_residual(wave_dis_move,wave_dis_fix,wave_move_env,wave_fix_env)

            was_list += [was]
            l2_org_list += [l2_org]
            l2_env_list += [l2_env]
            res_list += [res]

      was_array = np.array(was_list)
      l2_org_array = np.array(l2_org_list)
      l2_env_array = np.array(l2_env_list)
      res_array = np.array(res_list)
      
      np.save("./Python_Code/peer_reviewed/value/was_fks.npy",was_array)
      np.save("./Python_Code/peer_reviewed/value/l2_fks_org.npy",l2_org_array)
      np.save("./Python_Code/peer_reviewed/value/l2_fks_env.npy",l2_env_array)
      np.save("./Python_Code/peer_reviewed/value/residual_fks.npy",res_array)

def plot_shift(value):

      shift_time = np.arange(0,300,0.01)
      fig,ax = plt.subplots(figsize=(10,8))
      ax.plot(shift_time,value,color="black")
      ax.set_xlabel(r"$s$[s]",size=20)
      ax.set_ylabel("wasserstein metric",size=20)
      ax.tick_params(direction = "in", length = 10, labelsize=20)
      fig.savefig("./Python_Code/peer_reviewed/figure/residual_fks.png")
      plt.clf()
      plt.close()

was_array = np.load("./Python_Code/peer_reviewed/value/residual_fks.npy")
plot_shift(was_array)