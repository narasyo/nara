import numpy as np
from scipy import signal
import math
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import joblib
from typing import Optional
import contextlib
from joblib import Parallel,delayed
sys.path.append('./Python_Code')
from PySGM.vector import vector
from others.similarity import Similarity
from tqdm.auto import tqdm


@contextlib.contextmanager
def tqdm_joblib(total: Optional[int] = None, **kwargs):

    pbar = tqdm(total=total, miniters=1, smoothing=0, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        pbar.close()

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

def main_ricker(): 

      tim = np.arange(0,20,0.01)

      wave1 = -ricker(tim,0.5,2.5,10)
      wave2 = -ricker(tim,0.5,10,10)

      std_list = [0,5,10,15,20,25,30]
      length = len(std_list)

      for i,std in tqdm(enumerate(std_list)) :


            noise1500 = pink_noise(std,1500)
            noise2000_1 = pink_noise(std,2000)
            noise2000_2 = pink_noise(std,2000)

            noise1500_bandpass = vector.bandpass(noise1500,dt=0.01,low=0.1,high=1.0)
            noise2000_1_bandpass = vector.bandpass(noise2000_1,dt=0.01,low=0.1,high=1.0)
            noise2000_2_bandpass = vector.bandpass(noise2000_2,dt=0.01,low=0.1,high=1.0)
            
            wave_move = wave1 + noise2000_1_bandpass
            wave_fix = wave2 + noise2000_2_bandpass

            wave_move = np.hstack([noise1500_bandpass,wave_move])
            
            color=cm.plasma(i/length)
            plot_ricker(tim,wave_move,wave_fix,std,color)
            shift_ricker(tim,wave_move,wave_fix,std)

def plot_ricker(tim,wave1,wave2,std,color):

      from peer_reviewed.format_wave import Format
      Format.params()

      wave1 = wave1[1500:]

      fig,ax = plt.subplots()
      fleft_env = np.abs(signal.hilbert(wave1))
      fright_env = np.abs(signal.hilbert(wave2))

      ax.plot(tim,wave1,color=color,label=r"$f(t-s)$",linestyle="dashed")
      ax.plot(tim,wave2,color=color,label=r"$f(t)$")

      ax.plot(tim,fleft_env,color=color,label=r"$f(t-s)$_envelope",alpha=0.3,linestyle="dashed")
      ax.plot(tim,fright_env,color=color,label=r"$f(t)$_envelope",alpha=0.3)
      ax.plot(tim,np.zeros_like(tim),color="black")
      ax.set_xlim(0,20)
      point = {'start': [3, 3],'end': [16, 3]}
      ax.annotate('', xy=point['end'], xytext=point['start'],
                  arrowprops=dict(shrink=0, width=1, headwidth=40, 
                                    headlength=40, connectionstyle='arc3',
                                    facecolor='gray', edgecolor='gray'))
      ax.set_xlabel(r"$t$[s]")
      ax.set_ylabel("amplitude")
      ax.tick_params()
      ax.legend()
      plt.tight_layout()

      fig.savefig(f"./Python_Code/peer_reviewed/figure/real_noise3/ricker_{std}.png")
      plt.clf()
      plt.close()

      """

      class Format:
            def params():

                  ## Figureの設定 ##
                  plt.rcParams['figure.figsize'] = (35, 8)  # figure size in inch, 横×縦
                  plt.rcParams['figure.dpi'] = 300

                  ## フォントの種類 ##
                  rcParams['font.family'] = 'sans-serif'
                  rcParams['font.sans-serif'] = ["ヒラギノ丸ゴ ProN W4, 16"]
                  plt.rcParams['font.family'] = 'Times New Roman'  # font familyの設定
                  plt.rcParams['mathtext.fontset'] = 'stix'  # math fontの設定
                  
                  ## Axesの設定 ##
                  plt.rcParams["font.size"] = 40  # 全体のフォントサイズが変更されます
                  plt.rcParams['xtick.labelsize'] = 40  # 軸だけ変更されます
                  plt.rcParams['ytick.labelsize'] = 40  # 軸だけ変更されます
                  plt.rcParams['xtick.direction'] = 'in'  # x axis in
                  plt.rcParams['ytick.direction'] = 'in'  # y axis in
                  plt.rcParams['xtick.major.width'] = 3.0  # x軸主目盛り線の線幅
                  plt.rcParams['ytick.major.width'] = 3.0  # y軸主目盛り線の線幅
                  plt.rcParams['xtick.major.size'] = 20  #x軸目盛り線の長さ　
                  plt.rcParams['ytick.major.size'] = 20  #y軸目盛り線の長さ
                  plt.rcParams['axes.linewidth'] = 5  # axis line width
                  plt.rcParams['axes.grid'] = False # make grid
                  plt.rcParams['lines.linewidth'] = 4 #グラフの線の太さ

                  ## 凡例の設定 ##
                  plt.rcParams["legend.loc"] = 'upper right'
                  plt.rcParams["legend.fancybox"] = False  # 丸角
                  plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
                  plt.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更
                  plt.rcParams["legend.handlelength"] = 1  # 凡例の線の長さを調節
                  plt.rcParams["legend.labelspacing"] = 0.3  # 垂直（縦）方向の距離の各凡例の距離
                  plt.rcParams["legend.handletextpad"] = 1.  # 凡例の線と文字の距離の長さ
                  plt.rcParams["legend.markerscale"] = 1.  # 点がある場合のmarker scale
                  plt.rcParams["legend.borderaxespad"] = 1  # 凡例の端とグラフの端を合わせる
                  return

      """

def shift_ricker(tim,wave_left,wave_fix,std):
           
      wave_fix_env = np.abs(signal.hilbert(wave_fix))

      shift_time = np.arange(0,15,0.01)
      was_list,l2_org_list,l2_env_list = [],[],[]
      for i,s in tqdm(enumerate(shift_time)):

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

      was_array = np.array(was_list)
      l2_org_array = np.array(l2_org_list)
      l2_env_array = np.array(l2_env_list)
      
      np.save(f"./Python_Code/peer_reviewed/value/real_noise3/ricker_was_{std}.npy",was_array)
      np.save(f"./Python_Code/peer_reviewed/value/real_noise3/ricker_l2_{std}.npy",l2_org_array)
      np.save(f"./Python_Code/peer_reviewed/value/real_noise3/ricker_l2_env_{std}.npy",l2_env_array)

def plot_shift_ricker() :
      
      from peer_reviewed.format_shift import Format
      Format.params()

      shift = np.arange(-7.5,7.5,0.01)
      
      name1_list = ["l2","l2_env","was"]
      name2_list = ["MSE","MSE(envelope)","$W_{2}^2$"]
      loc_list = ["upper right","upper right","upper right"]

      for name1,name2,loc in zip(name1_list,name2_list,loc_list):
      
            fig,ax = plt.subplots()

            std_list = [0,5,10,15,20,25,30]
            length = len(std_list)

            for i,std in enumerate(std_list) :
                  value = np.load(f"./Python_Code/peer_reviewed/value/real_noise3/ricker_{name1}_{std}.npy") 
                  ax.plot(shift,value,color=cm.plasma(i/length),label=f"standard_deviation={std}")
            ax.set_xlabel(r"$s$[s]")
            ax.set_ylabel(fr"{name2}")
            ax.set_ylim(0,)
            ax.tick_params()
            ax.legend(loc=f"{loc}")
            fig.savefig(f"./Python_Code/peer_reviewed/figure/real_noise3/ricker_{name1}.png")
            plt.clf()
            plt.close()

def was_ricker():

      from peer_reviewed.format_shift import Format
      Format.params()

      shift = np.arange(-7.5,7.5,0.01)
      
      fig,ax = plt.subplots()

      std_list = [0,5,10,15,20,25,30]
      length = len(std_list)

      for i,std in enumerate(std_list) :
            value = np.load(f"./Python_Code/peer_reviewed/value/real_noise3/ricker_was_{std}.npy") 
            ax.plot(shift,value,color=cm.plasma(i/length),label=f"standard_deviation={std}")
      ax.set_xlabel(r"$s$[s]")
      ax.set_ylabel(r"$W_{2}^2$")
      ax.set_ylim(0,1)
      ax.tick_params()
      ax.legend()
      fig.savefig(f"./Python_Code/peer_reviewed/figure/real_noise3/ricker_was_part.png")
      plt.clf()
      plt.close()


#####

def main_synthetic1():

      tim = np.arange(0,20,0.01)

      wave1 = synthetic_ricker_left(tim)
      wave2 = synthetic_ricker_right(tim) 

      max_wave = np.max(wave1)
      wave1 = wave1*10/max_wave
      wave2 = wave2*10/max_wave

      std_list = [0,5,10,15,20,25,30]
      length = len(std_list)

      for i,std in tqdm(enumerate(std_list)) :

            noise1500 = pink_noise(std,1500)
            noise2000_1 = pink_noise(std,2000)
            noise2000_2 = pink_noise(std,2000)

            noise1500_bandpass = vector.bandpass(noise1500,dt=0.01,low=0.5,high=10)
            noise2000_1_bandpass = vector.bandpass(noise2000_1,dt=0.01,low=0.5,high=10)
            noise2000_2_bandpass = vector.bandpass(noise2000_2,dt=0.01,low=0.5,high=10)
            
            wave_move = wave1 + noise2000_1_bandpass
            wave_fix = wave2 + noise2000_2_bandpass

            wave_move = np.hstack([noise1500_bandpass,wave_move])

            color=cm.plasma(i/length)
            plot_synthetic1(tim,wave_move,wave_fix,std,color)
            shift_synthetic1(tim,wave_move,wave_fix,std)

def plot_synthetic1(tim,wave1,wave2,std,color):
      
      from peer_reviewed.format_wave import Format
      Format.params()

      wave1 = wave1[1500:]

      fig,ax = plt.subplots()
      fleft_env = np.abs(signal.hilbert(wave1))
      fright_env = np.abs(signal.hilbert(wave2))

      ax.plot(tim,wave1,color=color,label=r"$f(t-s)$",linestyle="dashed")
      ax.plot(tim,wave2,color=color,label=r"$f(t)$")

      ax.plot(tim,fleft_env,color=color,label=r"$f(t-s)$_envelope",alpha=0.3,linestyle="dashed")
      ax.plot(tim,fright_env,color=color,label=r"$f(t)$_envelope",alpha=0.3)
      ax.plot(tim,np.zeros_like(tim),color="black")
      ax.set_xlim(0,20)
      point = {'start': [3, 3],'end': [16, 3]}
      ax.annotate('', xy=point['end'], xytext=point['start'],
                  arrowprops=dict(shrink=0, width=1, headwidth=40, 
                                    headlength=40, connectionstyle='arc3',
                                    facecolor='gray', edgecolor='gray'))
      ax.set_xlabel(r"$t$[s]")
      ax.set_ylabel("amplitude")
      ax.tick_params()
      ax.legend()
      plt.tight_layout()

      fig.savefig(f"./Python_Code/peer_reviewed/figure/real_noise3/synthetic_ricker_{std}.png")
      plt.clf()
      plt.close()

def shift_synthetic1(tim,wave_left,wave_fix,std):

      wave_fix_env = np.abs(signal.hilbert(wave_fix))

      shift_time = np.arange(0,15,0.01)
      was_list,l2_org_list,l2_env_list = [],[],[]
      for i,s in tqdm(enumerate(shift_time)):

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

      was_array = np.array(was_list)
      l2_org_array = np.array(l2_org_list)
      l2_env_array = np.array(l2_env_list)
      
      np.save(f"./Python_Code/peer_reviewed/value/real_noise3/synthetic_was_{std}.npy",was_array)
      np.save(f"./Python_Code/peer_reviewed/value/real_noise3/synthetic_l2_{std}.npy",l2_org_array)
      np.save(f"./Python_Code/peer_reviewed/value/real_noise3/synthetic_l2_env_{std}.npy",l2_env_array)

def plot_shift_synthetic1() :

      from peer_reviewed.format_shift import Format
      Format.params()

      shift = np.arange(-7.5,7.5,0.01)
      
      name1_list = ["l2","l2_env","was"]
      name2_list = ["MSE","MSE(envelope)","$W_{2}^2$"]
      loc_list = ["lower right","lower right","upper center"]

      for name1,name2,loc in zip(name1_list,name2_list,loc_list):
      
            fig,ax = plt.subplots()

            std_list = [0,5,10,15,20,25,30]
            length = len(std_list)

            for i,std in enumerate(std_list) :
                  value = np.load(f"./Python_Code/peer_reviewed/value/real_noise3/synthetic_{name1}_{std}.npy") 
                  ax.plot(shift,value,color=cm.plasma(i/length),label=f"standard_deviation={std}")
            ax.set_xlabel(r"$s$[s]")
            ax.set_ylabel(fr"{name2}")
            ax.set_ylim(0,)
            ax.tick_params()
            ax.legend(loc=f"{loc}")
            fig.savefig(f"./Python_Code/peer_reviewed/figure/real_noise3/synthetic_{name1}.png")
            plt.clf()
            plt.close()

            """
 
            class Format:
            def params():

                  ## Figureの設定 ##
                  plt.rcParams['figure.figsize'] = (40, 32)  # figure size in inch, 横×縦
                  plt.rcParams['figure.dpi'] = 300

                  ## フォントの種類 ##
                  rcParams['font.family'] = 'sans-serif'
                  rcParams['font.sans-serif'] = ["ヒラギノ丸ゴ ProN W4, 16"]
                  plt.rcParams['font.family'] = 'Times New Roman'  # font familyの設定
                  plt.rcParams['mathtext.fontset'] = 'stix'  # math fontの設定
                  
                  ## Axesの設定 ##
                  plt.rcParams["font.size"] = 60  # 全体のフォントサイズが変更されます
                  plt.rcParams['xtick.labelsize'] = 60  # 軸だけ変更されます
                  plt.rcParams['ytick.labelsize'] = 60  # 軸だけ変更されます
                  plt.rcParams['xtick.direction'] = 'in'  # x axis in
                  plt.rcParams['ytick.direction'] = 'in'  # y axis in
                  plt.rcParams['xtick.major.width'] = 3.0  # x軸主目盛り線の線幅
                  plt.rcParams['ytick.major.width'] = 3.0  # y軸主目盛り線の線幅
                  plt.rcParams['xtick.major.size'] = 20  #x軸目盛り線の長さ　
                  plt.rcParams['ytick.major.size'] = 20  #y軸目盛り線の長さ
                  plt.rcParams['axes.linewidth'] = 5  # axis line width
                  plt.rcParams['axes.grid'] = False # make grid
                  plt.rcParams['lines.linewidth'] = 4 #グラフの線の太さ

                  ## 凡例の設定 ##
                  plt.rcParams["legend.loc"] = 'upper right'
                  plt.rcParams["legend.fancybox"] = False  # 丸角
                  plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
                  plt.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更
                  plt.rcParams["legend.handlelength"] = 1  # 凡例の線の長さを調節
                  plt.rcParams["legend.labelspacing"] = 0.3  # 垂直（縦）方向の距離の各凡例の距離
                  plt.rcParams["legend.handletextpad"] = 1.  # 凡例の線と文字の距離の長さ
                  plt.rcParams["legend.markerscale"] = 1.  # 点がある場合のmarker scale
                  plt.rcParams["legend.borderaxespad"] = 1  # 凡例の端とグラフの端を合わせる
                  plt.rcParams['legend.fontsize'] = "medium"
                  return
            
            """

####

def main_synthetic2():

      tim = np.arange(0,20,0.01)

      wave1 = synthetic_ricker_left(tim) - ricker(tim,0.5,2.5,10)
      wave2 = synthetic_ricker_right(tim) - ricker(tim,0.5,10,10)

      max_wave = np.max(wave1)
      wave1 = wave1*10/max_wave
      wave2 = wave2*10/max_wave

      std_list = [0,5,10,15,20,25,30]
      length = len(std_list)

      for i,std in tqdm(enumerate(std_list)) :

            noise1500 = pink_noise(std,1500)
            noise2000_1 = pink_noise(std,2000)
            noise2000_2 = pink_noise(std,2000)

            noise1500_bandpass = vector.bandpass(noise1500,dt=0.01,low=0.1,high=10)
            noise2000_1_bandpass = vector.bandpass(noise2000_1,dt=0.01,low=0.1,high=10)
            noise2000_2_bandpass = vector.bandpass(noise2000_2,dt=0.01,low=0.1,high=10)
            
            wave_move = wave1 + noise2000_1_bandpass
            wave_fix = wave2 + noise2000_2_bandpass

            wave_move = np.hstack([noise1500_bandpass,wave_move])

            color =cm.plasma(i/length)
            plot_synthetic2(tim,wave_move,wave_fix,std,color)
            shift_synthetic2(tim,wave_move,wave_fix,std)

def plot_synthetic2(tim,wave1,wave2,std,color):

      from peer_reviewed.format_wave import Format
      Format.params()

      wave1 = wave1[1500:]

      fig,ax = plt.subplots()
      fleft_env = np.abs(signal.hilbert(wave1))
      fright_env = np.abs(signal.hilbert(wave2))

      ax.plot(tim,wave1,color=color,label=r"$f(t-s)$",linestyle="dashed")
      ax.plot(tim,wave2,color=color,label=r"$f(t)$")

      ax.plot(tim,fleft_env,color=color,label=r"$f(t-s)$_envelope",alpha=0.3,linestyle="dashed")
      ax.plot(tim,fright_env,color=color,label=r"$f(t)$_envelope",alpha=0.3)
      ax.plot(tim,np.zeros_like(tim),color="black")
      ax.set_xlim(0,20)
      point = {'start': [3, 3],'end': [16, 3]}
      ax.annotate('', xy=point['end'], xytext=point['start'],
                  arrowprops=dict(shrink=0, width=1, headwidth=40, 
                                    headlength=40, connectionstyle='arc3',
                                    facecolor='gray', edgecolor='gray'))
      ax.set_xlabel(r"$t$[s]")
      ax.set_ylabel("amplitude")
      ax.tick_params()
      ax.legend()
      plt.tight_layout()

      fig.savefig(f"./Python_Code/peer_reviewed/figure/real_noise3/synthetic2_ricker_{std}.png")
      plt.clf()
      plt.close()

def shift_synthetic2(tim,wave_left,wave_fix,std):

      wave_fix_env = np.abs(signal.hilbert(wave_fix))

      shift_time = np.arange(0,15,0.01)
      was_list,l2_org_list,l2_env_list = [],[],[]
      for i,s in tqdm(enumerate(shift_time)):

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

      was_array = np.array(was_list)
      l2_org_array = np.array(l2_org_list)
      l2_env_array = np.array(l2_env_list)
      
      np.save(f"./Python_Code/peer_reviewed/value/real_noise3/synthetic2_was_{std}.npy",was_array)
      np.save(f"./Python_Code/peer_reviewed/value/real_noise3/synthetic2_l2_{std}.npy",l2_org_array)
      np.save(f"./Python_Code/peer_reviewed/value/real_noise3/synthetic2_l2_env_{std}.npy",l2_env_array)

def plot_shift_synthetic2() :

      from peer_reviewed.format_shift import Format
      Format.params()

      shift = np.arange(-7.5,7.5,0.01)
      
      name1_list = ["l2","l2_env","was"]
      name2_list = ["MSE","MSE(envelope)","$W_{2}^2$"]
      loc_list = ["lower right","lower right","upper center"]

      for name1,name2,loc in zip(name1_list,name2_list,loc_list):
      
            fig,ax = plt.subplots()

            std_list = [0,5,10,15,20,25,30]
            length = len(std_list)

            for i,std in enumerate(std_list) :
                  value = np.load(f"./Python_Code/peer_reviewed/value/real_noise3/synthetic2_{name1}_{std}.npy") 
                  ax.plot(shift,value,color=cm.plasma(i/length),label=f"standard_deviation={std}")
            ax.set_xlabel(r"$s$[s]")
            ax.set_ylabel(fr"{name2}")
            ax.set_ylim(0,)
            ax.tick_params()
            ax.legend(loc=f"{loc}")
            fig.savefig(f"./Python_Code/peer_reviewed/figure/real_noise3/synthetic2_{name1}.png")
            plt.clf()
            plt.close()

####

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
      length = len(std_list)

      def process(i,std):

            noise30000 = pink_noise(std,30000)
            noise40000_1 = pink_noise(std,40000)
            noise40000_2 = pink_noise(std,40000)

            noise30000_bandpass = vector.bandpass(noise30000,dt=0.01,low=0.1,high=10)
            noise40000_1_bandpass = vector.bandpass(noise40000_1,dt=0.01,low=0.1,high=10)
            noise40000_2_bandpass = vector.bandpass(noise40000_2,dt=0.01,low=0.1,high=10)

            wave_move = wave2 + noise40000_1_bandpass
            wave_fix = wave1 + noise40000_2_bandpass

            wave_move = np.hstack([noise30000_bandpass,wave_move])
            
            color=cm.plasma(i/length)

            plot_fks(tim,wave_move,wave_fix,std,color)
            shift_fks(tim,wave_move,wave_fix,std)
      
      Parallel(n_jobs=-1)([delayed(process)(i,std) for i,std in enumerate(std_list)])

def shift_fks(tim,wave_left,wave_fix,std):
           
      wave_fix_env = np.abs(signal.hilbert(wave_fix))

      shift_time = np.arange(0,300,0.01)
      was_list,l2_org_list,l2_env_list = [],[],[]
      for i,s in tqdm(enumerate(shift_time)):

            wave_move = wave_left[30000-i:70000-i]
                  
            wave_move_env = np.abs(signal.hilbert(wave_move))
            sim_org = Similarity(tim,wave_move,wave_fix,"")
            was = sim_org.wasserstein_softplus_normalizing(con=3.0)
            l2_org = sim_org.mse()
            sim_env = Similarity(tim,wave_move_env,wave_fix_env,"")
            l2_env = sim_env.mse()

            was_list += [was]
            l2_org_list += [l2_org]
            l2_env_list += [l2_env]

      was_array = np.array(was_list)
      l2_org_array = np.array(l2_org_list)
      l2_env_array = np.array(l2_env_list)
      
      np.save(f"./Python_Code/peer_reviewed/value/real_noise3/fks_was_{std}.npy",was_array)
      np.save(f"./Python_Code/peer_reviewed/value/real_noise3/fks_l2_{std}.npy",l2_org_array)
      np.save(f"./Python_Code/peer_reviewed/value/real_noise3/fks_l2_env_{std}.npy",l2_env_array)

def plot_fks(tim,wave1,wave2,std,color):

      from peer_reviewed.format_wave import Format
      Format.params()

      wave1 = wave1[30000:]

      fig,ax = plt.subplots()
      fleft_env = np.abs(signal.hilbert(wave1))
      fright_env = np.abs(signal.hilbert(wave2))

      ax.plot(tim,wave1,color=color,label=r"$f(t-s)$",linestyle="dashed")
      ax.plot(tim,wave2,color=color,label=r"$f(t)$",)

      ax.plot(tim,fleft_env,color=color,label=r"$f(t-s)$_envelope",alpha=0.3,linestyle="dashed")
      ax.plot(tim,fright_env,color=color,label=r"$f(t)$_envelope",alpha=0.3)
      ax.plot(tim,np.zeros_like(tim),color="black")
      point = {'start': [40, 75],'end': [300, 75]}
      ax.annotate('', xy=point['end'], xytext=point['start'],
                  arrowprops=dict(shrink=0, width=1, headwidth=40, 
                                    headlength=40, connectionstyle='arc3',
                                    facecolor='gray', edgecolor='gray'))
      ax.set_xlabel(r"$t$[s]")
      ax.set_ylabel("amplitude")
      ax.set_xlim(0,400)
      ax.tick_params()
      ax.legend()
      plt.tight_layout()

      fig.savefig(f"./Python_Code/peer_reviewed/figure/real_noise3/fks_{std}.png")
      plt.clf()
      plt.close()

def plot_shift_fks() :

      from peer_reviewed.format_shift import Format
      Format.params()

      shift = np.arange(-150,150,0.01)
      name1_list = ["l2","l2_env","was"]
      name2_list = ["MSE","MSE(envelope)","$W_{2}^2$"]
      loc_list = ["upper right","lower right","upper center"]

      for name1,name2,loc in zip(name1_list,name2_list,loc_list):
      
            fig,ax = plt.subplots()

            std_list = [0,40,80,120,160,200]
            length = len(std_list)

            for i,std in enumerate(std_list) :
                  value = np.load(f"./Python_Code/peer_reviewed/value/real_noise3/fks_{name1}_{std}.npy") 
                  ax.plot(shift,value,color=cm.plasma(i/length),label=f"standard_deviation={std}")
            ax.set_xlabel(r"$s$[s]")
            ax.set_ylabel(f"{name2}")
            ax.tick_params()
            ax.legend(loc=loc)
            fig.savefig(f"./Python_Code/peer_reviewed/figure/real_noise3/fks_{name1}.png")
            plt.clf()
            plt.close()

plot_shift_synthetic2()

