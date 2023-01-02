import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fractions import Fraction
from matplotlib.colors import ListedColormap
import matplotlib.patches as mp
from scipy import signal
from joblib import Parallel,delayed
import sys
sys.path.append('./Python_Code')
from PySGM.vector import vector
from others.similarity import Similarity
import scipy.optimize as op

def baseline_correct(wave) :

      ave = np.average(wave)
      new_wave = wave - ave

      return new_wave

def wave_acc_load(seismometer): #EW成分#

      if seismometer == "OSA" :
            acc_A = np.load("./Python_Code/peer_reviewed/value/semblance_analysis/acc_A.npy")
            return acc_A
      
      elif seismometer == "OSB" :
            acc_B = np.load("./Python_Code/peer_reviewed/value/semblance_analysis/acc_B.npy")
            return acc_B

      elif seismometer == "OSC" :
            acc_C = np.load("./Python_Code/peer_reviewed/value/semblance_analysis/acc_C.npy")
            return acc_C

      elif seismometer == "FKS" :
            acc_f = np.load("./Python_Code/peer_reviewed/value/semblance_analysis/acc_f.npy")
            return acc_f[6000:30000]

def wave_vel_load(seismometer):

      if seismometer == "OSA" :
            vel_A = np.load("./Python_Code/osaka_data/archive/osaka_wave/OSA_vel_ud.npy")
            return vel_A
      
      elif seismometer == "OSB" :
            vel_B = np.load("./Python_Code/osaka_data/archive/osaka_wave/OSB_vel_ud.npy")
            return vel_B

      elif seismometer == "OSC" :
            vel_C = np.load("./Python_Code/osaka_data/archive/osaka_wave/OSC_vel_ud.npy")
            return vel_C

      elif seismometer == "FKS" :
            vel_f = np.load("./Python_Code/osaka_data/archive/osaka_wave/FKS_vel_ud.npy")
            return vel_f

def vel_to_dis(seismometer):

      if seismometer == "OSA" :
            dis_A = np.load("./Python_Code/peer_reviewed/value/semblance_analysis/dis_A.npy")
            return dis_A
      
      elif seismometer == "OSB" :
            dis_B = np.load("./Python_Code/peer_reviewed/value/semblance_analysis/dis_B.npy")
            return dis_B

      elif seismometer == "OSC" :
            dis_C = np.load("./Python_Code/peer_reviewed/value/semblance_analysis/dis_C.npy")
            return dis_C

      elif seismometer == "FKS" :
            dis_f = np.load("./Python_Code/peer_reviewed/value/semblance_analysis/dis_f.npy")
            return dis_f[6000:]

def wave_select(seismometer):

      wave = wave_vel_load(seismometer)
      return wave

def angle(X,Y) :

    cos_fai = X/np.sqrt(X**2+Y**2)

    if Y >= 0 :
        ang = np.rad2deg(np.arccos(cos_fai))

    elif Y < 0 :
        ang = 360 - np.rad2deg(np.arccos(cos_fai))

    return ang

def length(x,y,ang) :

    len = np.abs(x*math.cos(math.radians(ang))+y*math.sin(math.radians(ang)))

    return len

def perpendicular_foot(x,y,ang) :

    x0 = (x*math.cos(math.radians(ang))+y*math.sin(math.radians(ang)))*(math.cos(math.radians(ang)))
    y0 = (x*math.cos(math.radians(ang))+y*math.sin(math.radians(ang)))*(math.sin(math.radians(ang)))

    return x0,y0

def time_lag(x,y,c,ang,len) :

    time = len/c

    if 0<=ang<90 or ang == 360 :
        if x>=0 :
            lag = time
        elif x<0 :
            lag = -time
    
    elif ang==90 :
        if y>=0 :
            lag = time
        elif y<0 :
            lag = -time

    elif 90<ang<270 :
        if x<=0 :
            lag = time
        elif x>0 :
            lag = -time 

    elif ang==270 :
        if y<=0 :
            lag = time
        elif y>0 :
            lag = -time
    
    elif 270<ang<360 :
        if x>=0 :
            lag = time
        elif x<0 :
            lag = -time

    return lag 

def wasserstein(x_list,y_list,seismometer,number_list,coo_dict,left,right,time_list) :

    was_mean_list_list = []
    for Y in tqdm(y_list) :
        was_mean_list = []
        for X in x_list :
            was_list = []
            for s1,n in zip(seismometer,number_list) :
                seismometer2 = seismometer[n:]
                s1_str = str(s1)
                for s2 in seismometer2 :
                        s2_str = str(s2)
                        x1 = coo_dict[f"{s1_str}_x"]
                        y1 = coo_dict[f"{s1_str}_y"]
                        x2 = coo_dict[f"{s2_str}_x"]
                        y2 = coo_dict[f"{s2_str}_y"]

                        ang = angle(X,Y)
                        vel = 1/(np.sqrt(X**2+Y**2))

                        len1 = length(x1,y1,ang)
                        len2 = length(x2,y2,ang)

                        x01,y01 = perpendicular_foot(x1,y1,ang)
                        x02,y02 = perpendicular_foot(x2,y2,ang)

                        lag1 = time_lag(x01,y01,vel,ang,len1)
                        lag2 = time_lag(x02,y02,vel,ang,len2)

                        lag1_level = int(format(lag1/0.01,'.0f'))
                        lag2_level = int(format(lag2/0.01,'.0f'))
      
                        wave1 = wave_select(s1)
                        wave2 = wave_select(s2)

                        wave1_lag = wave1[int(left-lag1_level):int(right-lag1_level)]
                        wave2_lag = wave2[int(left-lag2_level):int(right-lag2_level)]
                        
                        sim = Similarity(time_list,wave1_lag,wave2_lag,"",)
                        was = sim.wasserstein_softplus_normalizing(con=3)

                        was_list += [was]

            was_array = np.array(was_list)
            was_mean = np.mean(was_array)
            was_mean_list += [was_mean]
        was_mean_list_list += [was_mean_list]
    was_2Darray = np.array(was_mean_list_list)

    return was_2Darray

def semblance(x_list,y_list,seismometer,number_list,coo_dict,left,right) :

      sem_list_list = []
      for Y in tqdm(y_list) :
            sem_list = []
            for X in x_list :
                  numer_list = []
                  demon_list = []
                  for s1,n in zip(seismometer,number_list) :
                        seismometer2 = seismometer[n:]
                        s1_str = str(s1)
                        for s2 in seismometer2 :
                              s2_str = str(s2)
                              x1 = coo_dict[f"{s1_str}_x"]
                              y1 = coo_dict[f"{s1_str}_y"]
                              x2 = coo_dict[f"{s2_str}_x"]
                              y2 = coo_dict[f"{s2_str}_y"]

                              ang = angle(X,Y)
                              vel = 1/(np.sqrt(X**2+Y**2))

                              len1 = length(x1,y1,ang)
                              len2 = length(x2,y2,ang)

                              x01,y01 = perpendicular_foot(x1,y1,ang)
                              x02,y02 = perpendicular_foot(x2,y2,ang)

                              lag1 = time_lag(x01,y01,vel,ang,len1)
                              lag2 = time_lag(x02,y02,vel,ang,len2)

                              lag1_level = int(format(lag1/0.01,'.0f'))
                              lag2_level = int(format(lag2/0.01,'.0f'))
                        
                              wave1 = wave_select(s1)
                              wave2 = wave_select(s2)

                              wave1_lag = wave1[int(left-lag1_level):int(right-lag1_level)]
                              wave2_lag = wave2[int(left-lag2_level):int(right-lag2_level)]

                              numer = (wave1_lag-wave2_lag)**2
                              demon = (wave1_lag+wave2_lag)**2

                              numer_list += [numer]
                              demon_list += [demon]

                  numer_array = np.array(numer_list)
                  demon_array = np.array(demon_list)
                  numer_sum = np.sum(numer_array)
                  demon_sum = np.sum(demon_array)
                  lamda = numer_sum/demon_sum
                  sem = (1-lamda/2)/(1+lamda)
                  sem_list += [sem]
            sem_list_list += [sem_list]
      sem_2Darray = np.array(sem_list_list)

      return sem_2Darray

def semblance_env(x_list,y_list,seismometer,number_list,coo_dict,left,right) :

      sem_list_list = []
      for Y in tqdm(y_list) :
            sem_list = []
            for X in x_list :
                  numer_list = []
                  demon_list = []
                  for s1,n in zip(seismometer,number_list) :
                        seismometer2 = seismometer[n:]
                        s1_str = str(s1)
                        for s2 in seismometer2 :
                              s2_str = str(s2)
                              x1 = coo_dict[f"{s1_str}_x"]
                              y1 = coo_dict[f"{s1_str}_y"]
                              x2 = coo_dict[f"{s2_str}_x"]
                              y2 = coo_dict[f"{s2_str}_y"]

                              ang = angle(X,Y)
                              vel = 1/(np.sqrt(X**2+Y**2))

                              len1 = length(x1,y1,ang)
                              len2 = length(x2,y2,ang)

                              x01,y01 = perpendicular_foot(x1,y1,ang)
                              x02,y02 = perpendicular_foot(x2,y2,ang)

                              lag1 = time_lag(x01,y01,vel,ang,len1)
                              lag2 = time_lag(x02,y02,vel,ang,len2)

                              lag1_level = int(format(lag1/0.01,'.0f'))
                              lag2_level = int(format(lag2/0.01,'.0f'))
                        
                              wave1 = wave_select(s1,type)
                              wave2 = wave_select(s2,type)

                              wave1 = np.abs(signal.hilbert(wave1))
                              wave2 = np.abs(signal.hilbert(wave2))

                              wave1_lag = wave1[int(left-lag1_level):int(right-lag1_level)]
                              wave2_lag = wave2[int(left-lag2_level):int(right-lag2_level)]

                              numer = (wave1_lag-wave2_lag)**2
                              demon = (wave1_lag+wave2_lag)**2

                              numer_list += [numer]
                              demon_list += [demon]

                        numer_array = np.array(numer_list)
                        demon_array = np.array(demon_list)
                        numer_sum = np.sum(numer_array)
                        demon_sum = np.sum(demon_array)
                        lamda = numer_sum/demon_sum
                        sem = (1-lamda/2)/(1+lamda)
                        sem_list += [sem]
            sem_list_list += [sem_list]
      sem_2Darray = np.array(sem_list_list)

      return sem_2Darray     

def main() :

      coo_dict_obj = np.load("./Python_Code/osaka_data/archive/coordinate.npy",allow_pickle=True)
      coo_dict = coo_dict_obj.item()

      number_list = np.arange(1,4,1)
      seismometer1 = ["OSA","OSB","OSC","FKS"]

      wave_type = ["vel"]

      X_list = np.arange(-1,1.05,0.05) 
      Y_list = np.arange(1,-1.05,-0.05)
      tw = 200
      X_len = len(X_list)
      Y_len = len(Y_list)

      lag_list_left = np.arange(4000,4900,100)
      lag_list_right = np.arange(4200,5100,100)

      # def process(left,right) :
            
      #       time_list = 0.01*(np.arange(0,tw,1))
            
      #       new_tw = 0.01*tw
      #       new_left = 0.01*left
      #       new_right = 0.01*right
      
      #       was2D = wasserstein(X_list,Y_list,seismometer1,number_list,coo_dict,left,right,time_list)
      #       np.save(f"./Python_Code/peer_reviewed/value/semblance_analysis/was_ud_wid={new_tw}s_{new_left}s~{new_right}s.npy",was2D)

      #       sem2D = semblance(X_list,Y_list,seismometer1,number_list,coo_dict,left,right)
      #       np.save(f"./Python_Code/peer_reviewed/value/semblance_analysis/sem_ud_wid={new_tw}s_{new_left}s~{new_right}s.npy",sem2D)

      # Parallel(n_jobs=-1)([delayed(process)(left,right) for left,right in zip(lag_list_left,lag_list_right)])

      for left,right in zip(lag_list_left,lag_list_right) :
            
            time_list = 0.01*(np.arange(0,tw,1))
            
            new_tw = 0.01*tw
            new_left = 0.01*left
            new_right = 0.01*right
      
            was2D = wasserstein(X_list,Y_list,seismometer1,number_list,coo_dict,left,right,time_list)
            np.save(f"./Python_Code/peer_reviewed/value/semblance_analysis/was_ud_wid={new_tw}s_{new_left}s~{new_right}s.npy",was2D)

            sem2D = semblance(X_list,Y_list,seismometer1,number_list,coo_dict,left,right)
            np.save(f"./Python_Code/peer_reviewed/value/semblance_analysis/sem_ud_wid={new_tw}s_{new_left}s~{new_right}s.npy",sem2D)

def plot_colorbar(x_list,y_list,value,left,right,tim,name,X_len,image) :

    if name == "semblance" :

        fig2,ax2 = plt.subplots(figsize = (10,10))
        x, y = np.meshgrid(x_list, y_list)
        max_value_index = np.unravel_index(np.argmax(value),value.shape)
        image2 = ax2.pcolormesh(x, y, value, alpha=0.5, cmap='jet') # 等高線図の生成。cmapで色付けの規則を指定する。
        ax2.axis("image")
        divider = make_axes_locatable(ax2)
        ax2_cb = divider.new_horizontal(size="2%", pad=0.05)
        fig2.add_axes(ax2_cb)
        plt.colorbar(image2, cax=ax2_cb)
        # pp = fig.colorbar(image,orientation="vertical") # カラーバーの表示 
        ax2_cb.set_label(f"{name}") #カラーバーのラベル
        circle1 = mp.Circle(xy=(0,0),radius=1.0, fill=False, color='gray')
        circle2 = mp.Circle(xy=(0,0),radius=0.5, fill=False, color='gray')
        circle3 = mp.Circle(xy=(0,0),radius=0.25, fill=False, color='gray')
        circle4 = mp.Circle(xy=(0,0),radius=0.125, fill=False, color='gray')
        circle5_position_x = -1+0.05*max_value_index[1]+(0.05/2)
        circle5_position_y = 1-0.05*max_value_index[0]-(0.05/2)
        circle5 = mp.Circle(xy=(circle5_position_x,circle5_position_y),radius=0.075, fill=False, linewidth=2,color='white')
        ax2.add_patch(circle1)
        ax2.add_patch(circle2)
        ax2.add_patch(circle3)
        ax2.add_patch(circle4)
        ax2.add_patch(circle5)
        # ax2.set_title(f"semblance coefficient vel_{direction} timewidth={tw_new}[s] {left_new}[s]~{right_new}[s]")
        ax2.text(-0.05,0.95,r"$c=1$[km/s]",size=15)
        ax2.text(-0.05,0.45,r"$c=2$[km/s]",size=15)
        ax2.text(-0.05,0.20,r"$c=4$[km/s]",size=15)
        ax2.text(-0.05,0.075,r"$c=8$[km/s]",size=15)
        tw_new = 0.01*tim
        left_new = 0.01*left
        right_new = 0.01*right
      #   ax2.text(-1.0,1.2,f"semblance coefficient timewidth={tw_new}[s] {left_new}[s]~{right_new}[s]",size=20)
        ax2.axis("off")
        ax2.imshow(image, extent=[-1.0,-0.6,0.6,1.0])
        fig2.savefig(f"./Python_Code/peer_reviewed/figure/semblance_analysis/sem_ud_time_width={tw_new}s {left_new}s~{right_new}s.png")
        plt.clf()
        plt.close()

        return circle5_position_x,circle5_position_y
    
    elif name == "wasserstein metric" :

        fig2,ax2 = plt.subplots(figsize = (10,10))
        x, y = np.meshgrid(x_list, y_list)
        min_value_index = np.unravel_index(np.argmin(value),value.shape)
        image2 = ax2.pcolormesh(x, y, value, alpha=0.5, cmap='jet_r') # 等高線図の生成。cmapで色付けの規則を指定する。
        ax2.axis("image")
        divider = make_axes_locatable(ax2)
        ax2_cb = divider.new_horizontal(size="2%", pad=0.05)
        fig2.add_axes(ax2_cb)
        plt.colorbar(image2, cax=ax2_cb)
        # pp = fig.colorbar(image,orientation="vertical") # カラーバーの表示 
        ax2_cb.set_label(f"{name}") #カラーバーのラベル
        circle1 = mp.Circle(xy=(0,0),radius=1.0, fill=False, color='gray')
        circle2 = mp.Circle(xy=(0,0),radius=0.5, fill=False, color='gray')
        circle3 = mp.Circle(xy=(0,0),radius=0.25, fill=False, color='gray')
        circle4 = mp.Circle(xy=(0,0),radius=0.125, fill=False, color='gray')
        circle5_position_x = -1+0.05*min_value_index[1]+(0.05/2)
        circle5_position_y = 1-0.05*min_value_index[0]-(0.05/2)
        circle5 = mp.Circle(xy=(circle5_position_x,circle5_position_y),radius=0.075, fill=False, linewidth=2,color='white')
        ax2.add_patch(circle1)
        ax2.add_patch(circle2)
        ax2.add_patch(circle3)
        ax2.add_patch(circle4)
        ax2.add_patch(circle5)
        # ax2.set_title(f"wasserstein metric vel_{direction} timewidth={tw_new}[s] {left_new}[s]~{right_new}[s]")
        ax2.text(-0.05,0.95,r"$c=1$[km/s]",size=15)
        ax2.text(-0.05,0.45,r"$c=2$[km/s]",size=15)
        ax2.text(-0.05,0.20,r"$c=4$[km/s]",size=15)
        ax2.text(-0.05,0.075,r"$c=8$[km/s]",size=15)
        tw_new = 0.01*tim
        left_new = 0.01*left
        right_new = 0.01*right
      #   ax2.text(-1.0,1.2,f"wasserstein metric timewidth={tw_new}[s] {left_new}[s]~{right_new}[s]",size=20)
        ax2.axis("off")
        ax2.imshow(image, extent=[-1.0,-0.6,0.6,1.0])
        fig2.savefig(f"./Python_Code/peer_reviewed/figure/semblance_analysis/was_ud_time_width={tw_new}s {left_new}s~{right_new}s.png")
        plt.clf()
        plt.close()

        return circle5_position_x,circle5_position_y

def plot1():

      X_list = np.arange(-1,1.05,0.05) 
      Y_list = np.arange(1,-1.05,-0.05)
      tw = 200
      X_len = len(X_list)-1
      Y_len = len(Y_list)-1
      north_image = plt.imread("./Python_Code/osaka_data/archive/FourNorthArrow.png")

      lag_list_left = np.arange(4000,4900,100)
      lag_list_right = np.arange(4200,5100,100)

      k_was_list,k_sem_list = [],[]

      for left,right in tqdm(zip(lag_list_left,lag_list_right)) :
            tim = tw
            new_tw = 0.01*tw
            new_left = 0.01*left
            new_right = 0.01*right
            was2D = np.load(f"./Python_Code/peer_reviewed/value/semblance_analysis/was_ud_wid=2.0s_{new_left}s~{new_right}s.npy")
            name1 = "wasserstein metric"
            kx_was,ky_was = plot_colorbar(X_list,Y_list,was2D,left,right,tim,name1,X_len,north_image)
            k_was = [kx_was,ky_was]
            k_was_list += [k_was]
            sem2D = np.load(f"./Python_Code/peer_reviewed/value/semblance_analysis/sem_ud_wid=2.0s_{new_left}s~{new_right}s.npy")
            name2 = "semblance"
            kx_sem,ky_sem =plot_colorbar(X_list,Y_list,sem2D,left,right,tim,name2,X_len,north_image)
            k_sem = [kx_sem,ky_sem]
            k_sem_list += [k_sem]

      # np.save("./Python_Code/peer_reviewed/value/semblance_analysis/was_opt_10s_grid.npy",k_was_list)
      # np.save("./Python_Code/peer_reviewed/value/semblance_analysis/sem_opt_10s_grid.npy",k_sem_list)


##############
#  局所最適化  #
##############

def was_opt(seismometer,number_list,coo_dict,left,right,time_list,ini):

      def function_was(k):
            was_list = []
            for s1,n in zip(seismometer,number_list) :
                  seismometer2 = seismometer[n:]
                  s1_str = str(s1)
                  for s2 in seismometer2 :
                        s2_str = str(s2)
                        x1 = coo_dict[f"{s1_str}_x"]
                        y1 = coo_dict[f"{s1_str}_y"]
                        x2 = coo_dict[f"{s2_str}_x"]
                        y2 = coo_dict[f"{s2_str}_y"]

                        ang = angle(k[0],k[1])
                        vel = 1/(np.sqrt(k[0]**2+k[1]**2))

                        len1 = length(x1,y1,ang)
                        len2 = length(x2,y2,ang)

                        x01,y01 = perpendicular_foot(x1,y1,ang)
                        x02,y02 = perpendicular_foot(x2,y2,ang)

                        lag1 = time_lag(x01,y01,vel,ang,len1)
                        lag2 = time_lag(x02,y02,vel,ang,len2)

                        lag1_level = int(format(lag1/0.01,'.0f'))
                        lag2_level = int(format(lag2/0.01,'.0f'))

                        wave1 = wave_select(s1)
                        wave2 = wave_select(s2)

                        wave1_lag = wave1[int(left-lag1_level):int(right-lag1_level)]
                        wave2_lag = wave2[int(left-lag2_level):int(right-lag2_level)]
                        
                        sim = Similarity(time_list,wave1_lag,wave2_lag,"",)
                        was = sim.wasserstein_softplus_normalizing(con=3)

                        was_list += [was]

            was_array = np.array(was_list)
            was_mean = np.mean(was_array) 

            return was_mean
      
      def constraint_kx1(k):

            return k[0]+1

      def constraint_kx2(k):

            return -k[0]+1

      def constraint_ky1(k):

            return k[1]+1

      def constraint_ky2(k):

            return -k[1]+1

      cons = (
      {'type': 'ineq', 'fun': constraint_kx1},
      {'type': 'ineq', 'fun': constraint_kx2},
      {'type': 'ineq', 'fun': constraint_ky1},
      {'type': 'ineq', 'fun': constraint_ky2},
      )  

      result = op.minimize(function_was,x0=ini,constraints=cons,method="COBYLA")
      print(result.x)
      return result.x  

def sem_opt(seismometer,number_list,coo_dict,left,right,ini):

      def function_sem(k):
            numer_list,demon_list = [],[]
            for s1,n in zip(seismometer,number_list) :
                  seismometer2 = seismometer[n:]
                  s1_str = str(s1)
                  for s2 in seismometer2 :
                        s2_str = str(s2)
                        x1 = coo_dict[f"{s1_str}_x"]
                        y1 = coo_dict[f"{s1_str}_y"]
                        x2 = coo_dict[f"{s2_str}_x"]
                        y2 = coo_dict[f"{s2_str}_y"]

                        ang = angle(k[0],k[1])
                        vel = 1/(np.sqrt(k[0]**2+k[1]**2))

                        len1 = length(x1,y1,ang)
                        len2 = length(x2,y2,ang)

                        x01,y01 = perpendicular_foot(x1,y1,ang)
                        x02,y02 = perpendicular_foot(x2,y2,ang)

                        lag1 = time_lag(x01,y01,vel,ang,len1)
                        lag2 = time_lag(x02,y02,vel,ang,len2)

                        lag1_level = int(format(lag1/0.01,'.0f'))
                        lag2_level = int(format(lag2/0.01,'.0f'))

                        wave1 = wave_select(s1)
                        wave2 = wave_select(s2)

                        wave1_lag = wave1[int(left-lag1_level):int(right-lag1_level)]
                        wave2_lag = wave2[int(left-lag2_level):int(right-lag2_level)]


                        numer = (wave1_lag-wave2_lag)**2
                        demon = (wave1_lag+wave2_lag)**2

                        numer_list += [numer]
                        demon_list += [demon]

            numer_array = np.array(numer_list)
            demon_array = np.array(demon_list)
            numer_sum = np.sum(numer_array)
            demon_sum = np.sum(demon_array)
            lamda = numer_sum/demon_sum
            sem = (1-lamda/2)/(1+lamda)
            sem_rev = 1/sem

            return sem_rev

      def constraint_kx1(k):

            return k[0]+1

      def constraint_kx2(k):

            return -k[0]+1

      def constraint_ky1(k):

            return k[1]+1

      def constraint_ky2(k):

            return -k[1]+1

      cons = (
      {'type': 'ineq', 'fun': constraint_kx1},
      {'type': 'ineq', 'fun': constraint_kx2},
      {'type': 'ineq', 'fun': constraint_ky1},
      {'type': 'ineq', 'fun': constraint_ky2},
      )  

      result = op.minimize(function_sem,x0=ini,constraints=cons,method="COBYLA")
      print(result.x)
      return result.x  

def con_opt(seismometer,number_list,coo_dict,left,right,time_list,ini) :

      k1 = was_opt(seismometer,number_list,coo_dict,left,right,time_list,ini)
      k2 = sem_opt(seismometer,number_list,coo_dict,left,right,k1)

      return k2

def main2(ini_fix,sim):

      def sim_select(sim,seismometer1,number_list,coo_dict,left,right,time_list,ini) :

            if sim == "was" :
                  return was_opt(seismometer1,number_list,coo_dict,left,right,time_list,ini)

            elif sim == "sem" :
                  return sem_opt(seismometer1,number_list,coo_dict,left,right,ini)

            elif sim == "con" :
                  return con_opt(seismometer1,number_list,coo_dict,left,right,time_list,ini)

      coo_dict_obj = np.load("./Python_Code/osaka_data/archive/coordinate.npy",allow_pickle=True)
      coo_dict = coo_dict_obj.item()

      number_list = np.arange(1,4,1)
      seismometer1 = ["OSA","OSB","OSC","FKS"]

      tw = 1000

      lag_list_left = np.arange(3.5*tw,24000-15*tw,50)
      lag_list_right = np.arange(4.5*tw,24000-14*tw,50)

      # lag_list_left = np.arange(17.5*tw,24000-71*tw,10)
      # lag_list_right = np.arange(18.5*tw,24000-70*tw,10)

      ini = np.array([0.05,0.05])

      if ini_fix == True :

            def process(left,right) :

                  time_list = 0.01*(np.arange(0,tw,1))
                  
                  new_tw = 0.01*tw
                  new_left = 0.01*left
                  new_right = 0.01*right

                  k = sim_select(sim,seismometer1,number_list,coo_dict,left,right,time_list,ini)

                  np.save(f"./Python_Code/peer_reviewed/value/semblance_analysis/ini_fix/{sim}_ud_wid={new_tw}s_{new_left}s~{new_right}s.npy",k)
            
            Parallel(n_jobs=-1)([delayed(process)(left,right) for left,right in zip(lag_list_left,lag_list_right)])
      
      elif ini_fix == False :

            k_list = []
            for left,right in tqdm(zip(lag_list_left,lag_list_right)) :

                  time_list = 0.01*(np.arange(0,tw,1))

                  k = sim_select(sim,seismometer1,number_list,coo_dict,left,right,time_list,ini)

                  ini = k
                  k_list += [k]

            np.save(f"./Python_Code/peer_reviewed/value/semblance_analysis/{sim}_opt_10s_cobyla.npy",k_list)

def plot_time_series() :

      from peer_reviewed.format_time_series import Format
      Format.params()

      def angle_trans(theta):

            if 0<=theta<=230.7513 :

                  theta_ = 50.7513 - theta

            elif 230.7513<theta<360 :

                  theta_ = 410.7513 - theta

            return theta_ 

      def polar_trans(value_list) :

            value_x = [value_list[k][0] for k in range(len(value_list))]
            value_y = [value_list[k][1] for k in range(len(value_list))]

            fai_dash_list,vel_list = [],[]
            for x,y in zip(value_x,value_y) :

                  fai = angle(x,y)
                  fai_dash = angle_trans(fai)
                  vel = np.sqrt(x**2+y**2)
                  fai_dash_list += [fai_dash]
                  vel_list += [vel]
            
            return fai_dash_list

      def polar_trans_file(file_name):

            value = np.load(f"./Python_Code/peer_reviewed/value/semblance_analysis/{file_name}")
            value_x = [value[k][0] for k in range(len(value))]
            value_y = [value[k][1] for k in range(len(value))]

            fai_dash_list,vel_list = [],[]
            for x,y in zip(value_x,value_y) :

                  fai = angle(x,y)
                  fai_dash = angle_trans(fai)
                  vel = np.sqrt(x**2+y**2)
                  fai_dash_list += [fai_dash]
                  vel_list += [vel]
            
            vel_list = np.array(vel_list)
            return fai_dash_list,vel_list

      tw = 1000

      lag_list_left = np.arange(3.5*tw,24000-15*tw,50)
      lag_list_right = np.arange(4.5*tw,24000-14*tw,50)

      fai_was_loc,vel_was_loc = polar_trans_file("was_opt_10s_cobyla.npy")
      fai_sem_loc,vel_sem_loc = polar_trans_file("sem_opt_10s_cobyla.npy")
      fai_was_grid,vel_was_grid = polar_trans_file("was_opt_10s_grid.npy")
      fai_sem_grid,vel_sem_grid = polar_trans_file("sem_opt_10s_grid.npy")

      time_list = (lag_list_left + lag_list_right)/(2*100)

      fig1,ax1 = plt.subplots()
      ax1.plot(time_list,fai_was_loc,markersize=15,c="midnightblue",marker="o",label=r"$W$(COBYLA)")
      ax1.plot(time_list,fai_sem_loc,markersize=15,c="firebrick",marker="o",label=r"$Se$(COBYLA)")
      ax1.plot(time_list,fai_was_grid,c="dodgerblue",label=r"$W$(Grid Search)")
      ax1.plot(time_list,fai_sem_grid,c="red",label=r"$Se$(Grid Search)")

      ax1.set_ylim(-180,180)
      ax1.set_xlim(35,100)
      ax1.set_xlabel("Time[s]")
      ax1.set_ylabel("Direction")
      ax1.invert_yaxis()
      axis_array1 = np.array([-127.2487,-39.2487,0,50.7513,140.7513])
      ax1.set_yticks(axis_array1)
      ax1.set_yticklabels(["West","North","","East","South"])
      ax1.text(-0.050, 0.45, "★",transform=ax1.transAxes,fontname="MS Gothic")
      ax1.legend(frameon=False,loc=(0.25,0.57),fontsize=55)
      fig1.tight_layout()
      fig1.savefig(f"./Python_Code/peer_reviewed/figure/semblance_analysis/time_series_10s_cobyla_compare.png")
      plt.clf()
      plt.close()

      fig1,ax1 = plt.subplots()
      ax1.plot(time_list,1/vel_was_loc,markersize=15,c="midnightblue",marker="o",label=r"$W$(COBYLA)")
      ax1.plot(time_list,1/vel_sem_loc,markersize=15,c="firebrick",marker="o",label=r"$Se$(COBYLA)")
      ax1.plot(time_list,1/vel_was_grid,c="dodgerblue",label=r"$W$(Grid Search)")
      ax1.plot(time_list,1/vel_sem_grid,c="red",label=r"$Se$(Grid Search)")
      ax1.set_ylim(0.4,8)
      ax1.set_xlim(35,100)
      ax1.set_yscale("log")
      ax1.set_xlabel("Time[s]")
      ax1.set_ylabel("Velocity[km/s]")
      axis_array1 = np.array([0.5,1,2,4])
      ax1.set_yticks(axis_array1)
      ax1.set_yticklabels(["0.5","1","2","4"])
      ax1.legend(frameon=False,loc=(0.3,0.57),fontsize=55)
      fig1.tight_layout()
      fig1.savefig(f"./Python_Code/peer_reviewed/figure/semblance_analysis/time_series_10s_cobyla_vel_compare.png")
      plt.clf()
      plt.close()

def plot_cartesian(): 

      value = np.load("./Python_Code/peer_reviewed/value/semblance_analysis/con_opt_10s.npy")

      value_x = [value[k][0] for k in range(len(value))]
      value_y = [value[k][1] for k in range(len(value))]
      fig1,ax1 = plt.subplots(figsize=(10,10))
      ax1.plot(value_x,value_y)
      fig1.savefig(f"./Python_Code/peer_reviewed/figure/semblance_analysis/con_opt_10s.png")
      plt.clf()
      plt.close()

plot_time_series()