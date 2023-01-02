import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fractions import Fraction
from matplotlib.colors import ListedColormap
import matplotlib.patches as mp
from scipy import signal
import sys
sys.path.append('./Python_Code')
from others.similarity import Similarity


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

def wasserstein(x_list,y_list,seismometer,number_list,coo_dict,left,right,time_list,direction) :

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
    
                    wave1 = np.load(f"{s1}_vel_{direction}.npy")
                    wave2 = np.load(f"{s2}_vel_{direction}.npy")

                    wave1_lag = wave1[int(left-lag1_level):int(right-lag1_level)]
                    wave2_lag = wave2[int(left-lag2_level):int(right-lag2_level)]
                    
                    sim = Similarity(time_list,wave1_lag,wave2_lag)
                    was = sim.wasserstein_softplus_normalizing()

                    was_list += [was]

            was_array = np.array(was_list)
            was_mean = np.mean(was_array)
            was_mean_list += [was_mean]
        was_mean_list_list += [was_mean_list]
    was_2Darray = np.array(was_mean_list_list)

    return was_2Darray

def semblance(x_list,y_list,seismometer,number_list,coo_dict,left,right,direction) :

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
                
                    wave1 = np.load(f"{s1}_vel_{direction}.npy")
                    wave2 = np.load(f"{s2}_vel_{direction}.npy")

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

def plot_colorbar(x_list,y_list,value,left,right,tim,name) :

    if name == "semblance" :
        fig,ax = plt.subplots(figsize = (10,10))
        X, Y = np.meshgrid(x_list, y_list)
        max_value_index = np.unravel_index(np.argmax(value),value.shape)
        max_value = value[max_value_index]
        masked_Z = np.ma.masked_where(value < max_value*2/3,value)
        # cmap = plt.cm.jet
        # cmap.set_bad((0,0,0,0))
        # ax.imshow(masked_Z,cmap=cmap)
        image = ax.pcolormesh(X, Y, masked_Z, cmap='jet') # 等高線図の生成。cmapで色付けの規則を指定する。
        ax.axis("image")
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="2%", pad=0.05)
        fig.add_axes(ax_cb)
        plt.colorbar(image, cax=ax_cb)
        # pp = fig.colorbar(image,orientation="vertical") # カラーバーの表示 
        ax_cb.set_label(f"{name}") #カラーバーのラベル
        
        tw_new = 0.01*tim
        left_new = 0.01*left
        right_new = 0.01*right
        fig.savefig(f"sem_time_width={tw_new}s {left_new}s~{right_new}s.png")
        plt.clf()
        plt.close()
    
    elif name == "wasserstein metric" :
        fig,ax = plt.subplots(figsize = (10,10))
        X, Y = np.meshgrid(x_list, y_list)
        min_value_index = np.unravel_index(np.argmin(value),value.shape)
        min_value = value[min_value_index]
        masked_Z = np.ma.masked_where(value > min_value*3/2,value)
        # cmap = plt.cm.jet
        # cmap.set_bad((0,0,0,0))
        # ax.imshow(masked_Z,cmap=cmap)
        image = ax.pcolormesh(X, Y, masked_Z, cmap='jet') # 等高線図の生成。cmapで色付けの規則を指定する。
        ax.axis("image")
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="2%", pad=0.05)
        fig.add_axes(ax_cb)
        plt.colorbar(image, cax=ax_cb)
        # pp = fig.colorbar(image,orientation="vertical") # カラーバーの表示 
        ax_cb.set_label(f"{name}") #カラーバーのラベル
        tw_new = 0.01*tim
        left_new = 0.01*left
        right_new = 0.01*right
        fig.savefig(f"was_time_width={tw_new}s {left_new}s~{right_new}s.png")
        plt.clf()
        plt.close()

def main() :

    coo_dict_obj = np.load("./osaka_data/archive/coordinate.npy",allow_pickle=True)
    coo_dict = coo_dict_obj.item()

    number_list = np.arange(1,4,1)
    seismometer1 = ["OSA","OSB","OSC","FKS"]

    X_list = np.arange(-1,1.05,0.05) 
    Y_list = np.arange(1,-1.05,-0.05)
    time_width = [1000]
    time_width = np.array(time_width)
    X_len = len(X_list)
    Y_len = len(Y_list)
    direction = ["EW","NS","UD"]
    for dir in tqdm(direction) :
        for tw in tqdm(time_width) :
            width_qua = 24000/tw
            lag_qua = 48000/tw
            lag_list_left = np.arange(5*tw,24000-11*tw,tw/2)
            lag_list_right = np.arange(6*tw,24000-10*tw,tw/2)
            for left,right in tqdm(zip(lag_list_left,lag_list_right)) :
                time_list = 0.01*(np.arange(0,tw,1))
                
                was2D = wasserstein(X_list,Y_list,seismometer1,number_list,coo_dict,left,right,time_list,dir)
                
                new_tw = 0.01*tw
                new_left = 0.01*left
                new_right = 0.01*right
                np.save(f"was_vel_surface_{dir}_wid={new_tw}s_{new_left}s~{new_right}s",was2D)
                name1 = "wasserstein metric"
                # plot_colorbar(X_list,Y_list,was2D,left,right,tim,name1)

                sem2D = semblance(X_list,Y_list,seismometer1,number_list,coo_dict,left,right,dir)

                np.save(f"sem_vel_surface_{dir}_wid={new_tw}s_{new_left}s~{new_right}s",sem2D)
                name2 = "semblance"
                # # plot_colorbar(X_list,Y_list,sem2D,left,right,tim,name2)

