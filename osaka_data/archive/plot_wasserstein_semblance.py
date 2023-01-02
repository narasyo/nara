from re import A
from matplotlib import image
import numpy as np
import os 
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fractions import Fraction
from matplotlib.colors import ListedColormap
import matplotlib.patches as mp


def plot_colorbar(x_list,y_list,value,left,right,tim,name,len,image,direction) :

    if name == "semblance" :
        # fig1,ax1 = plt.subplots(figsize = (10,10))
        # x, y = np.meshgrid(x_list, y_list)
        # max_value_index = np.unravel_index(np.argmax(value),value.shape)
        # masked_value = value
        # for i in range(len*len-50) :
        #     min_value_index = np.unravel_index(np.argmin(masked_value),masked_value.shape)
        #     min_value = masked_value[min_value_index]
        #     masked_value = np.ma.masked_where(masked_value == min_value,masked_value)
        # image1 = ax1.pcolormesh(x, y, masked_value, alpha=0.5, cmap='jet') # 等高線図の生成。cmapで色付けの規則を指定する。
        # ax1.axis("image")
        # divider = make_axes_locatable(ax1)
        # ax1_cb = divider.new_horizontal(size="2%", pad=0.05)
        # fig1.add_axes(ax1_cb)
        # plt.colorbar(image1, cax=ax1_cb)
        # # pp = fig.colorbar(image,orientation="vertical") # カラーバーの表示 
        # ax1_cb.set_label(f"{name}") #カラーバーのラベル
        # circle1 = mp.Circle(xy=(0,0),radius=1.0, fill=False, color='gray')
        # circle2 = mp.Circle(xy=(0,0),radius=0.5, fill=False, color='gray')
        # circle3 = mp.Circle(xy=(0,0),radius=0.25, fill=False, color='gray')
        # circle4 = mp.Circle(xy=(0,0),radius=0.125, fill=False, color='gray')
        # circle5_position_x = -1+0.05*max_value_index[1]+(0.05/2)
        # circle5_position_y = 1-0.05*max_value_index[0]-(0.05/2)
        # circle5 = mp.Circle(xy=(circle5_position_x,circle5_position_y),radius=0.075, fill=False, linewidth=2,color='white')
        # tw_new = 0.01*tim
        # left_new = 0.01*left
        # right_new = 0.01*right
        # ax1.add_patch(circle1)
        # ax1.add_patch(circle2)
        # ax1.add_patch(circle3)
        # ax1.add_patch(circle4)
        # ax1.add_patch(circle5)
        # ax1.text(-0.05,0.95,r"$c=1$[km/s]",size=10)
        # ax1.text(-0.05,0.45,r"$c=2$[km/s]",size=10)
        # ax1.text(-0.05,0.20,r"$c=4$[km/s]",size=10)
        # ax1.text(-0.05,0.075,r"$c=8$[km/s]",size=10)
        # ax1.axis("off")
        # ax1.imshow(image, extent=[-1.0,-0.6,0.6,1.0])
        # fig1.savefig(f"sem_time_width={tw_new}s {left_new}s~{right_new}s_50th.png")
        # fig1.savefig(f"sem_time_width={tw_new}s {left_new}s~{right_new}s_50th.eps")
        # plt.clf()
        # plt.close()

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
        ax2.text(-1.0,1.2,f"semblance coefficient timewidth={tw_new}[s] {left_new}[s]~{right_new}[s] {direction}",size=20)
        ax2.axis("off")
        ax2.imshow(image, extent=[-1.0,-0.6,0.6,1.0])
        fig2.savefig(f"sem_vel_time_width={tw_new}s {left_new}s~{right_new}s_{direction}.png")
        fig2.savefig(f"sem_vel_time_width={tw_new}s {left_new}s~{right_new}s_{direction}.eps")
        plt.clf()
        plt.close()
    
    elif name == "wasserstein metric" :
        # fig1,ax1 = plt.subplots(figsize = (10,10))
        # x, y = np.meshgrid(x_list, y_list)
        # min_value_index = np.unravel_index(np.argmin(value),value.shape)
        # masked_value = value
        # for i in range(len*len-50) :
        #     max_value_index = np.unravel_index(np.argmax(masked_value),masked_value.shape)
        #     max_value = masked_value[max_value_index]
        #     masked_value = np.ma.masked_where(masked_value == max_value,masked_value)
        # image1 = ax1.pcolormesh(x, y, masked_value, alpha=0.5, cmap='jet_r') # 等高線図の生成。cmapで色付けの規則を指定する。
        # ax1.axis("image")
        # divider = make_axes_locatable(ax1)
        # ax1_cb = divider.new_horizontal(size="2%", pad=0.05)
        # fig1.add_axes(ax1_cb)
        # plt.colorbar(image1, cax=ax1_cb)
        # # pp = fig.colorbar(image,orientation="vertical") # カラーバーの表示 
        # ax1_cb.set_label(f"{name}") #カラーバーのラベル
        # circle1 = mp.Circle(xy=(0,0),radius=1.0, fill=False, color='gray')
        # circle2 = mp.Circle(xy=(0,0),radius=0.5, fill=False, color='gray')
        # circle3 = mp.Circle(xy=(0,0),radius=0.25, fill=False, color='gray')
        # circle4 = mp.Circle(xy=(0,0),radius=0.125, fill=False, color='gray')
        # circle5_position_x = -1+0.05*min_value_index[1]+(0.05/2)
        # circle5_position_y = 1-0.05*min_value_index[0]-(0.05/2)
        # circle5 = mp.Circle(xy=(circle5_position_x,circle5_position_y),radius=0.075, fill=False, linewidth=2,color='white')
        # ax1.add_patch(circle1)
        # ax1.add_patch(circle2)
        # ax1.add_patch(circle3)
        # ax1.add_patch(circle4)
        # ax1.add_patch(circle5)
        # ax1.text(-0.05,0.95,r"$c=1$[km/s]",size=10)
        # ax1.text(-0.05,0.45,r"$c=2$[km/s]",size=10)
        # ax1.text(-0.05,0.20,r"$c=4$[km/s]",size=10)
        # ax1.text(-0.05,0.075,r"$c=8$[km/s]",size=10)
        # ax1.axis("off")
        # ax1.imshow(image, extent=[-1.0,-0.6,0.6,1.0])
        # tw_new = 0.01*tim
        # left_new = 0.01*left
        # right_new = 0.01*right
        # fig1.savefig(f"was_time_width={tw_new}s {left_new}s~{right_new}s_50th.png")
        # fig1.savefig(f"was_time_width={tw_new}s {left_new}s~{right_new}s_50th.eps")
        # plt.clf()
        # plt.close()

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
        ax2.text(-1.0,1.2,f"wasserstein metric timewidth={tw_new}[s] {left_new}[s]~{right_new}[s] {direction}",size=20)
        ax2.axis("off")
        ax2.imshow(image, extent=[-1.0,-0.6,0.6,1.0])
        fig2.savefig(f"was_vel_time_width={tw_new}s {left_new}s~{right_new}s_{direction}.png")
        fig2.savefig(f"was_vel_time_width={tw_new}s {left_new}s~{right_new}s_{direction}.eps")
        plt.clf()
        plt.close()

os.chdir("./osaka_data/archive")

X_list = np.arange(-1,1.10,0.05) 
Y_list = np.arange(1,-1.10,-0.05)
time_width = [200]
time_width = np.array(time_width)
X_len = len(X_list)-1
Y_len = len(Y_list)-1
north_image = plt.imread("FourNorthArrow.png")
direction = ["EW","NS","UD"]
# os.mkdir(f"semblance_plot_circle_resolution={X_len}*{Y_len}_initial")
# os.mkdir(f"semblance_plot_circle_resolution={X_len}*{Y_len}_initial_pdf")
# os.mkdir(f"wasserstein_vel_EW={X_len}*{Y_len}_plot")
# os.mkdir(f"semblance_vel_EW={X_len}*{Y_len}_plot")
# os.mkdir(f"wasserstein_vel_NS={X_len}*{Y_len}_plot")
# os.mkdir(f"semblance_vel_NS={X_len}*{Y_len}_plot")
# os.mkdir(f"wasserstein_vel_UD={X_len}*{Y_len}_plot")
# os.mkdir(f"semblance_vel_UD={X_len}*{Y_len}_plot")
for dir in tqdm(direction):
    for tw in tqdm(time_width) :
        width_qua = 24000/tw
        lag_qua = 48000/tw
        lag_list_left = np.arange(19*tw,24000-95*tw,tw/2)
        lag_list_right = np.arange(20*tw,24000-94*tw,tw/2)
        for left,right in tqdm(zip(lag_list_left,lag_list_right)) :
            time_list = 0.01*(np.arange(0,tw,1))
            tim = tw
            
            os.chdir(f"./wasserstein_vel_{dir}={X_len}*{Y_len}")
            new_tw = 0.01*tw
            new_left = 0.01*left
            new_right = 0.01*right
            was2D = np.load(f"was_vel_{dir}_wid={new_tw}s_{new_left}s~{new_right}s.npy")
            name1 = "wasserstein metric"
            os.chdir("../")
            os.chdir(f"./wasserstein_vel_{dir}={X_len}*{Y_len}_plot")
            plot_colorbar(X_list,Y_list,was2D,left,right,tim,name1,X_len,north_image,dir)
            os.chdir("../")

            os.chdir(f"./semblance_vel_{dir}={X_len}*{Y_len}")
            sem2D = np.load(f"sem_vel_{dir}_wid={new_tw}s_{new_left}s~{new_right}s.npy")
            name2 = "semblance"
            os.chdir("../")
            os.chdir(f"semblance_vel_{dir}={X_len}*{Y_len}_plot")
            plot_colorbar(X_list,Y_list,sem2D,left,right,tim,name2,X_len,north_image,dir)
            os.chdir("../")


