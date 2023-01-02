import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

class Format:
    def params():

            ## Figureの設定 ##
            plt.rcParams['figure.figsize'] = (35,8)  # figure size in inch, 横×縦
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
            plt.rcParams['lines.linewidth'] = 3 #グラフの線の太さ

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