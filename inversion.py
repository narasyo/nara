import numpy as np
import similarity
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import scipy.optimize as opt
from scipy import interpolate, signal
# from format_wave import Format
import time
from sigfig import round
import math
from decimal import Decimal,getcontext
from joblib import Parallel,delayed
import multiprocessing as mp
import japanize_matplotlib


def plot_3D(data,title):
    xuniq = sorted(list(set(list(map(lambda x: x[0],data)))))
    yuniq = sorted(list(set(list(map(lambda x: x[1],data)))))
    X, Y = np.meshgrid(xuniq, yuniq)
     
    zDict = dict()
    for row in data:
        zDict[(row[0], row[1])] = row[2]
    Z = [] # Array<Array<number>>
    for yUniqIdx, y in enumerate(yuniq):
        Z.append([])
        for xUniqIdx, x in enumerate(xuniq):
            Z[yUniqIdx].append(zDict[(x, y)])
    Z = np.array(Z)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("tau1", fontname="MS Gothic")
    ax.set_ylabel("tau2", fontname="MS Gothic")
    ax.set_zlabel("was", fontname="MS Gothic")
    ax.set_title(title, fontname="MS Gothic")
    ax.plot_surface(X, Y, Z, cmap="summer")
    fig.savefig(f"./{title}.png")
    plt.show()

########## すべり分布の図を描画する
def slip_distribution(slip,splinetrans_slip,annotate_slip,file_name,kind):
 
    xuniq = sorted(list(set(list(map(lambda x: x[0],splinetrans_slip)))))
    yuniq = sorted(list(set(list(map(lambda x: x[1],splinetrans_slip)))))
    X, Y = np.meshgrid(xuniq, yuniq)
    zDict = dict()
    for row in splinetrans_slip:
        zDict[(row[0], row[1])] = row[2]
    Z = [] # Array<Array<number>>
    for yUniqIdx, y in enumerate(yuniq):
        Z.append([])
        for xUniqIdx, x in enumerate(xuniq):
            Z[yUniqIdx].append(zDict[(x, y)])
    Z = np.array(Z)  
    fig, ax = plt.subplots()
    im = ax.imshow(Z, interpolation='gaussian',cmap=cm.hot_r,
                origin='lower', extent=[0, 10, 0, 10],
                vmax=1.5, vmin=0)
    ax.contour(X, Y, Z, colors="black",levels=[0,0.2,0.4,0.6,0.8,1],linewidths=0.5)
    max_value = max(annotate_slip)
    for i in range(10):
        for j in range(10):
            if kind == "reverse":
                ax.annotate('', 
                            xy=[0.5+j+np.cos(63/180*np.pi)*annotate_slip[10*i+j],0.5+i-np.sin(63/180*np.pi)*annotate_slip[10*i+j]],
                            xytext=[0.5+j,0.5+i],
                            arrowprops=dict(shrink=0,
                                            width=annotate_slip[10*i+j]*5,
                                            headwidth=annotate_slip[10*i+j]*15, 
                                            headlength=annotate_slip[10*i+j]*10, 
                                            connectionstyle='arc3',
                                            facecolor='black', 
                                            edgecolor='black',
                                            fill=False)
                            )   
            elif kind == "strike":
                ax.annotate('', 
                            xy=[0.5+j+np.cos(164/180*np.pi)*annotate_slip[10*i+j],0.5+i-np.sin(164/180*np.pi)*annotate_slip[10*i+j]],
                            xytext=[0.5+j,0.5+i],
                            arrowprops=dict(shrink=0,
                                            # width=annotate_slip[10*i+j]/max_value*5,
                                            # headwidth=annotate_slip[10*i+j]/max_value*15, 
                                            # headlength=annotate_slip[10*i+j]/max_value*5, 
                                            width=annotate_slip[10*i+j]*5,
                                            headwidth=annotate_slip[10*i+j]*15,
                                            headlength=annotate_slip[10*i+j]*10,
                                            connectionstyle='arc3',
                                            facecolor='black', 
                                            edgecolor='black',
                                            fill=False)
                            )   
    # ax.annotate('', xy=[11+2*max_value,0], xytext=[11,0],
    #             arrowprops=dict(shrink=0, width=5, headwidth=15, 
    #                             headlength=5, connectionstyle='arc3',
    #                             facecolor='black', edgecolor='black',fill=False))
    
    ax.invert_yaxis()
    ax.set_xlabel('Along Strike(km)')
    ax.set_ylabel('Along Dip(km)')
    ax.set_title('Slip Distribution')
    ax.scatter(5,5,s=300,marker="*",color="black")
    for s in slip:
        if (s[0]==0 or s[1]==0):
            ax.scatter(s[0],s[1],s=100,marker=".",color="black") 
        elif (s[0]==10 or s[1]==10):
            ax.scatter(s[0],s[1],s=100,marker=".",color="black") 
        else:
            ax.scatter(s[0],s[1],s=100,marker=".",color="blue") 
    plt.colorbar(im,extend='both',label="Slip(m)",shrink=0.7)
    plt.savefig(f"{file_name}.png", bbox_inches = 'tight', dpi=150,pad_inches = 0)

def spline_dataset(slip, resolution_alpha,method):    

    spline_list = []
    for i in range(10*resolution_alpha+1):
        for j in range(10*resolution_alpha+1):
            sp = spline(slip,1/resolution_alpha*j,1/resolution_alpha*i,method)
            spline_list += [[1/resolution_alpha*j,1/resolution_alpha*i,sp]]

    return spline_list

def plot_slip_distribution(file_name,value,kind):
    
    slip = [[0,0,0],[1,0,0],[3,0,0],[5,0,0],[7,0,0],[9,0,0],[10,0,0],
            [0,1,0],[1,1,value[0]],[3,1,value[1]],[5,1,value[2]],[7,1,value[3]],[9,1,value[4]],[10,1,0],
            [0,3,0],[1,3,value[5]],[3,3,value[6]],[5,3,value[7]],[7,3,value[8]],[9,3,value[9]],[10,3,0],
            [0,5,0],[1,5,value[10]],[3,5,value[11]],[5,5,value[12]],[7,5,value[13]],[9,5,value[14]],[10,5,0],
            [0,7,0],[1,7,value[15]],[3,7,value[16]],[5,7,value[17]],[7,7,value[18]],[9,7,value[19]],[10,7,0],
            [0,9,0],[1,9,value[20]],[3,9,value[21]],[5,9,value[22]],[7,9,value[23]],[9,9,value[24]],[10,9,0],
            [0,10,0],[1,10,0],[3,10,0],[5,10,0],[7,10,0],[9,10,0],[10,10,0]]
    
    # taus  = [[0,0,value[0]],[5,0,value[1]],[10,0,value[2]],
    #         [0,5,value[3]],[5,5,value[4]],[10,5,value[5]],
    #         [0,10,value[6]],[5,10,value[7]],[10,10,value[8]]]

    spline_data = spline_dataset(slip,resolution_alpha=1,method="cubic")
    slip_distribution(slip,spline_data,spline_trans(slip,method="cubic"),file_name,kind=kind)
    
#########
    
def spline(value_array,x,y,method): #スプライン曲面を張るための基準点を与えると、求めたい点のz座標の値を返す
    
    xuniq = sorted(list(set(list(map(lambda x: x[0],value_array)))))
    yuniq = sorted(list(set(list(map(lambda x: x[1],value_array)))))
    X, Y = np.meshgrid(xuniq, yuniq)
     
    # z軸の辞書を作成。一々探索を走らせると本当に時間がかかるので全てマッピングしておく
    zDict = dict()
    for row in value_array:
       # x, y の値に対応する z の値
        zDict[(row[0], row[1])] = row[2]
    # メッシュ中の各部分に対応する値の入った二次元データを生成
    Z = [] # Array<Array<number>>
    for yUniqIdx, y_ in enumerate(yuniq):
        Z.append([])
        for xUniqIdx, x_ in enumerate(xuniq):
            Z[yUniqIdx].append(zDict[(x_, y_)])
    Z = np.array(Z)
    
    # f_cubic = interpolate.interp2d(X, Y, Z, kind=kind)
    # z_cubic = f_cubic(x, y)
    
    data_grid = tuple([yuniq,xuniq])
    f = interpolate.RegularGridInterpolator(data_grid,Z,method=method)
    point = tuple([(y,x)])
    interp_value = f(point)
    
    return interp_value

def spline_trans(value_array,method) : #スプライン曲面を張るための基準点を与えると、小断層における値を返す
    
    spline_list = [] 
    for i in range(10):
        for j in range(10):
            if spline(value_array,0.5+j,0.5+i,method) < 0:
                spline_list += [0]
            else:    
                spline_list += [spline(value_array,0.5+j,0.5+i,method)]
    spline_array = np.array(spline_list)
    
    return spline_array

def spline_integral(value_array) : #スプライン曲面を張るための基準点を与えると、中心点から各点への積分値を計算し、小断層における値を返す。
    
    spline_list = []
    for y in range(10):
        for x in range(10):
            y0 = y+0.5
            x0 = x+0.5
            d = np.sqrt((5-x0)**2+(5-y0)**2)
            sum = 0
            for i in range(100):
                xi = x0 + (5-x0)*i/100
                yi = y0 + (5-y0)*i/100
                sum += spline(value_array,xi,yi)*d/100
            spline_list += [sum]
    spline_array = np.array(spline_list)
    
    return spline_array   

def yoffe_function(t,dmax,taus,taur): #規格化 Yoffe function
    
    de_pi = Decimal(str(np.pi))
    de_0 = Decimal("0")
    de_2 = Decimal("2")
    
    def K(taus,taur) :
        return  2/(np.pi*taur*taus**2)
    
    def C1(t,taur,de_t,de_taur) :
        return (t/2+taur/4)*float(np.sqrt(de_t*(de_taur-de_t)))+(t*taur-taur**2)*np.arcsin(float(np.sqrt(de_t/de_taur)))-3/4*taur**2*np.arctan(float(np.sqrt((de_taur-de_t)/de_t)))
    
    def C2(taur) :
        return 3/8*np.pi*taur**2
    
    def C3(t,taus,taur,de_t,de_taus,de_taur) :

        return (taus-t-taur/2)*float(np.sqrt((de_t-de_taus)*(de_taur-de_t+de_taus)))+taur*(2*taur-2*t+2*taus)*np.arcsin(float(np.sqrt((de_t-de_taus)/de_taur)))+3/2*taur**2*np.arctan(float(np.sqrt((de_taur-de_t+de_taus)/(de_t-de_taus))))
    
    def C4(t,taus,taur,de_t,de_taus,de_taur) :
  
        return (-taus+t/2+taur/4)*float(np.sqrt((de_t-2*de_taus)*(de_taur-de_t+2*de_taus)))+taur*(-taur+t-2*taus)*np.arcsin(float(np.sqrt((de_t-2*de_taus)/de_taur)))-3/4*taur**2*np.arctan(float(np.sqrt((de_taur-de_t+2*de_taus)/(de_t-2*de_taus))))
    
    def C5(t,taur):
        return np.pi/2*taur*(t-taur)
    
    def C6(t,taus,taur) :
        return np.pi/2*taur*(2*taus-t+taur)
    
    de_dmax = Decimal(str(dmax))
    de_t = Decimal(str(t))
    de_taus = Decimal(str(taus))
    de_taur = Decimal(str(taur))
    
    if de_taur >= de_2*de_taus :
        # print("taur >= 2*taus")
        if de_t <= de_0 :
            S = 0
        elif de_0 < de_t <= de_taus :
            S = dmax*K(taus,taur)*(C1(t,taur,de_t,de_taur)+C2(taur))
        elif de_taus < de_t <= de_2*de_taus :
            S = dmax*K(taus,taur)*(C1(t,taur,de_t,de_taur)-C2(taur)+C3(t,taus,taur,de_t,de_taus,de_taur))
        elif de_2*de_taus < de_t <= de_taur :
            S = dmax*K(taus,taur)*(C1(t,taur,de_t,de_taur)+C3(t,taus,taur,de_t,de_taus,de_taur)+C4(t,taus,taur,de_t,de_taus,de_taur))
        elif de_taur < de_t <= de_taur+de_taus:
            S = dmax*K(taus,taur)*(C5(t,taur)+C3(t,taus,taur,de_t,de_taus,de_taur)+C4(t,taus,taur,de_t,de_taus,de_taur))
        elif de_taur+de_taus < de_t <= (de_taur + de_2*de_taus) :
            S = dmax*K(taus,taur) *(C4(t,taus,taur,de_t,de_taus,de_taur)+C6(t,taus,taur))
        elif (de_taur + de_2*de_taus) < de_t:
            S = 0
        return S

    elif de_taus <= de_taur < de_2*de_taus :
        
        if de_t <= de_0:
            S = 0
        elif de_0 < de_t <= de_taus :
            S = dmax*K(taus,taur)*(C1(t,taur,de_t,de_taur)+C2(taur))
        elif de_taus < de_t <= de_taur :
            S = dmax*K(taus,taur)*(C1(t,taur,de_t,de_taur)-C2(taur)+C3(t,taus,taur,de_t,de_taus,de_taur))
        elif de_taur < de_t <= de_2*de_taus :
            S = dmax*K(taus,taur)*(C5(t,taur)+C3(t,taus,taur,de_t,de_taus,de_taur)-C2(taur))
        elif de_2*de_taus < de_t <= de_taus+de_taur :
            S = dmax*K(taus,taur)*(C5(t,taur)+C3(t,taus,taur,de_t,de_taus,de_taur)+C4(t,taus,taur,de_t,de_taus,de_taur))
        elif de_taus+de_taur < de_t <= (de_taur + de_2*de_taus) :
            S = dmax*K(taus,taur) *(C4(t,taus,taur,de_t,de_taus,de_taur)+C6(t,taus,taur))
        elif (de_taur + de_2*de_taus) < de_t:
            S = 0
        return S       

def data_set(value,type) :  
    
    if type == "reverse":
        slip = [[0,0,0],[1,0,0],[3,0,0],[5,0,0],[7,0,0],[9,0,0],[10,0,0],
            [0,1,0],[1,1,value[0]],[3,1,value[1]],[5,1,value[2]],[7,1,value[3]],[9,1,value[4]],[10,1,0],
            [0,3,0],[1,3,value[5]],[3,3,value[6]],[5,3,value[7]],[7,3,value[8]],[9,3,value[9]],[10,3,0],
            [0,5,0],[1,5,value[10]],[3,5,value[11]],[5,5,value[12]],[7,5,value[13]],[9,5,value[14]],[10,5,0],
            [0,7,0],[1,7,value[15]],[3,7,value[16]],[5,7,value[17]],[7,7,value[18]],[9,7,value[19]],[10,7,0],
            [0,9,0],[1,9,value[20]],[3,9,value[21]],[5,9,value[22]],[7,9,value[23]],[9,9,value[24]],[10,9,0],
            [0,10,0],[1,10,0],[3,10,0],[5,10,0],[7,10,0],[9,10,0],[10,10,0]]
        ts = [[0,0,5*np.sqrt(2)/value[25]],[5,0,5/value[25]],[10,0,5*np.sqrt(2)/value[25]],
            [0,5,5/value[25]],[5,5,0],[10,5,5/value[25]],
            [0,10,5*np.sqrt(2)/value[25]],[5,10,5/value[25]],[10,10,5*np.sqrt(2)/value[25]]]
        taus = [[0,0,value[26]],[5,0,value[26]],[10,0,value[26]],
                [0,5,value[26]],[5,5,value[26]],[10,5,value[26]],
                [0,10,value[26]],[5,10,value[26]],[10,10,value[26]]]
        taur = [[0,0,value[27]],[5,0,value[27]],[10,0,value[27]],
                [0,5,value[27]],[5,5,value[27]],[10,5,value[27]],
                [0,10,value[27]],[5,10,value[27]],[10,10,value[27]]]
        
        spdmax = spline_trans(np.array(slip),method="cubic")
        # print(spdmax)
        slip_ini_mean = np.mean(spdmax)
        print("逆断層の平均滑り量は",slip_ini_mean)
        spts = spline_trans(np.array(ts),method="linear")
        sptaus = spline_trans(np.array(taus),method="linear")
        sptaur = spline_trans(np.array(taur),method="linear")
        
        return spdmax,spts,sptaus,sptaur
    
    elif type == "strike":
        slip = [[0,0,0],[1,0,0],[3,0,0],[5,0,0],[7,0,0],[9,0,0],[10,0,0],
            [0,1,0],[1,1,value[0]],[3,1,value[1]],[5,1,value[2]],[7,1,value[3]],[9,1,value[4]],[10,1,0],
            [0,3,0],[1,3,value[5]],[3,3,value[6]],[5,3,value[7]],[7,3,value[8]],[9,3,value[9]],[10,3,0],
            [0,5,0],[1,5,value[10]],[3,5,value[11]],[5,5,value[12]],[7,5,value[13]],[9,5,value[14]],[10,5,0],
            [0,7,0],[1,7,value[15]],[3,7,value[16]],[5,7,value[17]],[7,7,value[18]],[9,7,value[19]],[10,7,0],
            [0,9,0],[1,9,value[20]],[3,9,value[21]],[5,9,value[22]],[7,9,value[23]],[9,9,value[24]],[10,9,0],
            [0,10,0],[1,10,0],[3,10,0],[5,10,0],[7,10,0],[9,10,0],[10,10,0]]
        ts = [[0,0,5*np.sqrt(2)/value[25]+value[26]],[5,0,5/value[25]+value[26]],[10,0,5*np.sqrt(2)/value[25]+value[26]],
            [0,5,5/value[25]+value[26]],[5,5,value[26]],[10,5,5/value[25]+value[26]],
            [0,10,5*np.sqrt(2)/value[25]+value[26]],[5,10,5/value[25]+value[26]],[10,10,5*np.sqrt(2)/value[25]+value[26]]]
        taus = [[0,0,value[27]],[5,0,value[27]],[10,0,value[27]],
                [0,5,value[27]],[5,5,value[27]],[10,5,value[27]],
                [0,10,value[27]],[5,10,value[27]],[10,10,value[27]]]
        taur = [[0,0,value[28]],[5,0,value[28]],[10,0,value[28]],
                [0,5,value[28]],[5,5,value[28]],[10,5,value[28]],
                [0,10,value[28]],[5,10,value[28]],[10,10,value[28]]]
        
        spdmax = spline_trans(np.array(slip),method="cubic")
        slip_ini_mean = np.mean(spdmax)
        print("横ずれ断層の平均滑り量は",slip_ini_mean)
        spts = spline_trans(np.array(ts),method="linear")
        sptaus = spline_trans(np.array(taus),method="linear")
        sptaur = spline_trans(np.array(taur),method="linear")
        
        return spdmax,spts,sptaus,sptaur
    
################

def calculate_wave_new(spdmax,sptaus,sptaur,spts,obs,type):

    max_t = 0
    max_tau = 0
    for i in range(10) :
        for j in range(11)[1:]:
            if max_t < spts[10*i+j-1] :
                max_t = spts[10*i+j-1]
            if sptaus[10*i+j-1] <= 0 :
                sptaus[10*i+j-1] = 0.01
            tau = sptaus[10*i+j-1]+sptaur[10*i+j-1]
            if max_tau < tau :
                max_tau = tau
            if spdmax[10*i+j-1] < 0:
                spdmax[10*i+j-1] = 0
        
    add_nts = int(format(max_t/0.01,".0f"))
    add_ntau = int(format((max_tau+1)/0.01,".0f"))
    a = np.load(f"./green_function/{type}/wave_{obs}/time_{obs}.npy")
    a_len = len(a)

    t = np.arange(0,(a_len+add_nts+add_ntau)*0.01,0.01)
    new_x = np.append(np.zeros_like(a),np.zeros(int(add_nts+add_ntau)))
    # print(len(new_x))
    new_y = np.append(np.zeros_like(a),np.zeros(int(add_nts+add_ntau)))
    new_z = np.append(np.zeros_like(a),np.zeros(int(add_nts+add_ntau)))
    for i in range(10):
        for j in range(11)[1:]:
            x = np.load(f"./green_function/{type}/wave_{obs}/vx{10*i+j}_{obs}.npy")/10000
            y = np.load(f"./green_function/{type}/wave_{obs}/vy{10*i+j}_{obs}.npy")/10000
            z = np.load(f"./green_function/{type}/wave_{obs}/vz{10*i+j}_{obs}.npy")/10000
            mag_x = np.append(np.zeros_like(a),np.zeros(int(add_ntau)))
            mag_y = np.append(np.zeros_like(a),np.zeros(int(add_ntau)))
            mag_z = np.append(np.zeros_like(a),np.zeros(int(add_ntau)))
            yoffe = np.zeros(int(format((sptaus[10*i+j-1]+sptaur[10*i+j-1]+1)/0.01,".0f")))
            for before_ntau in range(int(format((sptaus[10*i+j-1]+sptaur[10*i+j-1]+1)/0.01,".0f"))):
                yoffe[before_ntau] = yoffe_function(0.01*before_ntau,spdmax[10*i+j-1]*100,sptaus[10*i+j-1],sptaur[10*i+j-1])
            mag_x = signal.fftconvolve(yoffe,x)
            mag_y = signal.fftconvolve(yoffe,y)
            mag_z = signal.fftconvolve(yoffe,z)
                # after_ntau = add_ntau-before_ntau
                # before_ntau_zero = np.zeros(before_ntau)
                # after_ntau_zero = np.zeros(after_ntau)
                # mag = yoffe_function(0.02*before_ntau,spdmax[10*i+j-1]*100,sptaus[10*i+j-1],sptaur[10*i+j-1])
                # mag_x += np.append(before_ntau_zero,np.append(x,after_ntau_zero))*mag
                # mag_y += np.append(before_ntau_zero,np.append(y,after_ntau_zero))*mag
                # mag_z += np.append(before_ntau_zero,np.append(z,after_ntau_zero))*mag
            # plt.plot(np.arange(0,0.01*len(mag_x),0.01),mag_x)
            # plt.show()   
            ts_number = spts[10*i+j-1]/0.01
            if ts_number < 0 :
                ts_number = 0 
            before_nts = math.floor(ts_number) #何マス分時間ずれがあるか
            after_nts = add_nts-before_nts
            before_nts_zero = np.zeros(before_nts)
            after_nts_zero = np.zeros(after_nts)
            shift_x = np.append(before_nts_zero,np.append(mag_x,after_nts_zero)) #時間ずれがある分、タイムウィンドウを増やす
            shift_y = np.append(before_nts_zero,np.append(mag_y,after_nts_zero)) 
            shift_z = np.append(before_nts_zero,np.append(mag_z,after_nts_zero))
            # print(len(shift_x))
            for n in range(len(shift_x)):
                if n <= before_nts :
                    new_x[n] += 0
                    new_y[n] += 0
                    new_z[n] += 0
                else :
                    new_x[n] += (shift_x[n]-shift_x[n-1])/0.01*(0.01-(spts[10*i+j-1]-0.01*before_nts)) + shift_x[n-1] 
                    new_y[n] += (shift_y[n]-shift_y[n-1])/0.01*(0.01-(spts[10*i+j-1]-0.01*before_nts)) + shift_y[n-1] 
                    new_z[n] += (shift_z[n]-shift_z[n-1])/0.01*(0.01-(spts[10*i+j-1]-0.01*before_nts)) + shift_z[n-1]
            # plt.plot(t,new_x)
            # plt.show()        
            # new_x += shift_x
            # new_y += shift_y
            # new_z += shift_z
    return t,new_x,new_y,new_z    
   
def l2(initial):
    
    print(initial)

    spdmax_re,spts_re,sptaus_re,sptaur_re = data_set(initial[0:28],"reverse")
    spdmax_st,spts_st,sptaus_st,sptaur_st = data_set(initial[28:57],"strike")
    
    obs_list = ["taka",
                "momo",
                "O2",
                "sira",
                "abu",
                "tama",
                "hig",
                "naka",
                "ban",
                "KH7",
                "O3",
                "K11",
                "O4",
                "O5",
                "KH8",
                "K13"]
    
    shift_dic = {"taka":1400,
                 "momo":1400,
                 "O2":1300,
                 "sira":9400,
                 "abu":3390,
                 "tama":2400,
                 "hig":1400,
                 "naka":2460,
                 "ban":1235,
                 "KH7":1210,
                 "O3":1110,
                 "K11":1110,
                 "O4":1210,
                 "O5":1110,
                 "KH8":1110,
                 "K13":1110
                 }
    
    range_dic = {"taka":[250,750],
                 "momo":[250,750],
                 "O2":[250,750],
                 "sira":[250,1000],
                 "abu":[250,750],
                 "tama":[250,1000],
                 "hig":[250,1000],
                 "naka":[250,1000],
                 "ban":[250,1000],
                 "KH7":[250,1000],
                 "O3":[400,1250],
                 "K11":[400,1000],
                 "O4":[300,1000],
                 "O5":[300,1250],
                 "KH8":[400,1250],
                 "K13":[350,1000]
                 }
    
    def process(obs):
        
        x_obs_ = np.load(f"./velocity/{obs}_ns.npy")[shift_dic[obs]:]
        y_obs_ = np.load(f"./velocity/{obs}_ew.npy")[shift_dic[obs]:]
        z_obs_ = np.load(f"./velocity/{obs}_ud.npy")[shift_dic[obs]:]
        x_obs = x_obs_[range_dic[obs][0]:range_dic[obs][1]]
        y_obs = y_obs_[range_dic[obs][0]:range_dic[obs][1]]
        z_obs = z_obs_[range_dic[obs][0]:range_dic[obs][1]]
        
        t_re,x_re,y_re,z_re = calculate_wave_new(spdmax=spdmax_re,
                                                sptaus=sptaus_re,
                                                sptaur=sptaur_re,
                                                spts=spts_re,
                                                obs=obs,
                                                type="reverse")
        t_st,x_st,y_st,z_st = calculate_wave_new(spdmax=spdmax_st,
                                                 sptaus=sptaus_st,
                                                 sptaur=sptaur_st,
                                                 spts=spts_st,
                                                 obs=obs,
                                                 type="strike")
        
        if len(t_re) >= len(t_st):
            add_zeros = np.zeros(len(t_re)-len(t_st))
            x_st_new = np.append(x_st,add_zeros)
            y_st_new = np.append(y_st,add_zeros)
            z_st_new = np.append(z_st,add_zeros)
            x_ = x_re + x_st_new
            y_ = y_re + y_st_new
            z_ = z_re + z_st_new
            ini_t = t_re
            
        else:
            add_zeros = np.zeros(len(t_st)-len(t_re))
            x_re_new = np.append(x_re,add_zeros)
            y_re_new = np.append(y_re,add_zeros)
            z_re_new = np.append(z_re,add_zeros)
            x_ = x_st + x_re_new
            y_ = y_st + y_re_new
            z_ = z_st + z_re_new
            ini_t = t_st
        
        x_syn = x_[range_dic[obs][0]:range_dic[obs][1]]
        y_syn = y_[range_dic[obs][0]:range_dic[obs][1]]
        z_syn = z_[range_dic[obs][0]:range_dic[obs][1]]
        
        t = np.arange(0,0.01*len(x_syn),0.01)
        
        sim_x = similarity.Similarity(t,x_obs,x_syn,"")
        sim_y = similarity.Similarity(t,y_obs,y_syn,"")
        sim_z = similarity.Similarity(t,z_obs,z_syn,"")
        l2_x = sim_x.l2_norm()
        l2_y = sim_y.l2_norm()
        l2_z = sim_z.l2_norm()
        l2 = l2_x + l2_y + l2_z

        return l2
    
    l2_list = Parallel(n_jobs=-1)([delayed(process)(obs) for obs in obs_list])
    l2_sum = sum(l2_list)
    
    print("l2=",l2_sum)
    with open("./convergence_txt/l2_range.txt",mode="a") as f:
        f.write(f"{l2_sum}\n")
        f.close()
    
    return l2_sum

def was(initial):

    print(initial)

    spdmax_re,spts_re,sptaus_re,sptaur_re = data_set(initial[0:28],"reverse")
    spdmax_st,spts_st,sptaus_st,sptaur_st = data_set(initial[28:57],"strike")
    
    obs_list = ["taka",
                "momo",
                "O2",
                "sira",
                "abu",
                "tama",
                "hig",
                "naka",
                "ban",
                "KH7",
                "O3",
                "K11",
                "O4",
                "O5",
                "KH8",
                "K13"]
    
    shift_dic = {"taka":1400,
                 "momo":1400,
                 "O2":1300,
                 "sira":9400,
                 "abu":3390,
                 "tama":2400,
                 "hig":1400,
                 "naka":2460,
                 "ban":1235,
                 "KH7":1210,
                 "O3":1110,
                 "K11":1110,
                 "O4":1210,
                 "O5":1110,
                 "KH8":1110,
                 "K13":1110
                 }
    
    range_dic = {"taka":[250,750],
                 "momo":[250,750],
                 "O2":[250,750],
                 "sira":[250,1000],
                 "abu":[250,750],
                 "tama":[250,1000],
                 "hig":[250,1000],
                 "naka":[250,1000],
                 "ban":[250,1000],
                 "KH7":[250,1000],
                 "O3":[400,1250],
                 "K11":[400,1000],
                 "O4":[300,1000],
                 "O5":[300,1250],
                 "KH8":[400,1250],
                 "K13":[350,1000]
                 }
    
    def process(obs):
        
        x_obs_ = np.load(f"./velocity/{obs}_ns.npy")[shift_dic[obs]:]
        y_obs_ = np.load(f"./velocity/{obs}_ew.npy")[shift_dic[obs]:]
        z_obs_ = np.load(f"./velocity/{obs}_ud.npy")[shift_dic[obs]:]
        x_obs = x_obs_[range_dic[obs][0]:range_dic[obs][1]]
        y_obs = y_obs_[range_dic[obs][0]:range_dic[obs][1]]
        z_obs = z_obs_[range_dic[obs][0]:range_dic[obs][1]]
        
        t_re,x_re,y_re,z_re = calculate_wave_new(spdmax=spdmax_re,
                                                sptaus=sptaus_re,
                                                sptaur=sptaur_re,
                                                spts=spts_re,
                                                obs=obs,
                                                type="reverse")
        t_st,x_st,y_st,z_st = calculate_wave_new(spdmax=spdmax_st,
                                                 sptaus=sptaus_st,
                                                 sptaur=sptaur_st,
                                                 spts=spts_st,
                                                 obs=obs,
                                                 type="strike")
        
        if len(t_re) >= len(t_st):
            add_zeros = np.zeros(len(t_re)-len(t_st))
            x_st_new = np.append(x_st,add_zeros)
            y_st_new = np.append(y_st,add_zeros)
            z_st_new = np.append(z_st,add_zeros)
            x_ = x_re + x_st_new
            y_ = y_re + y_st_new
            z_ = z_re + z_st_new
            ini_t = t_re
            
        else:
            add_zeros = np.zeros(len(t_st)-len(t_re))
            x_re_new = np.append(x_re,add_zeros)
            y_re_new = np.append(y_re,add_zeros)
            z_re_new = np.append(z_re,add_zeros)
            x_ = x_st + x_re_new
            y_ = y_st + y_re_new
            z_ = z_st + z_re_new
            ini_t = t_st
        
        x_syn = x_[range_dic[obs][0]:range_dic[obs][1]]
        y_syn = y_[range_dic[obs][0]:range_dic[obs][1]]
        z_syn = z_[range_dic[obs][0]:range_dic[obs][1]]
        
        t = np.arange(0,0.01*len(x_syn),0.01)

        sim_x = similarity.Similarity(t,x_obs,x_syn,"")
        sim_y = similarity.Similarity(t,y_obs,y_syn,"")
        sim_z = similarity.Similarity(t,z_obs,z_syn,"")
        was_x = sim_x.wasserstein_softplus_normalizing(con=3)
        was_y = sim_y.wasserstein_softplus_normalizing(con=3)
        was_z = sim_z.wasserstein_softplus_normalizing(con=3)
        was = was_x + was_y + was_z
        
        return was
        
    was_list = Parallel(n_jobs=-1)([delayed(process)(obs) for obs in obs_list])
    was_sum = sum(was_list)
    print("was=",was_sum)
    with open("./convergence_txt/was_range.txt",mode="a") as f:
        f.write(f"{was_sum}\n")
        f.close()
    
    return was_sum       

def main(evaluate_function,method):
    
    time_sta = time.time()

    initial_condition = initial_set()
    
    def cons_re(initial):
        
        slip_ini = [[0,0,0],[1,0,0],[3,0,0],[5,0,0],[7,0,0],[9,0,0],[10,0,0],
                [0,1,0],[1,1,initial[0]],[3,1,initial[1]],[5,1,initial[2]],[7,1,initial[3]],[9,1,initial[4]],[10,1,0],
                [0,3,0],[1,3,initial[5]],[3,3,initial[6]],[5,3,initial[7]],[7,3,initial[8]],[9,3,initial[9]],[10,3,0],
                [0,5,0],[1,5,initial[10]],[3,5,initial[11]],[5,5,initial[12]],[7,5,initial[13]],[9,5,initial[14]],[10,5,0],
                [0,7,0],[1,7,initial[15]],[3,7,initial[16]],[5,7,initial[17]],[7,7,initial[18]],[9,7,initial[19]],[10,7,0],
                [0,9,0],[1,9,initial[20]],[3,9,initial[21]],[5,9,initial[22]],[7,9,initial[23]],[9,9,initial[24]],[10,9,0],
                [0,10,0],[1,10,0],[3,10,0],[5,10,0],[7,10,0],[9,10,0],[10,10,0]]
        spdmax_ini = spline_trans(np.array(slip_ini),method="cubic")
        slip_ini_mean = np.mean(spdmax_ini)
        
        return slip_ini_mean - 0.0433
    
    def cons_st(initial):
        
        slip_ini = [[0,0,0],[1,0,0],[3,0,0],[5,0,0],[7,0,0],[9,0,0],[10,0,0],
                [0,1,0],[1,1,initial[25]],[3,1,initial[26]],[5,1,initial[27]],[7,1,initial[28]],[9,1,initial[29]],[10,1,0],
                [0,3,0],[1,3,initial[30]],[3,3,initial[31]],[5,3,initial[32]],[7,3,initial[33]],[9,3,initial[34]],[10,3,0],
                [0,5,0],[1,5,initial[35]],[3,5,initial[36]],[5,5,initial[37]],[7,5,initial[38]],[9,5,initial[39]],[10,5,0],
                [0,7,0],[1,7,initial[40]],[3,7,initial[41]],[5,7,initial[42]],[7,7,initial[43]],[9,7,initial[44]],[10,7,0],
                [0,9,0],[1,9,initial[45]],[3,9,initial[46]],[5,9,initial[47]],[7,9,initial[48]],[9,9,initial[49]],[10,9,0],
                [0,10,0],[1,10,0],[3,10,0],[5,10,0],[7,10,0],[9,10,0],[10,10,0]]
        spdmax_ini = spline_trans(np.array(slip_ini),method="cubic")
        slip_ini_mean = np.mean(spdmax_ini)
        
        return slip_ini_mean - 0.0738
    
    con = ({"type":"eq","fun":cons_re},
           {"type":"eq","fun":cons_st})
    bou = ((-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
           (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
           (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
           (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
           (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
           (2.5,3),
           (0.02,0.05),
           (0.2,0.3),
        #    (2.5,3),(2.5,3),(2.5,3),
        #    (2.5,3),(2.5,3),
        #    (2.5,3),(2.5,3),(2.5,3),
        #    (0.02,0.1),(0.02,0.1),(0.02,0.1),
        #    (0.02,0.1),(0.02,0.1),(0.02,0.1),
        #    (0.02,0.1),(0.02,0.1),(0.05,0.1),
        #    (0.2,0.3),(0.2,0.3),(0.2,0.3),
        #    (0.2,0.3),(0.2,0.3),(0.2,0.3),
        #    (0.2,0.3),(0.2,0.3),(0.2,0.3),
           (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
           (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
           (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
           (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
           (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
           (2.5,3),(0.3,0.4),
           (0.02,0.05),
           (0.2,0.3))
        #    (2.5,3),(2.5,3),(2.5,3),
        #    (0.02,0.1),(0.02,0.1),(0.02,0.1),
        #    (0.02,0.1),(0.02,0.1),(0.02,0.1),
        #    (0.02,0.1),(0.02,0.1),(0.05,0.1),
        #    (0.2,0.3),(0.2,0.3),(0.2,0.3),
        #    (0.2,0.3),(0.2,0.3),(0.2,0.3),
        #    (0.2,0.3),(0.2,0.3),(0.2,0.3)) 
    # options={"eps":0.01}
    if evaluate_function == "was" :        
        result = opt.minimize(fun=was,
                            x0=initial_condition,
                            bounds=bou,
                            constraints=con,
                            method=method
                            )
    elif evaluate_function == "l2" :
         result = opt.minimize(fun=l2,
                            x0=initial_condition,
                            bounds=bou,
                            constraints=con,
                            method=method
                            )       
    print(result["x"])
    times = result["nit"]
    time_end = time.time()
    tim = time_end - time_sta
    hour = tim/3600
    print(f"反復回数は{times}回です")
    print(f"計算時間は{hour}時間です")        

def initial_set():
    
    initial_reverse = [0.0433,0.0433,0.0433,0.0433,0.0433,
                       0.0433,0.0433,0.0433,0.0433,0.0433,
                       0.0433,0.0433,0.0433,0.0433,0.0433,
                       0.0433,0.0433,0.0433,0.0433,0.0433,
                       0.0433,0.0433,0.0433,0.0433,0.0433,
                       2.7,
                       0.03,
                       0.214]
                       
    
    initial_strike = [0.0738,0.0738,0.0738,0.0738,0.0738,
                      0.0738,0.0738,0.0738,0.0738,0.0738,
                      0.0738,0.0738,0.0738,0.0738,0.0738,
                      0.0738,0.0738,0.0738,0.0738,0.0738,
                      0.0738,0.0738,0.0738,0.0738,0.0738,
                      2.7,0.3,
                      0.04,
                      0.255]

    initial_reverse.extend(initial_strike)
    return initial_reverse

def main_basinhopping():
    
    time_sta = time.time()

    initial_condition = [0.2,0.2,0.2,
                         0.2,0.2,0.2,
                         0.2,0.2,0.2, #滑り量
                        5*np.sqrt(2)/1.8,5/1.8,5*np.sqrt(2)/1.8,
                        5/1.8,5/1.8,
                        5*np.sqrt(2)/1.8,5/1.8,5*np.sqrt(2)/1.8,
                         1.5,1.5,1.5,
                         1.5,1.5,1.5,
                         1.5,1.5,1.5,
                         3,3,3,
                         3,3,3,
                         3,3,3
                         ]
    
    def cons(initial):
        
        slip_ini = [[0,0,0],[0,2.5,0],[0,5,0],[0,7.5,0],[0,10,0],
                [2.5,0,0],[2.5,2.5,initial[0]],[2.5,5,initial[1]],[2.5,7.5,initial[2]],[2.5,10,0],
                [5,0,0],[5,2.5,initial[3]],[5,5,initial[4]],[5,7.5,initial[5]],[5,10,0],
                [7.5,0,0],[7.5,2.5,initial[6]],[7.5,5,initial[7]],[7.5,7.5,initial[8]],[7.5,10,0],
                [10,0,0],[10,2.5,0],[10,5,0],[10,7.5,0],[10,10,0]]
        spdmax_ini = spline_trans(np.array(slip_ini),kind="cubic")
        slip_ini_all = sum(spdmax_ini)
        
        model = [0.1,0.1,0.1,
                 0.1,0.5,0.1,
                 0.1,0.1,0.1]
        slip_model = [[0,0,0],[0,2.5,0],[0,5,0],[0,7.5,0],[0,10,0],
                    [2.5,0,0],[2.5,2.5,model[0]],[2.5,5,model[1]],[2.5,7.5,model[2]],[2.5,10,0],
                    [5,0,0],[5,2.5,model[3]],[5,5,model[4]],[5,7.5,model[5]],[5,10,0],
                    [7.5,0,0],[7.5,2.5,model[6]],[7.5,5,model[7]],[7.5,7.5,model[8]],[7.5,10,0],
                    [10,0,0],[10,2.5,0],[10,5,0],[10,7.5,0],[10,10,0]]
        spdmax_model = spline_trans(np.array(slip_model),kind="cubic")
        slip_model_all = sum(spdmax_model)
        
        return slip_ini_all - slip_model_all
    
    con = (
            # {"type":"ineq","fun":lambda x : x[26]-x[17]},
    #         {"type":"ineq","fun":lambda x : x[27]-x[18]},
    #         {"type":"ineq","fun":lambda x : x[28]-x[19]},
    #         {"type":"ineq","fun":lambda x : x[29]-x[20]},
    #         {"type":"ineq","fun":lambda x : x[30]-x[21]},
    #         {"type":"ineq","fun":lambda x : x[31]-x[22]},
    #         {"type":"ineq","fun":lambda x : x[32]-x[23]},
    #         {"type":"ineq","fun":lambda x : x[33]-x[24]},
    #         {"type":"ineq","fun":lambda x : x[34]-x[25]},
            {"type":"eq","fun":cons}
            )
    bou = ((0.05,1),(0.05,1),(0.05,1),
           (0.05,1),(0.05,1),(0.05,1),
           (0.05,1),(0.05,1),(0.05,1),
           (1.5,4),(1.5,4),(1.5,4),
           (1.5,4),(1.5,4),
           (1.5,4),(1.5,4),(1.5,4),
           (0.1,1.5),(0.1,1.5),(0.1,1.5),
           (0.1,1.5),(0.1,1.5),(0.1,1.5),
           (0.1,1.5),(0.1,1.5),(0.1,1.5),
           (1.5,4),(1.5,4),(1.5,4),
           (1.5,4),(1.5,4),(1.5,4),
           (1.5,4),(1.5,4),(1.5,4)) 
 
    result = opt.basinhopping(func=l2,
                            x0=initial_condition,
                            minimizer_kwargs={"method":"SLSQP","bounds":bou,"constraints":con},
                            )
    print(result.x)
    times = result.nit
    time_end = time.time()
    tim = time_end - time_sta
    hour = tim/3600
    print(f"反復回数は{times}回です")
    print(f"計算時間は{hour}時間です")   

def plot_wave(): #合成波形をすべて描画する

    initial = initial_set()

    spdmax_re,spts_re,sptaus_re,sptaur_re = data_set(initial[0:51],"reverse")
    spdmax_st,spts_st,sptaus_st,sptaur_st = data_set(initial[51:103],"strike")
    
    obs_list = ["taka","momo","take","O2","sira","abu","tama","hig","naka","ban","OH4","KH7","O3","K11","K12","OH2","O6","K9"]
    obs_name = {"taka":"高槻市",
                "momo":"桃園",
                "take":"竹ノ内",
                "O2":"OSK002(高槻)",
                "sira":"白川",
                "abu":"阿武山",
                "tama":"玉島",
                "hig":"東中条",
                "naka":"中穂積",
                "ban":"万博",
                "OH4":"OSKH04(交野)",
                "KH7":"KYTH07(久御山)",
                "O3":"OSK003(豊中)",
                "K11":"KYT011(亀岡)",
                "K12":"KYT012(京都)",
                "OH2":"OSKH02(此花)",
                "O6":"OSK006(堺)",
                "K9":"KYT009(日吉)"}
    
    
    def plot(tim,model_x,model_y,model_z):
        
        if i==0 :
            ax[i,0].plot(tim,model_y,color="black",label="model_wave",linewidth=1)
            ax[i,0].axis("off")
            ax[i,0].set_xlim(0,20)
            maxabsy = max(abs(model_y))
            ax[i,0].set_ylim(-maxabsy,maxabsy)
            ax[i,0].text(10, maxabsy*2, "EW(cm/s)", ha='center')
            ax[i,0].text(-10, 0, f"{obs_name[obs]}", va='center')
            ax[i,0].text(17,maxabsy,f"{round(maxabsy,sigfigs=3)}")
            
            ax[i,1].plot(tim,model_x,color="black",label="model_wave",linewidth=1)
            ax[i,1].axis("off")
            ax[i,1].set_xlim(0,20)
            maxabsx = max(abs(model_x))
            ax[i,1].set_ylim(-maxabsx,maxabsx)
            ax[i,1].text(10, maxabsx*2, "NS(cm/s)", ha='center')
            ax[i,1].text(17,maxabsx,f"{round(maxabsx,sigfigs=3)}")
            
            ax[i,2].plot(tim,model_z,color="black",label="model_wave",linewidth=1)
            ax[i,2].axis("off")
            ax[i,2].set_xlim(0,20)
            maxabsz = max(abs(model_z))
            ax[i,2].set_ylim(-maxabsz,maxabsz)
            ax[i,2].text(10, maxabsz*2, "UD(cm/s)", ha='center')
            ax[i,2].text(17,maxabsz,f"{round(maxabsz,sigfigs=3)}")
            
        elif i == 17:
            
            ax[i,0].plot(tim,model_y,color="black",label="model_wave",linewidth=1)
            ax[i,0].set_xlim(0,20)
            maxabsy = max(abs(model_y))
            ax[i,0].set_ylim(-maxabsy,maxabsy)
            [ax[i,0].spines[side].set_visible(False) for side in ['right','top',"left"]]
            ax[i,0].yaxis.set_visible(False)
            ax[i,0].set_xlabel("Time[s]")
            ax[i,0].text(-10, 0, f"{obs_name[obs]}", va='center')
            ax[i,0].text(17,maxabsy,f"{round(maxabsy,sigfigs=3)}")
            
            ax[i,1].plot(tim,model_x,color="black",label="model_wave",linewidth=1)
            ax[i,1].set_xlim(0,20)
            maxabsx = max(abs(model_x))
            ax[i,1].set_ylim(-maxabsx,maxabsx)
            [ax[i,1].spines[side].set_visible(False) for side in ['right','top',"left"]]
            ax[i,1].yaxis.set_visible(False)
            ax[i,1].set_xlabel("Time[s]")
            ax[i,1].text(17,maxabsx,f"{round(maxabsx,sigfigs=3)}")
        
            ax[i,2].plot(tim,model_z,color="black",label="model_wave",linewidth=1)
            ax[i,2].set_xlim(0,20)
            maxabsz = max(abs(model_z))
            ax[i,2].set_ylim(-maxabsz,maxabsz)
            [ax[i,2].spines[side].set_visible(False) for side in ['right','top',"left"]]
            ax[i,2].yaxis.set_visible(False)
            ax[i,2].set_xlabel("Time[s]")
            ax[i,2].text(17,maxabsz,f"{round(maxabsz,sigfigs=3)}")

        else :
            ax[i,0].plot(tim,model_y,color="black",label="model_wave",linewidth=1)
            ax[i,0].axis("off")
            ax[i,0].set_xlim(0,20)
            maxabsy = max(abs(model_y))
            ax[i,0].set_ylim(-maxabsy,maxabsy)
            ax[i,0].text(-10, 0, f"{obs_name[obs]}", va='center')
            ax[i,0].text(17,maxabsy,f"{round(maxabsy,sigfigs=3)}")
            
            ax[i,1].plot(tim,model_x,color="black",label="model_wave",linewidth=1)
            ax[i,1].axis("off")
            ax[i,1].set_xlim(0,20)
            maxabsx = max(abs(model_x))
            ax[i,1].set_ylim(-maxabsx,maxabsx)
            ax[i,1].text(17,maxabsx,f"{round(maxabsx,sigfigs=3)}")
            
            ax[i,2].plot(tim,model_z,color="black",label="model_wave",linewidth=1)
            ax[i,2].axis("off")
            ax[i,2].set_xlim(0,20)
            maxabsz = max(abs(model_z))
            ax[i,2].set_ylim(-maxabsz,maxabsz)
            ax[i,2].text(17,maxabsz,f"{round(maxabsz,sigfigs=3)}")
        
    fig, ax = plt.subplots(nrows=18,ncols=3,sharex=False,figsize=(10,8))
    for i,obs in tqdm(zip(range(18),obs_list)) : 
        t_re,x_re,y_re,z_re = calculate_wave_new(spdmax=spdmax_re,
                                                sptaus=sptaus_re,
                                                sptaur=sptaur_re,
                                                spts=spts_re,
                                                obs=obs,
                                                type="reverse")
        t_st,x_st,y_st,z_st = calculate_wave_new(spdmax=spdmax_st,
                                                 sptaus=sptaus_st,
                                                 sptaur=sptaur_st,
                                                 spts=spts_st,
                                                 obs=obs,
                                                 type="strike")
        
        if len(t_re) >= len(t_st):
            add_zeros = np.zeros(len(t_re)-len(t_st))
            x_st_new = np.append(x_st,add_zeros)
            y_st_new = np.append(y_st,add_zeros)
            z_st_new = np.append(z_st,add_zeros)
            x = x_re + x_st_new
            y = y_re + y_st_new
            z = z_re + z_st_new
            ini_t = t_re
            
        else:
            add_zeros = np.zeros(len(t_st)-len(t_re))
            x_re_new = np.append(x_re,add_zeros)
            y_re_new = np.append(y_re,add_zeros)
            z_re_new = np.append(z_re,add_zeros)
            x = x_st + x_re_new
            y = y_st + y_re_new
            z = z_st + z_re_new
            ini_t = t_st
            
        plot(ini_t,x,y,z)
                 
    fig.savefig("wave_test_.png")
  
def plot_wave_all(): #観測波形と合成波形をすべて描画する
    
    initial = initial_set()

    spdmax_re,spts_re,sptaus_re,sptaur_re = data_set(initial[0:25],"reverse")
    spdmax_st,spts_st,sptaus_st,sptaur_st = data_set(initial[25:51],"strike")   
    
    obs_name = {"taka":"高槻市",
                "momo":"桃園",
                "O2":"OSK002(高槻)",
                "sira":"白川",
                "abu":"阿武山",
                "tama":"玉島",
                "hig":"東中条",
                "naka":"中穂積",
                "ban":"万博",
                "KH7":"KYTH07(久御山)",
                "O3":"OSK003(豊中)",
                "K11":"KYT011(亀岡)",
                "O4":"OSK004(四條畷)",
                "O5":"OSK005(大阪)",
                "KH8":"KYTH08(京都)",
                "K13":"KYT013(宇治)"}
    
    shift_dic = {"taka":1400,
                 "momo":1400,
                 "O2":1300,
                 "sira":9400,
                 "abu":3390,
                 "tama":2400,
                 "hig":1400,
                 "naka":2460,
                 "ban":1235,
                 "KH7":1210,
                 "O3":1110,
                 "K11":1110,
                 "O4":1210,
                 "O5":1110,
                 "KH8":1110,
                 "K13":1110
                 }
    

    # con_spdmax,con_spts,con_sptaus,con_sptaur = data_set(con)
    # print(con_spts)
    def plot(tim,obs_x,obs_y,obs_z,ini_x,ini_y,ini_z):
        
        if i==0 :
            ax[i,0].plot(tim,obs_y,color="black",label="obs_wave",linewidth=0.5)
            ax[i,0].plot(tim,ini_y,color="red",label="syn_wave",linewidth=0.5)
            # ax[i,0].plot(tim,con_y,color="red",label="convergence_wave",linewidth=1)
            ax[i,0].axis("off")
            ax[i,0].set_xlim(0,20)
            maxabsy = max(max(abs(ini_y)),max(abs(obs_y)))
            ax[i,0].set_ylim(-maxabsy,maxabsy)
            ax[i,0].text(10, maxabsy*2, "EW(cm/s)", ha='center')
            ax[i,0].text(-10, 0, f"{obs_name[obs]}", va='center',fontsize=7)
            ax[i,0].text(15,maxabsy,f"{round(max(abs(ini_y)),sigfigs=3)}",fontsize=5,color="red")
            ax[i,0].text(17.5,maxabsy,f"{round(max(abs(obs_y)),sigfigs=3)}",fontsize=5)
            
            ax[i,1].plot(tim,obs_x,color="black",label="obs_wave",linewidth=0.5)
            ax[i,1].plot(tim,ini_x,color="red",label="syn_wave",linewidth=0.5)
            # ax[i,1].plot(tim,con_x,color="red",label="convergence_wave",linewidth=1)
            ax[i,1].axis("off")
            ax[i,1].set_xlim(0,20)
            maxabsx = max(max(abs(ini_x)),max(abs(obs_x)))
            ax[i,1].set_ylim(-maxabsx,maxabsx)
            ax[i,1].text(10, maxabsx*2, "NS(cm/s)", ha='center')
            ax[i,1].text(15,maxabsx,f"{round(max(abs(ini_x)),sigfigs=3)}",fontsize=5,color="red")
            ax[i,1].text(17.5,maxabsx,f"{round(max(abs(obs_x)),sigfigs=3)}",fontsize=5)
            
            ax[i,2].plot(tim,obs_z,color="black",label="obs_wave",linewidth=0.5)
            ax[i,2].plot(tim,ini_z,color="red",label="syn_wave",linewidth=0.5)
            # ax[i,2].plot(tim,con_z,color="red",label="convergence_wave",linewidth=1)
            ax[i,2].axis("off")
            ax[i,2].set_xlim(0,20)
            maxabsz = max(max(abs(ini_z)),max(abs(obs_z)))
            ax[i,2].set_ylim(-maxabsz,maxabsz)
            ax[i,2].text(10, maxabsz*2, "UD(cm/s)", ha='center')
            ax[i,2].text(15,maxabsz,f"{round(max(abs(ini_z)),sigfigs=3)}",fontsize=5,color="red")
            ax[i,2].text(17.5,maxabsz,f"{round(max(abs(obs_z)),sigfigs=3)}",fontsize=5)
            
        elif i == 17:
            
            ax[i,0].plot(tim,obs_y,color="black",label="obs_wave",linewidth=0.5)
            ax[i,0].plot(tim,ini_y,color="red",label="syn_wave",linewidth=0.5)
            # ax[i,0].plot(tim,con_y,color="red",label="convergence_wave",linewidth=1)
            ax[i,0].set_xlim(0,20)
            maxabsy = max(max(abs(ini_y)),max(abs(obs_y)))
            ax[i,0].set_ylim(-maxabsy,maxabsy)
            [ax[i,0].spines[side].set_visible(False) for side in ['right','top',"left"]]
            ax[i,0].yaxis.set_visible(False)
            ax[i,0].set_xlabel("Time[s]")
            ax[i,0].text(-10, 0, f"{obs_name[obs]}", va='center',fontsize=7)
            ax[i,0].text(15,maxabsy,f"{round(max(abs(ini_y)),sigfigs=3)}",fontsize=5,color="red")
            ax[i,0].text(17.5,maxabsy,f"{round(max(abs(obs_y)),sigfigs=3)}",fontsize=5)
            
            ax[i,1].plot(tim,obs_x,color="black",label="obs_wave",linewidth=0.5)
            ax[i,1].plot(tim,ini_x,color="red",label="syn_wave",linewidth=0.5)
            # ax[i,1].plot(tim,con_x,color="red",label="convergence_wave",linewidth=1)
            ax[i,1].set_xlim(0,20)
            maxabsx = max(max(abs(ini_x)),max(abs(obs_x)))
            ax[i,1].set_ylim(-maxabsx,maxabsx)
            [ax[i,1].spines[side].set_visible(False) for side in ['right','top',"left"]]
            ax[i,1].yaxis.set_visible(False)
            ax[i,1].set_xlabel("Time[s]")
            ax[i,1].text(15,maxabsx,f"{round(max(abs(ini_x)),sigfigs=3)}",fontsize=5,color="red")
            ax[i,1].text(17.5,maxabsx,f"{round(max(abs(obs_x)),sigfigs=3)}",fontsize=5)
        
            ax[i,2].plot(tim,obs_z,color="black",label="obs_wave",linewidth=0.5)
            ax[i,2].plot(tim,ini_z,color="red",label="syn_wave",linewidth=0.5)
            # ax[i,2].plot(tim,con_z,color="red",label="convergence_wave",linewidth=1)
            ax[i,2].set_xlim(0,20)
            maxabsz = max(max(abs(ini_z)),max(abs(obs_z)))
            ax[i,2].set_ylim(-maxabsz,maxabsz)
            [ax[i,2].spines[side].set_visible(False) for side in ['right','top',"left"]]
            ax[i,2].yaxis.set_visible(False)
            ax[i,2].set_xlabel("Time[s]")
            ax[i,2].text(15,maxabsz,f"{round(max(abs(ini_z)),sigfigs=3)}",fontsize=5,color="red")
            ax[i,2].text(17.5,maxabsz,f"{round(max(abs(obs_z)),sigfigs=3)}",fontsize=5)

        else :
            ax[i,0].plot(tim,obs_y,color="black",label="obs_wave",linewidth=0.5)
            ax[i,0].plot(tim,ini_y,color="red",label="syn_wave",linewidth=0.5)
            # ax[i,0].plot(tim,con_y,color="red",label="convergence_wave",linewidth=1)
            ax[i,0].axis("off")
            ax[i,0].set_xlim(0,20)
            maxabsy = max(max(abs(ini_y)),max(abs(obs_y)))
            ax[i,0].set_ylim(-maxabsy,maxabsy)
            ax[i,0].text(-10, 0, f"{obs_name[obs]}", va='center',fontsize=7)
            ax[i,0].text(15,maxabsy,f"{round(max(abs(ini_y)),sigfigs=3)}",fontsize=5,color="red")
            ax[i,0].text(17.5,maxabsy,f"{round(max(abs(obs_y)),sigfigs=3)}",fontsize=5)
            
            ax[i,1].plot(tim,obs_x,color="black",label="obs_wave",linewidth=0.5)
            ax[i,1].plot(tim,ini_x,color="red",label="syn_wave",linewidth=0.5)
            # ax[i,1].plot(tim,con_x,color="red",label="convergence_wave",linewidth=1)
            ax[i,1].axis("off")
            ax[i,1].set_xlim(0,20)
            maxabsx = max(max(abs(ini_x)),max(abs(obs_x)))
            ax[i,1].set_ylim(-maxabsx,maxabsx)
            ax[i,1].text(15,maxabsx,f"{round(max(abs(ini_x)),sigfigs=3)}",fontsize=5,color="red")
            ax[i,1].text(17.5,maxabsx,f"{round(max(abs(obs_x)),sigfigs=3)}",fontsize=5)
            
            ax[i,2].plot(tim,obs_z,color="black",label="obs_wave",linewidth=0.5)
            ax[i,2].plot(tim,ini_z,color="red",label="syn_wave",linewidth=0.5)
            # ax[i,2].plot(tim,con_z,color="red",label="convergence_wave",linewidth=1)
            ax[i,2].axis("off")
            ax[i,2].set_xlim(0,20)
            maxabsz = max(max(abs(ini_z)),max(abs(obs_z)))
            ax[i,2].set_ylim(-maxabsz,maxabsz)
            ax[i,2].text(15,maxabsz,f"{round(max(abs(ini_z)),sigfigs=3)}",fontsize=5,color="red")
            ax[i,2].text(17.5,maxabsz,f"{round(max(abs(obs_z)),sigfigs=3)}",fontsize=5)

    def len_fit(ini_t,ini_x,ini_y,ini_z,obs_t,obs_x,obs_y,obs_z):
        
        if len(ini_t) >= len(obs_t) :
            add_zeros = np.zeros(len(ini_t)-len(obs_t))
            obs_x_new = np.append(obs_x,add_zeros)
            obs_y_new = np.append(obs_y,add_zeros)
            obs_z_new = np.append(obs_z,add_zeros)
            time = ini_t
            
            return time,ini_x,ini_y,ini_z,obs_x_new,obs_y_new,obs_z_new
        
        elif len(obs_t) > len(ini_t) :
            add_zeros = np.zeros(len(obs_t)-len(ini_t))
            ini_x_new = np.append(ini_x,add_zeros)
            ini_y_new = np.append(ini_y,add_zeros)
            ini_z_new = np.append(ini_z,add_zeros)
            time = obs_t
            
            return time,ini_x_new,ini_y_new,ini_z_new,obs_x,obs_y,obs_z
        
    fig, ax = plt.subplots(nrows=18,ncols=3,sharex=False)
    for i,obs in tqdm(zip(range(17),["taka","momo","take","O2","sira","abu","tama","hig","naka","ban","OH4","KH7","O3","K11","K12","OH2","O6","K9"])) :

        t_re,x_re,y_re,z_re = calculate_wave_new(spdmax=spdmax_re,
                                                sptaus=sptaus_re,
                                                sptaur=sptaur_re,
                                                spts=spts_re,
                                                obs=obs,
                                                type="reverse")
        t_st,x_st,y_st,z_st = calculate_wave_new(spdmax=spdmax_st,
                                                    sptaus=sptaus_st,
                                                    sptaur=sptaur_st,
                                                    spts=spts_st,
                                                    obs=obs,
                                                    type="strike")
    
        if len(t_re) >= len(t_st):
            add_zeros = np.zeros(len(t_re)-len(t_st))
            x_st_new = np.append(x_st,add_zeros)
            y_st_new = np.append(y_st,add_zeros)
            z_st_new = np.append(z_st,add_zeros)
            ini_x = x_re + x_st_new
            ini_y = y_re + y_st_new
            ini_z = z_re + z_st_new
            ini_t = t_re
            
        else:
            add_zeros = np.zeros(len(t_st)-len(t_re))
            x_re_new = np.append(x_re,add_zeros)
            y_re_new = np.append(y_re,add_zeros)
            z_re_new = np.append(z_re,add_zeros)
            ini_x = x_st + x_re_new
            ini_y = y_st + y_re_new
            ini_z = z_st + z_re_new
            ini_t = t_st
        
        obs_x = np.load(f"./velocity/{obs}_ns.npy")[shift_dic[obs]:shift_dic[obs]+4096]
        obs_y = np.load(f"./velocity/{obs}_ew.npy")[shift_dic[obs]:shift_dic[obs]+4096]
        obs_z = np.load(f"./velocity/{obs}_ud.npy")[shift_dic[obs]:shift_dic[obs]+4096]
        obs_t = np.arange(0,0.01*len(obs_x),0.01)
        
        tim,ini_x,ini_y,ini_z,obs_x,obs_y,obs_z = len_fit(ini_t,ini_x,ini_y,ini_z,obs_t,obs_x,obs_y,obs_z)
        
        plot(tim,obs_x,obs_y,obs_z,ini_x,ini_y,ini_z)             
        
    fig.savefig("wave_compare.png")

def plot_wave_single(): #観測波形と合成波形を観測点ごとに描画する

    initial = initial_set()

    spdmax_re,spts_re,sptaus_re,sptaur_re = data_set(initial[0:25],"reverse")
    spdmax_st,spts_st,sptaus_st,sptaur_st = data_set(initial[25:51],"strike")   
    
    obs_name = {"taka":"高槻市",
                "momo":"桃園",
                "O2":"OSK002(高槻)",
                "sira":"白川",
                "abu":"阿武山",
                "tama":"玉島",
                "hig":"東中条",
                "naka":"中穂積",
                "ban":"万博",
                "KH7":"KYTH07(久御山)",
                "O3":"OSK003(豊中)",
                "K11":"KYT011(亀岡)",
                "O4":"OSK004(四條畷)",
                "O5":"OSK005(大阪)",
                "KH8":"KYTH08(京都)",
                "K13":"KYT013(宇治)"}
    
    obs_list = ["taka",
                "momo",
                "O2",
                "sira",
                "abu",
                "tama",
                "hig",
                "naka",
                "ban",
                "KH7",
                "O3",
                "K11",
                "O4",
                "O5",
                "KH8",
                "K13"]
    
    shift_dic = {"taka":1400,
                 "momo":1400,
                 "O2":1300,
                 "sira":9400,
                 "abu":3390,
                 "tama":2400,
                 "hig":1400,
                 "naka":2460,
                 "ban":1235,
                 "KH7":1210,
                 "O3":1110,
                 "K11":1110,
                 "O4":1210,
                 "O5":1110,
                 "KH8":1110,
                 "K13":1110
                 }
    
    for obs in obs_list:
        fig, ax = plt.subplots(nrows=3,ncols=1,sharex=False,figsize=(10,10))
        t_re,x_re,y_re,z_re = calculate_wave_new(spdmax=spdmax_re,
                                                sptaus=sptaus_re,
                                                sptaur=sptaur_re,
                                                spts=spts_re,
                                                obs=obs,
                                                type="reverse")
        t_st,x_st,y_st,z_st = calculate_wave_new(spdmax=spdmax_st,
                                                    sptaus=sptaus_st,
                                                    sptaur=sptaur_st,
                                                    spts=spts_st,
                                                    obs=obs,
                                                    type="strike")
        
        if len(t_re) >= len(t_st):
            add_zeros = np.zeros(len(t_re)-len(t_st))
            x_st_new = np.append(x_st,add_zeros)
            y_st_new = np.append(y_st,add_zeros)
            z_st_new = np.append(z_st,add_zeros)
            x = x_re + x_st_new
            y = y_re + y_st_new
            z = z_re + z_st_new
            ini_t = t_re
            
        else:
            add_zeros = np.zeros(len(t_st)-len(t_re))
            x_re_new = np.append(x_re,add_zeros)
            y_re_new = np.append(y_re,add_zeros)
            z_re_new = np.append(z_re,add_zeros)
            x = x_st + x_re_new
            y = y_st + y_re_new
            z = z_st + z_re_new
            ini_t = t_st
            
        # time_O2 = np.load("time_O2.npy") 
        x_O2 = np.load(f"./velocity/{obs}_ns.npy")[shift_dic[obs]:shift_dic[obs]+4096]
        y_O2 = np.load(f"./velocity/{obs}_ew.npy")[shift_dic[obs]:shift_dic[obs]+4096]
        z_O2 = np.load(f"./velocity/{obs}_ud.npy")[shift_dic[obs]:shift_dic[obs]+4096]
        time_O2 = np.arange(0,0.01*len(x_O2),0.01)
        maxabsx = max(abs(x))
        maxabsy = max(abs(y))
        maxabsz = max(abs(z))
        maxabsx_O2 = max(abs(x_O2))
        maxabsy_O2 = max(abs(y_O2))
        maxabsz_O2 = max(abs(z_O2))
        
        max_value = max(maxabsx,maxabsy,maxabsz,maxabsx_O2,maxabsy_O2,maxabsz_O2)
        
        ax[0].plot(ini_t,y,color="red",label=f"synthetic_wave {round(maxabsy,sigfigs=3)}(cm/s)",linewidth=1)
        ax[0].plot(time_O2,y_O2,color="black",label=f"observation_wave {round(maxabsy_O2,sigfigs=3)}(cm/s)",linewidth=1)
        ax[0].set_xlim(0,20)
        ax[0].set_ylim(-max_value,max_value)
        # ax[0].spines[side].set_visible(False) for side in ['right','top',"left"]
        # ax[0].yaxis.set_visible(False)
        ax[0].set_xlabel("Time[s]")
        ax[0].text(-3, 0, "EW(cm/s)")
        ax[0].set_title(f"{obs_name[obs]}")
        # ax[0].text(25,maxabsy,f"{round(maxabsy,sigfigs=3)}")
        
        ax[1].plot(ini_t,x,color="red",label=f"synthetic_wave {round(maxabsx,sigfigs=3)}(cm/s)",linewidth=1)
        ax[1].plot(time_O2,x_O2,color="black",label=f"observation_wave {round(maxabsx_O2,sigfigs=3)}(cm/s)",linewidth=1)
        ax[1].set_xlim(0,20)
        ax[1].set_ylim(-max_value,max_value)
        ax[1].text(-3, 0, "NS(cm/s)")
        # ax[i,1].spines[side].set_visible(False) for side in ['right','top',"left"]
        # ax[1].yaxis.set_visible(False)
        ax[1].set_xlabel("Time[s]")

        ax[2].plot(ini_t,z,color="red",label=f"synthetic_wave {round(maxabsz,sigfigs=3)}(cm/s)",linewidth=1)
        ax[2].plot(time_O2,z_O2,color="black",label=f"observation_wave {round(maxabsz_O2,sigfigs=3)}(cm/s)",linewidth=1)
        ax[2].set_xlim(0,20)
        ax[2].set_ylim(-max_value,max_value)
        ax[2].text(-3, 0, "UD(cm/s)")
        # ax[i,2].spines[side].set_visible(False) for side in ['right','top',"left"]
        # ax[2].yaxis.set_visible(False)
        ax[2].set_xlabel("Time[s]")
        
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.savefig(f"./wave_fig/{obs}_was.png")
        plt.close()
    
###########################

def para_change(): #あるパラメータを変化させたときの評価関数の動きをみる
    
    # taus = np.arange(0.1,0.5,0.05)
    # taur = np.arange(1.1,2,0.1)
    # # vel = np.arange(2,3,0.1)
    # # slip = np.arange(0.1,1,0.1)
    # l2_list,was_list = [],[]
    # for s in tqdm(taus) :
    #     for r in taur :
            
    #         initial_condition = [0.5,0,0,
    #                             0,0,0,
    #                             0.5,0,0,#滑り量
    #                             5*np.sqrt(2)/2.7,5/2.7,5*np.sqrt(2)/2.7,
    #                             5/2.7,5/2.7,
    #                             5*np.sqrt(2)/2.7,5/2.7,5*np.sqrt(2)/2.7,
    #                             s,s,s,
    #                             s,s,s,
    #                             s,s,s,
    #                             r,r,r,
    #                             r,r,r,
    #                             r,r,r
    #                             ]
        
    #         l2_sum = l2(initial_condition)
    #         was_sum = was(initial_condition)
            
    #         l2_list += [[s,r,l2_sum]]
    #         was_list += [[s,r,was_sum]]
    
    # l2_array = np.array(l2_list)
    # was_array = np.array(was_list)
    # np.save("./l2_array.npy",l2_array)
    # np.save("./was_array.npy",was_array)
    # fig1,ax1 = plt.subplots()
    # ax1.plot(slip,l2_list)
    # ax1.set_xlabel("slip[m]")
    # ax1.set_ylabel("l2")
    # fig1.savefig("./para_change/l2_slip.png")
    # plt.close()

    # fig2,ax2 = plt.subplots()
    # ax2.plot(slip,was_list)
    # ax2.set_xlabel("slip[m]")
    # ax2.set_ylabel("was")
    # fig2.savefig("./para_change/was_slip.png")
    # plt.close() 
    
    def plot_contour(value):
        xuniq = sorted(list(set(list(map(lambda x: x[0],value)))))
        yuniq = sorted(list(set(list(map(lambda x: x[1],value)))))
        X, Y = np.meshgrid(xuniq, yuniq)
        zDict = dict()
        for row in value:
            zDict[(row[0], row[1])] = row[2]
        Z = [] # Array<Array<number>>
        for yUniqIdx, y in enumerate(yuniq):
            Z.append([])
            for xUniqIdx, x in enumerate(xuniq):
                Z[yUniqIdx].append(zDict[(x, y)])
        Z = np.array(Z)  
        fig, ax = plt.subplots()
        im = ax.imshow(Z, interpolation='gaussian',cmap=cm.hot_r,
                    origin='lower')
        # ax.contour(X, Y, Z, colors="black",levels=6,linewidths=0.5)
        ax.set_xlabel(r'$\tau_s$')
        ax.set_ylabel(r'$\tau_r$')
        ax.set_xticks([0,2,4,6])
        ax.set_xticklabels(["0.1","0.2","0.3","0.4"])
        ax.set_yticks([0,2,4,6,8])
        ax.set_yticklabels(["1.2","1.4","1.6","1.8","2"])
        # ax.set_title('Slip Distribution')
        plt.colorbar(im,extend='both',label="ワッサースタイン計量",shrink=0.7)
        plt.savefig("./para_change/was_tau.png", bbox_inches = 'tight', dpi=150,pad_inches = 0)
    
    l2_array = np.load("l2_array.npy")
    was_array = np.load("was_array.npy")
    # plot_contour(l2_array)
    plot_contour(was_array)
     
def wave_check():
    
    model = [0.5,0,0, #############
             0,0,0, ##滑り量[m]##
             0.5,0,0, #############
             5*np.sqrt(2)/2.7,5/2.7,5*np.sqrt(2)/2.7,
             5/2.7,5/2.7,
             5*np.sqrt(2)/2.7,5/2.7,5*np.sqrt(2)/2.7,
             0.1,0.1,0.1,
             0.1,0.1,0.1,
             0.1,0.1,0.1,
            1.2,1.2,1.2,
            1.2,1.2,1.2,
            1.2,1.2,1.2]
    
    initial = [0,0,0.5,
                0,0,0,
                0,0,0.5,#滑り量
                    5*np.sqrt(2)/1.8,5/1.8,5*np.sqrt(2)/1.8,
                5/1.8,5/1.8,
                5*np.sqrt(2)/1.8,5/1.8,5*np.sqrt(2)/1.8,
                    0.4,0.4,0.4,
                    0.4,0.4,0.4,
                    0.4,0.4,0.4,
                    1.7,1.7,1.7,
                    1.7,1.7,1.7,
                    1.7,1.7,1.7,
                    ]
    
    model_spdmax,model_spts,model_sptaus,model_sptaur = data_set(model)
    ini_spdmax,ini_spts,ini_sptaus,ini_sptaur = data_set(initial)
    
    t_model,x_model,y_model,z_model = calculate_wave_new(model_spdmax,model_sptaus,model_sptaur,model_spts,"ABU")
    t_ini,x_ini,y_ini,z_ini = calculate_wave_new(ini_spdmax,ini_sptaus,ini_sptaur,ini_spts,"ABU")
    max_model_x = max(abs(x_model))
    max_ini_x = max(abs(x_ini))
    model_x = x_model*max_ini_x/max_model_x

    add_zeros = np.zeros(len(t_ini)-len(t_model))
    model_x = np.append(model_x,add_zeros)

    w1 = model_x
    w2 = x_ini
    con =3
    xmin = 0
    xmax = 0.02
    h = 0.02
    sum_time = 2388
    b = max(con/(max(abs(w1))),con/(max(abs(w2))))
    if (max(abs(w1)) or max(abs(w2))) == 0 :
        print("warning")

    pos_amp1_list = []
    for uni_amp1 in w1 :
        if b*uni_amp1 > 0 :
            pos_amp1 = b*uni_amp1+np.log(np.exp(-uni_amp1*b)+1)
            pos_amp1_list += [pos_amp1]
        else :
            pos_amp1 = np.log(np.exp(uni_amp1*b)+1)
            pos_amp1_list += [pos_amp1]

    pos_amp2_list = []
    for uni_amp2 in w2 :
        if b*uni_amp2 > 0 :
            pos_amp2 = b*uni_amp2+np.log(np.exp(-uni_amp2*b)+1)
            pos_amp2_list += [pos_amp2]
        else :
            pos_amp2 = np.log(np.exp(uni_amp2*b)+1)
            pos_amp2_list += [pos_amp2]
    
    w1_pos = np.array(pos_amp1_list)
    w2_pos = np.array(pos_amp2_list)

    s1 = h*(np.sum(w1_pos)-w1_pos[0]/2-w1_pos[sum_time-1]/2)
    s2 = h*(np.sum(w2_pos)-w2_pos[0]/2-w2_pos[sum_time-1]/2)
    p1 = [value1/s1 for value1 in w1_pos]
    p2 = [value2/s2 for value2 in w2_pos]
    
    plt.plot(t_ini,p1)
    plt.plot(t_ini,p2)
    plt.savefig("wave_wascheck_.png")
    plt.show()
   
# value_r = initial_set()[0:25]
# value_s = initial_set()[51:76]
# plot_slip_distribution("slip_reverse",value_r,"reverse")
# plot_slip_distribution("slip_strike",value_s)
# plot_wave_single()
main("was","SLSQP")
# para_change()
# wave_check()
# main_basinhopping()

# plot_wave_all()
# regulargrid

# value_r = [3.05480395e-18,1.63329081e-17,1.71736621e-17,1.12453047e-16,2.35267567e-02,
#            3.67087865e-17,8.15173134e-17,4.90549382e-02,0.00000000e+00,1.03163827e-16,
#            2.25959317e-16,3.11453748e-02,3.31711061e-01,2.69111431e-17,6.75262870e-03,
#            6.92637641e-18,4.06359339e-17,1.31895584e-16,1.26696027e-16,0.00000000e+00,
#            7.21775173e-17,4.10654070e-18,1.93006997e-17,5.82964673e-18,5.65146831e-01]
           
# value_s = [1.36923716e-18,0.00000000e+00,0.00000000e+00,1.38110768e-16,0.00000000e+00,
#            9.41246597e-18,8.39900336e-04,8.82705620e-02,2.15540757e-17,7.87583207e-17,
#            6.41331813e-17,0.00000000e+00,3.27861116e-01,1.55718398e-17,8.79553513e-17,
#            1.56604962e-16,2.81269245e-17,7.22937008e-17,1.71627384e-18,2.28165091e-17,
#            0.00000000e+00,1.15095211e+00,4.67281878e-18,0.00000000e+00,1.30049113e-16]
           
        #    3.99190490e-01]


# plot_slip_distribution("slip_reverse_was",value_r,"reverse")
# plot_slip_distribution("slip_strike_was",value_s,"strike")