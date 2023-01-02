from typing import ForwardRef
import numpy as np
import matplotlib.pyplot as plt
import ot
import os
from scipy.fftpack import fft,ifft,fftfreq
from library.output_format import Format
Format.params()

# with open("IES.SMART1.TAI01.C-00") as f1: #Update!
#     start1 = input('1つ目のファイルの始めの行番号を入力>>')
#     end1 = input('1つ目のファイルの終わりの行番号を入力>>')
#     data1 = f1.readlines()[int(start1):int(end1)] 

# g1 = open('C00_EW.txt', 'w')
# g1.writelines(data1)
# g1.close()

# with open('C00_EW.txt', 'r') as g1:
#     ampcha1_list = g1.read().split()

# amp1_list = []    
# for ampcha1 in ampcha1_list:
#     amp1 = float(ampcha1)*980/1024
#     amp1_list += [amp1]

N_input = input('サンプル数を入力>>') 
N = int(N_input)

time_dif_input = input('計測開始時刻の差[s]を入力>>')
time_dif = int(time_dif_input)


# amp1_mean = np.mean(amp1_list[time_dif:N+time_dif])

# mean1_list = np.linspace(amp1_mean,amp1_mean,N)

# def python_list_dif1(a1,b1):
#     uni_amp1_list = np.array(a1) - np.array(b1)
#     return uni_amp1_list.tolist()

# savedir1 = "./data"
# os.chdir(savedir1)
# uni_amp1_list = python_list_dif1(amp1_list[time_dif:N+time_dif],mean1_list)
# np.save('C00_EW_mean', uni_amp1_list)

# parameters
dt = 0.01 # data step [s]

t = np.arange(0, N*dt, dt) # time
#freq_list = np.linspace(0, 1.0/dt, N) # frequency step
freq1_list = fftfreq(N,d=dt) # frequency step
freq1_list = freq1_list[0:int(N/2)]
freq2_list = freq1_list + freq1_list[-1]
freq1_list = freq1_list.tolist()
freq2_list = freq2_list.tolist()
freq_list = freq1_list + freq2_list
freq_list = [x/2 for x in freq_list]
print(freq1_list)
print(freq2_list)
print(freq_list)

# yf_fft_list = fft(uni_amp1_list)
# yf = yf_fft_list/(N/2) # 離散フーリエ変換&規格化
# np.save('C00_EW_fft', yf)

savedir2="./data"
os.chdir(savedir2)

uni_amp1_list = np.load("C00_EW_mean.npy")
yf_fft_list = np.load("C00_EW_fft.npy")*N/2

r_list = np.linspace(0,300,301)
a = 10000 # 速さの次元[m/s]
was1_list = []
was2_list = []
decay_r_list = []
for r in r_list :
    decay_yf_fft_list = []
    for (f, yf_fft) in zip(freq_list, yf_fft_list):
        decay_yf_fft =  yf_fft*(np.exp(-4*(np.pi)*f*r/a))
        decay_yf_fft_list += [decay_yf_fft]
    
    # 逆フーリエ変換（ IFFT ）
    decay_yf_ifft_list = ifft(decay_yf_fft_list)
 
    # 実数部の取得
    decay_yf_ifft_real_list = np.real(decay_yf_ifft_list)

    xmin = 0
    xmax = 0.01*N
    h = (xmax-xmin)/N
    c1 = abs(min(uni_amp1_list[0:N]))
    c2 = abs(min(decay_yf_ifft_real_list[0:N]))
    c = max(c1,c2)

    pos_amp1_list = []
    for uni_amp1 in uni_amp1_list :
        if uni_amp1 >= 0 :
            pos_amp1 = uni_amp1+1/c
            pos_amp1_list += [pos_amp1]
        else :
            pos_amp1 = (np.exp(c*uni_amp1))/c
            pos_amp1_list += [pos_amp1]

    pos_amp2_list = []
    for uni_amp2 in decay_yf_ifft_real_list :
        if uni_amp2 >= 0 :
            pos_amp2 = uni_amp2+1/c
            pos_amp2_list += [pos_amp2]
        else :
            pos_amp2 = np.exp(c*uni_amp2)/c
            pos_amp2_list += [pos_amp2]
    #print(pos_amp1_list)
    #print(len(pos_amp1_list))
    s1 = h*(np.sum(pos_amp1_list)-pos_amp1_list[0]/2-pos_amp1_list[N-1]/2)

    s2 = h*(np.sum(pos_amp2_list)-pos_amp2_list[0]/2-pos_amp2_list[N-1]/2)

    def calc_devided_s1(n1):
        return n1/s1

    def calc_devided_s2(n2):
        return n2/s2

    amp1_norm_list = []
    for pos1 in pos_amp1_list:
        amp1_norm_list.append(calc_devided_s1(pos1))

    amp2_norm_list =[]
    for pos2 in pos_amp2_list:
        amp2_norm_list.append(calc_devided_s2(pos2))

    amp1_norm = np.array(amp1_norm_list)
    amp2_norm = np.array(amp2_norm_list)
    was1 = ot.emd2_1d(t,t,amp1_norm/sum(amp1_norm),amp2_norm/sum(amp2_norm),metric='minkowski',p=2)
    was1_list += [was1]
    
    decay_r_list += [decay_yf_ifft_real_list]

np.save('C00_EW_decay_list(r=300,a=10000)',decay_r_list)
    
os.chdir("../")
os.chdir("./image")

plt.figure()    
plt.subplot(2,1,1)
plt.plot(t,decay_r_list[0])
#plt.savefig('decay_C00_EW(r=0)')
plt.subplot(2,1,2)
plt.ylim(-100,100)
plt.plot(t,decay_r_list[1])
plt.savefig('decay_C00_EW(r=1)')

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,decay_r_list[0])
plt.subplot(2,1,2)
plt.ylim(-100,100)
plt.plot(t,decay_r_list[100])
plt.savefig('decay_C00_EW(r=100)')

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,decay_r_list[0])
plt.subplot(2,1,2)
plt.ylim(-100,100)
plt.plot(t,decay_r_list[200])
plt.savefig('decay_C00_EW(r=200)')

plt.figure()
plt.subplot(2,1,1)
plt.xlim
plt.plot(t,decay_r_list[0])
plt.subplot(2,1,2)
plt.ylim(-100,100)
plt.plot(t,decay_r_list[300])
plt.savefig('decay_C00_EW(r=300)')

    ###正負を入れ替えて計算###

def negative(neg):
    return -neg

rev_uni_amp1_list = []
for uni_amp1 in uni_amp1_list :
    reverse1 = negative(uni_amp1)
    rev_uni_amp1_list += [reverse1]

rev_yf_fft_list = fft(rev_uni_amp1_list)

for r in r_list :
    rev_decay_yf_fft_list = []
    for (f, rev_yf_fft) in zip(freq_list, rev_yf_fft_list):
        rev_decay_yf_fft =  rev_yf_fft*(np.exp(-4*(np.pi)*f*r/a))
        rev_decay_yf_fft_list += [rev_decay_yf_fft]
    
    # 逆フーリエ変換（ IFFT ）
    rev_decay_yf_ifft_list = ifft(rev_decay_yf_fft_list)
 
    # 実数部の取得
    rev_decay_yf_ifft_real_list = np.real(rev_decay_yf_ifft_list)
    rev_uni_amp2_list = rev_decay_yf_ifft_real_list

    d1 = abs(min(rev_uni_amp1_list[0:N]))
    d2 = abs(min(rev_uni_amp2_list[0:N]))
    d = max(d1,d2)

    rev_pos_amp1_list = []
    for rev1 in rev_uni_amp1_list :
        if rev1 >= 0 :
            rev_amp1 = rev1+1/d
            rev_pos_amp1_list += [rev_amp1]
        else :
            rev_amp1 = (np.exp(d*rev1))/d
            rev_pos_amp1_list += [rev_amp1]

    rev_pos_amp2_list = []
    for rev2 in rev_uni_amp2_list :
        if rev2 >= 0 :
            rev_amp2 = rev2+1/d
            rev_pos_amp2_list += [rev_amp2]
        else :
            rev_amp2 = np.exp(d*rev2)/d
            rev_pos_amp2_list += [rev_amp2]
    #print(pos_amp1_list)
    #print(len(pos_amp1_list))
    t1 = h*(np.sum(rev_pos_amp1_list)-rev_pos_amp1_list[0]/2-rev_pos_amp1_list[N-1]/2)

    t2 = h*(np.sum(rev_pos_amp2_list)-rev_pos_amp2_list[0]/2-rev_pos_amp2_list[N-1]/2)

    def calc_devided_t1(m1):
        return m1/t1

    def calc_devided_t2(m2):
        return m2/t2

    rev_amp1_norm_list = []
    for rev_pos1 in rev_pos_amp1_list:
        rev_amp1_norm_list.append(calc_devided_t1(rev_pos1))

    rev_amp2_norm_list =[]
    for rev_pos2 in rev_pos_amp2_list:
        rev_amp2_norm_list.append(calc_devided_t2(rev_pos2))

    rev_amp1_norm = np.array(rev_amp1_norm_list)
    rev_amp2_norm = np.array(rev_amp2_norm_list)
    was2 = ot.emd2_1d(t,t,rev_amp1_norm/sum(rev_amp1_norm),rev_amp2_norm/sum(rev_amp2_norm),metric='minkowski',p=2)
    was2_list += [was2]

was = np.array(was1_list) + np.array(was2_list)
print(was)

fig, ax = plt.subplots(facecolor="w")
ax.legend()
plt.xlabel(u'distance[m]')
plt.ylabel(u'wasserstein metric')
plt.plot(r_list/a,was)
plt.show()



