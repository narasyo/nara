import numpy as np
import matplotlib.pyplot as plt
import ot
import os
from scipy.fftpack import fft,ifft,fftfreq
from library.output_format import Format
Format.params()

N_input = input('サンプル数を入力>>') 
N = int(N_input)

tau_input = input('計測開始時刻の差[s]を入力>>')
tau = int(tau_input)

dt = 0.01 # data step [s]

t = np.arange(0, N*dt, dt) # time
# freq_list = np.linspace(0,1.0/dt,N) # frequency step
freq1_list = fftfreq(N,d=dt) # frequency step
freq1_list = freq1_list[0:int(N/2)]
freq2_list = freq1_list + freq1_list[-1]
freq1_list = freq1_list.tolist()
freq2_list = freq2_list.tolist()
freq_list = freq1_list + freq2_list
freq_list = [x/2 for x in freq_list]
# print(freq1_list)
# print(freq2_list)
# print(freq_list)

savedir2="./data"
os.chdir(savedir2)

uni_amp1_list = np.load("C00_EW_mean.npy")
yf_fft_list = np.load("C00_EW_fft.npy")*N/2

dlen = N
mean = 0.0
std_input = input('標準偏差を入力>>')
#print(freq_list)
std = float(std_input)
r_list = np.linspace(0,300,301)

was1_list = []
was2_list = []
phase_std_list = []

for r in r_list :
    norm = np.random.normal(mean,std,dlen)
    abs_norm = np.abs(norm)
    yf_fft_phase_list = []
    for norm_ele,yf_fft,freq in zip(abs_norm,yf_fft_list,freq_list) :
        yf_fft_phase = yf_fft*(np.cos(-4*(np.pi)*r*freq*(norm_ele-tau))+np.sin(-4*(np.pi)*r*freq*(norm_ele-tau))*1j)
        yf_fft_phase_list += [yf_fft_phase]

    yf_ifft_phase_list = ifft(yf_fft_phase_list)
    yf_ifft_phase_real_list = np.real(yf_ifft_phase_list)

    xmin = 0
    xmax = 0.01*N
    h = (xmax-xmin)/N
    c1 = np.abs(min(uni_amp1_list))
    c2 = np.abs(min(yf_ifft_phase_real_list))
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
    for uni_amp2 in yf_ifft_phase_real_list :
        if uni_amp2 >= 0 :
            pos_amp2 = uni_amp2+1/c
            pos_amp2_list += [pos_amp2]
        else :
            pos_amp2 = np.exp(c*uni_amp2)/c
            pos_amp2_list += [pos_amp2]
    
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

    phase_std_list += [yf_ifft_phase_real_list]

os.chdir("../")
os.chdir("./data")
np.save('C00_EW_phase(r=300,std=1.0e-04)',phase_std_list) 

os.chdir("../")
os.chdir("./image")

plt.figure()    
plt.subplot(2,1,1)
plt.plot(t,uni_amp1_list)
#plt.savefig('decay_C00_EW(r=0)')
plt.subplot(2,1,2)
plt.ylim(-100,100)
plt.plot(t,phase_std_list[0])
plt.savefig('phase_C00_EW(r=0)(std=1.0e-04).png')

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,phase_std_list[0])
plt.subplot(2,1,2)
plt.ylim(-100,100)
plt.plot(t,phase_std_list[99])
plt.savefig('phase_C00_EW(r=99)(std=1.0e-04).png')

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,phase_std_list[0])
plt.subplot(2,1,2)
plt.ylim(-100,100)
plt.plot(t,phase_std_list[199])
plt.savefig('phase_C00_EW(r=199)(std=1.0e-04).png')

plt.figure()
plt.subplot(2,1,1)
plt.xlim
plt.plot(t,phase_std_list[0])
plt.subplot(2,1,2)
plt.ylim(-100,100)
plt.plot(t,phase_std_list[299])
plt.savefig('phase_C00_EW(r=299)(std=1.0e-04).png')

   ###正負を入れ替えて計算###

def negative(neg):
    return -neg

rev_uni_amp1_list = []
for uni_amp1 in uni_amp1_list :
    reverse1 = negative(uni_amp1)
    rev_uni_amp1_list += [reverse1]

rev_yf_fft_list = fft(rev_uni_amp1_list)

for r in r_list :
    norm = np.random.normal(mean,std,dlen)
    abs_norm = np.abs(norm)
    rev_yf_fft_phase_list = []
    for norm_ele,rev_yf_fft,freq in zip(abs_norm,rev_yf_fft_list,freq_list) :
        rev_yf_fft_phase = rev_yf_fft*(np.cos(-4*(np.pi)*freq*r*(norm_ele-tau))+np.sin(-4*(np.pi)*freq*r*(norm_ele-tau))*1j)
        rev_yf_fft_phase_list += [rev_yf_fft_phase]

    # 逆フーリエ変換（ IFFT ）
    rev_yf_ifft_phase_list = ifft(rev_yf_fft_phase_list)
 
    # 実数部の取得
    rev_yf_ifft_phase_real_list = np.real(rev_yf_ifft_phase_list)
    rev_uni_amp2_list = rev_yf_ifft_phase_real_list

    d1 = abs(min(rev_uni_amp1_list))
    d2 = abs(min(rev_uni_amp2_list))
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
plt.xlabel(u'standard[s]')
plt.ylabel(u'wasserstein metric')
plt.plot(r_list,was)
plt.savefig('wasser(std=1.0e-04).png')
plt.show()
