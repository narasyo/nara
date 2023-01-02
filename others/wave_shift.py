import numpy as np
import ot
import os
import matplotlib.pyplot as plt 
import pywt 
from tqdm import tqdm

def wave_shift(wave1,wave2,shift,time):
    # time=100
    wave1_=np.zeros(len(wave1)+time*100)
    wave2_=np.zeros(len(wave2)+time*100)
    
    wave1_[int(time/2*100):int(time/2*100)+len(wave1)]+=wave1
    wave2_[int(time/2*100)+int(shift*100):int(time/2*100)+len(wave2)+int(shift*100)]+=wave2
    return wave1_,wave2_

def sort_wave(wave,wavelet) :

    ntim = len(wave)

    # Wavelet transform (Multi resolution analysis) #
    coeffs = pywt.wavedec(wave,wavelet)
    nlevel = pywt.dwt_max_level(ntim,wavelet) + 1
    ncoeffs = [len(c) for c in coeffs]
    zero_coeffs = [np.zeros_like(c) for c in coeffs]
    wave_levels = []
    for i in range(nlevel):
        coeffs_level = zero_coeffs.copy()
        #nl = int(len(coeffs_level[1])/2)
        #coeffs_level[1][nl] = 1.0
        coeffs_level[i] = coeffs[i].copy()
        wave_level = pywt.waverec(coeffs_level,wavelet)
        wave_levels += [wave_level]

    return wave_levels

def softplus_normalizing(wave1,wave2,sum_time) :

    xmin = 0
    xmax = 0.01*sum_time
    h = (xmax-xmin)/sum_time
    wave1 = np.array(wave1)
    wave2 = np.array(wave2)

    b =  3.0/(max([max(abs(wave1)),max(abs(wave2))]))
    w1_pos = np.log(np.exp(wave1*b)+1)
    w2_pos = np.log(np.exp(wave2*b)+1)
    s1 = h*(np.sum(w1_pos)-w1_pos[0]/2-w1_pos[sum_time-1]/2)
    s2 = h*(np.sum(w2_pos)-w2_pos[0]/2-w2_pos[sum_time-1]/2)
    p1 = [w1/s1 for w1 in w1_pos]
    p2 = [w2/s2 for w2 in w2_pos]

    return p1,p2



def sort_wave(wave,wavelet) :

    ntim = len(wave)

    # Wavelet transform (Multi resolution analysis) #
    coeffs = pywt.wavedec(wave,wavelet)
    nlevel = pywt.dwt_max_level(ntim,wavelet) + 1
    ncoeffs = [len(c) for c in coeffs]
    zero_coeffs = [np.zeros_like(c) for c in coeffs]
    wave_levels = []
    for i in range(nlevel):
        coeffs_level = zero_coeffs.copy()
        #nl = int(len(coeffs_level[1])/2)
        #coeffs_level[1][nl] = 1.0
        coeffs_level[i] = coeffs[i].copy()
        wave_level = pywt.waverec(coeffs_level,wavelet)
        wave_levels += [wave_level]

    return wave_levels

time = np.arange(0,360,0.01)

n1 = 3
n2 = 12

n1 = str(n1)
n2 = str(n2)

os.chdir("./data/furukawa_data/201308041229/20130804")

wave1 = np.load(f"201308041229_F{n1}_verticle_modified.npy")
wave2 = np.load(f"201308041229_F{n2}_verticle_modified.npy")



wavelet = pywt.Wavelet('dmey')

sort_wave1 = sort_wave(wave1,wavelet)
sort_wave2 = sort_wave(wave2,wavelet)
sort_wave1=sort_wave1[3]
sort_wave2=sort_wave2[3]
plt.figure()
plt.plot(sort_wave1)
plt.plot(sort_wave1+5)
plt.xlim(0,10000)
plt.show()
time=100
tim_list = np.arange(0,360+time,0.01)
timeshift=np.linspace(-5,50,100)
was_list,l2_list = [],[]
min=500
c_wave1=[]
c_wave2=[]
for i in tqdm(timeshift):
    w1,w2=wave_shift(sort_wave1,sort_wave2,i,time)
    # w1=sort_wave1
    # w2=sort_wave2
    w1_norm,w2_norm = softplus_normalizing(w1,w2,len(w1))
    # print(int(len(w1)/100))
    was_pos = ot.emd2_1d(tim_list,tim_list,w1_norm/sum(w1_norm),w2_norm/sum(w2_norm),metric='minkowski',p=2) 
    w1_norm,w2_norm = softplus_normalizing(-w1,-w2,len(w1))
    was_neg = ot.emd2_1d(tim_list,tim_list,w1_norm/sum(w1_norm),w2_norm/sum(w2_norm),metric='minkowski',p=2) 
    was = was_pos + was_neg
    was_list += [was]
    l2 = np.mean((w1-w2)**2)
    l2_list += [l2]
    if(min>was):
        min=was
        c_wave1=w1
        c_wave2=w2

plt.figure()
plt.plot(c_wave1)
plt.plot(c_wave2+5)
plt.xlim(0,15000)
plt.show()

plt.figure()
# ax1 = fig.add_subplot(2,1,1)
# ax1.tick_params(direction = "inout", length = 5, colors = "blue",labelsize=10)
plt.plot(timeshift, was_list)
# ax1.set_xlabel('time',fontsize=10)
# ax1.set_ylabel('gal',fontsize=10)
# ax1.grid(True)
plt.show()
# plt.figure()
# # ax1 = fig.add_subplot(2,1,1)
# # ax1.tick_params(direction = "inout", length = 5, colors = "blue",labelsize=10)
# plt.plot(timeshift, l2_list)
# # ax1.set_xlabel('time',fontsize=10)
# # ax1.set_ylabel('gal',fontsize=10)
# # ax1.grid(True)
# plt.show()
# fig = plt.figure()
# ax1 = fig.add_subplot(2,1,1)
# ax1.tick_params(direction = "inout", length = 5, colors = "blue",labelsize=10)
# ax1.plot(time[0:10000], wave1[0:10000], c='red',linewidth = 0.5)
# ax1.set_xlabel('time',fontsize=10)
# ax1.set_ylabel('gal',fontsize=10)
# ax1.grid(True)

# ax2 = fig.add_subplot(2,1,2)
# ax2.tick_params(direction = "inout", length = 5, colors = "blue",labelsize=10)
# ax2.plot(time[0:10000], wave1[0:10000], c='red',linewidth = 0.5)
# ax2.set_xlabel('time',fontsize=10)
# ax2.set_ylabel('gal',fontsize=10)
# ax2.grid(True)

# fig.tight_layout()
# # fig.savefig(f"F{n1}_F{n2}_wave.eps")
# # fig.savefig(f"F{n1}_F{n2}_wave.png")
# plt.clf()
# plt.close()

# fig = plt.figure()
# ax1 = fig.add_subplot(2,1,1)
# ax1.tick_params(direction = "inout", length import tqdm
# ax1.grid(True)

# ax2 = fig.add_subplot(2,1,2)
# ax2.tick_params(direction = "inout", length = 5, colors = "blue",labelsize=10)
# ax2.plot(time[0:10000], sort_wave2[4][0:10000], c='red',linewidth = 0.5)
# ax2.set_xlabel('time',fontsize=10)
# ax2.set_ylabel('gal',fontsize=10)
# ax2.grid(True)

# fig.tight_layout()
# # fig.savefig(f"F{n1}_F{n2}_wave_sorted.eps")
# # fig.savefig(f"F{n1}_F{n2}_wave_sorted.png")
# plt.clf()
# plt.close()


