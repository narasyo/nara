import numpy as np
import os 
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from library.output_format import Format
Format.params()

with open("IES.SMART1.TAI01.C-00") as f1: #Update!
    start1 = input('1つ目のファイルの始めの行番号を入力>>')
    end1 = input('1つ目のファイルの終わりの行番号を入力>>')
    data1 = f1.readlines()[int(start1):int(end1)] 

g1 = open('C00_EW.txt', 'w')
g1.writelines(data1)
g1.close()

with open('C00_EW.txt', 'r') as g1:
    ampcha1_list = g1.read().split()

amp1_list = []    
for ampcha1 in ampcha1_list:
    amp1 = float(ampcha1)*980/1024
    amp1_list += [amp1]

amp1_list = amp1_list[300:]

N_input = input('サンプル数を入力>>') 
N = int(N_input)

time_dif_input = input('計測開始時刻の差[s]を入力>>')
time_dif = int(time_dif_input)


amp1_mean = np.mean(amp1_list[time_dif:N+time_dif])

mean1_list = np.linspace(amp1_mean,amp1_mean,N)

def python_list_dif1(a1,b1):
    uni_amp1_list = np.array(a1) - np.array(b1)
    return uni_amp1_list.tolist()

savedir1 = "./data"
os.chdir(savedir1)
uni_amp1_list = python_list_dif1(amp1_list[time_dif:N+time_dif],mean1_list)
np.save('C00_EW_mean', uni_amp1_list)

# parameters
dt = 0.01 # data step [s]

t = np.arange(0, N*dt, dt) # time
freq = np.linspace(0, 1.0/dt, N) # frequency step

yf = fft(uni_amp1_list)/(N/2) # 離散フーリエ変換&規格化
np.save('C00_EW_fft', yf)
print(yf)    

os.chdir('../')
savedir2="./image"
os.chdir(savedir2)
plt.figure(1)
plt.subplot(211)
plt.plot(t, uni_amp1_list)
#plt.xlim(0, 1)
plt.xlabel("time")
plt.ylabel("amplitude")

plt.subplot(212)
plt.plot(freq, np.abs(yf))
plt.xlim(0, 50)
#plt.ylim(0, 5)
plt.xlabel("frequency")
plt.ylabel("amplitude")
plt.tight_layout()
plt.savefig("C00_EW_fft")
#plt.show()

plt.figure(2)
plt.subplot(211)
plt.plot(t, uni_amp1_list)
#plt.xlim(0, 1)
plt.xlabel("time")
plt.ylabel("amplitude")

plt.subplot(212)
plt.plot(freq, np.abs(yf)**2)
plt.xlim(0, 50)
#plt.ylim(0, 5)
plt.xlabel("time")
plt.ylabel("amplitude")
plt.tight_layout()
plt.savefig("C00_EW_power_spector")
#plt.show()