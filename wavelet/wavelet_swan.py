import matplotlib.pyplot as plt
import numpy as np
import os
import ot
from tqdm import tqdm
from swan import pycwt

os.chdir('/home/nara/デスクトップ/labo_document/Python_Code')
r_list = np.linspace(0,1000,1001)
tim = np.arange(0.01,23.96,0.01)
os.chdir('./data')
amp1 = np.load("C00_EW_mean.npy")
amp2_list = np.load('C00_EW_decay_phase(r=1000,std=1.0e-05).npy')

#plt.plot(tim, amp1)
#plt.show()

freq_input = input('周波数を入力>>') 
freq = float(freq_input)
N = 2396
Fs = 1/0.01
omega0 = 8
freqs=np.arange(0.1,50,0.025)
r1=pycwt.cwt_f(amp1,freqs,Fs,pycwt.Morlet(omega0))
rr1=np.abs(r1)
n = int(freq/0.025-4)

xmin = 0
xmax = 0.01*N
h = (xmax-xmin)/N
s1 = h*(np.sum(rr1[n])-rr1[n,0]/2-rr1[n,N-1]/2)
prob1 = rr1[n]/s1

was_list = []
for amp2 in tqdm(amp2_list) :
    r2=pycwt.cwt_f(amp2,freqs,Fs,pycwt.Morlet(omega0))
    rr2=np.abs(r2)
    s2 = h*(np.sum(rr2[n])-rr2[n,0]/2-rr2[n,N-1]/2)
    prob2 = rr2[n]/s2
    was = ot.emd2_1d(tim,tim,prob1/sum(prob1),prob2/sum(prob2),metric='minkowski',p=2)
    was_list += [was]

plt.plot(r_list,was_list)
plt.show()

# plt.rcParams['figure.figsize'] = (20, 6)
# fig = plt.figure()
# ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])
# ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax1)
# ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])

# ax1.plot(tim, amp1, 'k')

# img = ax2.imshow(np.flipud(rr1), extent=[0, 20, freqs[0], freqs[-1]],
#                  aspect='auto', interpolation='nearest')

# fig.colorbar(img, cax=ax3)

# plt.show()

#print(rr1[10])