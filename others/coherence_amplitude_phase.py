import numpy as np
import matplotlib.pyplot as plt
import PySGM.vector
import os

r_list = np.linspace(0,1000,1001)

os.chdir("./data")
acc1 = np.load('C00_EW_mean.npy')
acc2_list = np.load('C00_EW_decay_phase(r=1000,std=1.0e-05).npy')
fsamp = 100

tim = np.linspace(0,len(acc1)/fsamp,len(acc1),endpoint=False)
dt = 1/fsamp

# acc1 = scipy.signal.chirp(tim,0.01,1000,10,method='logarithmic')
# acc2 = scipy.signal.chirp(tim,0.01,1000,10,method='logarithmic')
# acc2 += np.random.normal(0.0,0.01,len(acc1))

# acc1 = PySGM.vector.vector.bandpass(acc1,dt,0.01,50)
# acc2 = PySGM.vector.vector.bandpass(acc2,dt,0.01,50)

wave1 = PySGM.vector.vector("",tim,acc1)

coh_list = []
for acc2 in acc2_list :
    wave2 = PySGM.vector.vector("",tim,acc2)
    #wave1.plot_with(wave2)
    freq,coh = wave1.coherence(wave2,2.0)
    coh_list += [coh]

print(freq)

xti = np.linspace(0,50,11)
os.chdir("../")
os.chdir("./image")
os.chdir("./rcons_freq_coh")
plt.plot(freq,coh_list[1000])
plt.xlabel('frequency[Hz]')
plt.ylabel('coherence')
plt.ylim([0,1])
plt.xticks(xti)
plt.savefig('r=1000[m].png')

os.chdir("../")
os.chdir("./freqcons_r_coh")
coh_r_w =[coh_r[1024] for coh_r in coh_list]
plt.figure()
plt.plot(r_list,coh_r_w)
plt.xlabel('distance[m]')
plt.ylabel('coherence')
plt.ylim([0,1])
plt.savefig('freq=50.0[Hz].png')
plt.show()