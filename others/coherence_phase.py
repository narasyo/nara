import numpy as np
import matplotlib.pyplot as plt
import PySGM.vector
import os

r_list = np.linspace(0,300,301)

os.chdir("./data")
acc1 = np.load('C00_EW_mean.npy')
acc2_list = np.load('C00_EW_phase(r=300,std=1.0e-04).npy')
fsamp = 100

tim = np.linspace(0,len(acc1)/fsamp,len(acc1),endpoint=False)
dt = 1/fsamp

print(tim)

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
    freq,coh = wave1.coherence(wave2,0.1)
    coh_list += [coh]

plt.plot(freq,coh_list[299])
print(freq)

coh_r_w =[coh_r[2000] for coh_r in coh_list]
plt.figure()
plt.plot(r_list,coh_r_w)
plt.ylim([0,1])
plt.show()