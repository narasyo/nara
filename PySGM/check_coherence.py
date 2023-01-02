import numpy as np
import matplotlib.pyplot as plt
import os
import vector

os.chdir("./data/furukawa_data/201212071719/20121207_modified")
acc1 = np.load('201212071719_F1_EW_modified.npy')
acc2 = np.load('201212071719_F3_EW_modified.npy')

fsamp = 100

tim = np.linspace(0,len(acc1)/fsamp,len(acc1),endpoint=False)
dt = 1/fsamp

# acc1 = scipy.signal.chirp(tim,0.01,1000,10,method='logarithmic')
# acc2 = scipy.signal.chirp(tim,0.01,1000,10,method='logarithmic')
# acc2 += np.random.normal(0.0,0.01,len(acc1))

# acc1 = PySGM.vector.vector.bandpass(acc1,dt,0.01,50)
# acc2 = PySGM.vector.vector.bandpass(acc2,dt,0.01,50)

wave1 = vector.vector("",tim,acc1)
wave2 = vector.vector("",tim,acc2)

print(wave1)
fig1 = plt.figure()

ax11 = fig1.add_subplot(2,1,1)
ax11.set_xlabel('time[s]')
ax11.set_ylabel('acceleration[gal]')
ax11.set_xlim(-20,500)
ax11.set_ylim(-150,250)
ax11.tick_params(direction = "inout", length = 5, colors = "blue")
ax11.plot(tim, acc1, color = "red")

ax12 = fig1.add_subplot(2,1,2)
ax12.set_xlabel('time[s]')
ax12.set_ylabel('acceleration[gal]')
ax12.set_xlim(-20,500)
ax12.set_ylim(-150,250)
ax12.tick_params(direction = "inout", length = 5, colors = "blue")
ax12.plot(tim, acc2, color = "red")

fig1.tight_layout()
fig1.savefig('20121207_F1_F3_waveform.png')

plt.clf()
plt.close()

freq,coh = wave1.coherence(wave2,0.4)
fig2 = plt.figure()
ax = fig2.add_subplot(1,1,1)
ax.set_xlabel('frequency[Hz]')
ax.set_ylabel('coherence')
ax.set_ylim(0.1)
ax.tick_params(direction = "inout", length = 5, colors = "blue")
ax.plot(freq,coh,color = "blue")

fig2.tight_layout()
fig2.savefig('20121207_F1_F3_coherence.png')

plt.clf()
plt.close()
