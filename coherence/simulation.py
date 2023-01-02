import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os


# Coherence model: Harichandran and Vanmarcke
def coherence_model(f,r,A=0.736,a=0.147,k=5210,f0=1.09,b=2.78):
    B = 1 - A + a*A
    nu = k*(1+(f/f0)**b)**(-0.5)

    coh = A*np.exp(-2*B*r/(a*nu)) + (1-A)*np.exp(-2*B*r/nu)
    return coh

os.chdir("./coherence")

wave_file = "data/1995JR_Takatori.acc"
tim,wave = np.loadtxt(wave_file,usecols=(0,2),unpack=True)
print(tim)
print(wave)
dt = tim[1]-tim[0]
ntim = len(wave)
print(ntim)
neq = int(ntim/2)

plt.plot(tim,wave)
plt.show()

# Set distance and simulate wave
r_list = [100,200,400,800,1600]  # [m]
freq = np.fft.fftfreq(ntim,dt)
print(freq)
print(len(freq))

wave_sim_list = []
for r in r_list:
    coh = coherence_model(np.abs(freq),r)
    W0 = np.fft.fft(wave)
    print(W0)
    W1 = W0*coh
    wave_sim = np.real(np.fft.ifft(W1))
    wave_sim_list += [wave_sim]

# plt.plot(tim,wave)
# plt.plot(tim,wave_sim_list[0])
# plt.show()

# Output
output_file = "result/wave_simulation.dat"
output_line = np.vstack([tim,wave_sim_list])
np.savetxt(output_file,output_line.T)


#Check coherence

nsamp = 1000
sigma = np.std(wave)
P0 = np.zeros_like(W0)
P1 = np.zeros_like(W0)
C01 = np.zeros_like(W0)
for i in range(nsamp):
    rand0 = np.random.normal(0,sigma,ntim)
    rand1 = np.random.normal(0,sigma,ntim)

    W0 = np.fft.fft(wave)
    W1 = np.fft.fft(wave_sim + rand1)

    P0 += np.conjugate(W0)*W0
    P1 += np.conjugate(W1)*W1
    C01 += np.conjugate(W0)*W1

P0 = P0/nsamp
P1 = P1/nsamp
C01 = C01/nsamp

coh = np.abs(C01)/np.sqrt(np.abs(P0)*np.abs(P1))

# plt.xlim([0,8])
# plt.ylim([0,1])
# plt.plot(freq[0:neq],coh[0:neq])
# plt.plot(freq[0:neq],coherence_model(freq[0:neq],r))
# plt.show()
