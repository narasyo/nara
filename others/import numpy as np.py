import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from library.output_format import Format
Format.params()

# parameters
N = 2**20 # data number
dt = 0.0001 # data step [s]
f1, f2 = 5, 8 # frequency[Hz]
A1, A2 = 5, 5 # Amplitude
p1, p2 = 0, 0 # phase

t = np.arange(0, N*dt, dt) # time
freq = np.linspace(0, 1.0/dt, N) # frequency step

y = A1*np.sin(2*np.pi*f1*t + p1) + A2*np.sin(2*np.pi*f2*t + p2) 

yf = fft(y)/(N/2) # 離散フーリエ変換&規格化

plt.figure(1)
plt.subplot(211)
plt.plot(t, y)
plt.xlim(0, 1)
plt.xlabel("time")
plt.ylabel("amplitude")

plt.subplot(212)
plt.plot(freq, np.abs(yf))
plt.xlim(0, 10)
#plt.ylim(0, 5)
plt.xlabel("frequency")
plt.ylabel("amplitude")
plt.tight_layout()
plt.savefig("01")
plt.show()