import matplotlib.pyplot as plt
import numpy as np
import math


mu = 0.0
std = 0.4

tim = np.linspace(0,10.24,1024)
dt = tim[1]-tim[0]

wn0 = np.random.randn(len(tim))
wn = wn0*std + mu

# ------------
Wn = np.fft.fft(wn)
freq = np.fft.fftfreq(len(wn),d=dt)
df = freq[1] - freq[0]

Wp = np.ones_like(Wn)
for (i,f) in enumerate(freq):
    if f > 1.e-8:
        # print(f,2.0*math.pi*f)
        coef = (0.0+1.0j)*np.sqrt(2.0*math.pi*f)
        Wp[i] = Wn[i] / coef
    elif f < -1.e-8:
        coef = -(0.0+1.0j)*np.sqrt(2.0*math.pi*abs(f))
        Wp[i] = Wn[i] / coef
    else:
        Wp[i] = 0.0 + 0.0j

wp = np.real(np.fft.ifft(Wp))
# ------------

std_sim = np.std(wp)
wp = std/std_sim * wp

plt.plot(tim,wn,c="k")
plt.plot(tim,wp,c="r")
plt.show()
