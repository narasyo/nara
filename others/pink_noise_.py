import numpy as np
import math
import matplotlib.pyplot as plt

std = 10
tim_len = 2000

wn = np.random.normal(0,std,tim_len)

Wn = np.fft.fft(wn)
freq = np.fft.fftfreq(len(wn),d=0.01)
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

tim = np.arange(0,20,0.01)

fig,ax = plt.subplots()

ax.plot(tim,wp)

fig.savefig("./fig.png")