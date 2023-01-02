import matplotlib.pyplot as plt
import numpy as np
import os

def ricker(tim,fp,tp,amp):

    t1 = ((tim-tp)*np.pi*fp)**2

    return (2*t1-1)*np.exp(-t1)*amp

os.chdir("./kansaishibu/figure")

N = 2000 # 波形のプロット数
xmin = -10
xmax = 10
tim = np.linspace(xmin,xmax,N+1)

f1 = ricker(tim,0.4,-0.5,15)
f2 = -ricker(tim,0.6,-0.25,25)
f3 = ricker(tim,0.8,0,40)
f4 = -ricker(tim,0.6,0.25,25)
f5 = ricker(tim,0.4,0.5,15)
f = f1 + f2 + f3 + f4 + f5

fig,ax = plt.subplots()
ax.plot(tim,f,color="blue")
ax.axis("off")
ax.set_xlim(-5,5)
fig.savefig("ricker_wave.png")
fig.savefig("ricker_wave.eps")