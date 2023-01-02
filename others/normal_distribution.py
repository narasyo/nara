import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats
from scipy.stats import norm
import os

x = np.linspace(-5,10,1000,endpoint=False)
norm1 = stats.norm.pdf(x=x, loc=0, scale=1.0)
norm2 = stats.norm.pdf(x=x, loc=4, scale=1.0)

point = {
    "start":[0,0.25],
    "end":[4,0.25]
}

plt.rcParams["font.size"] = 15

fig1, ax1 = plt.subplots()
ax1.set_xlabel(r"$x$")
ax1.set_ylabel("probability density")
ax1.plot(x, norm1, color="red")
ax1.plot(x, norm2, color="blue")
ax1.annotate("", xy=point["end"], xytext=point["start"],arrowprops=dict(shrink=0, width=1, headwidth=8, headlength=10, connectionstyle='arc3',facecolor='gray', edgecolor='gray'))
ax1.legend([r"$f(x)=N(0,1)$",r"$g(x)=N(4,1)$"], loc="upper right", fontsize=12)
fig1.tight_layout()

os.chdir("./soturon_tex/figure")
fig1.savefig("fig2-1.png")
fig1.savefig("fig2-1.eps")
plt.clf()
plt.close()

y1 = norm.cdf(x, loc=0, scale=1.0)
y2 = norm.cdf(x ,loc=4, scale=1.0)
x1 = np.linspace(0,4.0,1000)
y3 = np.linspace(0.5,0.5,1000)

fig2,ax2 = plt.subplots()
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"cumulative probability($t$)")
ax2.plot(x, y1, color="red")
ax2.plot(x, y2, color="blue")
ax2.plot(x1, y3, color="green")
ax2.text(0.3, 0.52, r"$|F^{-1}(t)-G^{-1}(t)|$",fontsize=10)
ax2.fill_between(x, y1, y2, facecolor="lime", alpha=0.5)
ax2.legend([r"$F(x)$",r"$G(x)$"], loc="upper left", fontsize=12)
fig2.tight_layout()

fig2.savefig("fig2-2.png")
fig2.savefig("fig2-2.eps")
plt.clf()
plt.close()
