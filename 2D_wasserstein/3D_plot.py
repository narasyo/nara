import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def gaussian_func(x, y, sigma):
    exponent = - (x ** 2 + y ** 2) / (2 * pow(sigma, 2))
    denominator = 2 * math.pi * pow(sigma, 2)
    return pow(math.e, exponent) / denominator

x = np.arange(-3.0, 3.0, 0.01)
y = np.arange(-3.0, 3.0, 0.01)
z = gaussian_func(x,y,1)
print(z)
print(z.shape)
X, Y = np.meshgrid(x, y)
Z = gaussian_func(X, Y, 1)

fig = plt.figure(figsize=(10, 10))
ax = Axes3D(fig)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")

ax.plot_wireframe(X, Y, Z)
plt.savefig("gaussian.jpg")
plt.show()