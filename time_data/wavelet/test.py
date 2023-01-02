import time
import numpy as np
import matplotlib.pyplot as plt
from CWavelet import CWavelet

fs = 100000
f = 440
t = np.arange(fs // 2)
x = np.zeros(3 * fs, dtype=np.float64)
x[fs//2:fs] = np.sin(2 * np.pi * f * t / fs)
x[fs + fs // 2:2 * fs] = np.sin(2 * np.pi * 2 * f * t / fs)
x[2 * fs + fs // 2:3 * fs] = np.sin(2 * np.pi * 4 * f * t / fs)

wavelet = CWavelet(fs)
start = time.time()
y = wavelet.transform(x, strides=1)
end = time.time()
print(end - start)

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(np.abs(y) ** 2, aspect='auto', origin='lower')
ax.set_xticks(np.arange(0, y.shape[1], fs // 2))
ax.set_xticklabels(map(str, np.arange(0, 3, 0.5)))
ax.set_yticks(np.arange(0, y.shape[0], 12))
ax.set_yticklabels(map(str, wavelet.freq_period[::12]))
ax.set_xlabel('Time [s]')
ax.set_ylabel('Frequency [Hz]')
plt.colorbar(im)
fig.tight_layout()
plt.show()

start = time.time()
x2 = wavelet.transform_inverse(y)
end = time.time()
print(end - start)

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(np.arange(x.shape[-1]) / fs, x)
ax.set_title('original')
ax = fig.add_subplot(212)
ax.plot(np.arange(x.shape[-1]) / fs, x2)
ax.set_title('recovered')
fig.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x[fs // 2:fs // 2 + 500], label='original')
ax.plot(x2[fs // 2:fs // 2 + 500], label='recovered')
ax.legend(loc='upper right')
plt.show()