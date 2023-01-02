import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sys
sys.path.append('./Python_Code')
from PySGM.vector import vector
from others.similarity import Similarity
from tqdm import tqdm
from peer_reviewed.format_time_series import Format
Format.params()

wave_fks = np.load("./Python_Code/osaka_data/archive/osaka_wave/FKS_vel_ud.npy")
wave_fks = wave_fks[3500:10000]
t = np.arange(35,100,0.01)
max_value = max(wave_fks)

fig1,ax1 = plt.subplots()
ax1.plot(t,wave_fks,color="black")
ax1.set_xlim(35,100)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.axes.yaxis.set_visible(False)
ax1.tick_params(axis='x')
ax1.set_xlabel("Time[s]")
# ax1.text(35,0.8*max_value,"FKS_vel_UD")

fig1.tight_layout()
fig1.savefig("./Python_Code/peer_reviewed/figure/osaka_wave/osaka_fks_ud.png")
