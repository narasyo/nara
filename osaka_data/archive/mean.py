import numpy as np 
import os

os.chdir("./osaka_data/archive")

list = ["OSA","OSB","OSC","FKS"]
for l in list :
    wave = np.load(f"{l}_acc_ud.npy")
    wave_mean = np.mean(wave)
    wave_mod = wave - wave_mean
    np.save(f"{l}_acc_ud_modified.npy",wave_mod)