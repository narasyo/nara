import numpy as np
import os
import glob

path=os.path.dirname(__file__)
os.chdir(path)
print(path)

file = sorted(glob.glob("../wavelet/input/201308041229/*.dat"))
print(file)

def averaging(x) :
   ave = np.average(x)
   wave_ave = x-ave
   return wave_ave

number_list = []
for file_ele in file :
    number = str(file_ele[-6:-4])
    number_list += [number]

for f,n in zip(file,number_list) :
    for i,name_list in zip([0,1,2],["_EW", "_NS", "_verticle"]):
        acc = np.loadtxt(f,usecols=(int(i)),unpack=True)
        averaged_acc = averaging(acc)
        name = '201308041229_F'+ str(n) + str(name_list)
        np.save(name,averaged_acc)

