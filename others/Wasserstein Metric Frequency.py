import numpy as np
import matplotlib.pyplot as plt
import ot
import os

n = 2696

xmin = 0
xmax = 0.01*n
h = (xmax-xmin)/n
p = np.linspace(xmin,xmax,n+1) 

savedir1 = "./data"
os.chdir(savedir1)

npy_list = ['I03_EW_fft.npy','M03_EW_fft.npy','O03_EW_fft.npy']
w_list = []
for npy in npy_list:
    fre1_list = np.load('C00_EW_fft.npy')
    fre2_list = np.load(npy)

    s1 = h*(np.sum(fre1_list)-fre1_list[0]/2-fre1_list[n-1]/2)
    s2 = h*(np.sum(fre2_list)-fre2_list[0]/2-fre2_list[n-1]/2)

    def calc_devided_s1(n1):
        return n1/s1

    def calc_devided_s2(n2):
        return n2/s2

    fre1_norm_list = []
    for fre1 in fre1_list:
        fre1_norm_list.append(calc_devided_s1(fre1))

    fre2_norm_list =[]
    for fre2 in fre2_list:
        fre2_norm_list.append(calc_devided_s2(fre2))

    tau = np.linspace(0.01,0.01*n,n)

    w = ot.emd2_1d(tau,tau,fre1_norm_list/sum(fre1_norm_list),fre2_norm_list/sum(fre2_norm_list),metric='minkowski',p=2)
    w_list += [w]

print(w_list)

