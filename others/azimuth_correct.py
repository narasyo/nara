import numpy as np
import os
import math

data_EW = ['201308041229_F01_EW.npy','201308041229_F02_EW.npy','201308041229_F03_EW.npy','201308041229_F04_EW.npy','201308041229_F05_EW.npy','201308041229_F08_EW.npy',\
           '201308041229_F09_EW.npy','201308041229_F11_EW.npy','201308041229_F12_EW.npy','201308041229_F14_EW.npy','201308041229_F15_EW.npy','201308041229_F16_EW.npy',\
           '201308041229_F17_EW.npy','201308041229_F19_EW.npy','201308041229_F20_EW.npy','201308041229_F22_EW.npy','201308041229_F23_EW.npy','201308041229_F24_EW.npy',\
           '201308041229_F25_EW.npy','201308041229_F26_EW.npy','201308041229_F27_EW.npy','201308041229_F28_EW.npy','201308041229_F29_EW.npy','201308041229_F30_EW.npy',\
           '201308041229_F31_EW.npy','201308041229_F32_EW.npy','201308041229_F34_EW.npy','201308041229_F35_EW.npy','201308041229_F36_EW.npy']

data_NS = ['201308041229_F01_NS.npy','201308041229_F02_NS.npy','201308041229_F03_NS.npy','201308041229_F04_NS.npy','201308041229_F05_NS.npy','201308041229_F08_NS.npy',\
           '201308041229_F09_NS.npy','201308041229_F11_NS.npy','201308041229_F12_NS.npy','201308041229_F14_NS.npy','201308041229_F15_NS.npy','201308041229_F16_NS.npy',\
           '201308041229_F17_NS.npy','201308041229_F19_NS.npy','201308041229_F20_NS.npy','201308041229_F22_NS.npy','201308041229_F23_NS.npy','201308041229_F24_NS.npy',\
           '201308041229_F25_NS.npy','201308041229_F26_NS.npy','201308041229_F27_NS.npy','201308041229_F28_NS.npy','201308041229_F29_NS.npy','201308041229_F30_NS.npy',\
           '201308041229_F31_NS.npy','201308041229_F32_NS.npy','201308041229_F34_NS.npy','201308041229_F35_NS.npy','201308041229_F36_NS.npy']

os.chdir('./data/furukawa_data')
azimuth = np.loadtxt('azimuth.txt')
azimuth = np.delete(azimuth, [5,6,9,12,17,20,32])
print(azimuth)

os.chdir('./20130804')
for ew,ns,az in zip(data_EW,data_NS,azimuth) :
    EW = np.load(ew)
    NS = np.load(ns)
    rad_az = math.radians(az)
    EW_list,NS_list = [],[]
    for EW_ele,NS_ele in zip(EW,NS) :
        mod_EW = -NS_ele*(math.sin(rad_az))+EW_ele*(math.cos(rad_az))
        mod_NS = NS_ele*(math.cos(rad_az))+EW_ele*(math.sin(rad_az))
        EW_list += [mod_EW]
        NS_list += [mod_NS]
    new_EW = np.array(EW_list)
    new_NS = np.array(NS_list)
    name_ew = ew[:19] + '_modified'
    name_ns = ns[:19] + '_modified'
    os.chdir('../20130804_modified')
    np.save(name_ew,new_EW)
    np.save(name_ns,new_NS)
    os.chdir('../20130804')






        



