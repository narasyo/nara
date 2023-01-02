import numpy as np
import matplotlib.pyplot as plt
import ot

with open("IES.SMART1.TAI01.C-00") as f1:
    data1 = f1.readlines()[694:1023] 

with open("IES.SMART1.TAI01.I-06") as f2: #Update!
    data2 = f2.readlines()[790:1167] #update!

g1 = open('C00_NS.txt', 'w')
g1.writelines(data1)
g1.close()

g2 = open('I06_NS.txt', 'w') #update!
g2.writelines(data2)
g2.close()

with open('C00_NS.txt', 'r') as g1:
    ampcha1_list = g1.read().split()

with open('I06_NS.txt', 'r') as g2: #update!
    ampcha2_list = g2.read().split()

amp1_list = []    
for ampcha1 in ampcha1_list:
    amp1 = float(ampcha1)*980/1024
    amp1_list += [amp1]

amp2_list = []    
for ampcha2 in ampcha2_list:
    amp2 = float(ampcha2)*980/1024
    amp2_list += [amp2]

N_time=3102 #update!
tim1_list = np.linspace(0.01,0.01*N_time,N_time)
tim2_list = np.linspace(0.01,0.01*N_time,N_time)

N = 3102 #update!
starttime_dif = 73 #update!
xmin = 0
xmax = 0.01*N
h = (xmax-xmin)/N
p = np.linspace(xmin,xmax,N+1) 
c1 = abs(min(amp1_list[0:N]))*1.1
c1_list = np.linspace(c1,c1,N)

c2 = abs(min(amp2_list[starttime_dif:starttime_dif+N]))*1.1 
c2_list = np.linspace(c2,c2,N)

def python_list_add1(a1,b1):
    pos_amp1_list = np.array(a1) + np.array(b1)
    return pos_amp1_list.tolist()
    
def python_list_add2(a2,b2):
    pos_amp2_list = np.array(a2) + np.array(b2)
    return pos_amp2_list.tolist()

pos_amp1_list = python_list_add1(amp1_list[0:N],c1_list)
s1 = h*(np.sum(pos_amp1_list)-pos_amp1_list[0]/2-pos_amp1_list[N-1]/2)

pos_amp2_list = python_list_add2(amp2_list[starttime_dif:starttime_dif+N],c2_list)  
s2 = h*(np.sum(pos_amp2_list)-pos_amp2_list[0]/2-pos_amp2_list[N-1]/2)

def calc_devided_s1(n1):
    return n1/s1

def calc_devided_s2(n2):
    return n2/s2

amp1_norm_list = []
for pos1 in pos_amp1_list:
    amp1_norm_list.append(calc_devided_s1(pos1))

amp2_norm_list =[]
for pos2 in pos_amp2_list:
    amp2_norm_list.append(calc_devided_s2(pos2))

tau = np.linspace(0.01,0.01*N_time,N_time)

w = ot.emd2_1d(tau,tau,amp1_norm_list/sum(amp1_norm_list),amp2_norm_list/sum(amp2_norm_list),metric='minkowski',p=2)
wa = np.sqrt(w**2+(s1-s2)**2)
print('w =',w)
print('wa =',wa)

fig, ax = plt.subplots(facecolor="w")
ax.plot(tim1_list,amp1_norm_list,label="C00")
ax.plot(tim2_list,amp2_norm_list,label="I06") #update!
ax.legend()
plt.xlabel(u'time[s]')
plt.ylabel(u'probability')
plt.title(u'SMART-1 Array C00,I06 NS Acceleration Converted by Probability Distribution') #update!
plt.show()


