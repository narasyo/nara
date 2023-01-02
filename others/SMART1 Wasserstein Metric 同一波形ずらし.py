import numpy as np
import matplotlib.pyplot as plt
import ot

with open("IES.SMART1.TAI01.C-00") as f1: #Update!
    start1 = input('1つ目のファイルの始めの行番号を入力>>')
    end1 = input('1つ目のファイルの終わりの行番号を入力>>')
    data1 = f1.readlines()[int(start1):int(end1)] 

start2 = input('2つ目のファイルの始めの行番号を入力>>')
end2 = input('2つ目のファイルの終わりの行番号を入力>>')

g1 = open('C00_EW.txt', 'w')
g1.writelines(data1)
g1.close()

with open('C00_EW.txt', 'r') as g1:
    ampcha1_list = g1.read().split()

N_input = input('サンプル数を入力>>') 
N = int(N_input)
starttime_dif_input = input('計測開始時刻の差[s]を入力>>')
starttime_dif = int(starttime_dif_input)

amp1_list = []    
for ampcha1 in ampcha1_list:
    amp1 = float(ampcha1)*980/1024
    amp1_list += [amp1]

list1 = np.linspace(0,0,3359)
for i,v in enumerate(list1):
        amp1_list.insert(0,v)

while len(amp1_list) < 10000:
    amp1_list.append(0)  
print('len1=' ,len(amp1_list))
amp1_mean = np.mean(amp1_list)
mean1_list = np.linspace(amp1_mean,amp1_mean,10000)

def python_list_dif1(a1,b1):
    uni_amp1_list = np.array(a1) - np.array(b1)
    return uni_amp1_list.tolist()

uni_amp1_list = python_list_dif1(amp1_list,mean1_list)

time = np.linspace(0.01,0.01*10000,10000)

xmin = 0
xmax = 0.01*10000
h = (xmax-xmin)/10000
p = np.linspace(xmin,xmax,10001) 
c1 = abs(min(uni_amp1_list))

N_dif_list =np.linspace(1,10000-N,10000-N)

w_list,l2_list,wa_list,amp2_list2 = [],[],[],[]

for N_dif in N_dif_list:
    with open("IES.SMART1.TAI01.C-00") as f2: #Update!
        data2 = f2.readlines()[int(start2):int(end2)]

    g2 = open('C00_EW.txt', 'w') #update!
    g2.writelines(data2)
    g2.close()

    with open('C00_EW.txt', 'r') as g2: #update!
        ampcha2_list = g2.read().split()

    amp2_list = []    
    for ampcha2 in ampcha2_list:
        amp2 = float(ampcha2)*980/1024
        amp2_list += [amp2]
    
    list7 = np.linspace(0,0,int(N_dif)-1)
    amp2_list2[0:9999] = amp2_list
    for list in list7:
        amp2_list2.insert(0,list)

    while len(amp2_list2) < 10000:
        amp2_list2.append(0)  
    
    dlen = 10000
    mean = 0.0
    std = 8.0


    y = np.random.normal(mean,std,dlen)

    amp2_noise_list = []
    amp2_noise_list = [y1 + y2 for (y1,y2) in zip(amp2_list2,y)]

    amp2_mean = np.mean(amp2_list2)
 
    mean2_list = np.linspace(amp2_mean,amp2_mean,10000)

    
    def python_list_dif2(a2,b2):
        uni_amp2_list = np.array(a2) - np.array(b2)
        return uni_amp2_list.tolist()

    uni_amp2_list = python_list_dif2(amp2_noise_list,mean2_list)

    """
    plt.plot(time,uni_amp1_list)
    plt.plot(time,uni_amp2_list)
    plt.show()
    """

    c2 = abs(min(uni_amp2_list))
    c = max(c1,c2)

    pos_amp1_list = []
    for uni_amp1 in uni_amp1_list :
        if uni_amp1 >= 0 :
            pos_amp1 = uni_amp1+1/c
            pos_amp1_list += [pos_amp1]
        else :
            pos_amp1 = (np.exp(c*uni_amp1))/c
            pos_amp1_list += [pos_amp1]

    s1 = h*(np.sum(pos_amp1_list)-pos_amp1_list[0]/2-pos_amp1_list[9999]/2)
    #print('s1 =',s1)

    def calc_devided_s1(n1):
        return n1/s1

    amp1_norm_list = []
    for pos1 in pos_amp1_list:
        amp1_norm_list.append(calc_devided_s1(pos1))

    pos_amp2_list = []
    for uni_amp2 in uni_amp2_list :
        if uni_amp2 >= 0 :
            pos_amp2 = uni_amp2+1/c
            pos_amp2_list += [pos_amp2]
        else :
            pos_amp2 = np.exp(c*uni_amp2)/c
            pos_amp2_list += [pos_amp2]

    s2 = h*(np.sum(pos_amp2_list)-pos_amp2_list[0]/2-pos_amp2_list[9999]/2)

    def calc_devided_s2(n2):
        return n2/s2

    amp2_norm_list =[]
    for pos2 in pos_amp2_list:
        amp2_norm_list.append(calc_devided_s2(pos2))

    tau = np.linspace(0.01,0.01*(10000-N),(10000-N))
    #plt.plot(time,amp2_norm_list)
    #plt.show()
    w = ot.emd2_1d(tau,tau,amp1_norm_list/sum(amp1_norm_list),amp2_norm_list/sum(amp2_norm_list),metric='minkowski',p=2)
    a_dif_list = []
    for a1,a2 in zip(amp1_norm_list,amp2_norm_list):
        a_dif = (a1**2-a2**2)**2
        a_dif_list += [a_dif]
    l2 = np.mean(a_dif_list)
    #print(len(amp2_list2))
    #print(amp2_list2)
    #print(w)
    w_list += [w]
    l2_list += [l2]

plt.plot(tau,w_list)
plt.ylim(0,1000)
#plt.plot(tau,wa_list)
plt.show()

plt.plot(tau,l2_list)
plt.show()
#print('w =',w)
#print('wa =',wa)

#fig, ax = plt.subplots(facecolor="w")
#ax.plot(tim1_list,amp1_norm_list,label="C00")
#ax.plot(tim2_list,amp2_norm_list,label="I03") #update!
#ax.legend()
#plt.xlabel(u'time[s]')
#plt.ylabel(u'probability')
#plt.title(u'SMART-1 Array C00,I03 NS Acceleration Converted by Probability Distribution') #update!
#plt.show()


