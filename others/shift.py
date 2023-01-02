import numpy as np 
import os
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import ot

def calc_devided(numer,demon) :

    return numer/demon

def normalized_linear(wave1,wave2,sum_time) :

    xmin = 0
    xmax = 0.01*sum_time
    h = (xmax-xmin)/sum_time
    c1 = np.abs(min(wave1))
    c2 = np.abs(min(wave2))
    const = max(c1,c2)
    pos_amp1_list = []
    for uni_amp1 in wave1 :
        pos_amp1 = uni_amp1 + const
        pos_amp1_list += [pos_amp1]
        
    pos_amp2_list = []
    for uni_amp2 in wave2 :
        pos_amp2 = uni_amp2 + const
        pos_amp2_list += [pos_amp2]

    s1 = h*(np.sum(pos_amp1_list)-pos_amp1_list[0]/2-pos_amp1_list[sum_time-1]/2)

    s2 = h*(np.sum(pos_amp2_list)-pos_amp2_list[0]/2-pos_amp2_list[sum_time-1]/2)

    amp1_norm_list = []
    for pos1 in pos_amp1_list:
        amp1_norm_list.append(calc_devided(pos1,s1))

    amp2_norm_list =[]
    for pos2 in pos_amp2_list:
        amp2_norm_list.append(calc_devided(pos2,s2))

    amp1_norm = np.array(amp1_norm_list)
    amp2_norm = np.array(amp2_norm_list)

    return amp1_norm,amp2_norm

os.chdir("./data/furukawa_data/201212071719/20121207_modified")

wave = np.load("201212071719_F1_EW_modified.npy")
t = np.arange(0,480,0.01)
time = 48000

N_list = np.arange(-10,10,0.1)

fig1,ax = plt.subplots()
ax.plot(t,wave)
fig1.savefig("wave.png")

was_list = []
for N in N_list :
    wave_mag = wave
    wave_mag = [x*N for x in wave]

    w1,w2 = normalized_linear(wave,wave_mag,time)

    was = ot.emd2_1d(t,t,w1/sum(w1),w2/sum(w2),metric='minkowski',p=2)
    was_list += [was]

fig3, ax =plt.subplots()
ax.plot(N_list,was_list,color = "blue")

N_len = len(N_list)

ims1 =[]
for i in range(N_len) :
    p = ax.plot(N_list[i],was_list[i],color = "darkblue",marker = 'o',markersize=10)
    ims1.append(p)

ani = animation.ArtistAnimation(fig3,ims1,interval=10)
ani.save("was_mag.gif",writer='imagemagick')

# fig2 = plt.figure()
# ims2 = []

# for N in N_list :
#     wave_part = wave
#     for i in array :
#         i = int(i)
#         wave_part = np.delete(wave_part,i)
#         wave_part = np.insert(wave_part,i,wave[i]*N)
    
#     im = plt.plot(t,wave_part,color="blue")

#     ims2.append(im)

# ani = animation.ArtistAnimation(fig2, ims2, interval=10)
# plt.show()
# ani.save("sample_part.gif", writer = "imagemagick")
    
