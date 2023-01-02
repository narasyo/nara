import numpy as np
import ot
import os
from scipy import signal

#######################################
##          Similarity    class      ##
#######################################
class Similarity:

    def __init__(self,tim,wave1,wave2,seismometer):
        self.tim = tim
        self.wave1 = wave1
        self.wave2 = wave2
        self.dt = tim[1]-tim[0]
        self.seismometer = seismometer
    
#-------------------------------------------#

    def linear_normalizing(w1,w2,sum_time) :

        xmin = 0
        xmax = 0.01*sum_time
        h = (xmax-xmin)/sum_time
        w1_min = min(w1)
        w2_min = min(w2)
        c = max(abs(w1_min),abs(w2_min))
        w1_pos = w1+c
        w2_pos = w2+c
        s1 = h*(np.sum(w1_pos)-w1_pos[0]/2-w1_pos[sum_time-1]/2)
        s2 = h*(np.sum(w2_pos)-w2_pos[0]/2-w2_pos[sum_time-1]/2)
        p1 = [value1/s1 for value1 in w1_pos]
        p2 = [value2/s2 for value2 in w2_pos]

        return p1,p2

    def softplus_normalizing(w1,w2,sum_time,con) :

        xmin = 0
        xmax = 0.01*sum_time
        h = (xmax-xmin)/sum_time

        b = con/(max(abs(w1)))

        pos_amp1_list = []
        for uni_amp1 in w1 :
            if b*uni_amp1 > 0 :
                pos_amp1 = b*uni_amp1+np.log(np.exp(-uni_amp1*b)+1)
                pos_amp1_list += [pos_amp1]
            else :
                pos_amp1 = np.log(np.exp(uni_amp1*b)+1)
                pos_amp1_list += [pos_amp1]

        pos_amp2_list = []
        for uni_amp2 in w2 :
            if b*uni_amp2 > 0 :
                pos_amp2 = b*uni_amp2+np.log(np.exp(-uni_amp2*b)+1)
                pos_amp2_list += [pos_amp2]
            else :
                pos_amp2 = np.log(np.exp(uni_amp2*b)+1)
                pos_amp2_list += [pos_amp2]
        
        w1_pos = np.array(pos_amp1_list)
        w2_pos = np.array(pos_amp2_list)

        s1 = h*(np.sum(w1_pos)-w1_pos[0]/2-w1_pos[sum_time-1]/2)
        s2 = h*(np.sum(w2_pos)-w2_pos[0]/2-w2_pos[sum_time-1]/2)
        p1 = [value1/s1 for value1 in w1_pos]
        p2 = [value2/s2 for value2 in w2_pos]

        return p1,p2

    def exponential_normalizing(w1,w2,sum_time) :

        xmin = 0
        xmax = 0.01*sum_time
        h = (xmax-xmin)/sum_time

        b = 3.0/(max(abs(w1)))

        pos_amp1_list = []
        for uni_amp1 in w1 :
            pos_amp1 = np.exp(b*uni_amp1)
            pos_amp1_list += [pos_amp1]

        pos_amp2_list = []
        for uni_amp2 in w2 :
            pos_amp2 = np.exp(b*uni_amp2)
            pos_amp2_list += [pos_amp2]

        w1_pos = np.array(pos_amp1_list)
        w2_pos = np.array(pos_amp2_list)

        s1 = h*(np.sum(w1_pos)-w1_pos[0]/2-w1_pos[sum_time-1]/2)
        s2 = h*(np.sum(w2_pos)-w2_pos[0]/2-w2_pos[sum_time-1]/2)
        p1 = [value1/s1 for value1 in w1_pos]
        p2 = [value2/s2 for value2 in w2_pos]

        return p1,p2
    
    def fourier_spectrum(wave,fs):

        def set_parameters(wave,dt):
            n = len(wave) / 2
            ntw = 2
            while ntw < n:
                ntw = ntw*2
            freq = abs(np.fft.fftfreq(ntw,d=dt))
            return freq, ntw

        dt = 1.0/float(fs)
        freq, ntw = set_parameters(wave,dt)
        w = np.fft.fft(wave[0:ntw])

        return freq, w
#-------------------------------------------------------#

    def wasserstein_linear_normalizing(self) :
       
        sum_time = len(self.tim)
        p1,p2 = Similarity.linear_normalizing(self.wave1,self.wave2,sum_time)
        was = ot.emd2_1d(self.tim,self.tim,p1/sum(p1),p2/sum(p2),metric='minkowski',p=2)

        return was

    def wasserstein_softplus_normalizing(self,con) :

        sum_time = len(self.tim)
        p1_pos,p2_pos = Similarity.softplus_normalizing(self.wave1,self.wave2,sum_time,con)
        was_pos = ot.emd2_1d(self.tim,self.tim,p1_pos/sum(p1_pos),p2_pos/sum(p2_pos),metric='minkowski',p=2)
        p1_neg,p2_neg = Similarity.softplus_normalizing(-self.wave1,-self.wave2,sum_time,con)
        was_neg = ot.emd2_1d(self.tim,self.tim,p1_neg/sum(p1_neg),p2_neg/sum(p2_neg),metric='minkowski',p=2)
        was = was_pos + was_neg

        return was

    def wasserstein_exponential_normalizing(self) :

        sum_time = len(self.tim)
        p1_pos,p2_pos = Similarity.exponential_normalizing(self.wave1,self.wave2,sum_time)
        was_pos = ot.emd2_1d(self.tim,self.tim,p1_pos/sum(p1_pos),p2_pos/sum(p2_pos),metric='minkowski',p=2)
        p1_neg,p2_neg = Similarity.exponential_normalizing(-self.wave1,-self.wave2,sum_time)
        was_neg = ot.emd2_1d(self.tim,self.tim,p1_neg/sum(p1_neg),p2_neg/sum(p2_neg),metric='minkowski',p=2)
        was = was_pos + was_neg

        return was

    def mse(self) :

        mse = np.mean((self.wave1-self.wave2)**2)

        return mse

    def coherence(self,window):

        freq,f0 = Similarity.fourier_spectrum(self.wave1,1/self.dt)
        freq,f1 = Similarity.fourier_spectrum(self.wave2,1/self.dt)
        arg =np.where(freq<0.2)
        f0[arg]=0.0
        p0 = np.real(np.conjugate(f0)*f0)
        p1 = np.real(np.conjugate(f1)*f1)
        c01 = np.conjugate(f0)*f1

        nw = int(window/(freq[1]-freq[0]))
        if nw%2 == 0:
            nw += 1
        nw2 = int((nw-1)/2)
        w = signal.parzen(nw,False)
        
        a = np.r_[c01[nw2:0:-1],c01,c01[0],c01[-1:-nw2:-1]]
        c01_s = np.convolve(w/w.sum(),a,mode='valid')

        a = np.r_[p0[nw2:0:-1],p0,p0[0],p0[-1:-nw2:-1]]
        p0_s = np.convolve(w/w.sum(),a,mode='valid')
        a = np.r_[p1[nw2:0:-1],p1,p1[0],p1[-1:-nw2:-1]]
        p1_s = np.convolve(w/w.sum(),a,mode='valid')

        c01_abs = np.real(np.conjugate(c01_s)*c01_s)

        return freq, c01_abs/(p0_s*p1_s) 









        
    





    