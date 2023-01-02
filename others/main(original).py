import numpy as np
import scipy.stats
import ot
import matplotlib.pyplot as plt

def ricker(tim,fp,tp,amp):
    t1 = ((tim-tp)*np.pi*fp)**2
    return (2*t1-1)*np.exp(-t1)*amp

N = 10**6
xmin = -2
xmax = 6
h = (xmax-xmin)/N
q = np.linspace(xmin,xmax,N+1)
f1 = -ricker(q,1.0,2.0,100)
f3 = -ricker(q,1.0,0.0,100)
print(f1)
c = abs(min(f1))*1.1
f2 = f1+c
s1 = h*(np.sum(f2)-f2[0]/2-f2[N]/2)
print(s1)

t = np.linspace(xmin,xmax,N+1)
f2nrm = f2/s1
a2 = f2nrm**2
a2_list =[]
for a2_ele in a2 :
    a2_list += [a2_ele]

#print(a2_list)
    

plt.plot(t,f1)
plt.plot(t,f3)
#plt.plot(t,f2nrm)
plt.show()

tau_list = np.linspace(-0.5,4.5,101)

w_list,l2_list = [],[]
for tau in tau_list:
    x = -ricker(q,1.0,tau,100.0)
    xc = x+c
    xnrm = xc/s1
    xnrm2 = xnrm**2
    #plt.plot(t,x2)
    #plt.plot(tau_list,l2_list)
    #plt.show()
    # Wp = scipy.stats.wasserstein_distance(t,t,x,y)
    l2 = np.mean((a2-xnrm2)**2)
    w = ot.emd2_1d(t,t,a2_list/sum(a2_list),xnrm2/sum(xnrm2),metric='minkowski',p=2)
    #print(tau,w,l2)
    w_list += [w]
    l2_list += [l2]
    #print(w_list)

#plt.plot(tau_list,w_list)
plt.plot(tau_list,l2_list)
plt.show()
