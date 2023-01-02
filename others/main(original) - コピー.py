import numpy as np
import scipy.stats
import ot
import matplotlib.pyplot as plt

N = 2696
xmin = 0
xmax = 26.96
h = (xmax-xmin)/N
p = np.linspace(xmin,xmax,N+1) 
f1 = -ricker(p,1.0,2.0,100)
c = abs(min(f1))*1.1
f2 = f1+c
s1 = h*(np.sum(f2)-f2[0]/2-f2[N]/2)
print(s1)

t = np.linspace(xmin,xmax,N+1)
f2nrm = f2/s1
a2 = f2nrm**2

#plt.plot(t,f1)
plt.plot(t,f2nrm)
plt.show()

tau_list = np.linspace(-0.5,4.5,101)

w_list,l2_list = [],[]
for tau in tau_list:
    x = -ricker(t,1.0,tau,100.0)
    xc = x+c
    xnrm = xc/s1
    xnrm2 = xnrm**2
    #plt.plot(t,x2)
    # # plt.plot(tau_list,l2_list)
    #plt.show()
    # Wp = scipy.stats.wasserstein_distance(t,t,x,y)
    l2 = np.mean((a2-xnrm2)**2)
    w = ot.emd2_1d(t,t,a2/a2.sum(),xnrm2/xnrm2.sum(),metric='minkowski',p=2)
    #print(tau,w,l2)
    w_list += [w]
    l2_list += [l2]
    #print(w_list)

#plt.plot(tau_list,w_list)
plt.plot(tau_list,l2_list)
plt.show()
