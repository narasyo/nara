import numpy as np
import scipy.stats
import ot
import matplotlib.pyplot as plt
import sympy as sym

def ricker_sp(tim,fp1,fp2,fp3,fp4,fp5,fp6,fp7,fp8,fp9,fp10,tp1,tp2,tp3,tp4,tp5,tp6,tp7,tp8,tp9,tp10,amp1,amp2,amp3,amp4,amp5,amp6,amp7,amp8,amp9,amp10):
    t1 = ((tim-tp1)*np.pi*fp1)**2
    t2 = ((tim-tp2)*np.pi*fp2)**2
    t3 = ((tim-tp3)*np.pi*fp3)**2
    t4 = ((tim-tp4)*np.pi*fp4)**2
    t5 = ((tim-tp5)*np.pi*fp5)**2
    t6 = ((tim-tp6)*np.pi*fp6)**2
    t7 = ((tim-tp7)*np.pi*fp7)**2
    t8 = ((tim-tp8)*np.pi*fp8)**2
    t9 = ((tim-tp9)*np.pi*fp9)**2
    t10 = ((tim-tp10)*np.pi*fp10)**2
    return (2*t1-1)*np.exp(-t1)*amp1+(2*t2-1)*np.exp(-t2)*amp2+(2*t3-1)*np.exp(-t3)*amp3+(2*t4-1)*np.exp(-t4)*amp4+(2*t5-1)*np.exp(-t5)*amp5+(2*t6-1)*np.exp(-t6)*amp6+(2*t7-1)*np.exp(-t7)*amp7+(2*t8-1)*np.exp(-t8)*amp8+(2*t9-1)*np.exp(-t9)*amp9+(2*t10-1)*np.exp(-t10)*amp10
    
N = 10**6
xmin = -30
xmax = 30
h = (xmax-xmin)/N
p = np.linspace(xmin,xmax,N+1)
f1 = -ricker_sp(p,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,2.0,4.0,6.0,8.0,10.0,10.0,8.0,6.0,4.0,2.0)
g1 = -ricker_sp(p,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,6.0,12.0,18.0,24.0,30.0,30.0,24.0,18.0,12.0,6.0)
c1 = abs(min(f1))*1.1
c2 = abs(min(g1))*1.1
f2 = f1+c1
g2 = g1+c2
s1 = h*(np.sum(f2)-f2[0]/2-f2[N]/2)
s2 = h*(np.sum(g2)-g2[0]/2-g2[N]/2)
print(s1)
print(s2)
t = np.linspace(xmin,xmax,N+1)
f2nrm = f2/s1
g2nrm = g2/s2
a2 = f2nrm**2

#plt.plot(t,f1)
plt.plot(t,f2nrm)
plt.plot(t,g2nrm)
plt.show()

tau_list = np.linspace(-10,10,100)

w_list,l2_list,wa_list = [],[],[]
for tau in tau_list:
    x = -ricker_sp(t,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,tau-5,tau-4,tau-3,tau-2,tau-1,tau,tau+1,tau+2,tau+3,tau+4,6.0,12.0,18.0,24.0,30.0,30.0,24.0,18.0,12.0,6.0)
    xc2 = x+c2
    xnrm = xc2/s2
    xnrm2 = xnrm**2
    #plt.plot(t,x2)
    # # plt.plot(tau_list,l2_list)
    #plt.show()
    # Wp = scipy.stats.wasserstein_distance(t,t,x,y)
    l2 = np.mean((a2-xnrm2)**2)
    w = ot.emd2_1d(t,t,a2/a2.sum(),xnrm2/xnrm2.sum(),metric='minkowski',p=2)
    wa = np.sqrt(w**2+(s1-s2)**2)
    #print(tau,w,l2)
    w_list += [w]
    l2_list += [l2]
    wa_list += [wa]
    #print(w_list)

plt.plot(tau_list,w_list)
plt.show()

plt.plot(tau_list,l2_list)
plt.show()

plt.plot(tau_list,wa_list)
plt.show()
