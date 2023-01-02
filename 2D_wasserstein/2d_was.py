import ot
import numpy as np
import math

def gauss(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2*sigma**2))

n = 500  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distribution

ls = np.arange(20, 500, 20)
nb = len(ls)
a = np.zeros((n, nb))
b = np.zeros((n, nb))
for i in range(nb):
      b[:, i] = gauss(n, mu=ls[i], sigma=10)
      a[:, i] = gauss(n, mu=ls[i], sigma=10)

print("a=",a)


M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
print("M=",M)

emd1 = ot.emd2(a, b, M, 1)
emdn = ot.emd2(a, b, M)

print("emd1=",emd1)
print("emdn=",emdn)


'''

# loss matrix
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
# M/=M.max()

print('Computing {} EMD '.format(nb))

# emd loss 1 proc
ot.tic()
emd1 = ot.emd2(a, b, M, 1)
ot.toc('1 proc : {} s')

# emd loss multipro proc
ot.tic()
emdn = ot.emd2(a, b, M)
ot.toc('multi proc : {} s')

np.testing.assert_allclose(emd1, emdn)

# emd loss multipro proc with log
ot.tic()
emdn = ot.emd2(a, b, M, log=True, return_matrix=True)
ot.toc('multi proc : {} s')

'''
# for i in range(len(emdn)):
#       emd = emdn[i]
#       log = emd[1]
#       cost = emd[0]
#       check_duality_gap(a, b[:, i], M, log['G'], log['u'], log['v'], cost)
#       emdn[i] = cost

# emdn = np.array(emdn)
# np.testing.assert_allclose(emd1, emdn) 