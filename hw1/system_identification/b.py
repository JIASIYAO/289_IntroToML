import numpy as np
import scipy.io

mdict = scipy.io.loadmat("b.mat")

x = mdict['x']
u = mdict['u']


# Your code to compute a and b
X = np.concatenate((x[:-1], u[:-1]), axis=1).T[0]
Y = x[1:].T[0]
X = np.matrix(X)
Y = np.matrix(Y)
w = Y @ X.T @ (X @ X.T).I 

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
print('A:')
print(w[:,0:3])
print('B:')
print(w[:,3:])
