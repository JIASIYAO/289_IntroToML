import numpy as np
import scipy.io

mdict = scipy.io.loadmat("a.mat")

x = mdict['x']
u = mdict['u']

# Your code to compute a and b
A = np.matrix([x[0][:-1], u[0][:-1]]).T
b = np.matrix([x[0][1:]]).T
sol = (A.T @ A).I @ A.T @ b
print('A: %.2f, B:%.2f'%(sol[0], sol[1]))
