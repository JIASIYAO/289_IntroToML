# Load mat file
import numpy as np
import scipy.io

mdict = scipy.io.loadmat("train.mat")

# Assemble xu matrix
x = mdict["x"]   # position of a car
v = mdict["xd"]  # velocity of the car
xprev = mdict["xp"]   # position of the car ahead
vprev = mdict["xdp"]  # velocity of the car ahead

acc = mdict["xdd"]  # acceleration of the car

a, b, c, d, e = 0, 0, 0, 0, 0

# Your code to compute a, b, c, d
X = np.matrix((x[0], v[0], xprev[0], vprev[0], np.ones(len(x[0])))).T
Y = np.matrix(acc[0]).T
sol = (X.T @ X).I @ X.T @ Y
a,b,c,d,e = sol
a = a[0,0]
b = b[0,0]
c = c[0,0]
d = d[0,0]
e = e[0,0]

print("Fitted dynamical system:")
print("xdd_i = {:.3f} x_i + {:.3f} xd_i + {:.3f} x_i-1 + {:.3f} xd_i-1 + {:.3f}".format(a, b, c, d, e))
