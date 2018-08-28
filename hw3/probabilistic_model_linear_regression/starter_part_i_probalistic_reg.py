import numpy as np
import matplotlib.pyplot as plt
import pdb

sample_size = [5,25,125,625]
sigma = 100
print('For sigma=%f' %sigma)

w_real = [1,2]
print('w_real is :')
print(w_real)

for k in range(4):
    n = sample_size[k]

    # generate data 
    # np.linspace, np.random.normal and np.random.uniform might be useful functions
    X = np.random.rand(n,2)
    Z = np.random.normal(0,1,n)
    Y = X @ w_real + Z

    
    # compute likelihood
    N = 1001
    W1s = np.linspace(0, 4,N)
    W2s = np.linspace(0, 4,N)
    likelihood = np.zeros([N,N]) # likelihood as a function of w_1 and w_0
                        
    for i1 in range(N):
        w_1 = W1s[i1]
        for i2 in range(N):
            w_2 = W2s[i2]
            # compute the likelihood here
            likelihood[i1][i2] += (np.linalg.norm(Y - X @ [w_1, w_2]))**2
            likelihood[i1][i2] += (np.linalg.norm([w_1, w_2]))**2/sigma**2 
            likelihood[i1][i2] /= -2

    # plotting the likelihood
    plt.figure()                          
    # for 2D likelihood using imshow
    plt.imshow(np.exp(likelihood.T), cmap='hot', aspect='auto',extent=[0,4,0,4])
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.show()
    plt.savefig('simga_%d_n_%d.png' %(sigma, n), format='png')
    print(n)
