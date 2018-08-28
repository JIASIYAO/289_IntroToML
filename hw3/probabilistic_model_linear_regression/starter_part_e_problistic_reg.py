import numpy as np
import matplotlib.pyplot as plt

sample_size = [5,25,125,625]
plt.figure(figsize=[12, 10])             
for k in range(4):
    n = sample_size[k]
    
    # generate data
    # np.linspace, np.random.normal and np.random.uniform might be useful functions
    X = np.linspace(1,100,n)
    Z = np.random.uniform(-0.5, 0.5, n)
    w_real = 3.2
    Y = X*w_real + Z
    
    W = np.linspace(3.18,3.22,10000)
    N = len(W)
    likelihood = np.ones(N) # likelihood as a function of w

    w_min = np.max((Y-0.5)/X)
    w_max = np.min((Y+0.5)/X)
    for i1 in range(N):
        # compute likelihood
        if W[i1]>w_min and W[i1]<w_max:
            likelihood[i1] = 1
        else:
            likelihood[i1] = 0
         

    likelihood /= sum(likelihood) # normalize the likelihood
    
    plt.figure(figsize=(15,10))
    # plotting likelihood for different n
    plt.plot(W, likelihood)
    plt.xlabel('w', fontsize=10)
    plt.title(['n=' + str(n)], fontsize=14)

plt.show()
