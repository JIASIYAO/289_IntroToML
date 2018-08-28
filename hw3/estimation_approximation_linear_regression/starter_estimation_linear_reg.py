import numpy as np
import matplotlib.pyplot as plt
import pdb

def que_e():
    degs = np.arange(1,10)
    step = 100
    ns = np.arange(100,1000,step)
    
    # assign problem parameters
    w1 = 1
    w0 =1
    
    
    errors = []
    for runs in range(50):
        for i in range(len(ns)):
            error = np.zeros((len(ns), len(degs)))
            n = ns[i]
            # generate data
            # np.random might be useful
            
            # generate alpha
            alpha = np.linspace(-1,1,n)
            
            # generate z
            sigma = 1
            z = np.random.normal(loc=0, scale=sigma, size=n)
            
            # generate y
            y_real = alpha*w1 + w0
            y = y_real + z
        
            for j in range(len(degs)):
                deg = degs[j]
                # fit data with different models
                # np.polyfit and np.polyval might be useful
                w = np.polyfit(alpha,y,deg)
                error[i][j] = np.linalg.norm(np.polyval(w, alpha) - y_real)/n
            errors.append(error)
    
    errors = np.mean(errors, axis=0)
    
    # plotting figures
    # sample code
    
    plt.figure(figsize=(20,10))
    plt.subplot(121)
    plt.plot(degs,errors[-1])
    plt.xlabel('degree of polynomial')
    plt.ylabel('error')
    plt.subplot(122)
    plt.loglog(ns, errors[:,-1])
    plt.xlabel('log(number of samples)')
    plt.ylabel('log of error')
    plt.tight_layout()
    plt.show()
    plt.savefig('4_e.png', format='png')
    plt.close()
    
def que_g():
    degs = np.arange(1,15)
    step = 10
    ns = np.arange(10,120,step)
    
    errors = []
    for runs in range(50):
        for i in range(len(ns)):
            error = np.zeros((len(ns), len(degs)))
            n = ns[i]
            # generate data
            # np.random might be useful
            
            # generate alpha
            alpha = np.linspace(-4,3,n)
            
            # generate z
            sigma = 1
            z = np.random.normal(loc=0, scale=sigma, size=n)
            
            # generate y
            y_real = np.exp(alpha)
            y = y_real + z
        
            for j in range(len(degs)):
                deg = degs[j]
                # fit data with different models
                # np.polyfit and np.polyval might be useful
                w = np.polyfit(alpha,y,deg)
                error[i][j] = np.linalg.norm(np.polyval(w, alpha) - y_real)/n
            errors.append(error)
    
    errors = np.mean(errors, axis=0)
    
    # plotting figures
    # sample code
    
    plt.figure(figsize=(20,10))
    plt.subplot(121)
    plt.plot(degs,errors[-1])
    plt.xlabel('degree of polynomial')
    plt.ylabel('error')
    plt.subplot(122)
    plt.loglog(ns, errors[:,-1])
    plt.xlabel('log(number of samples)')
    plt.ylabel('log of error')
    plt.tight_layout()
    plt.show()
    plt.savefig('4_g.png', format='png')
    plt.close()
