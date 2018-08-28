import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as spio

data = spio.loadmat('polynomial_regression_samples.mat', squeeze_me=True)
data_x = data['x']
data_y = data['y']
Kc = 4  # 4-fold cross validation
KD = 6  # max D = 6
LAMBDA = [0, 0.05, 0.1, 0.15, 0.2]

x1,x2,x3,x4,x5 = data_x.T
data_n = len(data_x)
step = int(data_n/Kc)

def multiPoly(D):
    order = []
    total_order = 0
    for i in np.arange(0,D+1):
        for j in np.arange(0,D+1):
            for k in np.arange(0,D+1):
                for m in np.arange(0,D+1):
                    for n in np.arange(0,D+1):
                        if i+j+k+m+n < D+1:
                            order.append([i, j, k, m, n])
    return np.array(order)

def fit(D, lambda_):
    # YOUR CODE TO COMPUTE THE AVERAGE ERROR PER SAMPLE
    orders = multiPoly(D)
    X = [x1**order[0] * x2**order[1] * x3**order[2] * x4**order[3] * x5**order[4] for order in orders]
    X = np.vstack(X).T
    train_err = []
    test_err = []
    for i in range(Kc):
        idx_test = np.arange(i*step,(i+1)*step)
        idx_train = np.setdiff1d(np.arange(0,data_n), idx_test) 
        X_train = X[idx_train]
        X_test = X[idx_test]
        y_train = data_y[idx_train]
        y_test = data_y[idx_test]
        w = np.linalg.inv(X_train.T @ X_train + lambda_  * np.identity(len(orders))) @ X_train.T @ y_train.T
        train_err.append(np.sum((X_train @ w - y_train)**2)/len(y_train))
        test_err.append(np.sum((X_test @ w - y_test)**2)/len(y_test))
    return(np.mean(train_err), np.mean(test_err)) 


def main():
    np.set_printoptions(precision=11)
    Etrain = np.zeros((KD+1, len(LAMBDA)))
    Evalid = np.zeros((KD+1, len(LAMBDA)))
    for D in range(KD+1):
        print(D)
        for i in range(len(LAMBDA)):
            Etrain[D, i], Evalid[D, i] = fit(D + 1, LAMBDA[i])

    print('Average train error:', Etrain, sep='\n')
    print('Average valid error:', Evalid, sep='\n')

    # for lambda = 0.1
    print('when lambda = 0.1, best D is when valid error is minimized')
    idx = np.where(np.array(LAMBDA)==0.1)[0][0]
    Evalid_3 = Evalid[:,idx]
    best_D = np.argmin(Evalid_3)
    print('best polynomial order is %d' %(best_D))
    #plt.plot(Evalid_3)
    #plt.xlabel('order')
    #plt.ylabel('Evalid')

    # YOUR CODE to find best D and i
    for i in range(len(LAMBDA)):
        print('For lambda = %f' %LAMBDA[i])
        for D in range(KD+1):
            print('D=%d, Etrain=%f, Evalid=%f' %(D, Etrain[D,i], Evalid[D,i]))

    best_D, best_i = np.where(Evalid==Evalid.min())
    print('best D = %d, lambda = %f' %(best_D, LAMBDA[best_i]))

if __name__ == "__main__":
    main()
