#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio


# There is numpy.linalg.lstsq, whicn you should use outside of this classs
def lstsq(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)


def main():
    data = spio.loadmat('1D_poly.mat', squeeze_me=True)
    x_train = np.array(data['x_train'])
    y_train = np.array(data['y_train']).T
    y_fresh = np.array(data['y_fresh']).T

    n = 20  # max degree
    err_train = np.zeros(n - 1)
    err_fresh = np.zeros(n - 1)

    # fill in err_fresh and err_train
    for D in np.arange(1,n):
        X = np.vstack(([x_train**i for i in range(D+1)])).T
        w = lstsq(X, y_train)
        y = X @ w
        err_train[D-1] = np.sum((y-y_train)**2)/n 
        err_fresh[D-1] = np.sum((y-y_fresh)**2)/n 

   # YOUR CODE HERE

    plt.figure()
    plt.ylim([0, 6])
    plt.plot(np.arange(1, n), err_train, label='train')
    plt.plot(np.arange(1, n), err_fresh, label='fresh')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
