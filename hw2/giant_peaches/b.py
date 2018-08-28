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

    n = 20  # max degree
    err = np.zeros(n - 1)

    # fill in err
    # YOUR CODE HERE
    for D in np.arange(1,n):
        X = np.vstack(([x_train**i for i in range(D+1)])).T
        w = lstsq(X, y_train)
        y = X @ w
        err[D-1] = np.sum((y-y_train)**2)/n 

    plt.plot(np.arange(1, n), err)
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Training Error')
    plt.show()


if __name__ == "__main__":
    main()
