#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# choose the data you want to load
#data = np.load('circle.npz')
#pic = 'circle_0.15_training'
data = np.load('heart.npz')
pic = 'heart_noclip'
#data = np.load('asymmetric.npz')
#pic = 'asymmetric'

SPLIT = 0.15
X = data["x"]
y = data["y"]
X /= np.max(X)  # normalize the data

n_train = int(X.shape[0] * SPLIT)
X_train = X[:n_train:, :]
X_valid = X[n_train:, :]
y_train = y[:n_train]
y_valid = y[n_train:]
n_valid = len(y_valid)

LAMBDA = 0.001

orders = np.arange(1,24)

def lstsq(A, b, lambda_=0):
    return np.linalg.solve(A.T @ A + lambda_ * np.eye(A.shape[1]), A.T @ b)


def heatmap(f, X, y, p, pic, clip=5):
    # example: heatmap(lambda x, y: x * x + y * y)
    # clip: clip the function range to [-clip, clip] to generate a clean plot
    #   set it to zero to disable this function

    xx = yy = np.linspace(np.min(X), np.max(X), 72)
    x0, y0 = np.meshgrid(xx, yy)
    x0, y0 = x0.ravel(), y0.ravel()
    z0 = f(x0, y0)

    if clip:
        z0[z0 > clip] = clip
        z0[z0 < -clip] = -clip

    plt.clf()
    plt.hexbin(x0, y0, C=z0, gridsize=50, cmap=cm.jet, bins=None)
    plt.colorbar()
    cs = plt.contour(
        xx, yy, z0.reshape(xx.size, yy.size), [-2, -1, -0.5, 0, 0.5, 1, 2], cmap=cm.jet)
    plt.clabel(cs, inline=1, fontsize=10)

    pos = y[:] == +1.0
    neg = y[:] == -1.0
    plt.scatter(X[pos, 0], X[pos, 1], c='red', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], c='blue', marker='v')
    plt.show()
    #plt.savefig('pre_kernal_%s_order%d.png' %(pic,p), format='png')
    plt.savefig('pre_kernal_%s_sigma%.2f.png' %(pic,p), format='png')


def main():
    # loop through different orders
    

    ########   poly kernal #########
    #print('***  %s  *** '  %pic)
    #print('order     training_error      valid_error')
    #for p in orders:
    #    # derive kernal matrix
    #    K = (X_train @ X_train.T + 1)**p

    #    # calculate predicted position for training data
    #    temp =  np.linalg.inv((K+LAMBDA*np.identity(n_train))) @ y_train 
    #    y_pre_train = [((z @ X_train.T+1)**p) @ temp for z in X_train]
    #    y_pre_train = np.sign(y_pre_train) 

    #    # calculate predicted position for valid data
    #    y_pre_valid = [((z @ X_train.T+1)**p) @ temp for z in X_valid]
    #    y_pre_valid = np.sign(y_pre_valid) 

    #    train_err = np.sum((y_pre_train - y_train)**2)/n_train
    #    valid_err = np.sum((y_pre_valid - y_valid)**2)/n_valid

    #    print('%5d   %e   %e' %(p, train_err, valid_err))
    #
    #    # example usage of heatmap
    #    heatmap(lambda x, y: x * x + y * y, X_valid, y_pre_valid, p, pic)

    ######   RBF kernal #########
    print('sigma          valid_error')
    sigmas = [10,3,1,0.3,0.1,0.03]
    for sigma in sigmas:
        # derive kernal matrix
        K = np.zeros((n_train, n_train)) 
        for i in range(n_train):
            for j in range(n_train):
                v = X_train[i] - X_train[j]
                K[i][j] = np.exp(-(v @ v.T)/(2*sigma**2))
        # calculate predicted position for valid data
        temp = np.zeros((n_valid, n_train))
        for i in range(n_valid):
            for j in range(n_train):
                v = X_valid[i] - X_train[j]
                temp[i][j] = np.exp(-(v @ v.T)/(2*sigma**2))
    
    
        y_pre_valid =  temp @ np.linalg.inv((K+LAMBDA*np.identity(n_train))) @ y_train
        y_pre_valid = np.sign(y_pre_valid) 
    
        valid_err = np.sum((y_pre_valid - y_valid)**2)/n_valid
    
        print('%.2f    %e' %(sigma, valid_err))
    
        # example usage of heatmap
        heatmap(lambda x, y: x * x + y * y, X_valid, y_pre_valid, sigma, pic)



if __name__ == "__main__":
    main()

