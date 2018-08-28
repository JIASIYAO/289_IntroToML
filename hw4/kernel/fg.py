import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as spio
from matplotlib import cm
import pdb

# choose the data you want to load
#data = np.load('circle.npz')
#pic = 'circle'
#data = np.load('heart.npz')
#pic = 'heart'
#data = np.load('asymmetric.npz')
#pic = 'asymmetric'


def multiPoly(D):
    order = []
    total_order = 0
    for i in np.arange(0,D+1):
        for j in np.arange(0,D+1):
            if i+j < D+1:
                order.append([i, j])
    return np.array(order)

def fit(X_train, y_train, X_valid, y_valid, D, LAMBDA):
    n_train = len(y_train)
    n_valid = len(y_valid)
                
    x1_train,x2_train = X_train.T
    x1_valid,x2_valid = X_valid.T

    # YOUR CODE TO COMPUTE THE AVERAGE ERROR PER SAMPLE
    # construct X matrix
    orders = multiPoly(D)
    X_D = [x1_train**order[0] * x2_train**order[1]  for order in orders]
    X_train_D = np.vstack(X_D).T
    X_D = [x1_valid**order[0] * x2_valid**order[1]  for order in orders]
    X_valid_D = np.vstack(X_D).T

    # use MLE to 
    w = np.linalg.inv(X_train_D.T @ X_train_D + LAMBDA  * np.identity(len(orders))) @ X_train_D.T @ y_train.T
    y_pre_train = np.sign(X_train_D @ w) 
    y_pre_valid = np.sign(X_valid_D @ w) 
    train_err = np.sum((y_pre_train - y_train)**2)/n_train
    valid_err = np.sum((y_pre_valid - y_valid)**2)/n_valid

    #heatmap(lambda x, y: x * x + y * y, X_valid, y_pre_valid, D, pic)
    return(train_err, valid_err) 

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
    plt.savefig('pre_non_kernal_%s_order%d.png' %(pic,p), format='png')



def main():
    np.set_printoptions(precision=11)
    data = np.load('asymmetric.npz')
    pic = 'asymmetric'
    SPLIT = 0.8
    X = data["x"]
    y = data["y"]
    X /= np.max(X)  # normalize the data
    n_train = int(X.shape[0] * SPLIT)
    
    #poly_orders = np.arange(1,16)
    #LAMBDA = 0.001
    LAMBDA = np.array([0.0001,0.001, 0.01])
    poly_orders = np.array([5,6])
    #n_data = (np.arange(0.1,1,0.05)*n_train).astype('int')
    n_data = np.arange(100,10000,1000)
    n_circle = 100


    for i,p in enumerate(poly_orders):
        #print('***  %s  *** '  %pic)
        #print('order     training_error      valid_error')
        plt.figure()
        for j,l in enumerate(LAMBDA):
            train_errors = np.zeros((len(n_data),n_circle))
            valid_errors = np.zeros((len(n_data),n_circle))

            for k,n in enumerate(n_data): 

                # sample training data
                n_train = int(X.shape[0] * SPLIT)
                X_train = X[:n_train:, :]
                y_train = y[:n_train]

                X_valid = X[n_train:, :]
                y_valid = y[n_train:]
    
                for h in range(n_circle):
                    idx = np.random.choice(np.arange(n_train), n)
                    train_errors[k][h], valid_errors[k][h] = fit(X_train[idx], y_train[idx], X_valid, y_valid, p, l)
                    #print('%5d   %e   %e' %(p, error[0], error[1]))

            mean_train_error = np.mean(train_errors, axis=1)
            mean_valid_error = np.mean(valid_errors, axis=1)
            plt.plot(n_data, mean_valid_error, label=r'$\lambda=%f$' %l)
        plt.legend()
        plt.xlabel('number of training data')
        plt.ylabel('validation squared loss')
        plt.title('order = %d' %p)
        plt.gca().set_xscale('log')
        plt.savefig('order_%d_valid_err_vs_num_train.png' %p, format='png')
        plt.close()

if __name__ == "__main__":
    main()
