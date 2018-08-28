import matplotlib
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

def generate_data(n):
    """
    This function generates data of size n.
    """
    #TODO implement this
    X = np.random.normal(loc=0, scale=5, size=(2,n))
    Z = np.random.normal(loc=0, scale=1, size=n)
    y = X[0] + X[1]+ Z
    return (X.T,y.T)

def tikhonov_regression(X,Y,Sigma):
    """
    This function computes w based on the formula of tikhonov_regression.
    """
    #TODO implement this
    w = np.linalg.inv((X.T @ X + np.linalg.inv(Sigma))) @ X.T @ Y
    return w

def compute_mean_var(X,y,Sigma):
    """
    This function computes the mean and variance of the posterior
    """
    #TODO implement this
    mux, muy = tikhonov_regression(X, y, Sigma)
    cov = (X.T @ X + np.linalg.inv(Sigma))
    sigmax = cov[0][0]
    sigmay = cov[1][1]
    sigmaxy = cov[0][1]
    return mux,muy,sigmax,sigmay,sigmaxy

Sigmas = [np.array([[1,0],[0,1]]), np.array([[1,0.25],[0.25,1]]),
          np.array([[1,0.9],[0.9,1]]), np.array([[1,-0.25],[-0.25,1]]),
          np.array([[1,-0.9],[-0.9,1]]), np.array([[0.1,0],[0,0.1]])]
names = [str(i) for i in range(1,6+1)]

for num_data in [5,50,500]:
    X,Y = generate_data(num_data)
    for i,Sigma in enumerate(Sigmas):

        mux,muy,sigmax,sigmay,sigmaxy = compute_mean_var(X, Y, Sigma)# TODO compute the mean and covariance of posterior.

        x = np.arange(0.5, 1.5, 0.01)
        y = np.arange(0.5, 1.5, 0.01)
        X_grid, Y_grid = np.meshgrid(x, y)

        Z = np.exp(-((X_grid-mux)**2*sigmax + (Y_grid-muy)**2*sigmay + 2*(X_grid-mux)*(Y_grid-muy)*sigmaxy)/2)
        # TODO Generate the function values of bivariate normal.

        # plot
        plt.figure(figsize=(10,10))
        CS = plt.contour(X_grid, Y_grid, Z/Z.max(),
                         levels = np.concatenate([np.arange(0,0.05,0.01),np.arange(0.05,1,0.05)]))
        plt.clabel(CS, inline=1, fontsize=10)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Sigma'+ names[i] + ' with num_data = {}'.format(num_data))
        plt.savefig('Sigma'+ names[i] + '_num_data_{}.png'.format(num_data))
        plt.close()
