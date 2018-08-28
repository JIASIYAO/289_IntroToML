import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import sklearn.linear_model
from sklearn.model_selection import train_test_split


######## PROJECTION FUNCTIONS ##########

## Random Projections ##
def random_matrix(d, k):
    '''
    d = original dimension
    k = projected dimension
    '''
    return 1./np.sqrt(k)*np.random.normal(0, 1, (d, k))

def random_proj(X, k):
    _, d= X.shape
    return X.dot(random_matrix(d, k))

## PCA and projections ##
def my_pca(X, k):
    '''
    compute PCA components
    X = data matrix (each row as a sample)
    k = #principal components
    '''
    n, d = X.shape
    assert(d>=k)
    _, _, Vh = np.linalg.svd(X)    
    V = Vh.T
    return V[:, :k]

def pca_proj(X, k):
    '''
    compute projection of matrix X
    along its first k principal components
    '''
    P = my_pca(X, k)
    # P = P.dot(P.T)
    return X.dot(P)


######### LINEAR MODEL FITTING ############

def rand_proj_accuracy_split(X, y, k):
    '''
    Fitting a k dimensional feature set obtained
    from random projection of X, versus y
    for binary classification for y in {-1, 1}
    '''
    
    # test train split
    _, d = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # random projection
    J = np.random.normal(0., 1., (d, k))
    rand_proj_X = X_train.dot(J)
    
    # fit a linear model
    line = sklearn.linear_model.LinearRegression(fit_intercept=False)
    line.fit(rand_proj_X, y_train)
    
    # predict y
    y_pred=line.predict(X_test.dot(J))
    
    # return the test error
    return 1-np.mean(np.sign(y_pred)!= y_test)

def pca_proj_accuracy(X, y, k):
    '''
    Fitting a k dimensional feature set obtained
    from PCA projection of X, versus y
    for binary classification for y in {-1, 1}
    '''

    # test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # pca projection
    P = my_pca(X_train, k)
    P = P.dot(P.T)
    pca_proj_X = X_train.dot(P)
                
    # fit a linear model
    line = sklearn.linear_model.LinearRegression(fit_intercept=False)
    line.fit(pca_proj_X, y_train)
    
     # predict y
    y_pred=line.predict(X_test.dot(P))
    

    # return the test error
    return 1-np.mean(np.sign(y_pred)!= y_test)


######## LOADING THE DATASETS #########

# to load the data:
data = np.load('data1.npz')
X = data['X']
y = data['y']
n, d = X.shape


n_trials = 10  # to average for accuracies over random projections

######### YOUR CODE GOES HERE ##########

# Using PCA and Random Projection for:
# Visualizing the datasets 
#X_pca = pca_proj(X, 2)
#plt.clf()
#plt.plot(X_pca.T[0], X_pca.T[1], 'bo', mec='none', label='pca')
#plt.legend()
#plt.title('data3')
#plt.savefig('pca_3.png', format='png')
#
#X_rand = random_proj(X, 2)
#plt.clf()
#plt.plot(X_rand.T[0], X_rand.T[1], 'ro', mec='none', label='random')
#plt.legend()
#plt.title('data3')
#plt.savefig('rand_3.png', format='png')


# Computing the accuracies over different datasets.
pca_acc = np.zeros((d, n_trials))
rand_acc = np.zeros((d, n_trials))
for k in range(d):
    for i in range(n_trials):
        pca_acc[k][i] = pca_proj_accuracy(X, y, k+1)
        rand_acc[k][i] = rand_proj_accuracy_split(X, y, k+1)
plt.clf()
# Don't forget to average the accuracy for multiple
# random projections to get a smooth curve.
plt.plot(np.arange(d)+1, np.mean(pca_acc, axis=1), 'bo-', mec='none', label='pca')
plt.plot(np.arange(d)+1, np.mean(rand_acc, axis=1), 'ro-', mec='none', label='random')
plt.legend(loc='best')
plt.title('data1')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.savefig('acc_k_1.png', format='png')


# And computing the SVD of the feature matrix
data = np.load('data1.npz')
X = data['X']
u,s1,v = np.linalg.svd(X)

data = np.load('data2.npz')
X = data['X']
u,s2,v = np.linalg.svd(X)

data = np.load('data3.npz')
X = data['X']
u,s3,v = np.linalg.svd(X)

plt.clf()
plt.plot(s1, 'ro-', label='data1')
plt.plot(s2, 'b^-',label='data2')
plt.plot(s3, 'gs-', label='data3')
plt.legend()
plt.ylabel('sigular values')
plt.savefig('sv.png', format='png')






