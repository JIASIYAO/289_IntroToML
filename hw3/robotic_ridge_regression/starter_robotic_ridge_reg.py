import pickle
import matplotlib.pyplot as plt
import numpy as np

class HW3_Sol(object):

    def __init__(self):
        pass

    def load_data(self):
        self.x_train = pickle.load(open('x_train.p','rb'), encoding='latin1').astype(float)
        self.y_train = pickle.load(open('y_train.p','rb'), encoding='latin1').astype(float)
        self.x_test = pickle.load(open('x_test.p','rb'), encoding='latin1').astype(float)
        self.y_test = pickle.load(open('y_test.p','rb'), encoding='latin1').astype(float)


if __name__ == '__main__':

    hw3_sol = HW3_Sol()

    hw3_sol.load_data()

    # Your solution goes here

    ###### a ########
    # visulize data
    image0 = hw3_sol.x_train[0]
    control0 = hw3_sol.y_train[0]
    plt.imshow(image0)
    image10 = hw3_sol.x_train[10]
    control10 = hw3_sol.y_train[10]
    plt.imshow(image10)
    image20 = hw3_sol.x_train[20]
    control20 = hw3_sol.y_train[20]
    plt.imshow(image20)

    ###### b ########
    n = len(hw3_sol.x_train)
    X = hw3_sol.x_train.reshape(n, 2700)
    U = hw3_sol.y_train
    #pi = np.linalg.inv(X.T @ X ) @ X.T @ U

    ###### c ########
    lambs = np.array([0.1,  1, 10, 100, 1000])
    train_error = []
    pi_c = []
    for lamb in lambs:
        pi = np.linalg.inv(X.T @ X + lamb * np.identity(2700)) @ X.T @ U
        pi_c.append(pi)
        train_error.append(np.linalg.norm(X @ pi - U)**2/n)
    print('avg training error without standardization')
    print(train_error)

    ###### d ########
    Xs = X/255.*2 -1 
    train_error = []
    pi_d = []
    for lamb in lambs:
        pi = np.linalg.inv(Xs.T @ Xs + lamb * np.identity(2700)) @ Xs.T @ U
        pi_d.append(pi)
        train_error.append(np.linalg.norm(Xs @ pi - U)**2/n)
    print('avg training error with standardization')
    print(train_error)

    ###### e ########
    n_test = len(hw3_sol.x_test)
    X_test = hw3_sol.x_test.reshape(n_test, 2700)
    y_test = hw3_sol.y_test
    print('avg test error without standardization')
    test_error = []
    for i in range(len(lambs)):
        pi = pi_c[i]
        test_error.append(np.linalg.norm(X_test @ pi - y_test)**2/n_test)
    print(test_error)

    print('avg test error with standardization')
    test_error = []
    Xs_test = X_test/255.*2 -1 
    for i in range(len(lambs)):
        pi = pi_d[i]
        test_error.append(np.linalg.norm(Xs_test @ pi - y_test)**2/n_test)
    print(test_error)
   
    ###### e ########
    lamb = 100
    sv = np.linalg.cond(X.T @ X + lamb * np.identity(2700))
    print('condition number without standardization')
    print(sv)
    svs = np.linalg.cond(Xs.T @ Xs + lamb * np.identity(2700))
    print('condition number with standardization')
    print(svs)
