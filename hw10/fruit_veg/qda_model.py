
import random
import time


import numpy as np
import numpy.linalg as LA


from numpy.linalg import inv
from numpy.linalg import det

from projection import Project2D, Projections


class QDA_Model(): 
    def __init__(self,class_labels):
        ###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
        self.reg_cov = 0.01
        self.NUM_CLASSES = len(class_labels)



    def train_model(self,X,Y): 
        ''''
        FILL IN CODE TO TRAIN MODEL
        MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

        '''
        self.u = []
        self.sig = []
        for i in range(self.NUM_CLASSES):
            idx = np.where(np.array(Y)==i)[0]
            self.u.append(np.mean(np.array(X)[idx], axis=0))
            sig = np.cov(np.array(X)[idx].T)
            sig += np.identity(len(sig))*self.reg_cov
            self.sig.append(sig)

    def eval(self,x):
        ''''
        Fill in code to evaluate model and return a prediction
        Prediction should be an integer specifying a class
        '''
        y = []
        for i in range(self.NUM_CLASSES): 
            y.append((x-self.u[i]) @ inv(self.sig[i]) @ (x-self.u[i]).T + np.log(det(self.sig[i])))
        return np.argmin(y)


        
