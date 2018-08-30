import random
import time


import glob
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys
from numpy.linalg import inv
from numpy.linalg import det
from sklearn.svm import LinearSVC
from projection import Project2D, Projections

import pdb

class LDA_Model():

    def __init__(self,class_labels):
        ###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
        self.reg_cov = 0.001
        self.NUM_CLASSES = len(class_labels)



    def train_model(self,X,Y): 
        ''''
        FILL IN CODE TO TRAIN MODEL
        MAKE SURE TO ADD HYPERPARAMTER TO MODEL 
        '''
        self.u = []
        for i in range(self.NUM_CLASSES):
            idx = np.where(np.array(Y)==i)[0]
            self.u.append(np.mean(np.array(X)[idx], axis=0))
        self.sig = np.cov(np.array(X).T)
        self.sig += np.identity(len(self.sig))*self.reg_cov
            

    def eval(self,x):
        ''''
        Fill in code to evaluate model and return a prediction
        Prediction should be an integer specifying a class
        '''
        y = []
        for i in range(self.NUM_CLASSES): 
            y.append((x-self.u[i]) @ inv(self.sig) @ (x-self.u[i]).T)
        return np.argmin(y)

    
