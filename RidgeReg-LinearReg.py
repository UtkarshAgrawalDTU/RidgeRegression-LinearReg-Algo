import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def costfunc(h,y,theta, lam):
    m = y.shape[0]
    cf = (np.dot((h-y).transpose(), h-y) + lam*np.dot(theta.transpose(), theta))/(2*m)
    return cf


def RidgeReg(X, y, alpha = 0.01, niter = 1000000):
    m = X.shape[0]
    X = np.insert(X, 0, values = 1, axis = 1)
    theta = np.zeros(shape = (X.shape[1],1))
    h = np.dot(X, theta)
    JHist = [0 for i in range(0, niter)]
    lam = 0.00001*m/alpha
    
    for i in range(0, niter):
        J = costfunc(h,y, theta[1:], lam)
        JHist[i] = J
        temp = theta[0,0]
        theta = theta*(0.999) - alpha*np.dot(X.transpose(),h-y)/m
        theta[0,0] = temp - alpha*np.dot(X.transpose(),h-y)[0,0]/m
        h = np.dot(X, theta)
    
    return (JHist, theta)

#X = np.array([1,2,2,3,3,4,4,5]).reshape(4,2)
#y = np.array([0,1,2,3]).reshape(4,1)
#(J, theta) = RidgeReg(X, y)