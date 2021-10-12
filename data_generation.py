# -*- coding: utf-8 -*-
"""
Data generation for logistic regression
"""
import math

import numpy as np
from numpy.random import multivariate_normal
from scipy.linalg.special_matrices import toeplitz

n_features = 50  # The dimension of the feature is set to 50
n_samples = 1000 # Generate 1000 training data

idx = np.arange(n_features)
coefs = ((-1) ** idx) * np.exp(-idx/10.)
coefs[20:] = 0.


def sigmoid(t):
    """Sigmoid function"""
    return 1. / (1. + np.exp(-t))

def sim_logistic_regression(coefs, n_samples=1000, corr=0.5):
    """"
    Simulation of a logistic regression model
    
    Parameters
    coefs: `numpy.array', shape(n_features,), coefficients of the model
    n_samples: `int', number of samples to simulate
    corr: `float', correlation of the features
    
    Returns
    A: `numpy.ndarray', shape(n_samples, n_features)
       Simulated features matrix. It samples of a centered Gaussian vector with covariance 
       given bu the Toeplitz matrix
    
    b: `numpy.array', shape(n_samples,), Simulated labels
    """
    cov = toeplitz(corr ** np.arange(0, n_features))
    A = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    p = sigmoid(A.dot(coefs))
    b = np.random.binomial(1, p, size=n_samples)
    b = 2 * b - 1
    return A, b


A, b = sim_logistic_regression(coefs)

#run the 4 methods, and plot


#run for Subgradient, proximal gradient, accelrated with momentum, Nesterov's
for ld in [0.005, 0.01, 0.05, 0.1]:
    print("lambda: ", ld)
    for it in [500, 5000, 7500, 9000, 9500, 9750, 9990, 9995, 10000]:
        print("iterations: ", it)


def F1(A, b, x, ld):
    print("L1 Reg")
    ret = 0
    for i in range(len(b)):
        term = 1 + -b[i] * A[i].T @ x
        ret += math.log(term, 2)
    ret = ret / (len(b) * 2)
    return ret + ld * np.linalg.norm(x, ord=1)

def F2(A, b, x, ld):
    print("L2 Reg")
    ret = 0
    for i in range(len(b)):
        term = 1 + -b[i] * A[i].T @ x
        ret += math.log(term, 2)
    ret = ret / (len(b) * 2)
    return ret + ld * np.linalg.norm(x)