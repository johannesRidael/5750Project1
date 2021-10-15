# -*- coding: utf-8 -*-
"""
Data generation for logistic regression
"""
import math

import numpy as np
from numpy.random import multivariate_normal
from scipy.linalg.special_matrices import toeplitz
from scipy import optimize
import matplotlib.pyplot as plt

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
A=np.array(A)
b = np.array(b)
x0=[]
for i in range(50):
    x0.append(1)

def F1(x):
    #print("L1 Reg")
    ret = 0
    for i in range(len(b)):
        hold = -b[i] * A[i].T @ x
        term = 1 + np.exp(hold)
        ret += math.log(term, 2)
    ret = ret / (len(b) * 2)
    return (ret/(2*len(x))) + ld * np.linalg.norm(x, ord=1)

def F1_prime(x):
    """
        :x: a vector d=50
        :return: a vector with the gradient for each x
        currently implemented for log_2
        """
    ret = list()
    for i in range(50):
        curr = 0
        for j in range(50):
            u = -b[j] * (x @ A[j])
            t = 0
            for k in range(50):
                t += A[j][k]
            a = (math.log(2, np.e) * (1 + np.exp(u)))
            curr += u * -b[j] * np.exp(u) * (t + A[j][i] * x[i] - A[j][i]) / (math.log(2, np.e) * (1 + np.exp(u)))
        # now we add the gradient of the regularization term
        curr = curr / (2 * len(x))
        curr += 1
        ret.append(curr)
    #print("L1 Dev")
    return ret


def F2(x):
    print("L2 Reg")
    ret = 0
    for i in range(len(b)):
        term = 1 + math.exp(-b[i] * A[i].T @ x)
        ret += math.log(term, 2)
    ret = ret / (len(b) * 2)
    return ret + ld * np.linalg.norm(x)

def F2_prime(x):
    """
        :x: a vector d=50
        :return: a vector with the gradient for each x
        currently implemented for log_2
        """
    ret = []
    for i in range(50):
        curr = None
        for j in range(50):
            u = -b[j] * (x @ A[j])
            t = 0
            for k in range(50):
                t += A[j][k]
            curr += u * -b[j] * np.exp(u) * (t + A[j][i] * x[i] - A[j][i]) / (np.log(2, np.e) * (1 + np.exp(u)))
        # now we add the gradient of the regularization term
        u = 0
        for k in range(50):
            u += x[k] * x[k]
        curr += x[i] / np.sqrt(u)
        ret.append(curr)
    print("L2 Dev")
    return ret


#from https://scipy-lectures.org/advanced/mathematical_optimization/auto_examples/plot_gradient_descent.html
#TODO--Convert to arbitrary dimension,
def gradient_descent(x0, f, f_prime, hessian=None, adaptative=False):
    #x_i, y_i = x0
    x_i = x0
    all_x_i = []
    all_f_i = []
    for i in range(1, 10000):
        all_x_i.append(x_i)
        all_f_i.append(f(x_i))
        dx_i = f_prime(x_i)
        if adaptative:
            # Compute a step size using a line_search to satisfy the Wolf
            # conditions
            #step = optimize.line_search(f, f_prime,
            #                    np.r_[x_i, y_i], -np.r_[dx_i, dy_i],
            #                    np.r_[dx_i, dy_i], c2=.05)
            step = 1#step[0]
            if step is None:
                step = 0
        else:
            step = 1
        #x_i += - step*dx_i
        for j in range(len(dx_i)):
            x_i[j] -= step*dx_i[j]
        if math.fabs(f(x_i) - all_f_i[i-1]) < 1e-16:
            break
    return all_x_i, all_f_i


#run the 4(8) methods, and plot
ld = .001
exes, wise = gradient_descent(x0, F1, F1_prime)
print(wise)
print(exes)


#run for Subgradient, proximal gradient, accelrated with momentum, Nesterov's
for ld in [0.005, 0.01, 0.05, 0.1]:
    print("lambda: ", ld)
    for it in [500, 5000, 7500, 9000, 9500, 9750, 9990, 9995, 10000]:
        print("iterations: ", it)
