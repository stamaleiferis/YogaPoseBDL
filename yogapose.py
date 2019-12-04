import sys, os, operator, time, math, datetime, json
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
import src.bnn.hmc as hmcnn
import src.bnn.bnn as bnn


# Data functions
def command_line():
    J = int(sys.argv[1])
    K = int(sys.argv[2])
    L = int(sys.argv[3])
    e = float(sys.argv[4])
    return J, K, L, e

def load_data():
    mat1 = np.loadtxt('mat1.csv', delimiter=',')
    mat2 = np.loadtxt('mat2.csv', delimiter=',')
    mat3 = np.loadtxt('mat3.csv', delimiter=',')
    mat4 = np.loadtxt('mat4.csv', delimiter=',')
    return [mat1, mat2, mat3, mat4]

def split_data(mat):
    X = mat[:, :-1]
    Y = mat[:, -1]
    Y = onehot(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=.125)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def onehot(Y):
    n = int(np.max(Y)) + 1
    ret = []
    for i in range(len(Y)):
        vec = np.zeros(n)
        vec[int(Y[i])] = 1
        ret.append(vec)
    return ret

# Network functions

J, K, L, e = command_line()

mats = load_data()

priors = [tfp.distributions.Normal(0, 1.0), tfp.distributions.Laplace(0, 1.0)]

for prior in priors:
    for mat in mats:
        X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(mat)
        input = [len(X_train[0])]
        hidden = [K] * J
        output = [len(Y_train[0])]
        architecture = input + hidden + output
        init_state = bnn.get_random_initial_state(prior, prior, architecture, overdisp=1.0)
        Y_pred,trace, final_kernel_results = hmcnn.hmc_predict(prior, prior, init_state, X_train, Y_train, X_val, Y_val)
        
