import sys, os, operator, time, math, datetime, json
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm

import src.bnn.hmc as hmcnn
import src.bnn.bnn as bnn

J = 2
K = 30
L = 1000
e = .00001

Js = [1, 2, 3, 4, 5]
Ks = [10, 20, 30, 40, 50]
Ls = [10, 50, 100, 500, 1000]
es = [.00001, .0001, .001, .01, .1]

# Data Functions
def load_data():
    mat1 = np.loadtxt('mat1.csv', delimiter=',', dtype='float32')
    mat2 = np.loadtxt('mat2.csv', delimiter=',', dtype='float32')
    mat3 = np.loadtxt('mat3.csv', delimiter=',', dtype='float32')
    mat4 = np.loadtxt('mat4.csv', delimiter=',', dtype='float32')
    return [mat1, mat2, mat3, mat4]

def split_data(mat, train, val, test):
    X = mat[:, :-1]
    Y = mat[:, -1]
    X_train = X[train]
    X_val = X[val]
    X_test = X[test]
    Y_train = Y[train]
    Y_val = Y[val]
    Y_test = Y[test]
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

# Learn functions
def accuracy(Y_true, Y_pred):
    means, stds = eval(Y_true, Y_pred)
    return np.trace(means) / np.sum(means)

def eval(Y_true, Y_pred):
    cms = []
    for Y in Y_pred:
        cms.append(cm(Y_test, Y))
    cms = np.array(cms)
    means = np.mean(cms, axis=0)
    stds = np.std(cms, axis=0)
    return means, stds

def val_model(X_train, X_val, Y_train, Y_val, prior):
    start = time.time()
    best_params = [] 
    best_acc = 0
    for J in Js:
        for K in Ks:
            for L in Ls:
                for e in es:
                    architecture = [len(X_train[0])] + [K] * J + [int(np.max(Y_train)) + 1]
                    init = bnn.get_random_initial_state(prior, prior, architecture, overdisp=1.0)
                    Y_pred, trace, k, s = hmcnn.hmc_predict(prior, prior, init, X_train, Y_train, X_val, L, e)
                    Y_pred = tf.math.argmax(Y_pred, axis=2)
                    acc = accuracy(Y_val, Y_pred)
                    if (acc > best_acc):
                        best_params = [J, K, L, e]
                        best_acc = acc
    end = time.time()
    print("Validated in " + str(datetime.timedelta(seconds=int(end-start))))
    return best_params

def test_model(X_train, X_test, Y_train, Y_test, prior, J, K, L, e):
    start = time.time()
    architecture = [len(X_train[0])] + [K] * J + [int(np.max(Y_train)) + 1]
    init = bnn.get_random_initial_state(prior, prior, architecture, overdisp=1.0)
    Y_pred, trace, k, s = hmcnn.hmc_predict(prior, prior, init, X_train, Y_train, X_test, L, e)
    log_prob = trace[0].inner_results.accepted_results.target_log_prob.numpy()
    Y_pred = tf.math.argmax(Y_pred, axis=2)
    means, stds = eval(Y_test, Y_pred)
    acc = accuracy(Y_test, Y_pred)
    end = time.time()
    print("Tested in " + str(datetime.timedelta(seconds=int(end-start))))
    return means, stds, acc, log_prob

mats = load_data()
train, test = train_test_split(np.arange(len(mats[0])), test_size=.2)
train, val = train_test_split(train, test_size=.125)

priors = [tfp.distributions.Normal(0, 1.0), tfp.distributions.Laplace(0, 1.0)]

for mat in mats:
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(mat, train, val, test)
    for prior in priors:
        [J, K, L, e] = val_model(X_train, X_val, Y_train, Y_val, prior)
        means, stds, acc, log_prob = test_model(X_train, X_test, Y_train, Y_test, prior, J, K, L, e)
        print(means)
        print(stds)
        print(acc)  
