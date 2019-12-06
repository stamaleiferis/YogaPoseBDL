import sys, os, operator, time, math, datetime, json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_data():
    mat1 = np.loadtxt('mat1.csv', delimiter=',',dtype='float32')
    #mat2 = np.loadtxt('mat2.csv', delimiter=',',dtype='float32')
    #mat3 = np.loadtxt('mat3.csv', delimiter=',')
    #mat4 = np.loadtxt('mat4.csv', delimiter=',')
    #return mat1, mat2, mat3, mat4
    return mat1

def split_data(mat):
    X = mat[:, :-1]
    Y = mat[:, -1]
    #Y = onehot(Y)
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

'''mat1, mat2, mat3, mat4 = load_data()
X1_train, X1_val, X1_test, Y1_train, Y1_val, Y1_test = split_data(mat1)
X2_train, X2_val, X2_test, Y2_train, Y2_val, Y2_test = split_data(mat2)
X3_train, X3_val, X3_test, Y3_train, Y3_val, Y3_test = split_data(mat3)
X4_train, X4_val, X4_test, Y4_train, Y4_val, Y4_test = split_data(mat4)'''

def get_data():
    return split_data(load_data())
     

