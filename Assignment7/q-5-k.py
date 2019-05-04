import numpy as np
import pandas as pd
import csv
import itertools as its
import math
import sys
import random
import operator
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def load_file(filename):
    train_data, test_data = [], []
    dataset = []
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = np.array(list(lines))

    dataset = np.array(dataset[1:, :], dtype=float)

    np.random.shuffle(dataset)

    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size, :]
    train_data = np.array(train_data)
    test_data = dataset[train_size:, :]
    test_data = np.array(test_data)

    return train_data, test_data

def classifyXY(data):
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y

def normalizeData(data):
    mean = np.ones(data.shape[1])
    std = np.ones(data.shape[1])

    for i in range(0, data.shape[1]):
        mean[i] = np.mean(data.transpose()[i])
        std[i] = np.std(data.transpose()[i])
        for j in range(0, data.shape[0]):
            data[j][i] = (data[j][i] - mean[i])/std[i]

    return data

def hypothesis(theta, X, n):
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n+1)
    h = np.dot(theta, X.T)
    h = h.reshape(X.shape[0])
    return h

def BGD(theta, alpha, num_iters, h, X, y, n, lamb):
    cost = np.ones(num_iters)
    for i in range(0,num_iters):
        theta[0] = theta[0] - (alpha/(2*X.shape[0])) * sum(h - y)
        theta = theta - (alpha / (2 * X.shape[0])) * np.sum(np.dot((h - y), X) + 2 *lamb * theta)
        h = hypothesis(theta, X, n)
        cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y) + lamb * np.sum(theta.T * theta))
    theta = theta.reshape(1,n+1)
    return theta, cost

def linear_regression(X, y, alpha, num_iters, lamb):
    n = X.shape[1]
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis=1)
    theta = np.zeros(n+1)
    # lamb = 0.01
    h = hypothesis(theta, X, n)
    theta, cost = BGD(theta,alpha,num_iters,h,X,y,n,lamb)
    return theta, cost

def classifier():
    filename = sys.argv[1]
    train_data, test_data = load_file(filename)

    X_train, y_train = classifyXY(train_data)
    X_test, y_test = classifyXY(test_data)

    X_train = normalizeData(X_train)
    X_test = normalizeData(X_test)

    num_iters, alpha = 20000, 0.001
    fin_theta = None
    min_mse = 100000

    lamb = 0.005

    kf = KFold(n_splits=5)
    kf.get_n_splits(X_train)
    print(kf)

    i = 0

    for train_ind, test_ind in kf.split(X_train):
        X_train_split, X_val_split = X_train[train_ind], X_train[test_ind]
        y_train_split, y_val_split = y_train[train_ind], y_train[test_ind]

        ones = np.ones((X_val_split.shape[0],1))
        X_val_split = np.concatenate((ones, X_val_split), axis=1)

        theta, cost = linear_regression(X_train_split, y_train_split, alpha, num_iters, lamb)
        predictions = hypothesis(theta, X_val_split, X_val_split.shape[1]-1)
        finalCost_mse = (1/X_val_split.shape[0]) * 0.5 * sum(np.square(predictions - y_val_split))

        print("Iteration Number =", i)
        print("MSE Error =", finalCost_mse)

        if finalCost_mse < min_mse:
            fin_theta = theta
            min_mse = finalCost_mse

        i += 1

    ones = np.ones((X_test.shape[0],1))
    X_test = np.concatenate((ones, X_test), axis=1)
    
    final_pred = hypothesis(fin_theta, X_test, X_test.shape[1]-1)

    final_mse = (1/X_test.shape[0]) * 0.5 * sum(np.square(final_pred - y_test))

    print("Final MSE Error on Test Data =",final_mse)

classifier()