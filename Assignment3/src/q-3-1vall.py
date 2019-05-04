import numpy as np
import pandas as pd
import sys
import csv
import itertools as its
import math
import random
import operator
import matplotlib.pyplot as plt
from matplotlib import interactive
from collections import Counter

def load_file(filename):
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')

        dataset = list(readCSV)
        total_len = len(dataset)

    dataset = np.array(dataset[1:])
    np.random.shuffle(dataset)

    train_len = int(0.8 * total_len)
    train_data = np.array(dataset[:train_len], dtype=float)
    test_data =  np.array(dataset[train_len:], dtype=float)

    # print(train_data[0])

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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def hypothesis(theta, X, n):
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n+1)
    h = np.dot(theta, X.T)
    h = sigmoid(h)
    h = h.reshape(X.shape[0])
    return h

def loss(h, y):
    return np.mean(-y * np.log(h) - (1-y) * np.log(1 - h))

def predict_probs(X, theta):
    return sigmoid(np.dot(theta, X.T))

def predictProb(X, theta, threshold=0.5):
    return predict_probs(theta, X) >= threshold

def gradient_descent(theta, alpha, num_iters, h, X, y, n):
    cost = np.ones(num_iters)
    for i in range(num_iters):
        gradient = np.dot((h-y), X) / y.shape[0]
        theta = theta - alpha * gradient
        h = hypothesis(theta, X, n)
        # print(h)
        # break
        p = predictProb(X, theta, 0.5)
        p[p == True] = 1
        p[p == False] = 0
        cost[i] = loss(h, y)
    theta = theta.reshape(1, n+1)
    # print(theta)
    return theta, cost

def logistic_regression(X, y, alpha, num_iters):
    n = X.shape[1]
    # print(n)
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis=1)
    theta = np.zeros(n+1)
    h = hypothesis(theta, X, n)
    theta, cost = gradient_descent(theta,alpha,num_iters,h,X,y,n)
    return theta, cost

def performance(tn, tp, fn, fp):
    # recall = float(float(tp) / float(tp + fn))
    # precision = float(float(tp) / float(tp + fp))
    accuracy = float(float(tp+tn) / float(tp+tn+fp+fn))
    # f1 = float(float(2*tp) + float(2*tp + fp + fn))

    return accuracy

def calculate_tntp(predictions, y_test):
    tn, tp, fn, fp = 0, 0, 0, 0

    for i in range(len(y_test)):
        if predictions[i]==0 and y_test[i]==0:
            tn += 1
        elif predictions[i]==0 and y_test[i]==1:
            fp += 1
        elif predictions[i]==1 and y_test[i]==0:
            fn += 1
        elif predictions[i]==1 and y_test[i]==1:
            tp += 1

    return tn, tp, fn, fp

def one_v_all(X_train, y_train, X_test, y_test, num_iters, alpha):
    for i in range(3,10):
        y_train_round = np.array(y_train)

        y_train_round[y_train_round != float(i)] = 0
        y_train_round[y_train_round == float(i)] = 1

        theta, cost = logistic_regression(X_train, y_train_round, alpha, num_iters)

        orig_pred = hypothesis(theta, X_test, X_test.shape[1]-1)

        # print(Counter(orig_pred))

        y_test_round = np.array(y_test)
        orig_pred[orig_pred >= 0.5] = 1
        orig_pred[orig_pred < 0.5] = 0
        y_test_round[y_test_round == float(i)] = 1
        y_test_round[y_test_round != float(i)] = 0

        tn, tp, fn, fp = calculate_tntp(orig_pred, y_test_round)
        accuracy = performance(tn, tp, fn, fp)

        print("For output label " + str(i) + ", accuracy is " + str(accuracy))

def classifier():
    filename = sys.argv[1]
    train_data, test_data = load_file(filename)

    X_train, y_train = classifyXY(train_data)
    X_test, y_test = classifyXY(test_data)

    X_train = normalizeData(X_train)
    X_test = normalizeData(X_test)

    ones = np.ones((X_test.shape[0],1))
    X_test = np.concatenate((ones, X_test), axis=1)

    num_iters, alpha = 20000, 0.005

    print(Counter(y_train))

    print('One vs All Classifier')
    one_v_all(X_train, y_train, X_test, y_test, num_iters, alpha)

if __name__ == "__main__":
    classifier()