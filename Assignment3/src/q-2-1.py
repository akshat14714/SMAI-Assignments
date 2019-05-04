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
    return (-y * np.log(h) - (1-y) * np.log(1 - h)).mean()

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
    recall = float(float(tp) / float(tp + fn))
    precision = float(float(tp) / float(tp + fp))
    accuracy = float(float(tp+tn) / float(tp+tn+fp+fn))
    f1 = float(float(2*tp) / float(2*tp + fp + fn))

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

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

    theta, cost = logistic_regression(X_train, y_train, alpha, num_iters)

    predictions = hypothesis(theta, X_test, X_test.shape[1]-1)

    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    y_test_round = np.array(y_test)

    y_test_round[y_test_round >= 0.5] = 1
    y_test_round[y_test_round < 0.5] = 0

    # errors = y_test - predictions

    # print(predictions)

    # print(predictions)
    tn, tp, fn, fp = 0, 0, 0, 0

    for i in range(len(y_test_round)):
        # if predictions[i] == y_test_round[i]:
        #     correct += 1

        if predictions[i]==0 and y_test_round[i]==0:
            tn += 1
        elif predictions[i]==0 and y_test_round[i]==1:
            fp += 1
        elif predictions[i]==1 and y_test_round[i]==0:
            fn += 1
        elif predictions[i]==1 and y_test_round[i]==1:
            tp += 1

    performance(tn, tp, fn, fp)

    cost = list(cost)
    # n_iterations = [x for x in range(1,300001)]
    plt.figure(1)
    plt.plot(np.arange(20000), cost)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.title('Error vs Iterations')
    # interactive(True)
    plt.show()

classifier()