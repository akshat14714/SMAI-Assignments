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
    X = data[:, :11]
    Y = data[:, 11:]
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


def loss(h, y):
    return np.mean(-y * np.log(h) - (1-y) * np.log(1 - h))

def predict_probs(X, theta):
    return sigmoid(np.dot(X, theta))

def gradientdescent(X, y, theta, alpha, num_iter):
    cost = []
    for _ in range(num_iter):
        h = predict_probs(X, theta)
        gradient = np.dot(X.T, (h-y))/len(y)
        theta -= alpha * gradient
        cost.append(loss(h, y))
    return theta, cost

if __name__ == '__main__':
    filename = sys.argv[1]
    train_data, test_data = load_file(filename)

    X_train, y_train = classifyXY(train_data)
    X_test, y_test = classifyXY(test_data)

    X_train = normalizeData(X_train)
    X_test = normalizeData(X_test)

    ones = np.ones((X_test.shape[0],1))
    X_test = np.concatenate((ones, X_test), axis=1)

    num_iters, alpha = 100000, 0.001

    no_of_features, p, list_feature = {}, [], []

    theta = np.zeros([12,1])
    theta_composite = [theta] * 21

    for i in range(len(train_data)):
        if train_data[i][11] not in no_of_features.keys():
            no_of_features[train_data[i][11]] = 1
        else:
            no_of_features[train_data[i][11]] += 1

    for i in range(len(test_data)):
        if test_data[i][11] not in no_of_features.keys():
            no_of_features[test_data[i][11]] = 1
        else:
            no_of_features[test_data[i][11]] += 1

    for i in no_of_features:
        list_feature.append(i)

    list_feature = np.array(list_feature, dtype=float)

    l, m = 0, 0
    for i in range(len(no_of_features)):
        j = i+1
        while j < len(no_of_features):
            new_inp_train = np.array(p)
            new_out_train = np.array(y_train)
            new_inp_train = list(new_inp_train)
            # new_out_train = list(new_out_train)
            l = 0
            for k in range(len(train_data)):
                if y_train[k][-1] == list_feature[i]:
                    new_inp_train.append(train_data[k])
                    new_out_train[l][0] = 1
                    l += 1
                elif y_train[k][-1] == list_feature[j]:
                    new_inp_train.append(train_data[k])
                    new_out_train[l][0] = 0
                    l += 1

            new_out_train = np.array(new_out_train[:l])

            new_inp_train = np.array(new_inp_train, dtype = np.float)
            new_out_train = np.array(new_out_train, dtype = np.float)
            
            theta = np.zeros([12, 1])
            theta_composite[m], cost = gradientdescent(new_inp_train, new_out_train, theta, alpha, num_iter)
            j += 1
            m += 1

            if j > 3:
                break
        if j > 3:
            break

    l, m, j, i = 0, 0, 0, 0

    pred_count_class = {}
    for i in range(len(no_of_features)):
        # print(list_feature[i])
        j = i+1
        while j < len(no_of_features):
            new_inp_test = np.array(p)
            new_out_test = np.array(y_train)
            new_inp_test = list(new_inp_test)
            l = 0

            k = 0
            for k in range(len(test_data)):
                if y_test[k][-1] == list_feature[i]:
                    new_inp_test.append(test_data[k])
                    new_out_test[l][0] = 1
                    l += 1
                elif y_test[k][-1] == list_feature[j]:
                    new_inp_test.append(test_data[k])
                    new_out_test[l][0] = 0
                    l += 1
            
            new_out_test = np.array(new_out_test[:l])

            new_inp_test = np.array(new_inp_test, dtype = np.float)
            new_out_test = np.array(new_out_test, dtype = np.float)
            k = 0
            pred = predict_probs(new_inp_test, theta_composite[m])
            # pred = sigmoid(new_inp_test, theta_composite[m])
            for k in range(pred.shape[0]):
                if pred[k][0] < 0.5:
                    pred[k][0] = 0
                else:
                    pred[k][0] = 1
            count = 0
            accuracy = 0
            for k in range(pred.shape[0]):
                if pred[k][0] == new_out_test[k][0]:
                    count += 1
        accuracy = count / float(pred.shape[0])
        print("Accuracy in round ", m, "is:", accuracy)

        m += 1