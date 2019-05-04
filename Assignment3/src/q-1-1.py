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
from sklearn.preprocessing import StandardScaler
import bisect

def load_file(filename):
    train_data, test_data = [], []
    dataset = []
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = np.array(list(lines))

    dataset = np.array(dataset[1:, :])

    np.random.shuffle(dataset)

    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size, :]
    train_data = np.array(train_data)
    test_data = dataset[train_size:, :]
    test_data = np.array(test_data)

    return dataset, train_data, test_data

def classifyXY(data):
    X = np.array(data[:, :-1], dtype=np.float)
    Y = data[:, -1]
    return X, Y

def PCA(dataset):
    dataset = StandardScaler().fit_transform(dataset)
    dataset = np.array(dataset, dtype=float)
    M = np.mean(dataset.T, axis=1)
    C = dataset - M
    V = np.cov(C.T)
    
    eig_vals, eig_vecs = np.linalg.eig(V)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort()
    eig_pairs.reverse()

    # print(eig_pairs)

    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    plt.figure()
    # plt.ion()
    plt.plot(cum_var_exp)
    plt.xlabel('Number of features')
    plt.ylabel('Cumulative Variance %')
    plt.show()

    # print(cum_var_exp)

    # values = [x[0] for x in eig_pairs]
    vectors = np.array([x[1] for x in eig_pairs])

    # print(vectors)
    
    col_num =  bisect.bisect_left(cum_var_exp, 90)

    vectors = vectors[:, :col_num]

    new_mat = np.matmul(dataset, vectors)

    return new_mat


def classifier():
    filename = sys.argv[1]
    dataset, train_data, test_data = load_file(filename)
    # X_train, y_train = classifyXY(train_data)
    # X_test, y_test = classifyXY(test_data)
    X, Y = classifyXY(dataset)
    print(X.shape)

    X_reg = PCA(X)
    # X_reg = np.matmul(X, eig_vecs)
    # X_test_reg = np.matmul(X, eig_vecs)
    # print(X[:5])
    # print(X_reg[:5])

    # X_reg = np.asarray(X_reg)

    print(X_reg.shape)
    # print(X_test_reg.shape)

if __name__ == "__main__":
    classifier()

# classifier()