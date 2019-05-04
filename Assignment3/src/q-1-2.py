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
import copy

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

    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    vectors = np.array([x[1] for x in eig_pairs])

    col_num =  bisect.bisect_left(cum_var_exp, 90)

    vectors = vectors[:, :col_num]

    new_mat = np.dot(dataset, vectors)
    return new_mat

def near_centroid(k_vals, entries):
    min_dist = sys.maxsize
    result = -1

    for j in range(len(k_vals)):
        temp = 0
        for i in range(len(k_vals[j])):
            temp += (k_vals[j][i] - entries[i])**2

        temp = math.sqrt(temp)

        if temp < min_dist:
            min_dist = temp
            result = j

    return result

def new_means(dataset):
    mean_vals = []
    dataset = np.array(dataset)
    
    for i in range(len(dataset[0])):
        mean_vals.append(np.mean(dataset[:, i]))

    return mean_vals

def KMeans_classifier():
    filename = sys.argv[1]
    dataset, train_data, test_data = load_file(filename)
    X, Y = classifyXY(dataset)
    X_reg = PCA(X)
    
    k = 5

    centroids, done_k = [], []

    while len(centroids)<k:
        temp = random.randint(1, len(X_reg))
        if temp not in done_k:
            centroids.append(X_reg[temp])
            done_k.append(temp)

    z = 0
    purity = []

    maxP = 0

    while True and z<10:
        corresp_k = []
        for row in X_reg:
            corresp_k.append(near_centroid(centroids, row))
        
        new_k = []

        # print(corresp_k)

        for i in range(k):
            arr = []
            for j in range(len(X_reg)):
                # print(corresp_k[j])
                if corresp_k[j] == i:
                    arr.append(X_reg[j])
            new_k.append(new_means(arr))

        flag = 0

        for i in range(k):
            for j in range(len(centroids[i])):
                if centroids[i][j] != new_k[i][j]:
                    flag = 1
                    break
            if flag:
                break

        centroids.clear()
        centroids = new_k

        purity.clear()
        for i in range(k):
            check_arr = []
            for j in range(len(corresp_k)):
                if corresp_k[j] == i:
                    check_arr.append(Y[j])

            unique_arr = set(check_arr)
            max_count = 0
            for label in unique_arr:
                max_count = max(max_count, check_arr.count(label))

            purity.append(max_count)

        print(np.sum(purity) / len(X_reg))

        if maxP < np.sum(purity)/len(X_reg):
            maxP = np.sum(purity)/len(X_reg)

        if flag==0:
            break
        
        print(z)
        z += 1

    print(maxP)

if __name__ == "__main__":
    KMeans_classifier()