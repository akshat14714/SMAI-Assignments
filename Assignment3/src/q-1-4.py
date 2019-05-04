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
from sklearn.cluster import AgglomerativeClustering
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

    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    vectors = np.array([x[1] for x in eig_pairs])

    col_num =  bisect.bisect_left(cum_var_exp, 90)

    vectors = vectors[:, :col_num]

    new_mat = np.matmul(dataset, vectors)
    return new_mat

def Hierarchical_clustering():
    filename = sys.argv[1]
    dataset, train_data, test_data = load_file(filename)
    X, Y = classifyXY(dataset)

    X_reg = PCA(X)

    Hierrachical_model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single')
    predictions = Hierrachical_model.fit_predict(X_reg)

    purity = []
    unique_preds = set(predictions)

    for preds in unique_preds:
        check_arr = []
        for j in range(len(X_reg)):
            if predictions[j] == preds:
                check_arr.append(Y[j])

        unique_arr = set(check_arr)
        max_count = 0
        for label in unique_arr:
            max_count = max(max_count, check_arr.count(label))

        purity.append(max_count)

    print(np.sum(purity) / len(X_reg))

    

if __name__ == "__main__":
    Hierarchical_clustering()