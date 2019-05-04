import numpy as np
import pandas as pd
import sys
import csv
import random
import math
from operator import itemgetter
import matplotlib.pyplot as plt

def load_file(filename):
    train_data, test_data = [], []
    dataset = []
    distinct = []
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = np.array(list(lines))

    dataset = np.array(dataset)
    
    for i in range(len(dataset)):
        dataset[i][:-1] = dataset[i][:-1].astype(float)
        if dataset[i][-1] not in distinct:
            distinct.append(dataset[i][-1])


    random.shuffle(dataset)
    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size, :]
    test_data = dataset[train_size:, :]

    return train_data, test_data, distinct

def calculate_distance(x, y):
    dists = []
    x1 = np.array(x[:-1])
    x1 = x1.astype(float)

    for elem in y:
        elem1 = np.array(elem[0:len(elem)-1])
        elem1 = elem1.astype(float)
        diff = x1 - elem1
        dists.append(tuple((elem[len(elem)-1], math.sqrt(np.dot(diff.T, diff)))))

    return sorted(dists, key=itemgetter(1))

def classifier():
    filename = sys.argv[1]
    train_data, test_data, distincts = load_file(filename)

    dict_vals = {'Iris-setosa':0, 'Iris-virginica':1, 'Iris-versicolor':2}

    k_vals = np.linspace(1, 25, 13)

    accuracies = []
 
    for k in k_vals:
        k = int(k)
        correct = 0

        tpfn_mat = np.zeros((3, 3))

        for elem in test_data:
            dict_val = {}

            for elements in distincts:
                dict_val[elements] = 0
            dists = calculate_distance(elem, train_data)
            for i in range(k):
                dict_val[dists[i][0]] += 1

            pred = max(dict_val.items(), key=itemgetter(1))[0]

            if pred==elem[-1]:
                correct += 1

            # print(pred)

            tpfn_mat[dict_vals[elem[-1]]][dict_vals[pred]] += 1
        
        accuracy = float(float(correct) / float(len(test_data)))

        tpfn_sums = [np.sum(tpfn_mat, axis=1), np.sum(tpfn_mat, axis=0)]

        for i in range(3):
            if i==0:
                print('Class label = Iris-setosa')
            elif i==1:
                print('Class Label = Iris-virginica')
            elif i==2:
                print('Class Label = Iris-versicolor')
            precision = float(float(tpfn_mat[i][i]) / float(np.sum(tpfn_sums[0][i])))
            recall = float(float(tpfn_mat[i][i]) / float(np.sum(tpfn_sums[1][i])))
            f1 = float(float(2 * precision * recall) / float(precision + recall))
            print("Precision =", precision)
            print("Recall =", recall)
            print("F1 Score =", f1)

        accuracies.append(accuracy)

        print("Accuracy =", accuracy)

    plt.figure()
    plt.scatter(k_vals, accuracies, color='b')
    plt.xlabel('K Values')
    plt.ylabel('Accuracy Values')
    plt.show()

classifier()