import numpy as np
import pandas as pd
import sys
import random
import math
import matplotlib.pyplot as plt
from operator import itemgetter

def load_file(filename):
    with open(filename) as file:
        data = [[str(elem) for elem in line.split()] for line in file]

    train_size = int(0.8 * len(data))
    random.shuffle(data)
    train_data = data[:train_size]
    val_data = data[train_size:]

    return train_data, val_data

def calculate_distance(x, y):
    dists = []
    x1 = np.array(x[1:-1])
    x1 = x1.astype(int)
    for elem in y:
        elem1 = np.array(elem[1:-1])
        elem1 = elem1.astype(int)
        diff = (x1 - elem1)
        dists.append(tuple((int(elem[0]), math.sqrt(np.dot(diff.T, diff)))))

    return sorted(dists, key=itemgetter(1))

def performance(tn, tp, fn, fp):
    recall = float(float(tp) / float(tp + fn))
    precision = float(float(tp) / float(tp + fp))
    accuracy = float(float(tp+tn) / float(tp+tn+fp+fn))
    f1 = float(float(2*tp) + float(2*tp + fp + fn))

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    return precision, recall, accuracy, f1

def classifier():
    filename = sys.argv[1]
    train_r1, val_r1 = load_file(filename)
    # train_r2, val_r2 = load_file('../RobotDataset/Robot2')

    k_vals = np.linspace(1, 25, 13)

    accuracies = []

    for k in k_vals:
        k = int(k)
        tn, tp, fn, fp = 0, 0, 0, 0

        for elem in val_r1:
            dict_val = {1: 0, 0: 0}

            dists = calculate_distance(elem, train_r1)

            # k = 5

            for i in range(k):
                dict_val[dists[i][0]] += 1

            pred = max(dict_val.items(), key=itemgetter(1))[0]

            # if pred==int(elem[0]):
            #     correct += 1

            if pred==0 and int(elem[0])==0:
                tn += 1
            elif pred==0 and int(elem[0])==1:
                fp += 1
            elif pred==1 and int(elem[0])==0:
                fn += 1
            elif pred==1 and int(elem[0])==1:
                tp += 1
    
        precision, recall, accuracy, f1 = performance(tn, tp, fn, fp)
        accuracies.append(accuracy)

    plt.figure()
    plt.scatter(k_vals, accuracies, color='b')
    plt.xlabel('K Values')
    plt.ylabel('Accuracy Values')
    plt.show()

classifier()