import numpy as np
import pandas as pd
import sys
import csv
import itertools as its
import math
import random
import operator

def load_file(filename):
    train_data, test_data = [], []
    dataset = []
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = np.array(list(lines))

    dataset = np.array(dataset)
    dataset = dataset.astype(np.float)

    dataset[:, :5] = dataset[:, 0:5].astype(np.int)
    dataset[:, 7:] = dataset[:, 7:].astype(np.int)
    
    # print(dataset[0][7])

    random.shuffle(dataset)
    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size, :]
    test_data = dataset[train_size:, :]

    return train_data, test_data

def splitByClass(dataset):
    separated = {}
    for row in dataset:
        if int(row[9]) not in separated:
            separated[int(row[9])] = []
        separated[int(row[9])].append(row)
    return separated

def mean(arr):
    return np.mean(arr, axis=0)

def stdDev(arr):
    return np.std(arr, axis=0)

def calcProb(arr, dataset):
    probabilities = {}

    for key, value in dataset.items():
        prob = 1
        for col in range(len(arr)):
            if col != 9:
                exponent = math.exp(-(math.pow(arr[col]-value[0][col],2)/(2*math.pow(value[1][col],2))))
                prob *= (1 / (math.sqrt(2*math.pi) * value[1][col])) * exponent
            # elif col > 9:

        probabilities[key] = prob

    return probabilities

def predict(train_set, arr):
    probabilities = calcProb(arr, train_set)

    bestLabel, bestProb = None, -1

    for key, value in probabilities.items():
        if bestLabel is None or bestProb < value:
            bestLabel = key
            bestProb = value

    return bestLabel

def classifier():
    filename = sys.argv[1]
    train_set, test_set = load_file(filename)
    train_set = splitByClass(train_set)
    labelProbs = {}
    for key, value in train_set.items():
        labelProbs[key] = [mean(value), stdDev(value)]

    correct = 0

    for row in test_set:
        label = predict(labelProbs, row)
        if int(label) == int(row[9]):
            correct += 1

    accuracy = float(float(correct) / float(len(test_set)))

    print(accuracy)

classifier()