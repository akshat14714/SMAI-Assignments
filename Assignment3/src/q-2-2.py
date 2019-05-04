import csv
import matplotlib.pyplot as plt
import math
import operator
import numpy as np
import sys

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

def euclidian_dist(tdata, target):
	dist = 0
	length = len(target) - 1
	for i in range(length):
		tdata[i] = float(tdata[i])
		target[i] = float(target[i])
		dist = dist + pow((tdata[i] - target[i]), 2)
	return math.sqrt(dist)

def kneighbors(tdata, target, k):
	distances = []
	for i in range(len(tdata)):
		# dist = euclidian_dist(tdata[i], target)
		distances.append((tdata[i], euclidian_dist(tdata[i], target)))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	k = int(k)
	for i in range(k):
		neighbors.append(distances[i][0])
	return neighbors

def response(neighbors):
	classvotes = {}
	for x in range(len(neighbors)):
		response = float(neighbors[x][-1])
		if response > 0.5:
			response = 1
		else:
			response = 0
		if response in classvotes:
			classvotes[response] += 1
		else:
			classvotes[response] = 1
	sortedvotes = sorted(classvotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedvotes

def performance(tn, tp, fn, fp):
	recall = float(float(tp) / float(tp + fn))
	precision = float(float(tp) / float(tp + fp))
	accuracy = float(float(tp+tn) / float(tp+tn+fp+fn))
	f1 = float(float(2*tp) / float(2*tp + fp + fn))

	print("Accuracy:", accuracy)
	print("Precision:", precision)
	print("Recall:", recall)
	print("F1-Score:", f1)

	return accuracy

def classifier():
	filename = sys.argv[1]

	train_data, test_data = load_file(filename)

	accuracy_array = []

	k_val = np.linspace(1, 25, 13)
	
	for k in k_val:
		tn, tp, fn, fp = 0, 0, 0, 0
		
		accuracy = 0

		for i in range(len(test_data)):
			test_data[i][-1] = float(test_data[i][-1])

			if test_data[i][-1] > 0.5:
				test_data[i][-1] = 1
			else:
				test_data[i][-1] = 0

			neighbors = kneighbors(train_data, test_data[i], k)
			result = response(neighbors)

			if int(result[0][0]) == 0 and int(test_data[i][-1]) == 0:
				tn += 1
			elif int(result[0][0]) == 1 and int(test_data[i][-1]) == 1:
				tp += 1
			elif int(result[0][0]) == 0 and int(test_data[i][-1]) == 1:
				fn += 1
			else:
				fp += 1
		
		accuracy = performance(tn, tp, fn, fp)
		accuracy_array.append(accuracy)

	plt.scatter(k_val, accuracy_array, color='b')
	plt.xlabel("Value of k")
	plt.ylabel("Accuracy")
	plt.show()

if __name__ == "__main__":
	classifier()