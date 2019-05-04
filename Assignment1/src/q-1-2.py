import numpy as np
import pandas as pd
import random
import operator
import pprint
import matplotlib.pyplot as plt
from matplotlib import interactive
import math

def is_numeric(val):
	return isinstance(val, int) or isinstance(val, float)

def get_unique_labels(data, attributes, attr):
    index = attributes.index(attr)
    df = pd.DataFrame(data)
    values = list(set(df[index]))
    return values

def count_labels(data, target_ind):
    freq = {}
    for row in data:
        if row[target_ind] in freq:
            freq[row[target_ind]] += 1
        else:
            freq[row[target_ind]] = 1

    major = max(freq.items(), key=operator.itemgetter(1))[0]

    return freq, major

class Question:
	def __init__(self, column, value):
		self.column = column
		self.value = value

	def match(self, example):
		val = example[self.column]
		if is_numeric(val):
			return val <= self.value
		else:
			return val == self.value

def split_node(row, question):
	left, right = [], []

	for element in row:
		if question.match(element):
			left.append(element)
		else:
			right.append(element)

	return left, right


def gini(data, attributes, target):
	label_counts, major = count_labels(data, attributes.index(target))
	impurity = 1
	for label in label_counts:
		pl = label_counts[label]/(float)(len(data))
		impurity -= float(pl * pl)
	return impurity

def info_gain(data, left, right, attributes, target, current_uncertainty):
	p = float((len(left)) / (len(left) + len(right)))
	return current_uncertainty - p * gini(left, attributes, target) - (1-p) * gini(right, attributes, target)

def entropy(data, attributes, target):
    freq = {}
    dataEntropy = 0.0

    i = 0

    for attr in attributes:
        if attr==target:
            continue
        i += 1

    i -= 1

    for row in data:
        if row[i] in freq:
            freq[row[i]] += 1.0
        else:
            freq[row[i]] = 1.0

    for val in freq.values():
        dataEntropy += (-val/len(data)) * math.log(val/len(data), 2)

    return dataEntropy

def misClassification(data, attributes, target):
    freq = {}

    i = 0

    for attr in attributes:
        if attr==target:
            continue
        i += 1

    i -= 1

    for row in data:
        if row[i] in freq:
            freq[row[i]] += 1.0
        else:
            freq[row[i]] = 1.0

    p = float(0)

    for val in freq.values():
        myP = float(float(val) / float(len(data)))
        if myP > p:
            p = myP

    return 1-p

def best_split(data, attributes, target, numerical, categorical):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(data, attributes, target)

    for attr in attributes:
        column = attributes.index(attr)

        if attr!=target:
            values = get_unique_labels(data, attributes, attr)

            for val in values:
                question = Question(column, val)
                left, right = split_node(data, question)

                if len(left)!=0 and len(right)!=0:
                    gain = info_gain(data, left, right, attributes, target, current_uncertainty)

                    if gain >= best_gain:
                        best_gain = gain
                        best_question = question

    return best_gain, best_question

class Leaf:
	def __init__ (self, data, target_ind):
		self.predictions, self.major = count_labels(data, target_ind)

class Decision_Node:
    def __init__(self, question, left, right):
        self.question = question
        self.left = left
        self.right = right

xerr, ydepth = [], []

def build_tree(data, attributes, target, numerical, categorical, nodes, max_nodes):
    gain, question = best_split(data, attributes, target, numerical, categorical)

    if gain==0 or nodes>=max_nodes:
        return Leaf(data, attributes.index(target))
    
    left, right = split_node(data, question)

    left_branch = build_tree(left, attributes, target, numerical, categorical, nodes, max_nodes)
    nodes = nodes + 1
    right_branch = build_tree(right, attributes, target, numerical, categorical, nodes, max_nodes)
    nodes = nodes + 1

    return Decision_Node(question, left_branch, right_branch)

def predict(data, node):
	if isinstance(node, Leaf):
		return node.major

	if node.question.match(data):
		return predict(data, node.left)
	else:
		return predict(data, node.right)

def performance(tn, tp, fn, fp):
    total = tn+tp+fn+fp
    # print("True Negatives:", tn)
    # print("True Positives:", tp)
    # print("False Negatives:", fn)
    # print("False Positives:", fp)

    accuracy = float(float(tn+tp) / float(total))
    error = float(1.0 - accuracy)
    precision = float(float(tp) / float(tp+fp))
    recall = float(float(tp) / float(tp+fn))
    f1_score = float(float(2*tp) / float(2*tp+fp+fn))

    # print("Accuracy:", accuracy)
    # print("Error:", error)
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1-Score:", f1_score)

    out_file = open('../output_data/part2.txt', 'a')
    out_file.write("True Negatives: " + str(tn) + '\n')
    out_file.write("True Positives: " + str(tp) + '\n')
    out_file.write("False Negatives: " + str(fn) + '\n')
    out_file.write("False Positives: " + str(fp) + '\n')
    out_file.write("Accuracy: " + str(accuracy) + '\n')
    out_file.write("Error: " + str(error) + '\n')
    out_file.write("Precision: " + str(precision) + '\n')
    out_file.write("Recall: " + str(recall) + '\n')
    out_file.write("F1-Score: " + str(f1_score) + '\n')
    out_file.write('\n')

def check_accuracy(data, tree, idx):
    tp, tn, fp, fn = 0, 0, 0, 0

    for row in data:
        prediction = predict(row, tree)

        if row[idx]==0 and prediction==0:
            tn += 1
        elif row[idx]==0 and prediction==1:
            fp += 1
        elif row[idx]==1 and prediction==0:
            fn += 1
        elif row[idx]==1 and prediction==1:
            tp += 1

    return tn, tp, fn, fp

def decision_tree():
    data = pd.read_csv('../input_data/train.csv')
    attributes = list(data)
    subset = data[attributes]
    data = [tuple(x) for x in subset.values]
    target = attributes[-4]
    categorical = ['Work_accident', 'promotion_last_5years', 'sales', 'salary']
    numerical = ['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company']

    random.shuffle(data)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    idx = attributes.index(target)

    sizes = np.linspace(1, 20, 10, dtype=int)

    error_labelv, error_labelt, nodes = [], [], []

    outfile = open('../output_data/part2.txt', 'w')
    outfile.write("Result of part 2 - Both numerical and categorical" + '\n')

    for size in sizes:

        tree = build_tree(train_data, attributes, target, numerical, categorical, 0, size)

        tpv, tnv, fpv, fnv = check_accuracy(val_data, tree, idx)
        tpt, tnt, fpt, fnt = check_accuracy(train_data, tree, idx)

        outfile = open('../output_data/part2.txt', 'a')
        outfile.write("Max Nodes = " + str(size) + '\n')

        performance(tnv, tpv, fnv, fpv)

        errorv = float(1.0) - float(float(tnv+tpv) / float(tnv+tpv+fnv+fpv))
        error_labelv.append(errorv)
        errort = float(1.0) - float(float(tnt+tpt) / float(tnt+tpt+fnt+fpt))
        error_labelt.append(errort)
        nodes.append(size)

    outfile = open('../output_data/part2.txt', 'a')
    outfile.write("Entropy: " + str(entropy(train_data, attributes, target)))
    outfile.write("Misclassification: " + str(misClassification(train_data, attributes, target)))

    xlabel1, ylabel1 = [], []
    xlabel2, ylabel2 = [], []

    for row in train_data:
        if row[6]==1:
            xlabel1.append(row[0])
            ylabel1.append(row[1])
        else:
            xlabel2.append(row[0])
            ylabel2.append(row[1])

    plt.figure(1)
    plt.scatter(xlabel1, ylabel1, color='b')
    plt.scatter(xlabel2, ylabel2, color='r')
    plt.title('satisfaction_level vs last_evaluation')
    plt.xlabel('Satisfaction Level')
    plt.ylabel('Last Evaluation')
    interactive(True)
    plt.show()

    plt.figure(2)
    plt.plot(error_labelv, nodes, color='g')
    plt.title('Validation Error vs Number of Nodes')
    plt.xlabel('Error')
    plt.ylabel('Nodes')
    interactive(True)
    plt.show()

    plt.figure(3)
    plt.plot(error_labelt, nodes, color='g')
    plt.title('Train Error vs Number of Nodes')
    plt.xlabel('Error')
    plt.ylabel('Nodes')
    interactive(False)
    plt.show()

if __name__=="__main__":
    decision_tree()
