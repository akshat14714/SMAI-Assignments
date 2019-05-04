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

        if attr!=target and attr in categorical:
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

def build_tree(data, attributes, target, numerical, categorical):
    gain, question = best_split(data, attributes, target, numerical, categorical)

    if gain==0:
        return Leaf(data, attributes.index(target))
    
    left, right = split_node(data, question)

    left_branch = build_tree(left, attributes, target, numerical, categorical)
    right_branch = build_tree(right, attributes, target, numerical, categorical)

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
    if tp!=0 or fp!=0:
        precision = float(float(tp) / float(tp+fp))
    else:
        precision = -1
    recall = float(float(tp) / float(tp+fn))
    f1_score = float(float(2*tp) / float(2*tp+fp+fn))

    # print("Accuracy:", accuracy)
    # print("Error:", error)
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1-Score:", f1_score)

    out_file = open('../output_data/part1.txt', 'w')
    out_file.write("Accuracy: " + str(accuracy) + '\n')
    out_file.write("Error: " + str(error) + '\n')
    if precision!=-1:
        out_file.write("Precision: " + str(precision) + '\n')
    out_file.write("Recall: " + str(recall) + '\n')
    out_file.write("F1-Score: " + str(f1_score) + '\n')

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

    entr = entropy(train_data, attributes, target)
    misclassification = misClassification(train_data, attributes, target)

    tree = build_tree(train_data, attributes, target, numerical, categorical)

    idx = attributes.index(target)

    tp, tn, fp, fn = 0, 0, 0, 0

    for row in val_data:
        prediction = predict(row, tree)

        if row[idx]==0 and prediction==0:
            tn += 1
        elif row[idx]==0 and prediction==1:
            fp += 1
        elif row[idx]==1 and prediction==0:
            fn += 1
        elif row[idx]==1 and prediction==1:
            tp += 1

    outfile = open('../output_data/part1.txt', 'a')
    outfile.write("True Negatives: " + str(tn) + '\n')
    outfile.write("True Positives: " + str(tp) + '\n')
    outfile.write("False Negatives: " + str(fn) + '\n')
    outfile.write("False Positives: " + str(fp) + '\n')
    outfile.write("Entropy: " + str(entr) + '\n')
    outfile.write("Misclassification: " + str(misclassification) + '\n')

    performance(tn, tp, fn, fp)

if __name__=="__main__":
    decision_tree()
