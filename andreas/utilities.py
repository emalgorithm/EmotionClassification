import math
import numpy as np

from scipy.io import loadmat
from collections import Counter

def load_and_clean_data():
    clean_data = loadmat('../Data/cleandata_students.mat')
    noisy_data = loadmat('../Data/noisydata_students.mat')

    X_clean = clean_data['x']
    X_noisy = noisy_data['x']

    attributes = [i for i in range(len(X_clean[0]) - 1)]

    y_clean = [y[0] for y in clean_data['y']]
    y_noisy = [y[0] for y in noisy_data['y']]

    return (X_clean, y_clean, X_noisy, y_noisy, attributes)

def majoirty_value(y):
    return Counter(y).most_common(1)[0][0]

def filter_examples(X, y, attribute, value):
    X_new = []
    y_new = []

    for (index, x) in enumerate(X):
        if (x[attribute] == value):
            X_new.append(x)
            y_new.append(y[index])

    return (X_new, y_new)

def unique_attr_vals(X):
    X = np.array(X)
    X = X.reshape(X.shape[1], X.shape[0])
    return [list(set(list(attr_vec))) for attr_vec in list(X)]

def format_train_targets(y, target_class):
    return [1 if c == target_class else 0 for c in y]

def E(y):
    counts = [c[1] for c in Counter(y).most_common(2)]
    total = sum(counts)

    # normalize
    counts = [c / total for c in counts]

    return sum([-c * math.log(c, 2) for c in counts])

def IG(X, attr, y):
    total = len(X)

    new_entropy = 0

    y_on = []
    y_off = []

    for (index, x) in enumerate(X):
        if x[attr] == 1:
            y_on.append(y[index])
        else:
            y_off.append(y[index])

    new_entropy = (len(y_on) / total)*E(y_on) + (len(y_off) / total) * E(y_off)

    return E(y) - new_entropy

def predict_single(tree, x):
    if tree.op == -1:
        return tree.value

    return predict_single(tree.kids[x[tree.op]], x)

def choose_best_attribute(X, attributes, y):
    igs = [IG(X, a, y) for a in attributes]
    return attributes[igs.index(max(igs))]

class Node:
    def __init__(self, op, value=None):
        self.op = op
        self.value = value
        self.kids = [None, None]

    def add_child(self, value, node):
        self.kids[value] = node
