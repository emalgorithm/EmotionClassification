import sys
from scipy.stats import mode
import numpy as np
from node import Node


# returns the mode of the array (whether 0 or 1 is the majority element)
def majority_value(binary_targets):
    no_ones = np.count_nonzero(binary_targets == 1)
    no_zeroes = len(binary_targets) - no_ones
    if no_ones > no_zeroes:
        return 1, no_ones / len(binary_targets)
    # ? no_ones == no_zeroes
    return 0, no_zeroes / len(binary_targets)

def entropy(p, n):
    if p == 0 or n == 0:
        return 0
    f_p = p / (p + n)
    f_n = n / (p + n)
    return f_p * np.log2(1 / f_p) + f_n * np.log2(1 / f_n)

def compute_gain(p, n, p_0, n_0, p_1, n_1):
    i = entropy(p, n)
    r = (p_0 + n_0) / (p + n) * entropy(p_0, n_0) + (p_1 + n_1) / (p + n) * entropy(p_1, n_1)
    gain = i - r
    return gain

def binarise_data(targets, emotion):
    return np.array(list(map((lambda x: 1 if x == emotion else 0), targets)))


def choose_best_decision_attribute(examples, attributes, binary_targets):
    m = np.c_[examples, binary_targets]
    positives = m[m[:, -1] == 1]
    negatives = m[m[:, -1] == 0]
    p = len(positives)
    n = len(negatives)
    # TODO: check gain values
    max_gain = -1
    best_attribute = None
    for attribute in attributes:
        # no. of positive examples for which the selected attribute is 0
        p_0 = len(positives[positives[:, attribute] == 0])
        # no. of positive examples for which the selected attribute is 1
        p_1 = p - p_0
        # no. of negative examples for which the selected attribute is 0
        n_0 = len(negatives[negatives[:, attribute] == 0])
        # no. of negative examples for which the selected attribute is 1
        n_1 = n - n_0
        gain = compute_gain(p, n, p_0, n_0, p_1, n_1)
        if gain > max_gain:
            max_gain = gain
            best_attribute = attribute
    return best_attribute


def split_data(examples, best_attribute, binary_targets, i):
    # join examples with their labels
    m = np.c_[examples, binary_targets]
    # print("EXAMPLES")
    # print(examples)
    # print("LLAST COL")
    # print(binary_targets)
    # print(m)
    # print(best_attribute)
    # filter the ones which have the right value of best_attribute
    m = m[m[:, best_attribute] == i]
    return m[:, 0:-1], m[:, -1]

def decision_tree_learning(examples, attributes, binary_targets):
    # stop if all examples have the same value of binary targets
    if (binary_targets == binary_targets[0]).sum() == len(binary_targets):
        return Node(None, [], binary_targets[0])
    elif attributes.size == 0:
        classification, confidence = majority_value(binary_targets)
        return Node(None, [], classification, confidence)
    else:
        best_attribute = choose_best_decision_attribute(examples, attributes, binary_targets)
        # a new decision tree with root as best_attribute
        root = Node(best_attribute, [], None)
        # there can be two possible values of best_attribute, 0 and 1
        for i in range(2):
            # add children to this node corresponding to best_attribute = i
            examples_i, binary_targets_i = split_data(examples, best_attribute, binary_targets, i)

            if examples_i.size == 0:
                classification, confidence = majority_value(binary_targets)
                root.kids[i] = Node(None, [], classification, confidence)
            else:
                index = np.argwhere(attributes == best_attribute)
                attributes = np.delete(attributes, index)
                root.kids[i] = decision_tree_learning(examples_i, attributes, binary_targets_i)            
    return root

def predict_helper(tree, data_point):
    if tree.classification is not None:
        return tree.classification, tree.confidence
    if data_point[tree.op] == 0:
        return predict_helper(tree.kids[0], data_point)
    return predict_helper(tree.kids[1], data_point)
