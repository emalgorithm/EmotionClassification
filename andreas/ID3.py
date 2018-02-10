from utilities import *

from sklearn.model_selection import train_test_split

import math
import numpy as np

def build_decision_tree(X, attributes, y, attr_vals):
    unique_y = set(y)

    if len(unique_y) == 1:
        return Node(-1, unique_y.pop())
    elif len(attributes) == 0:
        return Node(-1, majoirty_value(y))

    best_attr = choose_best_attribute(X, attributes, y)

    node = Node(best_attr)

    for val in attr_vals[best_attr]:
        (X_v, y_v) = filter_examples(X, y, best_attr, val)

        if len(X_v) == 0:
            node.add_child(val, Node(-1, majoirty_value(y)))
        else:
            new_attr = [a for a in attributes if a != best_attr]
            child_node = build_decision_tree(X_v, new_attr, y_v, attr_vals)
            node.add_child(val, child_node)

    return node

def build_tree(X, y, target, attributes):
    attr_vals = unique_attr_vals(X)
    y_formatted = format_train_targets(y, target)

    return build_decision_tree(X, attributes, y_formatted, attr_vals)

def predict(trees, x):
    predictions = [predict_single(tree, x) for tree in trees]

    index = predictions.index(max(predictions))

    classes = list(range(1, 7))

    return classes[index]

def evaluate_accuracy(X, actual_y, trees):
    predicted = [predict(trees, x) for x in X]

    correct = sum([1 if y == predicted[i] else 0 for (i, y) in enumerate(actual_y)])

    return round(correct / len(actual_y), 4) * 100

def build_trees_and_evaluate(X, y, attributes):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    trees = [build_tree(X_train, y_train, target, attributes) for target in list(set(y_noisy))]

    accuracy = evaluate_accuracy(X_test, y_test, trees)

    print("{}%".format(accuracy))

    return (trees, accuracy)

(X_clean, y_clean, X_noisy, y_noisy, attributes) = load_and_clean_data()

(trees, accuracy) = build_trees_and_evaluate(X_clean, y_clean, attributes)
