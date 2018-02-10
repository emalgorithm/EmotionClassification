import numpy as np
from decision_tree_learning import *

class EmotionPredictor(object):

    def __init__(self):
        self.attributes = np.array([i for i in range(45)])
        self.emotions = [i + 1 for i in range(6)]
        # List of trees containing one tree for each emotion
        self.trees = []

    def train_trees(self, train_data_x, train_data_y):
        self.trees = []
        for emotion in self.emotions:
            binary_targets = binarise_data(train_data_y, emotion)
            tree = decision_tree_learning(train_data_x, self.attributes, binary_targets)
            self.trees.append(tree)


    def train_and_test_trees(self, train_data_x, train_data_y, test_data_x, test_data_y):
        self.train_trees(train_data_x, train_data_y)
        correct = 0
        total = 0
        for i, example in enumerate(test_data_x):
            result = self.predict(test_data_x[i])
            if result == test_data_y[i]:
                correct += 1
            total += 1
        return correct * 100 / total

    def predict(self, example):
        max_confidence = 0
        result = 1

        for i, tree in enumerate(self.trees):
            emotion, confidence = predict_helper(tree, example)
            if emotion == 0:
                confidence = 1 - confidence

            if confidence >= max_confidence:
                max_confidence = confidence
                result = i + 1

        return result