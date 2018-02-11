import random
import numpy as np
from tree import Tree

class RandomForest(object):
    def __init__(self, num_of_trees = 100):
        self.trees = []
        self.num_of_trees = num_of_trees

    # It receives a binarized training dataset
    def fit(self, predictors, X, y):
        for i in range(self.num_of_trees):
            sample_X, sample_y = self.get_new_sample(X, y)
            tree = Tree(random_forest = True)
            tree.fit(predictors, sample_X, sample_y)
            self.trees.append(tree)

    def predict(self, data_point):
        confidence = sum([tree.predict(data_point)[1] for tree in self.trees]) / len(self.trees)

        emotion = 1 if confidence >= 0.5 else 0

        return emotion, confidence


    def get_new_sample(self, X, y):
        sample_X, sample_y = [], [] 

        for i in range(len(X)):
            index = random.randint(0, len(X) - 1)
            sample_X.append(X[index])
            sample_y.append(y[index])

        return np.array(sample_X), np.array(sample_y)
