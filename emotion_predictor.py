from tree import Tree
from util import binarize_y

class EmotionPredictor(object):
    def __init__(self, target, predictors):
        self.target = target
        self.predictors = predictors
        # List of trees containing one tree for each emotions
        self.trees = []

    # Train one binary tree for each target value
    def fit(self, emotion_values, X, y):
        for emotion_number in emotion_values:
            # Create binary data for given emotion_number
            binary_y = binarize_y(y, emotion_number) 
            
            tree = Tree(self.target)
            tree.fit(self.predictors, emotion_number, X, binary_y)
            self.trees.append(tree)

    # Predict emotion for a given data_point
    def predict(self, data_point):
        for i, tree in enumerate(self.trees):
            if tree.predict(data_point):
                return i + 1
        return 1
