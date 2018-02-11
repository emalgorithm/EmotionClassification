from tree import Tree
from random_forest import RandomForest
from util import binarize_y
from multiprocessing import Pool

class EmotionPredictor(object):
    def __init__(self, predictors, random_forest = False, use_confidence = False):
        self.predictors = predictors
        # List of trees containing one tree for each emotions
        self.trees = []
        self.random_forest = random_forest
        self.use_confidence = use_confidence

    # Train one binary tree for each target value
    def fit(self, emotion_values, X, y):
        self.trees = [self.fit_emotion(emotion_number, X, y) for emotion_number in emotion_values]

    def fit_emotion(self, emotion_number, X, y):
        binary_y = binarize_y(y, emotion_number) 
        tree = Tree()

        if self.random_forest:
            tree = RandomForest(num_of_trees=50)
        
        tree.fit(self.predictors, X, binary_y)

        return tree

    # Predict emotion for a given data_point
    def predict(self, data_point):
        if self.use_confidence:
            return self.predict_with_confidence(data_point)
        return self.predict_without_confidence(data_point)


    def predict_with_confidence(self, data_point):
        predicted_emotion = 1
        max_confidence = 0

        for i, tree in enumerate(self.trees):
            emotion, confidence = tree.predict(data_point)

            if confidence >= max_confidence:
                predicted_emotion = i + 1
                max_confidence = confidence
        
        return predicted_emotion

    # Returns the index of the first tree with a positive result
    def predict_without_confidence(self, data_point):
        for i, tree in enumerate(self.trees):
            emotion, confidence = tree.predict(data_point)

            if emotion:
                return i + 1
        
        return 1
