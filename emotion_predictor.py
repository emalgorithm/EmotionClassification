from tree import Tree
from random_forest import RandomForest
from util import binarize_y

class EmotionPredictor(object):
    def __init__(self, predictors):
        self.predictors = predictors
        # List of trees containing one tree for each emotions
        self.trees = []

    # Train one binary tree for each target value
    def fit(self, emotion_values, X, y):
        for emotion_number in emotion_values:
            # Create binary data for given emotion_number
            binary_y = binarize_y(y, emotion_number) 
            
            # tree = Tree()
            tree = RandomForest(num_of_trees=10)
            tree.fit(self.predictors, X, binary_y)
            self.trees.append(tree)

    # Predict emotion for a given data_point
    def predict(self, data_point):
        predicted_emotion = 1
        max_confidence = 0

        for i, tree in enumerate(self.trees):
            emotion, confidence = tree.predict(data_point)

            if confidence >= max_confidence:
                predicted_emotion = i + 1
                max_confidence = confidence
        
        return predicted_emotion
