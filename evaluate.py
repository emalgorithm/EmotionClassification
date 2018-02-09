import scipy.io
import numpy as np
import math
import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split

from emotion_predictor import EmotionPredictor
from tree import Tree
from util import get_clean_dataframe, get_target, get_predictors, get_emotion_values

def binarize_dataset(dataset, emotion_number):
    binary_dataset = dataset.copy()
    binary_dataset['emotion'] = binary_dataset['emotion'].apply(lambda emotion: 1 if emotion == emotion_number else 0)

    return binary_dataset

def cross_validation(k, dataset):
    for i in range(k):
        target = get_target()
        predictors = get_predictors()
        emotion_values = get_emotion_values()

        #TODO: Maybe generate our own splits to control that they are different (even if these should be as well)
        X_train, X_test, y_train, y_test = train_test_split(dataset[predictors], dataset[target], test_size=0.2)

        emotion_predictor = EmotionPredictor(target, predictors)
        emotion_predictor.fit(emotion_values, X_train, y_train)
        
        correct = 0
        total = 0

        for index, row in X_test.iterrows():
            if emotion_predictor.predict(row) == y_test.loc[index]:
                correct += 1
            total += 1
                
        print("Accuracy for round " + str(i) + " is " + str(correct * 100 / total) + "%")

# target = get_target()
# predictors = get_predictors()

# clean_dataset = get_clean_dataframe()
# X_train, X_test, y_train, y_test = train_test_split(clean_dataset[predictors], clean_dataset[target], test_size=0.2)

# emotion_predictor = EmotionPredictor(target, predictors)
# emotion_predictor.fit(emotion_values, X_train, y_train)

cross_validation(10, get_clean_dataframe()[:30])


# cross_validation(10, emotion_number, clean_dataset)

# bin_dataset = binarize_dataset(clean_dataset[:10], emotion)
# X_train, X_test, y_train, y_test = train_test_split(bin_dataset[predictors], bin_dataset[target], test_size=0.2)

# tree = Tree(target)
# tree.fit(predictors, emotion, X_train[:10], y_train[:10])
# # Training Data Accuracy

# correct = 0
# total = 0

# for i in range(noisy_dataset.shape[0]):
#     positive = noisy_dataset.iloc[i]['emotion'] == emotion
#     if tree.predict(noisy_dataset.iloc[i]) == positive:
#         correct += 1
#     total += 1
        
# print("Accuracy is " + str(correct * 100 / total) + "%")














