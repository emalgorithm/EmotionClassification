import scipy.io
import numpy as np
import math
import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split

from emotion_predictor import EmotionPredictor
from tree import Tree
from util import get_clean_dataframe, get_target, get_predictors, get_emotion_values

def cross_validation(k, dataset):
    accuracies = []
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
                
        accuracy = float(correct * 100) / float(total)
        accuracies.append(accuracy)
        print("Accuracy for round " + str(i) + " is " + str(accuracy) + "%")

    print("Result for cross validation: Accuracy has a mean of {} and a std of {}".format(np.mean(accuracies), np.std(accuracies)))


cross_validation(10, get_clean_dataframe()[:200])













