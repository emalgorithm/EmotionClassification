import numpy as np
import math
import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from emotion_predictor import EmotionPredictor
from tree import Tree
from util import get_clean_dataframe, get_noisy_dataframe, get_target, get_predictors, get_emotion_values, plot_confusion_matrix, get_precision, get_recall, get_f1_score
from draw_tree import visualise

def cross_validation(k, dataset):
    accuracies = []
    y_pred = []
    y_true = []
    
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

        predictions = []

        for index, row in X_test.iterrows():
            prediction = emotion_predictor.predict(row)
            if prediction == y_test.loc[index]:
                correct += 1
            y_pred.append(prediction)
            y_true.append(y_test.loc[index])
            total += 1
                
        accuracy = float(correct * 100) / float(total)
        accuracies.append(accuracy)
        print("Accuracy for round " + str(i) + " is " + str(accuracy) + "%")
    
    print("Result for cross validation: Accuracy has a mean of {} and a std of {}".format(np.mean(accuracies), np.std(accuracies)))

    for emotion_number in emotion_values:
        print("Precision for emotion {} is {}".format(emotion_number, get_precision(y_true, y_pred, emotion_number)))
        print("Recall for emotion {} is {}".format(emotion_number, get_recall(y_true, y_pred, emotion_number)))
        print("f1 score for emotion {} is {}".format(emotion_number, get_f1_score(y_true, y_pred, emotion_number)))
    
    plt.figure()
    cfm = confusion_matrix(y_true, y_pred) / k
    plot_confusion_matrix(cfm, classes=["1", "2", "3", "4", "5", "6"])
    plt.show()






# cross_validation(5, get_noisy_dataframe())

# dataset = get_clean_dataframe()[:50]
# target = get_target()
# predictors = get_predictors()
# emotion_values = get_emotion_values()

# X_train, X_test, y_train, y_test = train_test_split(dataset[predictors], dataset[target], test_size=0.0)

# emotion_predictor = EmotionPredictor(target, predictors)
# emotion_predictor.fit(emotion_values, X_train, y_train)
# visualise(emotion_predictor.trees[0])














