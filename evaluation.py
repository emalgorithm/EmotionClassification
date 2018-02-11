from util import get_clean_dataframe, get_noisy_dataframe, get_target, get_predictors, get_emotion_values
import numpy as np
import math
import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from time import gmtime, strftime


from emotion_predictor import EmotionPredictor

def get_train_test_split(X_splits, y_splits, index):
    X_test = X_splits[index]
    y_test = y_splits[index]

    y_train = []
    X_train = []

    for i, elem in enumerate(y_splits):
        if i != index:
            for j, row in enumerate(X_splits[i]):
                y_train.append(elem[j])
                X_train.append(row)

    return np.array(X_train), X_test, np.array(y_train), y_test

def cross_validation(k, X, y, random_forest = False, use_confidence = False):
    accuracies = []
    y_pred = []
    y_true = []
    predictors = get_predictors()
    emotion_values = get_emotion_values()
    
    X_splits = np.array_split(X, k)
    y_splits = np.array_split(y, k)

    for i in range(k):
        #TODO: Maybe generate our own splits to control that they are different (even if these should be as well)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        X_train, X_test, y_train, y_test = get_train_test_split(X_splits, y_splits, i)

        emotion_predictor = EmotionPredictor(predictors, random_forest, use_confidence)
        emotion_predictor.fit(emotion_values, X_train, y_train)
        
        correct = 0
        total = 0

        predictions = []

        for index, row in enumerate(X_test):
            prediction = emotion_predictor.predict(row)
            if prediction == y_test[index]:
                correct += 1
            y_pred.append(prediction)
            y_true.append(y_test[index])
            total += 1

        accuracy = float(correct * 100) / float(total)
        accuracies.append(accuracy)
        print("Accuracy for round {0} is {1:.2f}".format(i + 1, accuracy))
    
    print("Cross Validation accuracy has a mean of {0:.2f} and a std of {1:.2f}".format(np.mean(accuracies), np.std(accuracies)))

    print("          prec, rec, f1")
    for emotion_number in emotion_values:
        print("Emotion {0}: {1:.2f}, {2:.2f}, {3:.2f}".format(emotion_number, 
        	get_precision(y_true, y_pred, emotion_number), get_recall(y_true, y_pred, emotion_number), get_f1_score(y_true, y_pred, emotion_number)))
    
    plt.figure()
    cfm = confusion_matrix(y_true, y_pred) / k
    plot_confusion_matrix(cfm, classes=["1", "2", "3", "4", "5", "6"])
    plt.show()

def get_precision(y_true, y_predicted, emotion_number):
	predicted_positive = sum([1 for prediction in y_predicted if prediction == emotion_number])
	true_positive = sum([1 for i, prediction in enumerate(y_predicted) if prediction == emotion_number and y_true[i] == emotion_number])

	if predicted_positive == 0:
		return 1.0

	return float(true_positive) / float(predicted_positive)

def get_recall(y_true, y_predicted, emotion_number):
	positive = sum([1 for elem in y_true if elem == emotion_number])
	true_positive = sum([1 for i, prediction in enumerate(y_predicted) if prediction == emotion_number and y_true[i] == emotion_number])

	if positive == 0:
		return 1.0

	return float(true_positive) / float(positive)

def get_f1_score(y_true, y_predicted, emotion_number):
	precision = get_precision(y_true, y_predicted, emotion_number)
	recall = get_recall(y_true, y_predicted, emotion_number)

	if precision + recall == 0:
		return 1.0

	return 2 * (precision * recall) / (precision + recall)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
