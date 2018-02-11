import numpy as np
import math
import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from emotion_predictor import EmotionPredictor
from tree import Tree
from util import get_clean_dataframe, get_noisy_dataframe, get_target, get_predictors, get_emotion_values
from util import get_clean_data, get_noisy_data
from draw_tree import visualise
from evaluation import cross_validation, plot_confusion_matrix, get_precision, get_recall, get_f1_score




X, y = get_clean_data()
clf = DecisionTreeClassifier(random_state=0)
scores = cross_val_score(clf, X, y, cv=10)
print("Average accuracy for sklearn decision tree is {} and std is {}".format(np.mean(scores), np.std(scores)))

rf = RandomForestClassifier(random_state=0)
scores = cross_val_score(rf, X, y, cv=10)
print("Average accuracy for sklearn random forest is {} and std is {}".format(np.mean(scores), np.std(scores)))

cross_validation(10, X, y)

# dataset = get_clean_dataframe()[:50]
# target = get_target()
# predictors = get_predictors()
# emotion_values = get_emotion_values()

# X_train, X_test, y_train, y_test = train_test_split(dataset[predictors], dataset[target], test_size=0.0)

# emotion_predictor = EmotionPredictor(target, predictors)
# emotion_predictor.fit(emotion_values, X_train, y_train)
# visualise(emotion_predictor.trees[0])














