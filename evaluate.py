import numpy as np
import math
import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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


cross_validation(10, X, y, random_forest = True, use_confidence = True)

# clf = DecisionTreeClassifier(random_state=0)
# scores = cross_val_score(clf, X, y, cv=10)
# print("Average accuracy for sklearn decision tree is {} and std is {}".format(np.mean(scores), np.std(scores)))

# rf = RandomForestClassifier(random_state=0)
# scores = cross_val_score(rf, X, y, cv=10)
# print("Average accuracy for sklearn random forest is {} and std is {}".format(np.mean(scores), np.std(scores)))

