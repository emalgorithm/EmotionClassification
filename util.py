import scipy.io
import numpy as np
import math
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
import itertools

def binarize_y(y, emotion_number):
	binary_y = y.apply(lambda emotion: 1 if emotion == emotion_number else 0)

	return binary_y

def get_clean_dataframe():
	clean_data = scipy.io.loadmat('data/cleandata_students.mat')

	# clean_X and noisy_X are numpy matrixes with dimensions (# of training examples, number of features) containing training data
	clean_X = clean_data['x']

	# clean_Y and noisy_Y are numpy matrixes with dimensions (# of training examples, 1) 
	# clean_Y[k][0] contains the target emotion for training example k
	clean_y = np.array([array[0] for array in clean_data['y']])

	# clean_dataset is the pandas dataframe we will use to manipulate our data
	clean_dataset = pd.DataFrame(clean_X)
	clean_dataset[get_target()] = Series(clean_y, index=clean_dataset.index)

	return clean_dataset

def get_noisy_dataframe():
	noisy_data = scipy.io.loadmat('data/noisydata_students.mat')

	# clean_X and noisy_X are numpy matrixes with dimensions (# of training examples, number of features) containing training data
	noisy_X = noisy_data['x']

	# clean_Y and noisy_Y are numpy matrixes with dimensions (# of training examples, 1) 
	# clean_Y[k][0] contains the target emotion for training example k
	noisy_y = np.array([array[0] for array in noisy_data['y']])

	# noisy_dataset is the pandas dataframe we will use to manipulate our data
	noisy_dataset = pd.DataFrame(noisy_X)
	noisy_dataset[get_target()] = Series(noisy_y, index=noisy_dataset.index)	

	return noisy_dataset

def get_emotion_values():
	clean_data = scipy.io.loadmat('data/cleandata_students.mat')
	clean_y = np.array([array[0] for array in clean_data['y']])

	return list(set(clean_y))

def get_target():
	return 'emotion'

def get_predictors():
	clean_dataset = get_clean_dataframe()
	predictors = list(clean_dataset.columns)
	predictors.remove(get_target())

	return predictors

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

    print(cm)

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