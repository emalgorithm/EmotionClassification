import scipy.io
import numpy as np
import math
import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split


from tree import Tree

def binarize_dataset(dataset, emotion_number):
	binary_dataset = dataset.copy()
	binary_dataset['emotion'] = binary_dataset['emotion'].apply(lambda emotion: 1 if emotion == emotion_number else 0)

	return binary_dataset

clean_data = scipy.io.loadmat('data/cleandata_students.mat')
noisy_data = scipy.io.loadmat('data/noisydata_students.mat')

# clean_X and noisy_X are numpy matrixes with dimensions (# of training examples, number of features) containing training data
clean_X = clean_data['x']
noisy_X = noisy_data['x']

# clean_Y and noisy_Y are numpy matrixes with dimensions (# of training examples, 1) 
# clean_Y[k][0] contains the target emotion for training example k
clean_y = np.array([array[0] for array in clean_data['y']])
noisy_y = np.array([array[0] for array in noisy_data['y']])

target = 'emotion'

# clean_dataset is the pandas dataframe we will use to manipulate our data
clean_dataset = pd.DataFrame(clean_X)
clean_dataset[target] = Series(clean_y, index=clean_dataset.index)

# noisy_dataset is the pandas dataframe we will use to manipulate our data
noisy_dataset = pd.DataFrame(noisy_X)
noisy_dataset[target] = Series(noisy_y, index=noisy_dataset.index)

# Store list of all possible classes
classes = list(set(clean_y))

# Store list of all predictors
predictors = list(clean_dataset.columns)
predictors.remove(target)

emotion = 6

bin_dataset = binarize_dataset(clean_dataset[:10], emotion)
X_train, X_test, y_train, y_test = train_test_split(bin_dataset[predictors], bin_dataset[target], test_size=0.2)

tree = Tree(target)
tree.fit(predictors, emotion, X_train[:10], y_train[:10])
# Training Data Accuracy

correct = 0
total = 0

for i in range(noisy_dataset.shape[0]):
    positive = noisy_dataset.iloc[i]['emotion'] == emotion
    if tree.predict(noisy_dataset.iloc[i]) == positive:
        correct += 1
    total += 1
        
print("Accuracy is " + str(correct * 100 / total) + "%")



# def cross_validation(k, emotion, dataset):
# 	for i in range(k):
# 		#TODO: Maybe generate our own splits to control that they are different (even if these should be as well)
# 		X_train, X_test, y_train, y_test = train_test_split(dataset[predictors], dataset[target], test_size=0.2)
# 		tree = Tree(target)
# 		tree.fit(predictors, emotion, dataset)
# 		# Training Data Accuracy

# 		correct = 0
# 		total = 0

# 		for i in range(noisy_dataset.shape[0]):
# 		    positive = noisy_dataset.iloc[i]['emotion'] == emotion
# 		    if tree.predict(noisy_dataset.iloc[i]) == positive:
# 		        correct += 1
# 		    total += 1
		        
# 		print("Accuracy is " + str(correct * 100 / total) + "%")










