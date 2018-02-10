import scipy.io
import numpy as np
from emotion_predictor import EmotionPredictor
from draw_tree import visualise
from sklearn.model_selection import train_test_split

# dictionary corresponding to the files
clean_data = scipy.io.loadmat('Data/cleandata_students.mat')
noisy_data = scipy.io.loadmat('Data/noisydata_students.mat')

# clean_x and noisy_x are numpy matrixes corresponding to the x variable in the dataset
# the dimensions are N x 45 (no. examples, number of AUs)
clean_data_x = clean_data['x']
noisy_data_x = noisy_data['x']

# clean_y and noisy_y are vectors N x 1
clean_data_y = np.squeeze(np.asarray(clean_data['y']))
noisy_data_y = np.squeeze(np.asarray(noisy_data['y']))


def main():

    emotion_predictor = EmotionPredictor()
    folds = 10
    num_data = len(clean_data_x)
    accuracy = []

    for fold in range(folds):
        #train_x, test_x, train_y, test_y = train_test_split(clean_data_x, clean_data_y, test_size=0.2)
        low = int(fold * (num_data / folds))
        high = int(low + (num_data / folds))
        print("FOR THIS RUN {} {}".format(low, high))
        train_x = np.concatenate((clean_data_x[:low], clean_data_x[high:]), axis = 0)
        train_y = np.concatenate((clean_data_y[:low], clean_data_y[high:]), axis = 0)
        test_x = clean_data_x[low:high]
        test_y = clean_data_y[low:high]
        accuracy.append(emotion_predictor.train_and_test_trees(train_x, train_y, test_x, test_y))
        print("Accuracy for this round " + str(accuracy[-1]) + "%")

    emotion_predictor.train_trees(clean_data_x, clean_data_y)
    avg = 0
    for a in accuracy:
        avg += a
    avg /= len(accuracy)
    print("Average accuracy " + str(avg) + "%")

    # visualise(emotion_predictor.trees[0])

if __name__ == "__main__":
    main()
