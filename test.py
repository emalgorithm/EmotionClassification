import pickle
import sys

from util import get_data

def main():
    test_data_path = sys.argv[1]
    print(test_data_path)
    X_test, y_test = get_data(test_data_path)

    with open('emotion_predictor.pickle', 'rb') as f:
        emotion_predictor = pickle.load(f)

    y_pred = emotion_predictor.predict(X_test)

    correct = sum([1 for i, prediction in enumerate(y_pred) if prediction == y_test[i]])
    accuracy = float(correct * 100) / len(y_test)
    
    print("Accuracy is {0:.2f}".format(accuracy))

if __name__ == "__main__":
    # execute only if run as a script
    main()

