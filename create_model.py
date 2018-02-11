import pickle
from emotion_predictor import EmotionPredictor

from util import get_clean_data, get_predictors, get_emotion_values

X, y = get_clean_data()
predictors = get_predictors()
emotion_values = get_emotion_values()

emotion_predictor = EmotionPredictor(predictors, random_forest=True, use_confidence=True, num_of_trees=200)
emotion_predictor.fit(emotion_values, X, y)

with open('emotion_predictor.pickle', 'wb') as f:
    pickle.dump(emotion_predictor, f, pickle.HIGHEST_PROTOCOL)

