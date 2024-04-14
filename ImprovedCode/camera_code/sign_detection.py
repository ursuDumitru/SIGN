import numpy as np
from keras.models import load_model


class Model:
    def __init__(self, model_path) -> None:
        self.model = load_model(model_path)


    def make_prediction(self, normalized_landmarks_list, status):
        prediction = np.zeros((1, 5))

        if status.MODE in ('d', 'w'):
            prediction = self.model.predict(np.array([normalized_landmarks_list]))

        return prediction