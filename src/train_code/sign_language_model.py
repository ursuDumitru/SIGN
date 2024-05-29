import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input


class Model:
    def __init__(self, sign_labels_file_path, data_set_path, model_save_path, random_state) -> None:
        self.data_set_path = data_set_path
        self.model_save_path = model_save_path
        self.sign_labels = self.get_sign_labels(sign_labels_file_path)
        self.random_state = random_state

        self.model = Sequential([
            Input((21 * 2, )), # TODO: cannot be fused with the following layer
            Dense(80, activation='relu'), # TODO: change neurons to multiples of 2
            Dropout(0.2),
            Dense(40, activation='relu'),
            Dense(20, activation='relu'),
            Dense(len(self.sign_labels), activation='softmax') # experiment with different activation functions
        ])


    def get_sign_labels(self, sign_labels_file_path):
        # check if file exists
        try:
            with open(sign_labels_file_path, 'r') as file:
                pass
        except FileNotFoundError:
            print(f'File : [{sign_labels_file_path}] not found!')
            exit(1) # FIXME: maybe handle this better ?

        sign_labels = []
        with open(sign_labels_file_path, 'r') as file:
            sign_labels = file.read().splitlines()
        file.close()

        return np.array(sign_labels)


    def load_data_set(self):
        # TODO: see how you can change these
        X_data = np.loadtxt(self.data_set_path, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
        y_data = np.loadtxt(self.data_set_path, delimiter=',', dtype='int32', usecols=(0))

        return train_test_split(X_data, y_data, test_size=0.2, random_state=55) # TODO: try different number


    def save_model(self):
        self.model.save(self.model_save_path)