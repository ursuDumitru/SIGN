import numpy as np
import os

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, Input, Embedding
from keras.utils import to_categorical


class Model:
    def __init__(self, sign_labels_file_path, data_set_path, model_save_path, random_state) -> None:
        self.sign_labels_file_path = sign_labels_file_path
        self.data_set_path = data_set_path
        self.model_save_path = model_save_path
        self.sign_labels = []
        self.random_state = random_state

        self.model = None

    def get_sign_labels(self):
        """
        This method is used to get the sign labels from the CSV file and
        automatically updates the sign_labels attribute.
        """
        # check if file exists
        try:
            with open(self.sign_labels_file_path, 'r') as file:
                sign_labels = file.read().splitlines()
                file.close()
                self.sign_labels = np.array(sign_labels)

        except FileNotFoundError:
            print(f"Error: File '{self.sign_labels_file_path}' not found!")
            exit(1)  # FIXME: maybe handle this better ?

    def save_model(self):
        self.model.save(self.model_save_path)

    def load_data_set(self):
        pass


class ModelStatic(Model):
    def __init__(self, sign_labels_file_path, data_set_path, model_save_path, random_state) -> None:
        super().__init__(sign_labels_file_path, data_set_path, model_save_path, random_state)

        self.model = Sequential([
            Input((21 * 2,)),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(len(self.sign_labels), activation='softmax')  # TODO get act func from d.Holban research
        ])

    def load_data_set(self):
        x_data = np.loadtxt(self.data_set_path, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
        y_data = np.loadtxt(self.data_set_path, delimiter=',', dtype='int32', usecols=0)

        return train_test_split(x_data, y_data, test_size=0.2, random_state=55)  # TODO: try different number


class ModelDynamic(Model):
    def __init__(self, sign_labels_file_path, data_set_path, model_save_path, random_state) -> None:
        super().__init__(sign_labels_file_path, data_set_path, model_save_path, random_state)
        self.data_set_signs_path = []

        self.model = Sequential([
            # Input((30, 21, 2)),
            # GRU(activation='relu', input_shape=(30, 42), units=256),
            # GRU(activation='relu', units=128),
            # GRU(activation='relu', units=64),
            Embedding(input_dim=30*21*2, output_dim=64),
            GRU(256, return_sequences=True),
            GRU(128, return_sequences=True),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(len(self.sign_labels), activation='softmax')
        ])

    def get_data_set_dirs(self):
        """
        This method is used to create a directory for each sign label for data collecting.
        """
        self.get_sign_labels()
        try:
            if not os.path.isdir(self.data_set_path):
                raise FileNotFoundError
            else:
                for sign_label in self.sign_labels:
                    self.data_set_signs_path.append(self.data_set_path + "/" + sign_label)
        except FileNotFoundError:
            print(f"Directory '{self.data_set_path}' does not exist.")

    def load_data_set(self):
        x_data = []
        y_data = []

        self.get_data_set_dirs()
        for i, sign_dir in enumerate(self.data_set_signs_path):
            for file in os.listdir(sign_dir):
                data = np.load(sign_dir + "/" + file)
                x_data.append(data)
                y_data.append(i)
        
        return train_test_split(np.array(x_data), to_categorical(y_data).astype(int), test_size=0.2, random_state=55)
        # return train_test_split(np.array(x_data), y_data, test_size=0.2, random_state=55)
