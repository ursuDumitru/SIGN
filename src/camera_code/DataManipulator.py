import os
import numpy as np

from abc import ABC, abstractmethod


class DataManipulator(ABC):
    """
    This class is used to manage the data set and the sign labels.
    It has the following attributes:
        - data_set_file_path: str, path to the data set file
        - sign_labels_file_path: str, path to the sign labels file
        - sign_labels: list, all the sign labels available currently
        - sign_labels_counted: list, the count of each sign label
        - sign_labels_index: int, the index of the currently selected sign label
    """

    def __init__(self, data_set_file_path, sign_labels_file_path) -> None:
        """
        Initialize the DataManipulator object.
        - data_set_file_path: str, path to the data set file
        - sign_labels_file_path: str, path to the sign labels file
        """
        # file paths
        self.data_set_file_path = data_set_file_path
        self.sign_labels_file_path = sign_labels_file_path

        # control attributes for saving mode, both static and dynamic
        self.sign_labels = []
        self.sign_labels_counted = []
        self.sign_labels_index = -1

    @abstractmethod
    def convert_detected_landmarks_to_dict(self, mediapipe_results):
        pass

    @abstractmethod
    def normalize_landmarks(self, landmarks_dict):
        pass

    @abstractmethod
    def get_sign_labels_counted(self):
        pass

    @abstractmethod
    def get_sign_labels(self):
        pass

    @abstractmethod
    def move_sign_labels_index(self, key_input):
        pass


class DataManipulatorStatic(DataManipulator):
    """
    This class is an extension of the DataManipulator class.
    It is used to manage the data set and the sign labels for the static mode.
    """

    def __init__(self, data_set_file_path, sign_labels_file_path) -> None:
        """
        Initialize the DataManipulatorStatic object.
        :param data_set_file_path: str, path to the static data set file
        :param sign_labels_file_path: str, path to the static sign labels file
        """
        super(DataManipulatorStatic, self).__init__(data_set_file_path, sign_labels_file_path)

    def convert_detected_landmarks_to_dict(self, mediapipe_results):
        """
        This method is used to convert the detected landmarks to a dictionary made of
        2D points(x and y coordinates).
        :param mediapipe_results: ...NormalizedLandmarkList, the detected landmarks.
        :return landmarks_dict: dict, the detected landmarks in a dictionary format.
        """
        landmarks_dict = []

        for hand_landmarks in mediapipe_results.multi_hand_landmarks:
            print(type(hand_landmarks))
            for landmark in hand_landmarks.landmark:
                landmarks_dict.append({
                    'x': landmark.x,
                    'y': landmark.y
                })

        return landmarks_dict

    def normalize_landmarks(self, landmarks_dict):
        """
        This method is used to normalize the landmarks.
        The wrist landmark is used as the origin(0, 0).
        :param landmarks_dict: dict, the landmarks to be normalized.
        :return normalized_landmarks: dict, the normalized landmarks.
        """
        wrist = landmarks_dict[0]
        normalized_landmarks = []

        for landmark in landmarks_dict:
            for key, _ in landmark.items():
                normalized_landmarks.append(landmark[key] - wrist[key])

        return normalized_landmarks

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
                self.sign_labels = sign_labels

        except FileNotFoundError:
            print(f"Error: File '{self.sign_labels_file_path}' not found!")
            exit(1)  # FIXME: maybe handle this better ?

    def move_sign_labels_index(self, key_input):
        """
        This method is used to change the sign that we want to save landmarks for
            - '<': move to the previous sign
            - '>': move to the next sign.
        :param key_input: int, the key input from the user.
        """
        if key_input == ord('>') and self.sign_labels_index < (len(self.sign_labels)):
            self.sign_labels_index += 1
        if key_input == ord('<') and self.sign_labels_index > -1:
            self.sign_labels_index -= 1

    def get_sign_labels_counted(self):
        """
        This method is used to count the number of each sign label in the data set
        and automatically updates the sign_labels_counted attribute.
        """
        # check if file exists
        try:
            with open(self.data_set_file_path, 'r') as file:
                counted_signs = [0] * len(self.sign_labels)
                for line in file.readlines():
                    counted_signs[int(line.split(',')[0])] += 1
                self.sign_labels_counted = counted_signs

        except FileNotFoundError:
            print(f"Error: File '{self.data_set_file_path}' not found !")
            exit(1)

    def save_landmarks_to_csv_file(self, normalized_landmarks, key_input):
        """
        This method is used to save the landmarks to a CSV file.
            - 'c': save the landmarks to the CSV file.
        :param normalized_landmarks: list, the normalized landmarks to be saved.
        :param key_input: int, the key input from the user.
        """
        if self.sign_labels_index in range(0, len(self.sign_labels)):
            if key_input == ord('c'):
                # check if file exists
                try:
                    with open(self.data_set_file_path, 'a') as file:
                        # add comma after each landmark
                        string_to_save = f"{self.sign_labels_index}," + ','.join(
                            str(landmark) for landmark in normalized_landmarks)
                        # write the last landmark
                        file.write(string_to_save + '\n')
                        file.close()

                except FileNotFoundError:
                    print(f"Error: File '{self.data_set_file_path}' not found.")
                    exit(1)


class DataManipulatorDynamic(DataManipulator):
    """
    This class is an extension of the DataManipulator class.
    It is used to manage the data set and the sign labels for the dynamic mode.
    """

    def __init__(self, data_set_file_path, sign_labels_file_path) -> None:
        """
        Initialize the DataManipulatorDynamic object.
        :param data_set_file_path: str, path to the dynamic data set file
        :param sign_labels_file_path: str, path to the sign labels file
        """
        super(DataManipulatorDynamic, self).__init__(data_set_file_path, sign_labels_file_path)
        self.data_dirs_paths = []
        self.sequence = []
        self.number_of_frames_per_sequence = 30
        self.current_sequence_frame = 0
        self.SEQUENCE_ONGOING = False

    def convert_detected_landmarks_to_dict(self, mediapipe_results):
        """
        This method is used to convert the detected landmarks to a dictionary made of
        2D points(x and y coordinates).
        :param mediapipe_results: ...NormalizedLandmarkList, the detected landmarks.
        :return landmarks_dict: dict, the detected landmarks in a dictionary format.
        """
        landmarks_dict = []

        for hand_landmarks in mediapipe_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks_dict.append({
                    'x': landmark.x,
                    'y': landmark.y
                })

        return landmarks_dict

    def normalize_landmarks(self, landmarks_dict):
        """
        This method is used to normalize the landmarks.
        The wrist landmark is used as the origin(0, 0).
        :param landmarks_dict: dict, the landmarks to be normalized.
        :return normalized_landmarks: dict, the normalized landmarks.
        """
        wrist = landmarks_dict[0]
        normalized_landmarks = []

        for landmark in landmarks_dict:
            for key, _ in landmark.items():
                normalized_landmarks.append(landmark[key] - wrist[key])

        return normalized_landmarks

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
                self.sign_labels = sign_labels

        except FileNotFoundError:
            print(f"Error: File '{self.sign_labels_file_path}' not found!")
            exit(1)  # FIXME: maybe handle this better ?

    def move_sign_labels_index(self, key_input):
        """
        This method is used to change the sign that we want to save landmarks for
            - '<': move to the previous sign
            - '>': move to the next sign.
        :param key_input: int, the key input from the user.
        """
        if key_input == ord('>') and self.sign_labels_index < (len(self.sign_labels)):
            self.sign_labels_index += 1
        if key_input == ord('<') and self.sign_labels_index > -1:
            self.sign_labels_index -= 1

    def create_dir_for_each_sign(self):
        """
        This method is used to create a directory for each sign label for data collecting.
        """
        try:
            if not os.path.isdir(self.data_set_file_path):
                raise FileNotFoundError
            else:
                for sign_label in self.sign_labels:
                    path = self.data_set_file_path + "\\" + sign_label
                    if not os.path.isdir(path):
                        os.makedirs(path)
                    self.data_dirs_paths.append(path)
        except FileNotFoundError:
            print(f"Directory '{self.data_set_file_path}' does not exist.")

    def get_sign_labels_counted(self):
        self.create_dir_for_each_sign()
        self.sign_labels_counted = []
        for dir_path in self.data_dirs_paths:
            self.sign_labels_counted.append(len(os.listdir(dir_path)))

    def save_landmark_sequence_to_npy_file(self, normalized_landmarks, key_input):
        """
        This method is used to save the landmark sequence to a .npy file.
            - 'c': start to save 30 frames in one .npy file
        :param normalized_landmarks: list, the normalized landmarks to be saved.
        :param key_input: int, the key input from the user.
        """
        if self.sign_labels_index in range(0, len(self.sign_labels)):
            if key_input == ord('c') and not self.SEQUENCE_ONGOING:
                self.SEQUENCE_ONGOING = True

            if self.SEQUENCE_ONGOING:
                self.sequence.append(normalized_landmarks)
                self.current_sequence_frame += 1

                if self.current_sequence_frame == self.number_of_frames_per_sequence:
                    np.save(self.data_dirs_paths[self.sign_labels_index] + "\\" +
                            str(len(os.listdir(self.data_dirs_paths[self.sign_labels_index]))) +
                            ".npy", np.array(self.sequence))
                    self.sequence = []
                    self.SEQUENCE_ONGOING = False
                    self.current_sequence_frame = 0
