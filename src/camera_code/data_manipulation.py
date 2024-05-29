import cv2 as cv
import mediapipe as mp


class HandDetector:
    def __init__(self, use_static_image, detection_confidence, tracking_confidence, num_of_hands, sign_labels_file_path,
                 data_set_file_path):
        # useful objects
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.data_set_file_path = data_set_file_path
        self.sign_labels_file_path = sign_labels_file_path
        self.sign_labels = self.get_sign_labels()  # list of sign labels
        # self.sign_labels_counted = self.get

        # colors
        self.purple = (153, 0, 153)
        self.white = (255, 255, 255)

        # mediapipe model
        self.model = self.mp_hands.Hands(
            static_image_mode=use_static_image,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            max_num_hands=num_of_hands
        )

    def mediapipe_detect(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        mediapipe_results = self.model.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        return image, mediapipe_results

    def draw_landmarks(self, image, mediapipe_results):
        if mediapipe_results.multi_hand_landmarks:
            for hand_landmarks in mediapipe_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=self.purple, thickness=2, circle_radius=1),
                    self.mp_drawing.DrawingSpec(color=self.white, thickness=2, circle_radius=1))

        return image

    def get_landmarks_as_dict(self, mediapipe_results):
        landmarks_dict = []
        if mediapipe_results.multi_hand_landmarks:  # len = 1 or 2
            for hand_landmarks in mediapipe_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks_dict.append({
                        'x': landmark.x,
                        'y': landmark.y})

        return landmarks_dict  # len =  21 always(for one hand)

    def convert_landmark_to_list(self, landmarks_dict):
        # wrist will be 0.0, 0.0, 0.0
        wrist = landmarks_dict[0]
        normalized_landmarks = []  # values between -1 and 1, wrist being 0, 0, 0, len = 21(for one hand)

        for landmark in landmarks_dict:
            normalized_landmarks.append({
                'x': landmark['x'] - wrist['x'],
                'y': landmark['y'] - wrist['y']})
            # no 'z' as I do not care about the depth

        return normalized_landmarks

    def normalize_landmarks(self, landmarks_dict):
        normalized_landmarks = self.convert_landmark_to_list(landmarks_dict)
        normalized_landmarks_list = []

        for landmark in normalized_landmarks:
            for key, _ in landmark.items():
                normalized_landmarks_list.append(landmark[key])

        return normalized_landmarks_list

    def save_landmarks_to_csv_file(self, normalized_landmarks_list, key_input, status):
        if status.MODE == 's':
            if status.selected_sub_list_sign is not None:
                status.set_real_list_index()
                if status.real_list_index < len(self.sign_labels):
                    if key_input == ord('c'):
                        # check if file exists
                        try:
                            with open(self.data_set_file_path, 'r') as file:
                                pass
                        except FileNotFoundError:
                            print(f'File : {self.data_set_file_path} not found.')
                            exit(0)

                        # add comma after each landmark
                        string_to_save = f"{status.real_list_index}," + ','.join(
                            str(landmark) for landmark in normalized_landmarks_list)

                        # write the last landmark
                        with open(self.data_set_file_path, 'a') as file:  # a = append
                            file.write(str(string_to_save) + '\n')
                        file.close()

    def get_sign_labels(self):
        # check if file exists
        try:
            with open(self.sign_labels_file_path, 'r') as file:
                pass
        except FileNotFoundError:
            print(f'File : {self.sign_labels_file_path} not found!')
            exit(1)  # FIXME: maybe handle this better ?

        with open(self.sign_labels_file_path, 'r') as file:
            sign_labels = file.read().splitlines()
        file.close()

        return sign_labels

    def count_number_of_saved_landmarks(self):
        # check if file exists
        try:
            with open(self.data_set_file_path, 'r') as file:
                pass
        except FileNotFoundError:
            print(f'File : {self.data_set_file_path} not found while trying to count!')
            return None

        list_of_counted_signs = [0] * len(self.get_sign_labels())

        with open(self.data_set_file_path, 'r') as file:
            for line in file.readlines():
                list_of_counted_signs[int(line.split(',')[0])] += 1

        return list_of_counted_signs

    def find_min_and_max_for_x_and_y(self, landmarks_dict):
        min_x = min_y = 1
        max_x = max_y = 0

        for landmark in landmarks_dict:
            min_x = min(min_x, landmark['x'])
            min_y = min(min_y, landmark['y'])
            max_x = max(max_x, landmark['x'])
            max_y = max(max_y, landmark['y'])

        return min_x, min_y, max_x, max_y

    # make a list of sub-lists, each 10 elements
    # for easier sign selection while creating the data-set
    def reshape_sign_labels(self):
        return [self.sign_labels[i:i + 10] for i in range(0, len(self.sign_labels), 10)]
