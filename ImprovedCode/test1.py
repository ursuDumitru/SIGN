import cv2 as cv
import numpy as np
import mediapipe as mp

class HandDetector:
    def __init__(self, use_static_image, detection_confidence, tracking_confidence, num_of_hands, sign_labels_file_path, data_set_file_path) -> None:
        # useful objects
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.data_set_file_path = data_set_file_path
        self.sign_labels_file_path = sign_labels_file_path
        self.sign_labels = self.get_sign_labels() # list of sign labels

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
                    self.mp_drawing.DrawingSpec(color=(153,0,153), thickness=2, circle_radius=1),
                    self.mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1)
                )

        return image


    def get_landmarks_as_dict(self, mediapipe_results):
        landmarks_dict = []
        if mediapipe_results.multi_hand_landmarks: # len = 1 or 2
            for hand_landmarks in mediapipe_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks_dict.append({
                        'x': landmark.x,
                        'y': landmark.y
                        # 'z': landmark.z
                    })

        return landmarks_dict # len =  21 always(for one hand???)


    def convert_landmark_to_list(self, landmarks_dict):
        # wrist will be 0.0, 0.0, 0.0
        wrist = landmarks_dict[0]
        normalized_landmarks = [] # values between -1 and 1, wrist being 0, 0, 0, len = 21(for one hand)

        for landmark in landmarks_dict:
            normalized_landmarks.append({
                'x': landmark['x'] - wrist['x'],
                'y': landmark['y'] - wrist['y']
                # 'z': landmark['z'] - wrist['z']
            })

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
            if status.sign_to_save_landmarks_for is not None and status.sign_to_save_landmarks_for < len(self.sign_labels): # make sign_labels part of the class
                if key_input == ord('c'):
                    # check if file exists
                    try:
                        with open(self.data_set_file_path, 'r') as file:
                            pass
                    except FileNotFoundError:
                        print(f'File : {self.data_set_file_path} not found, it will be created.')

                    # add comma after each landmark
                    string_to_save = f"{status.sign_to_save_landmarks_for}," + ','.join(str(landmark) for landmark in normalized_landmarks_list)

                    # write the last landmark
                    with open(self.data_set_file_path, 'a') as file: # a = append
                        file.write(str(string_to_save) + '\n')
                    file.close()


    def get_sign_labels(self):
        # check if file exists
        try:
            with open(self.sign_labels_file_path, 'r') as file:
                pass
        except FileNotFoundError:
            print(f'File : {self.sign_labels_file_path} not found!')
            exit(1) # FIXME: maybe handle this better ?

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
                list_of_counted_signs[int(line[0])] += 1

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

class Status:
    def __init__(self) -> None:
        self.MODE = None
        self.sign_to_save_landmarks_for = None

        self.DOT = [.5, .5]
        pass


    def set_status_mode(self, key_input):
        if key_input == ord('s'):
            self.MODE = 's' # save mode
            self.DOT = [0, 0]
        elif key_input == ord('d'):
            self.MODE = 'd' # detect mode
        elif key_input == ord('f'):
            self.MODE = 'f' # free camera mode
            self.DOT = [0, 0]

        if self.MODE == 's': # make a way to shift sign groups(if there are more than 10 signs)
            if ord('0') <= key_input <= ord('9'):
                self.sign_to_save_landmarks_for = int(chr(key_input))
        else:
            self.sign_to_save_landmarks_for = None


    def set_status_text(self, image, sign_labels, hands):
        list_of_counted_signs = hands.count_number_of_saved_landmarks()
        text = None

        cv.rectangle(image, (0, 0), (image.shape[1], 35), (255,255,255), -1)
        if self.MODE == 's':
            if self.sign_to_save_landmarks_for is not None and list_of_counted_signs is not None:
                if 0 <= self.sign_to_save_landmarks_for < len(sign_labels):
                    text = f'Saving landmarks for sign: {sign_labels[self.sign_to_save_landmarks_for]}' \
                           f'({list_of_counted_signs[self.sign_to_save_landmarks_for]})'
                else:
                    text = 'Invalid sign'
            else:
                text = 'Saving landmarks mode'
        elif self.MODE == 'd':
            text = 'Detect mode'
        else:
            text = 'Free camera mode'

        text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        cv.putText(image, text, (int((image.shape[1] - text_size[0]) / 2), 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (153,0,153), 2, cv.LINE_AA)

        return image


    def draw_rectangle_around_hand(self, image, hands, landmarks_dict, prediction, status):
        if status.MODE == 'd':
            min_x, min_y, max_x, max_y = hands.find_min_and_max_for_x_and_y(landmarks_dict)
            cv.rectangle(image,
                        (int(min_x * image.shape[1] - 10), int(min_y * image.shape[0] - 10)), # TODO: why * image.shape[1] and * image.shape[0] and not reverse ???
                        (int(max_x * image.shape[1] + 10), int(max_y * image.shape[0] + 10)),
                        (153,0,153), 2)

            if np.max(prediction) > 0.8:
                label, accuracy = hands.sign_labels[np.argmax(prediction)], np.max(prediction)
            else:
                label, accuracy = 'Unknown sign', np.prod(1 - prediction) # FIXME if I change how the probabilities are calculated, I need to change this too

            # move the dot according to the sign detected
            if label == hands.sign_labels[0] and status.DOT[1] >= 0.01:
                status.DOT[1] += -.01
            elif label == hands.sign_labels[1] and status.DOT[1] <= 0.99:
                status.DOT[1] += .01
            elif label == hands.sign_labels[2] and status.DOT[0] <= 0.99:
                status.DOT[0] += .01
            elif label == hands.sign_labels[3] and status.DOT[0] >= 0.01:
                status.DOT[0] += -.01

            cv.circle(image, (int(status.DOT[0] * image.shape[1]), int(status.DOT[1] * image.shape[0])), 5, (255,255,153), -1)


            cv.putText(image, f'{label} ({accuracy:.2f})', (int(min_x * image.shape[1]), int(min_y * image.shape[0]) - 15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.75, (153,0,153), 2, cv.LINE_AA)

        return image

from keras.models import load_model

class Model:
    def __init__(self, model_path) -> None:
        self.model = load_model(model_path)

    def make_prediction(self, normalized_landmarks_list, status):
        prediction = np.zeros((1, 5))

        if status.MODE == 'd':
            prediction = self.model.predict(np.array([normalized_landmarks_list]))

        return prediction

status = Status()

hands = HandDetector(use_static_image=True,
                     detection_confidence=0.5,
                     tracking_confidence=0.5,
                     num_of_hands=2, # FIXME may wanna make this work with only one hand first
                     sign_labels_file_path='.\data\sign_labels3.csv',
                     data_set_file_path='.\data\data_set3.csv')

model = Model('.\models\model16_2.h5')

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()

    key_input = cv.waitKey(10)

    # quit camera
    if key_input == ord('q'):
        break

    status.set_status_mode(key_input)
    frame, mediapipe_results = hands.mediapipe_detect(frame)
    frame = status.set_status_text(frame, hands.sign_labels, hands)

    if mediapipe_results.multi_hand_landmarks is not None:

        # make a dict of the basic coordinates of the landmarks
        landmarks_dict = hands.get_landmarks_as_dict(mediapipe_results) # values between 0 and 1, starting from the top left corner

        # make landmarks visible
        frame = hands.draw_landmarks(frame, mediapipe_results)

        # normalize landmarks to local(relative) axis and convert to 1d list
        normalized_landmarks_list = hands.normalize_landmarks(landmarks_dict)

        # save the last landmarks to a csv file
        hands.save_landmarks_to_csv_file(normalized_landmarks_list, key_input, status)

        # FIXME:crashes if there are 2 hands at the same time
        # make a prediction
        prediction = model.make_prediction(normalized_landmarks_list, status)

        # highlight the predicted sign
        frame = status.draw_rectangle_around_hand(frame, hands, landmarks_dict, prediction, status)

    cv.imshow('App', frame) # SIRS: Sistem Inteligent de Recunoastere a Semnelor ???

cap.release()
cv.destroyAllWindows()