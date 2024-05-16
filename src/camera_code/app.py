import cv2 as cv
import multiprocessing
import time
import os

from data_manipulation import HandDetector
from frame_status import FrameStatus
from sign_detection import Model


def toggle_insert_letter(insert_letter):
        while True:
            if insert_letter.value == False:
                time.sleep(2)
                insert_letter.value = True


base_dir = os.path.dirname(os.path.realpath(__file__)) + '\\..\\..\\'
sign_labels_file_path = base_dir + 'data\sign_labels\sign_labels_abc_3.csv'
data_set_file_path = base_dir + 'data\data_set\data_set_abc_3.csv'
model_weights_file_path = base_dir + 'models\model_abc_3.h5'

# create a HandDetector object
hands = HandDetector(
    use_static_image=False,
    detection_confidence=0.5,
    tracking_confidence=0.5,
    num_of_hands=1,
    sign_labels_file_path=sign_labels_file_path,
    data_set_file_path=data_set_file_path
    )

# create a FrameStatus object
status = FrameStatus()
status.sub_lists_of_signs = hands.reshape_sign_labels()

# create a Model object
model = Model(model_weights_file_path)

if __name__ == "__main__":

    # start the camera
    cap = cv.VideoCapture(0)

    insert_letter = multiprocessing.Value('b', False)
    process = multiprocessing.Process(target=toggle_insert_letter, args=(insert_letter,))

    while True:
        _, frame = cap.read()
        key_input = cv.waitKey(10)

        # quit camera
        if key_input == ord('q'):
            break

        status.set_status_mode(key_input)
        status.get_sign_from_key_input(key_input)
        frame, mediapipe_results = hands.mediapipe_detect(frame)
        frame = status.set_status_text(frame, key_input, hands)

        if mediapipe_results.multi_hand_landmarks is not None:

            # make a dict of the basic coordinates of the landmarks
            # values between 0 and 1, starting from the top left corner
            landmarks_dict = hands.get_landmarks_as_dict(mediapipe_results)

            # make landmarks visible
            frame = hands.draw_landmarks(frame, mediapipe_results)

            # normalize landmarks to local(relative) axis and convert to 1d list
            normalized_landmarks_list = hands.normalize_landmarks(landmarks_dict)

            # save the last landmarks to a csv file
            hands.save_landmarks_to_csv_file(normalized_landmarks_list, key_input, status)

            # make a prediction
            prediction = model.make_prediction(normalized_landmarks_list, status)

            # take the predicted sign and manage it accordingly
            frame = status.manage_prediction(frame, landmarks_dict, prediction, hands, insert_letter, process)

        cv.imshow('App', frame) # SIRS: Sistem Inteligent de Recunoastere a Semnelor ???

    cap.release()
    cv.destroyAllWindows()

    if process.is_alive():
        process.terminate()