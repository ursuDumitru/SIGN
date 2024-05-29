import numpy as np
import cv2 as cv
import os

from HandsDetector import HandsDetector
from ApplicationMode import ApplicationMode
from DataManipulator import DataManipulatorStatic, DataManipulatorDynamic
from SignDetector import SignDetectorStatic, SignDetectorDynamic

if __name__ == "__main__":

    # get the base directory of the project
    base_dir = os.path.dirname(os.path.realpath(__file__)) + '\\..\\..\\'

    # get the paths for the static mode
    static_sign_labels_file_path = base_dir + r"data\static\sign_labels\sign_labels_abc_4.csv"
    static_data_set_file_path = base_dir + r"data\static\data_set\data_set_abc_4.csv"
    static_model_weights_file_path = base_dir + r"models\static\model_abc_3.h5"

    # get the paths for the dynamic mode
    dynamic_sign_labels_file_path = base_dir + r"data\dynamic\sign_labels\sign_labels_1.csv"
    dynamic_data_set_dir_path = base_dir + r"data\dynamic\data_set\data_set_1"
    dynamic_model_weights_file_path = base_dir + r"models\dynamic\model_dynamic_1.h5"

    # create the important objects
    app_mode = ApplicationMode()

    hands_detector = HandsDetector(min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5,
                                   max_num_hands=1)

    data_manipulator_static = DataManipulatorStatic(static_data_set_file_path,
                                                    static_sign_labels_file_path)
    data_manipulator_dynamic = DataManipulatorDynamic(dynamic_data_set_dir_path,
                                                      dynamic_sign_labels_file_path)

    sign_detector_static = SignDetectorStatic(static_model_weights_file_path)
    # sign_detector_dynamic = SignDetectorDynamic(dynamic_model_weights_file_path)

    # start the camera
    cap = cv.VideoCapture(0)

    # set up a while loop to process the frames
    while True:
        _, frame = cap.read()

        # get the key input from the user and set the app mode
        key_input = cv.waitKey(10)
        app_mode.get_app_mode(key_input)
        frame = app_mode.set_app_mode(frame, key_input, data_manipulator_static, data_manipulator_dynamic)
        if app_mode.MODE == 'q':
            break

        # detect hands in the frame
        mediapipe_results = hands_detector.mediapipe_hands_detect(frame)

        if mediapipe_results.multi_hand_landmarks is not None:

            # draw the landmarks of the hands on the frame
            if app_mode.MODE != 'f' and app_mode.SHOW_LANDMARKS:
                frame = hands_detector.draw_hands_landmarks(frame, mediapipe_results)

            # get the landmarks from the hand detector model
            landmarks_dictionary = data_manipulator_static.convert_detected_landmarks_to_dict(mediapipe_results)
            normalized_landmarks = data_manipulator_static.normalize_landmarks(landmarks_dictionary)

            # control the data flow based on the application mode
            if app_mode.MODE in {'s', 'l'}:
                # save the landmarks of the detected static sign
                if app_mode.MODE == 's':
                    data_manipulator_static.save_landmarks_to_csv_file(normalized_landmarks, key_input)

                # make a prediction based on the detected static sign
                if app_mode.MODE == 'l':
                    prediction = sign_detector_static.make_prediction(normalized_landmarks)
                    label = data_manipulator_static.sign_labels[np.argmax(prediction)]
                    confidence = np.max(prediction)

            if app_mode.MODE in {'d', 'w'}:
                # get the landmarks as a sequence from the hand detector model
                if app_mode.MODE == 'd':
                    data_manipulator_dynamic.save_landmark_sequence_to_npy_file(normalized_landmarks, key_input)

                if app_mode.MODE == 'w':
                    # ...
                    pass

        cv.imshow('SIGN', frame)

    cap.release()
    cv.destroyAllWindows()
