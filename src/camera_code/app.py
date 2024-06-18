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
    TRY = 5
    static_sign_labels_file_path = base_dir + f"data\\static\\sign_labels\\sign_labels_{TRY}.csv"
    static_data_set_file_path = base_dir + f"data\\static\\data_set\\data_set_{TRY}.csv"
    static_model_weights_file_path = base_dir + f"models\\static\\model_static_{TRY}_1.h5"

    # get the paths for the dynamic mode
    TRY = 3
    dynamic_sign_labels_file_path = base_dir + f"data\\dynamic\\sign_labels\\sign_labels_{TRY}.csv"
    dynamic_data_set_dir_path = base_dir + f"data\\dynamic\\data_set\\data_set_{TRY}"
    dynamic_model_weights_file_path = base_dir + f"models\\dynamic\\model_dynamic_2_2.h5"

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
    sign_detector_dynamic = SignDetectorDynamic(dynamic_model_weights_file_path)

    # start the camera
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 2)

    # set up a while loop to process the frames
    while True:
        _, frame = cap.read()

        # get the key input from the user and set the app mode
        key_input = cv.waitKey(10)
        app_mode.get_app_mode(key_input)
        frame = app_mode.set_app_mode(frame.copy(), key_input, data_manipulator_static, data_manipulator_dynamic)
        if app_mode.MODE == 'q':
            break

        # detect hands in the frame
        mediapipe_results = hands_detector.mediapipe_hands_detect(frame)

        if mediapipe_results.multi_hand_landmarks is not None:
            # get the landmarks from the hand detector model
            landmarks_dictionary = data_manipulator_static.convert_detected_landmarks_to_dict(mediapipe_results)
            normalized_landmarks = data_manipulator_static.normalize_landmarks(landmarks_dictionary)

            # set up the variables used by the prediction
            label = confidence = None
            accepted_word_labels = data_manipulator_static.sign_labels[:26] + data_manipulator_dynamic.sign_labels

            # control the data flow based on the application mode
            # save the landmarks of the detected static sign
            if app_mode.MODE == '2':
                data_manipulator_static.save_landmarks_to_csv_file(normalized_landmarks, key_input)

            # make a prediction based on the detected static sign
            if app_mode.MODE == '4':
                label, confidence = sign_detector_static.get_label_and_prediction(normalized_landmarks,
                                                                                  data_manipulator_static)
                # display the prediction above the detected hand
                frame = hands_detector.display_prediction_on_frame(frame.copy(), label, confidence,
                                                                   landmarks_dictionary)
                # sentence mode
                if app_mode.SENTENCE_MODE:
                    app_mode.create_word(label, accepted_word_labels)
                    app_mode.create_sentence(label)

            # save the landmarks as a sequence of the detected dynamic sign
            if app_mode.MODE == '3':
                data_manipulator_dynamic.save_landmark_sequence_to_npy_file(normalized_landmarks, key_input)

            # make a prediction based on the detected dynamic sign
            if app_mode.MODE == '5':
                label, confidence = sign_detector_dynamic.get_label_and_prediction(normalized_landmarks,
                                                                                   data_manipulator_dynamic)
                # display the prediction above the detected hand
                frame = hands_detector.display_prediction_on_frame(frame.copy(), label, confidence,
                                                                   landmarks_dictionary)
                # sentence mode
                if app_mode.SENTENCE_MODE:
                    app_mode.create_word(label, accepted_word_labels)
                    app_mode.create_sentence(label)

            # draw the landmarks of the hands on the frame
            if app_mode.MODE != '1' and app_mode.SHOW_LANDMARKS:
                frame = hands_detector.draw_hands_landmarks(frame.copy(), mediapipe_results)
                if app_mode.MODE in {'4', '5'}:
                    frame = hands_detector.draw_rectangle_around_hand(frame.copy(), landmarks_dictionary)

        cv.imshow('SIGN', frame)

    cap.release()
    cv.destroyAllWindows()
