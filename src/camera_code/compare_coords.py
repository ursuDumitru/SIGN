import cv2 as cv
import os

from HandsDetector import HandsDetector
from DataManipulator import DataManipulatorStatic


if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.realpath(__file__)) + '\\..\\..\\'

    static_sign_labels_file_path = base_dir + r"data\static\sign_labels\sign_labels_5.csv"
    static_data_set_file_path = base_dir + r"data\static\data_set\data_set_5.csv"
    frame_save_file_path = base_dir + r"images\frames"

    hands_detector = HandsDetector(min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5,
                                   max_num_hands=1)

    data_manipulator_static = DataManipulatorStatic(static_data_set_file_path,
                                                    static_sign_labels_file_path)

    cap = cv.VideoCapture(0)
    TAKE = 0

    while True:
        _, frame = cap.read()
        frame1 = frame.copy()
        # frame2 = frame.copy()
        # frame3 = frame.copy()

        key_input = cv.waitKey(10)
        if key_input == ord('q'):
            break

        mediapipe_results = hands_detector.mediapipe_hands_detect(frame)

        if mediapipe_results.multi_hand_landmarks is not None:
            landmarks_dictionary = data_manipulator_static.convert_detected_landmarks_to_dict(mediapipe_results)

            frame1 = hands_detector.draw_hands_landmarks(frame1, mediapipe_results)
            # frame1 = hands_detector.draw_original_coord(frame.copy(), landmarks_dictionary)
            # frame2 = hands_detector.draw_normalized_coord(frame.copy(), landmarks_dictionary)
            # frame3 = hands_detector.draw_normalized_to_wrist_coord(frame.copy(), landmarks_dictionary)

            if key_input == ord('c'):
                cv.imwrite(frame_save_file_path + f"\\semn_{TAKE}.jpg", frame1)
                TAKE += 1
                # cv.imwrite(frame_save_file_path + f"\\normalized_coords{TAKE}.jpg", frame2)
                # cv.imwrite(frame_save_file_path + f"\\normalized_to_wrist_coords{TAKE}.jpg", frame3)

        cv.imshow('original_coords', frame1)
        # cv.imshow('normalized_coord', frame2)
        # cv.imshow('normalized_to_wrist_coord', frame3)

    cap.release()
    cv.destroyAllWindows()
