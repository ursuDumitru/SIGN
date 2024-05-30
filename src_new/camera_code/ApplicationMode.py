import cv2 as cv
import numpy as np


class ApplicationMode:
    """
    This class is used to manage how the application behaves.
    It has the following modes:
        - 'f' : free camera mode
        - 's' : save static landmarks mode(letter)
        - 'd' : save dynamic landmarks mode(word)
        - 'l' : detect static signs mode(letter)
        - 'w' : detect dynamic signs mode(word)
        - 'q' : quit application
        - 'e' : enable/hide landmarks
    Also, this class provides methods that help and/or change
    the flow of data in the application.
    """

    def __init__(self) -> None:
        """
        Initialize necessary variables for the class and some
        colors that will be used in editing the frame GUI.
        Sets the application mode with the default mode 'f'.
        Landmarks are not shown by default.
        """
        self.MODE = 'f'
        self.SHOW_LANDMARKS = False

        # colors:
        self.purple = (128, 0, 128)
        self.white = (255, 255, 255)

    def get_app_mode(self, key_input):
        """
        This method is used to change the application mode
        based on the user's key input.

        :param key_input: int, unicode code of the key pressed
        """
        if key_input == ord('f'):
            self.MODE = 'f'
        elif key_input == ord('s'):
            self.MODE = 's'
        elif key_input == ord('d'):
            self.MODE = 'd'
        elif key_input == ord('l'):
            self.MODE = 'l'  # TODO make a key_input that start the sentence creation
        elif key_input == ord('w'):
            self.MODE = 'w'
        elif key_input == ord('q'):
            self.MODE = 'q'
        elif key_input == ord('e'):
            self.SHOW_LANDMARKS = not self.SHOW_LANDMARKS
        else:
            self.MODE = self.MODE
            self.SHOW_LANDMARKS = self.SHOW_LANDMARKS

    def set_text_for_save_mode(self, key_input, dm, data_type):
        """
        This method is used to set the text for either the static or dynamic mode

        :param key_input: int, unicode value of user input.
        :param dm: DataManipulator, Static or Dynamic.
        :param data_type: str, "static" or "dynamic"
        :return:
        """
        dm.get_sign_labels()
        dm.get_sign_labels_counted()
        dm.move_sign_labels_index(key_input)

        if dm.sign_labels_index in range(0, len(dm.sign_labels)) and len(dm.sign_labels_counted) != 0:
            text = f"Saving {data_type} landmark for: " \
                   f"{dm.sign_labels[dm.sign_labels_index]}" \
                   f"({dm.sign_labels_counted[dm.sign_labels_index]}), " \
                   f"sign: {dm.sign_labels_index + 1}/{len(dm.sign_labels)}"

            if self.MODE == 'd' and dm.SEQUENCE_ONGOING:
                print(dm.current_sequence_frame)  # developer needs
                text = text + f", frame: {dm.sign_labels_counted[dm.sign_labels_index] + 1}/" \
                              f"{dm.number_of_frames_per_sequence}"

        else:
            text = f"Save {data_type} Landmarks Mode(letter)"

        return text

    def set_app_mode(self, frame, key_input, data_manipulator_static, data_manipulator_dynamic):
        """
        This method is used to display the current application
        mode on the GUI of the frame.

        :param frame: np.array, the frame to be displayed.
        :param key_input: int, unicode value of user input.
        :param data_manipulator_static: DataManipulatorStatic, blabla.
        :param data_manipulator_dynamic: DataManipulatorStatic, blabla
        """
        cv.rectangle(frame, (0, 0), (frame.shape[1], 20), (255, 255, 255), -1)
        text = ""

        if self.MODE == 'f':
            text = "Free Camera Mode"
        elif self.MODE == 's':
            text = self.set_text_for_save_mode(key_input, data_manipulator_static, 'static')
        elif self.MODE == 'd':
            text = self.set_text_for_save_mode(key_input, data_manipulator_dynamic, 'dynamic')
        elif self.MODE == 'l':
            text = "Detect Static Signs Mode(letter)"
        elif self.MODE == 'w':
            text = "Detect Dynamic Signs Mode(word)"
        elif self.MODE == 'q':
            text = "Quit Application"

        text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv.putText(frame, text, (int((frame.shape[1] - text_size[0]) / 2), 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, self.purple, 1, cv.LINE_AA)

        return frame
