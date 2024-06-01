import cv2 as cv


class ApplicationMode:
    """
    This class is used to manage how the application behaves.
    It has the following modes:
        - '1' : free camera mode
        - '2' : save static landmarks mode(letter)
        - '3' : save dynamic landmarks mode(word)
        - '4' : detect static signs mode(letter)
        - '5' : detect dynamic signs mode(word)
        - 'q' : quit application
        - 's' : enable/disable sentence mode
        - 'l' : enable/disable landmarks
    Also, this class provides methods that help and/or change
    the flow of data in the application.
    """

    def __init__(self) -> None:
        """
        Initialize necessary variables for the class and some
        colors that will be used in editing the frame GUI.
        Sets the application mode with the default mode '1'.
        Landmarks are not shown by default.
        Sentence mode is not available by default.
        """
        self.MODE = '1'
        self.SHOW_LANDMARKS = False
        self.SENTENCE_MODE = False

        # control attributes for sentence mode
        self.INSERT_DELAY = 30  # 30 frames delay
        self.WORD = ""
        self.SENTENCE = ""
        self.MAX_WORD_LENGTH = 30
        self.MAX_SENTENCE_LENGTH = 60

        # colors:
        self.purple = (128, 0, 128)
        self.white = (255, 255, 255)

    def get_app_mode(self, key_input):
        """
        This method is used to change the application mode
        based on the user's key input.
        :param key_input: int, unicode code of the key pressed
        """
        if key_input == ord('1'):
            self.SENTENCE_MODE = False
            self.MODE = '1'
        elif key_input == ord('2'):
            self.SENTENCE_MODE = False
            self.MODE = '2'
        elif key_input == ord('3'):
            self.SENTENCE_MODE = False
            self.MODE = '3'
        elif key_input == ord('4'):
            self.MODE = '4'
            self.INSERT_DELAY = 30
        elif key_input == ord('5'):
            self.MODE = '5'
            self.INSERT_DELAY = 30
        elif key_input == ord('q'):
            self.MODE = 'q'
        elif key_input == ord('s') and self.MODE in {'4', '5'}:
            self.SENTENCE_MODE = not self.SENTENCE_MODE
        elif key_input == ord('l'):
            self.SHOW_LANDMARKS = not self.SHOW_LANDMARKS
        else:
            self.MODE = self.MODE
            self.SHOW_LANDMARKS = self.SHOW_LANDMARKS
            self.SENTENCE_MODE = self.SENTENCE_MODE

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

            if self.MODE == '3' and dm.SEQUENCE_ONGOING:
                text = text + f", frame: {dm.current_sequence_frame}/{dm.number_of_frames_per_sequence}"

        else:
            text = f"Save {data_type} Landmarks Mode(letter)"

        return text

    def set_sentence_mode(self, frame):
        """
        This method is used to display the sentence mode on the GUI of the frame.
        :param frame: np.array, the frame to be displayed.
        :return: frame: np.Array, the frame with the sentence mode displayed on it.
        """
        # draw box at the bottom of the frame
        cv.rectangle(frame, (0, frame.shape[0]), (frame.shape[1], frame.shape[0] - 50),
                     (255, 255, 255), -1)

        # display sentence at the bottom of the frame
        cv.putText(frame, f'sentence: {self.SENTENCE}',
                   (10, frame.shape[0] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   self.purple, 1, cv.LINE_AA)

        # display word above sentence
        cv.putText(frame, f'word: {self.WORD}',
                   (10, frame.shape[0] - 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   self.purple, 1, cv.LINE_AA)

        return frame

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

        if self.MODE == '1':
            text = "Free Camera Mode"

        elif self.MODE == '2':
            text = self.set_text_for_save_mode(key_input, data_manipulator_static,
                                               'static')

        elif self.MODE == '3':
            text = self.set_text_for_save_mode(key_input, data_manipulator_dynamic,
                                               'dynamic')

        elif self.MODE == '4':
            text = "Detect Static Signs Mode(letter)"
            if self.SENTENCE_MODE:
                frame = self.set_sentence_mode(frame)

        elif self.MODE == '5':
            text = "Detect Dynamic Signs Mode(word)"
            if self.SENTENCE_MODE:
                frame = self.set_sentence_mode(frame)

        elif self.MODE == 'q':
            text = "Quit Application"

        text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv.putText(frame, text, (int((frame.shape[1] - text_size[0]) / 2), 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, self.purple, 1, cv.LINE_AA)

        return frame

    def create_word(self, label):
        """
        This method is used to create a word from the detected signs.
        :param label: str, the detected sign label.
        :return: None
        """
        # FIXME needs implementation for word actions
        if self.INSERT_DELAY == 0:
            if len(self.WORD) < self.MAX_WORD_LENGTH:
                self.WORD += label
                if self.MODE == '5':
                    self.WORD += " "
                self.INSERT_DELAY = 30
        else:
            self.INSERT_DELAY -= 1

    def create_sentence(self, label, dm):  # TODO this is implemented using the custom signs
        """
        This method is used to create a sentence from the detected words.
        :param label:
        :param dm: DataManipulator, Static or Dynamic.
        :return: None
        """
        if dm.WORD != "":
            dm.SENTENCE += dm.WORD + " "
            dm.WORD = ""
