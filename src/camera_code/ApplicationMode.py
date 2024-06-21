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
        self.SENTENCE_MOVE_INDEX = -1  # FIXME, has to be reset somewhere
        self.MAX_WORD_LENGTH = 30
        self.MAX_SENTENCE_LENGTH = 60

        # colors:
        self.purple = (128, 0, 128)
        self.white = (255, 255, 255)

        self.TAKE = 0

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
        :param data_manipulator_static: DataManipulatorStatic, object that stores static data.
        :param data_manipulator_dynamic: DataManipulatorStatic, object that stores dynamic data.
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

    def add_to_word(self, label):
        """
        This method is used to add a letter/entire word to the active word attribute.
        :param label: str, the detected sign label.
        :return:
        """
        if len(self.WORD) < self.MAX_WORD_LENGTH:
            self.WORD += label
            if self.MODE == '5':
                self.WORD += " "
            self.INSERT_DELAY = 30

    def delete_letter_from_word(self):
        """
        This method is used to delete the last letter from the active word.
        :return:
        """
        if len(self.WORD) > 0:
            self.WORD = self.WORD[:-1]
            self.INSERT_DELAY = 30

    def delete_word(self):
        """
        This method is used to delete the entire active word.
        :return:
        """
        self.WORD = ""
        self.INSERT_DELAY = 30

    def create_word(self, label, accepted_word_labels):
        """
        This method is used to create a word from the detected signs.
        :param accepted_word_labels: list, accepted sign labels for the word.
        :param label: str, the detected sign label.
        :return: None
        """
        if self.INSERT_DELAY == 0:
            if label in accepted_word_labels:
                self.add_to_word(label)
            elif label == "delete_letter_from_word":
                self.delete_letter_from_word()
            elif label == "delete_word":
                self.delete_word()
            # if the detected sign is not in the accepted_word_labels do nothing
            else:
                return
        else:
            self.INSERT_DELAY -= 1

    def add_word_to_sentence(self):
        """
        This method is used to add the active word to the sentence.
        :return:
        """
        if len(self.WORD) > 0 and len(self.SENTENCE) + len(self.WORD) < self.MAX_SENTENCE_LENGTH:
            if " " not in self.WORD:
                self.WORD += " "
            self.SENTENCE += self.WORD
            self.WORD = ""
            self.INSERT_DELAY = 30
            self.SENTENCE_MOVE_INDEX = -1

    def move_to_left_word(self):
        """
        This method is used to move to the next left word in the sentence.
        :return:
        """
        buff_split_sentence = self.SENTENCE.lower().split(" ")
        if len(buff_split_sentence) >= 1 and self.SENTENCE_MOVE_INDEX >= 0:
            self.INSERT_DELAY = 30
            self.SENTENCE_MOVE_INDEX -= 1
            if self.SENTENCE_MOVE_INDEX > -1:
                buff_split_sentence[self.SENTENCE_MOVE_INDEX] = buff_split_sentence[self.SENTENCE_MOVE_INDEX].upper()
            self.SENTENCE = " ".join(buff_split_sentence)

    def move_to_right_word(self):
        """
        This method is used to move to the next right word in the sentence.
        :return:
        """
        buff_split_sentence = self.SENTENCE.lower().split(" ")
        if len(buff_split_sentence) >= 1 and self.SENTENCE_MOVE_INDEX < len(buff_split_sentence) - 1:
            self.INSERT_DELAY = 30
            self.SENTENCE_MOVE_INDEX += 1
            if self.SENTENCE_MOVE_INDEX < len(buff_split_sentence):
                buff_split_sentence[self.SENTENCE_MOVE_INDEX] = buff_split_sentence[self.SENTENCE_MOVE_INDEX].upper()
            self.SENTENCE = " ".join(buff_split_sentence)

    def select_word_from_sentence(self):
        """
        This method is used to select the highlighted word from the sentence.
        :return:
        """
        buff_split_sentence = self.SENTENCE.lower().split(" ")
        if len(buff_split_sentence) > 0 and self.SENTENCE_MOVE_INDEX in range(0, len(buff_split_sentence) - 1):
            self.WORD = buff_split_sentence[self.SENTENCE_MOVE_INDEX]
            buff_split_sentence.pop(self.SENTENCE_MOVE_INDEX)
            self.SENTENCE = " ".join(buff_split_sentence)
            self.INSERT_DELAY = 30
            self.SENTENCE_MOVE_INDEX = -1

    def delete_word_from_sentence(self):
        """
        This method is used to delete the highlighted word from the sentence.
        :return:
        """
        buff_split_sentence = self.SENTENCE.lower().split(" ")
        if len(buff_split_sentence) > 0 and self.SENTENCE_MOVE_INDEX in range(0, len(buff_split_sentence) - 1):
            buff_split_sentence.pop(self.SENTENCE_MOVE_INDEX)
            self.SENTENCE = " ".join(buff_split_sentence)
            self.INSERT_DELAY = 30
            self.SENTENCE_MOVE_INDEX = -1

    def delete_sentence(self):
        """
        This method is used to delete the entire sentence.
        :return:
        """
        self.SENTENCE = ""
        self.INSERT_DELAY = 30
        self.SENTENCE_MOVE_INDEX = -1

    def create_sentence(self, label):
        """
        This method is used to create a sentence from the detected words.
        :param label: str, the detected word label.
        :return: None
        """
        if self.INSERT_DELAY == 0:
            if label == "add_word_to_sentence":
                self.add_word_to_sentence()
            if label == "move_to_left_word" and self.WORD == "":  # only if active word is empty
                self.move_to_left_word()
            if label == "move_to_right_word" and self.WORD == "":  # only if active word is empty
                self.move_to_right_word()
            if label == "select_word_from_sentence" and self.WORD == "":
                self.select_word_from_sentence()
            if label == "delete_word_from_sentence" and self.WORD == "":
                self.delete_word_from_sentence()
            elif label == "delete_sentence":
                self.delete_sentence()
            else:
                return
        else:
            self.INSERT_DELAY -= 1
