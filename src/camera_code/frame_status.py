import cv2 as cv
import numpy as np


class FrameStatus:
    def __init__(self) -> None:
        self.sub_lists_of_signs = []
        self.selected_sub_list_index = 0
        self.selected_sub_list_sign = None
        self.real_list_index = 0

        self.THRESHOLD = 0.75
        self.WORD = ''
        self.SENTENCE = ''
        self.sentence_index = 0
        self.MODE = None

        self.DOT = [.5, .5]  # center of frame for the dot

        self.purple = (153, 0, 153)
        self.blue = (255, 255, 0)

    def set_status_mode(self, key_input):
        if key_input == ord('s'):
            self.MODE = 's'  # save mode
            self.DOT = [0, 0]
        elif key_input == ord('d'):
            self.MODE = 'd'  # detect mode
        elif key_input == ord('w'):
            self.MODE = 'w'  # word mode
        elif key_input == ord('f'):
            self.MODE = 'f'  # free camera mode
            self.DOT = [0, 0]

    def get_sign_from_key_input(self, key_input):
        if self.MODE == 's':
            if ord('0') <= key_input <= ord('9'):
                self.selected_sub_list_sign = int(chr(key_input))
        else:
            self.selected_sub_list_sign = None

    def move_between_sub_lists(self, key_input):
        if key_input == ord('>') and self.selected_sub_list_index < (len(self.sub_lists_of_signs) - 1):
            self.selected_sub_list_index += 1
            self.selected_sub_list_sign = 0
        if key_input == ord('<') and self.selected_sub_list_index > 0:
            self.selected_sub_list_index -= 1
            self.selected_sub_list_sign = 0

    def set_real_list_index(self):
        self.real_list_index = 10 * self.selected_sub_list_index + self.selected_sub_list_sign

    def display_word_and_sentence(self, image):
        # draw box at the bottom of the frame
        cv.rectangle(image,
                     (0, image.shape[0]), (image.shape[1], image.shape[0] - 50),
                     (255, 255, 255), -1)

        # display sentence at the bottom of the frame
        cv.putText(image, f'sentence: {self.SENTENCE}',
                   (10, image.shape[0] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, self.purple, 1, cv.LINE_AA)

        # display word above sentence
        cv.putText(image, f'word: {self.WORD}',
                   (10, image.shape[0] - 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, self.purple, 1, cv.LINE_AA)

        return image

    def set_status_text(self, image, key_input, hands):
        text = ''
        list_of_signs_counted = hands.count_number_of_saved_landmarks()
        cv.rectangle(image, (0, 0), (image.shape[1], 20), (255, 255, 255), -1)

        if self.MODE == 'd':
            text = 'Detect mode'
        elif self.MODE == 'w':
            text = 'Word mode'
            image = self.display_word_and_sentence(image)
        elif self.MODE == 's':
            if self.selected_sub_list_sign is not None and list_of_signs_counted is not None:
                if 0 <= self.selected_sub_list_sign < len(self.sub_lists_of_signs[self.selected_sub_list_index]):
                    self.move_between_sub_lists(key_input=key_input)
                    self.set_real_list_index()
                    text = f'Saving landmarks for sign: {self.sub_lists_of_signs[self.selected_sub_list_index][self.selected_sub_list_sign]}' \
                           f'({list_of_signs_counted[self.real_list_index]}), page[{self.selected_sub_list_index + 1}/{len(self.sub_lists_of_signs)}]'
                else:
                    text = 'Invalid sign'
            else:
                text = 'Saving landmarks mode'
        else:
            text = 'Free camera mode'

        text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv.putText(image, text, (int((image.shape[1] - text_size[0]) / 2), 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, self.purple, 1, cv.LINE_AA)

        return image

    def create_word_from_signs(self, label, insert_letter):
        if insert_letter.value == True and self.MODE == 'w':
            # return if the detected sign is not relevant to word creation
            non_relevant_labels = ('up',
                                   'down',
                                   'left',
                                   'right',
                                   'delete_sentence',
                                   'delete_word_from_sentence',
                                   'add_word_to_sentence')
            if label in non_relevant_labels:
                return

            if label == 'delete_letter_from_active_word' and len(self.WORD) != 0:
                self.WORD = self.WORD[:-1]
                insert_letter.value = False
                return

            if label == 'delete_active_word':
                self.WORD = ''
                insert_letter.value = False
                return

            self.WORD += label
            insert_letter.value = False

    def create_sentence_from_words(self, label, insert_letter):
        if insert_letter.value == True and self.MODE == 'w':
            # select word from sentence
            if label in ('up', 'down') and self.SENTENCE != '':
                # reset any selected words
                self.SENTENCE = self.SENTENCE.lower()
                sentence_as_list = self.SENTENCE.split(' ')

                # move the selection
                if label == 'up' and self.sentence_index < len(sentence_as_list) - 1:
                    self.sentence_index += 1
                elif label == 'down' and self.sentence_index > 0:
                    self.sentence_index -= 1

                # create sentence with selected word
                sentence_as_list[self.sentence_index] = sentence_as_list[self.sentence_index].upper()
                self.SENTENCE = ' '.join(sentence_as_list)
                insert_letter.value = False

            if label == 'delete_word_from_sentence' and self.SENTENCE != '':
                sentence_as_list = self.SENTENCE.split(' ')
                sentence_as_list.pop(self.sentence_index)

                self.sentence_index = 0
                self.SENTENCE = ' '.join(sentence_as_list)
                insert_letter.value = False

            if label == 'delete_sentence':
                self.sentence_index = 0
                self.SENTENCE = ''
                insert_letter.value = False

            if label == 'add_word_to_sentence' and self.WORD != '':
                self.sentence_index = 0
                self.SENTENCE = self.SENTENCE.lower()
                self.SENTENCE += self.WORD if self.SENTENCE == '' else ' ' + self.WORD
                self.WORD = ''
                insert_letter.value = False

    def manage_prediction(self, image, landmarks_dict, prediction, hands, insert_letter, process):
        if self.MODE in ('d', 'w'):
            # draw rectangle around hand
            min_x, min_y, max_x, max_y = hands.find_min_and_max_for_x_and_y(landmarks_dict)
            cv.rectangle(image,
                         (int(min_x * image.shape[1] - 10), int(min_y * image.shape[0] - 10)),
                         # TODO: why * image.shape[1] and * image.shape[0] and not reverse ???
                         (int(max_x * image.shape[1] + 10), int(max_y * image.shape[0] + 10)),
                         self.purple, 2)

            if np.max(prediction) > self.THRESHOLD:
                label, accuracy = hands.sign_labels[np.argmax(prediction)], np.max(prediction)

                if process.is_alive() == False:
                    process.start()

                self.create_sentence_from_words(label, insert_letter)
                self.create_word_from_signs(label, insert_letter)
            else:
                label, accuracy = 'Unknown sign', np.prod(1 - prediction)  # FIXME ???

            # display the label and accuracy
            cv.putText(image, f'prediction: {label} ({accuracy:.2f})',
                       (int(min_x * image.shape[1]), int(min_y * image.shape[0]) - 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, self.purple, 1, cv.LINE_AA)
        else:
            if process.is_alive():
                process.terminate()

        return image
