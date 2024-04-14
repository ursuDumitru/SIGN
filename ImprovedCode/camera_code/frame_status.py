import cv2 as cv
import numpy as np
import time


class FrameStatus:
    def __init__(self) -> None:
        self.sub_lists_of_signs = []
        self.selected_sub_list_index = 0
        self.selected_sub_list_sign = None
        self.real_list_index = 0
        
        self.THRESHOLD = 0.75
        self.WORD = ''
        self.SENTENCE = ''
        self.MODE = None
        
        self.DOT = [.5, .5] # center of frame
        pass


    def set_status_mode(self, key_input):
        if key_input == ord('s'):
            self.MODE = 's' # save mode
            self.DOT = [0, 0]
        elif key_input == ord('d'):
            self.MODE = 'd' # detect mode
        elif key_input == ord('w'):
            self.MODE = 'w' # word mode
        elif key_input == ord('f'):
            self.MODE = 'f' # free camera mode
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


    def set_status_text(self, image, key_input, hands):
        list_of_signs_counted = hands.count_number_of_saved_landmarks()
        text = ''

        cv.rectangle(image, (0, 0), (image.shape[1], 35), (255,255,255), -1)
        if self.MODE == 'd':
            text = 'Detect mode'
        elif self.MODE == 'w':
            text = 'Word mode'
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
        
        text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        cv.putText(image, text, (int((image.shape[1] - text_size[0]) / 2), 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (153,0,153), 2, cv.LINE_AA)

        return image
    
    
    # TODO: make it to be a MODE
    def move_dot(self, label, image):
        # move the dot according to the sign detected
        if label == self.hands.sign_labels[0] and self.DOT[1] >= 0.01:
            self.DOT[1] += -.01
        elif label == self.hands.sign_labels[1] and self.DOT[1] <= 0.99:
            self.DOT[1] += .01
        elif label == self.hands.sign_labels[2] and self.DOT[0] <= 0.99:
            self.DOT[0] += .01
        elif label == self.hands.sign_labels[3] and self.DOT[0] >= 0.01:
            self.DOT[0] += -.01
        
        cv.circle(image, (int(self.DOT[0] * image.shape[1]), int(self.DOT[1] * image.shape[0])), 5, (255,255,153), -1)
        
        return image


    def create_word_from_signs(self, label, insert_letter, process):
        if process.is_alive() == False:
            process.start()

        if insert_letter.value == True and self.MODE == 'w':
            if label != 'z': # FIXME: change from 'z'
                self.WORD += label
                insert_letter.value = False

    def create_sentence_from_words(self, hands, prediction):
        label, accuracy = hands.sign_labels[np.argmax(prediction)], np.max(prediction)
        if label == 'z' and accuracy > self.THRESHOLD and self.WORD != '': # FIXME: change from 'z'
            self.SENTENCE += self.WORD + ' '
            self.WORD = ''

    
    def display_word_and_sentence(self, image, hands, landmarks_dict):
        if self.MODE == 'w':
            min_x, min_y, _, _ = hands.find_min_and_max_for_x_and_y(landmarks_dict)

            # display word above detexted sign
            cv.putText(image, f'word: {self.WORD}',
                       (int(min_x * image.shape[1]), int(min_y * image.shape[0]) - 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2, cv.LINE_AA)

            # display sentence at the bottom of the frame
            cv.rectangle(image,
                         (0, image.shape[0]), (image.shape[1], image.shape[0] - 35),
                         (255,255,255), -1)
            cv.putText(image, f'sentence: {self.SENTENCE}',
                       (10 , image.shape[0] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2, cv.LINE_AA)
        
        return image


    def draw_rectangle_around_hand(self, image, landmarks_dict, prediction, hands, insert_letter, process):
        if self.MODE in ('d', 'w'):
            min_x, min_y, max_x, max_y = hands.find_min_and_max_for_x_and_y(landmarks_dict)
            cv.rectangle(image,
                        (int(min_x * image.shape[1] - 10), int(min_y * image.shape[0] - 10)), # TODO: why * image.shape[1] and * image.shape[0] and not reverse ???
                        (int(max_x * image.shape[1] + 10), int(max_y * image.shape[0] + 10)),
                        (153,0,153), 2)

            if np.max(prediction) > self.THRESHOLD:
                label, accuracy = hands.sign_labels[np.argmax(prediction)], np.max(prediction)
                self.create_sentence_from_words(hands, prediction)
                self.create_word_from_signs(label, insert_letter, process)
            else:
                label, accuracy = 'Unknown sign', np.prod(1 - prediction) # FIXME if I change how the probabilities are calculated, I need to change this too
            
            # image = self.move_dot(label, image)

            # display the label and accuracy
            cv.putText(image, f'prediction: {label} ({accuracy:.2f})',
                       (int(min_x * image.shape[1]), int(min_y * image.shape[0]) - 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, (153,0,153), 2, cv.LINE_AA)
        else:
            if process.is_alive() == True:
                process.terminate()

        return image
