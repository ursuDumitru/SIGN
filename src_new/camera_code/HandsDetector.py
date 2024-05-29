import mediapipe as mp
import cv2 as cv


class HandsDetector:
    """
    This class is used to detect hands in a frame using the mediapipe framework.
    It has the following attributes:
        - mp_hands: mediapipe.solutions.hands, the hands module from the mediapipe framework
        - mp_drawing: mediapipe.solutions.drawing_utils, the drawing_utils module from the mediapipe framework
        - model: mediapipe.solutions.hands.Hands, the hands model from the mediapipe framework
    """
    def __init__(self, min_detection_confidence, min_tracking_confidence, max_num_hands) -> None:
        """
        Initialize the HandsDetector object.
        :param min_detection_confidence: float, the minimum confidence value for hand detection
        :param min_tracking_confidence: float, the minimum confidence value for hand tracking
        :param max_num_hands: int, the maximum number of hands to detect
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.model = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=max_num_hands
        )

        # colors
        self.purple = (128, 0, 128)
        self.white = (255, 255, 255)

    def mediapipe_hands_detect(self, frame):
        """
        This method is used to detect hands in a frame using the mediapipe hands model.
        :param frame: np.array, one frame from the video feed used to detect the hands.
        :return mediapipe_results: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList,
        the landmarks of the hands.
        """
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame.flags.writeable = False
        mediapipe_results = self.model.process(frame)

        return mediapipe_results

    def draw_hands_landmarks(self, frame, mediapipe_results):
        """
        This method is used to draw the landmarks of the hands on the frame.
        :param frame: np.array, one frame from the video feed.
        :param mediapipe_results: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList,
        the landmarks of the hands.
        :return frame: np.array, one frame with the landmarks drawn on it.
        """
        if mediapipe_results.multi_hand_landmarks:
            for hand_landmarks in mediapipe_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=self.purple, thickness=2, circle_radius=1),
                    self.mp_drawing.DrawingSpec(color=self.white, thickness=2, circle_radius=1))

        return frame
