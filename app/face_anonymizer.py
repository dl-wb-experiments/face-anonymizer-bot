from abc import ABCMeta, abstractmethod
import cv2
import numpy as np

from app.constants import Emotion
from app.engines import EmotionRecognizer, FaceDetector
from app.utils import prepare_output_video_stream


class BaseFaceAnonymizer(metaclass=ABCMeta):
    _face_detector = FaceDetector()

    @classmethod
    def process_single_frame(cls, original_frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def load_image(cls, image_path: str) -> np.ndarray:
        # Use OpenCV to load the input image
        return cv2.imread(image_path)

    @classmethod
    @abstractmethod
    def process_image(cls, input_file_path: str, output_file_path: str):
        original_image = cls.load_image(input_file_path)

        processed_image = cls.process_single_frame(original_image) if cls._face_detector.is_ready else original_image

        cv2.imwrite(filename=output_file_path, img=processed_image)

    @staticmethod
    def pre_process_input_image(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        # Resize the image dimensions from image to model input w x h
        resized_image = cv2.resize(image, (target_width, target_height))

        # Change data layout from HWC to CHW
        transposed_image = resized_image.transpose((2, 0, 1))

        n = 1  # Batch is always 1 in our case
        c = 3  # Channels is always 3 in our case

        # Reshape to input dimensions
        reshaped_image = transposed_image.reshape((n, c, target_height, target_width))
        return reshaped_image

    @classmethod
    def face_detector_inference(cls, image: np.ndarray) -> np.ndarray:
        *_, face_detector_width, face_detector_height = cls._face_detector.input_shape

        # 1. Prepare the image
        pre_processed_image = cls.pre_process_input_image(image,
                                                          face_detector_height,
                                                          face_detector_width)

        # 2. Infer the model
        face_detection_inference_results = cls._face_detector.inference(pre_processed_image)

        return face_detection_inference_results

    @classmethod
    def parse_face_detection_results(cls, inference_results: np.ndarray,
                                     original_image_width: int,
                                     original_image_height: int,
                                     prob_threshold: float = 60) -> list:

        # Prepare a list to save the detected faces
        detected_faces = []

        # Iterate through all the detected faces
        for inference_result in inference_results[0][0]:

            # Get the probability of the detected face and convert it to percent
            probability = inference_result[2]
            confidence = round(probability * 100, 1)

            # If confidence is more than the specified threshold, draw and label the box
            if confidence < prob_threshold:
                continue

            # Get coordinates of the box containing the detected object
            xmin = int(inference_result[3] * original_image_width)
            ymin = int(inference_result[4] * original_image_height)
            xmax = int(inference_result[5] * original_image_width)
            ymax = int(inference_result[6] * original_image_height)

            detected_face = (xmin, ymin, xmax, ymax, confidence)
            detected_faces.append(detected_face)

        return detected_faces

    @classmethod
    def process_video(cls, input_file_path: str, output_file_path: str):
        input_video_stream = cv2.VideoCapture(input_file_path)
        output_video_stream = prepare_output_video_stream(input_video_stream, output_file_path)

        try:
            while input_video_stream.isOpened():
                return_code, original_frame = input_video_stream.read()
                if not return_code:
                    break

                processed_image = cls.process_single_frame(original_frame) if cls._face_detector.is_ready else original_frame

                output_video_stream.write(processed_image)
        finally:
            input_video_stream.release()
            output_video_stream.release()


class BlurFaceAnonymizer(BaseFaceAnonymizer):

    @classmethod
    def process_single_frame(cls, original_frame: np.ndarray) -> np.ndarray:
        *_, face_detector_width, face_detector_height = cls._face_detector.input_shape

        pre_processed_image = cls.pre_process_input_image(original_frame,
                                                          face_detector_height,
                                                          face_detector_width)

        face_detection_inference_results = cls._face_detector.inference(pre_processed_image)

        processed_image = cls.blur_faces(face_detection_inference_results, original_frame)

        return processed_image

    @classmethod
    def blur_faces(cls, face_detection_inference_results: np.ndarray, original_image: np.ndarray):
        return original_image

    @classmethod
    def blur(cls, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        pixels_count = 16

        # Resize the image to pixels_count*pixels_count with interpolation to blur the image
        resized_image = cv2.resize(image, (pixels_count, pixels_count), interpolation=cv2.INTER_LINEAR)
        # Resize the image to original image size
        blurry_image = cv2.resize(resized_image, (width, height), interpolation=cv2.INTER_NEAREST)

        return blurry_image


class EmotionFaceAnonymizer(BaseFaceAnonymizer):
    _emotions = (Emotion.neutral, Emotion.happy, Emotion.sad, Emotion.surprise, Emotion.anger)
    _face_detector = FaceDetector()
    _emotion_recognizer = EmotionRecognizer()

    @classmethod
    def emotion_recognizer_inference(cls, face_frame: np.ndarray) -> np.ndarray:
        *_, emotion_recognizer_input_height, emotion_recognizer_input_width = cls._emotion_recognizer.input_shape

        prepared_frame = cls.pre_process_input_image(face_frame,
                                                     target_width=emotion_recognizer_input_width,
                                                     target_height=emotion_recognizer_input_height)

        # Run the inference the same way you did before
        inference_results = cls._emotion_recognizer.inference(prepared_frame)

        return inference_results

    @classmethod
    def process_single_frame(cls, original_frame: np.ndarray) -> np.ndarray:
        return original_frame

    @staticmethod
    def get_emoji_by_index(emotion_inference_result: np.ndarray) -> np.ndarray:
        emotions = ['neutral', 'happy', 'sad', 'surprised', 'angry']
        # Get the index of the emotion with the highest probability
        emotion_index = np.argmax(emotion_inference_result.flatten())
        emoji_path = f'./data/{emotions[emotion_index]}.png'
        return cv2.imread(emoji_path, -1)

    @staticmethod
    def put_emoji_on_top_of_face(inference_results: np.ndarray, face: np.ndarray) -> np.ndarray:
        result_face = face.copy()

        # Get width and height of the face
        height, width, _ = face.shape

        # Get an emoji by inference results
        emoji = EmotionFaceAnonymizer.get_emoji_by_index(inference_results)

        # Resize the emoji to the face shape
        resized_emoji = cv2.resize(emoji, (width, height))

        # Put the emoji over the face
        alpha_s = resized_emoji[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            result_face[:, :, c] = alpha_s * resized_emoji[:, :, c] + alpha_l * face[:, :, c]

        return result_face
