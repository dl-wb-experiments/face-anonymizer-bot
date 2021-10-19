from abc import ABCMeta
import cv2
import numpy as np

from app.constants import Emotion
from app.frame_utils import blur_faces, emotion_recognition_processing, face_detector_inference
from app.utils import prepare_output_video_stream


class BaseFaceAnonymizer(metaclass=ABCMeta):
    @classmethod
    def process_single_frame(cls, original_frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def load_image(cls, image_path: str) -> np.ndarray:
        # Use OpenCV to load the input image
        return cv2.imread(image_path)

    @classmethod
    def process_image(cls, input_file_path: str, output_file_path: str):
        original_image = cls.load_image(input_file_path)

        processed_image = cls.process_single_frame(original_image)

        cv2.imwrite(filename=output_file_path, img=processed_image)

    @classmethod
    def process_video(cls, input_file_path: str, output_file_path: str):
        input_video_stream = cv2.VideoCapture(input_file_path)
        output_video_stream = prepare_output_video_stream(input_video_stream, output_file_path)

        try:
            while input_video_stream.isOpened():
                return_code, original_frame = input_video_stream.read()
                if not return_code:
                    break

                processed_image = cls.process_single_frame(original_frame)

                output_video_stream.write(processed_image)
        finally:
            input_video_stream.release()
            output_video_stream.release()


class BlurFaceAnonymizer(BaseFaceAnonymizer):

    @classmethod
    def process_single_frame(cls, original_frame: np.ndarray) -> np.ndarray:
        face_detection_inference_results = face_detector_inference(original_frame)

        processed_image = blur_faces(face_detection_inference_results, original_frame)

        return processed_image


class EmotionFaceAnonymizer(BaseFaceAnonymizer):
    _emotions = (Emotion.neutral, Emotion.happy, Emotion.sad, Emotion.surprise, Emotion.anger)

    @classmethod
    def process_single_frame(cls, original_frame: np.ndarray) -> np.ndarray:
        return emotion_recognition_processing(original_frame)
