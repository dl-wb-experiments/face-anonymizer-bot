from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
# Import the Inference Engine
from openvino.inference_engine import IECore, IENetwork

from app.config import Config, OpenVINOModelPaths
from app.constants import Emotion
from app.functions import pre_process_input_image, face_detector_input_height, face_detector_input_width, \
    face_detector_inference, blur_faces, emotion_recognizer_inference, parse_face_detection_results, \
    put_emoji_on_top_of_face
from app.utils import blur, prepare_output_video_stream


class BaseFaceAnonymizer(metaclass=ABCMeta):
    _ie_core = IECore()

    _face_detection_probability_threshold = 50

    @classmethod
    def _load_image(cls, image_path: str) -> np.ndarray:
        # Use OpenCV to load the input image
        return cv2.imread(image_path)

    @classmethod
    @abstractmethod
    def process_image(cls, input_file_path: str, output_file_path: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def process_video(cls, input_file_path: str, output_file_path: str) -> None:
        raise NotImplementedError


class BlurFaceAnonymizer(BaseFaceAnonymizer):

    @classmethod
    def process_image(cls, input_file_path: str, output_file_path: str) -> None:
        original_image = cls._load_image(input_file_path)

        face_detection_inference_results = face_detector_inference(original_image)

        processed_image = blur_faces(face_detection_inference_results, original_image)

        cv2.imwrite(filename=output_file_path, img=processed_image)

    @classmethod
    def process_video(cls, input_file_path: str, output_file_path: str) -> None:
        input_video_stream = cv2.VideoCapture(input_file_path)
        output_video_stream = prepare_output_video_stream(input_video_stream, output_file_path)

        try:
            while input_video_stream.isOpened():
                return_code, original_frame = input_video_stream.read()
                if not return_code:
                    break

                face_detection_inference_results = face_detector_inference(original_frame)

                processed_image = blur_faces(face_detection_inference_results, original_frame)

                output_video_stream.write(processed_image)
        finally:
            input_video_stream.release()
            output_video_stream.release()


class EmotionFaceAnonymizer(BaseFaceAnonymizer):
    _emotions = (Emotion.neutral, Emotion.happy, Emotion.sad, Emotion.surprise, Emotion.anger)

    @classmethod
    def process_image(cls, input_file_path: str, output_file_path: str) -> None:
        original_image = cls._load_image(input_file_path)
        height, width = original_image.shape[:2]
        processed_image = original_image.copy()

        face_detection_inference_results = face_detector_inference(original_image)

        detected_faces = parse_face_detection_results(face_detection_inference_results, width, height)

        for detected_face in detected_faces:
            xmin, ymin, xmax, ymax, confidence = detected_face
            face = original_image[ymin:ymax, xmin:xmax]
            detected_emotions = emotion_recognizer_inference(face)

            processed_image[ymin:ymax, xmin:xmax] = put_emoji_on_top_of_face(detected_emotions, face)

        cv2.imwrite(filename=output_file_path, img=processed_image)

    @classmethod
    def process_video(cls, input_file_path: str, output_file_path: str) -> None:

        input_video_stream = cv2.VideoCapture(input_file_path)
        output_video_stream = prepare_output_video_stream(input_video_stream, output_file_path)

        try:
            while input_video_stream.isOpened():
                return_code, original_frame = input_video_stream.read()
                if not return_code:
                    break

                height, width = original_frame.shape[:2]
                processed_image = original_frame.copy()

                face_detection_inference_results = face_detector_inference(original_frame)

                detected_faces = parse_face_detection_results(face_detection_inference_results, width, height)

                for detected_face in detected_faces:
                    xmin, ymin, xmax, ymax, confidence = detected_face
                    face = original_frame[ymin:ymax, xmin:xmax]
                    detected_emotions = emotion_recognizer_inference(face)

                    processed_image[ymin:ymax, xmin:xmax] = put_emoji_on_top_of_face(detected_emotions, face)

                output_video_stream.write(processed_image)
        finally:
            input_video_stream.release()
            output_video_stream.release()
