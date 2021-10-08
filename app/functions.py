# Define the function to pre-process (resize, transpose) the input image
import cv2
import numpy as np
from .ie_utils import *

prob_threshold = 50


def pre_process_input_image(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    return image


def parse_face_detection_results(inference_results: np.ndarray, original_image_width: int,
                                 original_image_height: int) -> list:
    return []


def blur(image: np.ndarray) -> np.ndarray:
    return image


def face_detector_inference(image: np.ndarray) -> np.ndarray:
    return np.array([])


def blur_faces(face_detection_inference_result: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    return original_image


def emotion_recognizer_inference(face_frame: np.ndarray) -> np.ndarray:
    return np.array([])


def get_emoji_by_index(emotion_inference_result: np.ndarray) -> np.ndarray:
    return np.array([])


def put_emoji_on_top_of_face(inference_results: np.ndarray, face: np.ndarray) -> np.ndarray:
    return face
