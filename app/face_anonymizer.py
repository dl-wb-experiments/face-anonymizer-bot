from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
# Import the Inference Engine
from openvino.inference_engine import IECore, IENetwork

from app.config import Config, OpenVINOModelPaths
from app.constants import Emotion
from app.utils import blur, prepare_output_video_stream


class BaseFaceAnonymizer(metaclass=ABCMeta):
    _ie_core = IECore()

    _face_detection_probability_threshold = 50

    @classmethod
    def _load_image(cls, image_path: str) -> np.ndarray:
        # Use OpenCV to load the input image
        return cv2.imread(image_path)

    @classmethod
    def _read_openvino_model(cls, model_path: OpenVINOModelPaths) -> IENetwork:
        model_xml = model_path['xml_path']
        model_bin = model_path['bin_path']
        return cls._ie_core.read_network(model=model_xml, weights=model_bin)

    @classmethod
    def _pre_process_input_image(cls, image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
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
    def _infer_network(cls, image: np.ndarray, network_model: IENetwork) -> np.ndarray:
        network_input_name = next(iter(network_model.input_info))
        network_output_name = next(iter(network_model.outputs))

        executable_network = cls._ie_core.load_network(network=network_model, device_name=Config.device)

        # Read the input dimensions: n=batch size, c=number of channels, h=height, w=width
        network_input_shape = network_model.input_info[network_input_name].input_data.shape
        n, c, network_input_height, network_input_width = network_input_shape

        # 1. Prepare the image
        input_frame = cls._pre_process_input_image(image=image,
                                                   target_height=network_input_height,
                                                   target_width=network_input_width)

        # 2. Infer the model
        network_inference_results = executable_network.infer(inputs={network_input_name: input_frame})
        return network_inference_results[network_output_name]

    @classmethod
    def _parse_face_detection_results(cls, inference_results: np.ndarray, original_image_width: int,
                                      original_image_height: int) -> list:
        # Prepare list to save detected faces
        detected_faces = []

        # Iterate through all detected faces
        for inference_result in inference_results[0][0]:

            # Get the probability of the detected face and convert it to percent
            probability = inference_result[2]
            confidence = round(probability * 100, 1)

            # If confidence is more than the specified threshold, draw and label the box
            if confidence < cls._face_detection_probability_threshold:
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
    @abstractmethod
    def process_image(cls, input_file_path: str, output_file_path: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def process_video(cls, input_file_path: str, output_file_path: str) -> None:
        raise NotImplementedError


class BlurFaceAnonymizer(BaseFaceAnonymizer):
    @classmethod
    def _blur_faces(cls, face_detection_inference_result: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        # Get original image size
        original_image_height, original_image_width, _ = original_image.shape

        # Prepare result image to not affect the original image
        processed_image = original_image.copy()

        detected_faces = cls._parse_face_detection_results(face_detection_inference_result, original_image_width,
                                                           original_image_height)

        for detected_face in detected_faces:
            xmin, ymin, xmax, ymax, _ = detected_face
            # Get the face from the original image
            face = original_image[ymin:ymax, xmin:xmax]
            # Blur face using blur function
            blurry_face = blur(face)
            # replace the face to the blurry face in the result image
            processed_image[ymin:ymax, xmin:xmax] = blurry_face

        return processed_image

    @classmethod
    def process_image(cls, input_file_path: str, output_file_path: str) -> None:
        face_detection_model = cls._read_openvino_model(Config.face_detection_model)

        original_image = cls._load_image(input_file_path)

        inference_result = cls._infer_network(image=original_image, network_model=face_detection_model)

        processed_image = cls._blur_faces(face_detection_inference_result=inference_result,
                                          original_image=original_image)

        cv2.imwrite(filename=output_file_path, img=processed_image)

    @classmethod
    def process_video(cls, input_file_path: str, output_file_path: str) -> None:
        face_detection_model = cls._read_openvino_model(Config.face_detection_model)

        input_video_stream = cv2.VideoCapture(input_file_path)
        output_video_stream = prepare_output_video_stream(input_video_stream, output_file_path)

        try:
            while input_video_stream.isOpened():
                return_code, original_frame = input_video_stream.read()
                if not return_code:
                    break

                inference_result = cls._infer_network(image=original_frame, network_model=face_detection_model)

                processed_image = cls._blur_faces(face_detection_inference_result=inference_result,
                                                  original_image=original_frame)

                output_video_stream.write(processed_image)
        finally:
            input_video_stream.release()
            output_video_stream.release()


class EmotionFaceAnonymizer(BaseFaceAnonymizer):
    _emotions = (Emotion.neutral, Emotion.happy, Emotion.sad, Emotion.surprise, Emotion.anger)

    @classmethod
    def _get_emotion_image(cls, emotion_index: int) -> np.ndarray:
        emotion = cls._emotions[emotion_index]
        emotion_image_path = Config.emotions_images[emotion]
        return cv2.imread(filename=str(emotion_image_path), flags=cv2.IMREAD_UNCHANGED)

    @classmethod
    def _emotion_recognition_inference_postprocess(cls, image: np.ndarray, emotion_recognition_results: np.ndarray,
                                                   xmin: int, ymin: int, xmax: int, ymax: int):
        width = xmax - xmin
        height = ymax - ymin

        x_center = xmax - xmin
        x_delta = int(height / 2)

        # xmin = max(x_center - x_delta, 0)
        # max_image_width = image.shape[1]
        # xmax = min(xmin + height, max_image_width)

        emotion_index = np.argmax(emotion_recognition_results.flatten())
        emotion_image = cls._get_emotion_image(emotion_index)
        resized_emotion_image = cv2.resize(emotion_image, (width, height))
        # resized_emotion_image = cv2.resize(emotion_image, (height, height))

        alpha_s = resized_emotion_image[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            image[ymin:ymax, xmin:xmax, c] = (
                    alpha_s * resized_emotion_image[:, :, c] + alpha_l * image[ymin:ymax, xmin:xmax, c])

    @classmethod
    def _add_emotions(cls, face_detection_inference_result: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        # Get original image size
        original_image_height, original_image_width, _ = original_image.shape

        # Prepare result image to not affect the original image
        processed_image = original_image.copy()

        detected_faces = cls._parse_face_detection_results(face_detection_inference_result, original_image_width,
                                                           original_image_height)

        # TODO Move outside of this method + consider using singleton pattern
        emotion_recognition_model = cls._read_openvino_model(Config.emotion_recognition_model)

        for detected_face in detected_faces:
            xmin, ymin, xmax, ymax, _ = detected_face
            # Add emotion image to recognized face
            emotion_recognition_results = cls._infer_network(image=original_image,
                                                             network_model=emotion_recognition_model)
            cls._emotion_recognition_inference_postprocess(image=processed_image,
                                                           emotion_recognition_results=emotion_recognition_results,
                                                           xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

        return processed_image

    @classmethod
    def process_image(cls, input_file_path: str, output_file_path: str) -> None:
        face_detection_model = cls._read_openvino_model(Config.face_detection_model)

        original_image = cls._load_image(input_file_path)

        inference_result = cls._infer_network(image=original_image, network_model=face_detection_model)

        processed_image = cls._add_emotions(face_detection_inference_result=inference_result,
                                            original_image=original_image)

        cv2.imwrite(filename=output_file_path, img=processed_image)

    @classmethod
    def process_video(cls, input_file_path: str, output_file_path: str) -> None:
        face_detection_model = cls._read_openvino_model(Config.face_detection_model)

        input_video_stream = cv2.VideoCapture(input_file_path)
        output_video_stream = prepare_output_video_stream(input_video_stream, output_file_path)

        try:
            while input_video_stream.isOpened():
                return_code, original_frame = input_video_stream.read()
                if not return_code:
                    break

                face_detection_inference_result = cls._infer_network(image=original_frame,
                                                                     network_model=face_detection_model)

                processed_image = cls._add_emotions(face_detection_inference_result=face_detection_inference_result,
                                                    original_image=original_frame)

                output_video_stream.write(processed_image)
        finally:
            input_video_stream.release()
            output_video_stream.release()
