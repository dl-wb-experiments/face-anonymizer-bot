import cv2
import numpy as np

from app.engines import EmotionRecognizer, FaceDetector


def blur_faces(face_detection_inference_results: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    return original_image


def emotion_recognition_processing(original_frame: np.ndarray) -> np.ndarray:
    return original_frame


def face_detector_inference(image: np.ndarray) -> np.ndarray:
    face_detector = FaceDetector()
    *_, face_detector_width, face_detector_height = face_detector.input_shape

    # 1. Prepare the image
    pre_processed_image = pre_process_input_image(image,
                                                  face_detector_height,
                                                  face_detector_width)

    # 2. Infer the model
    face_detection_inference_results = face_detector.inference(pre_processed_image)

    return face_detection_inference_results


def emotion_recognizer_inference(cls, face_frame: np.ndarray) -> np.ndarray:
    emotion_recognizer = EmotionRecognizer()
    *_, emotion_recognizer_input_height, emotion_recognizer_input_width = emotion_recognizer.input_shape

    prepared_frame = pre_process_input_image(face_frame,
                                             target_width=emotion_recognizer_input_width,
                                             target_height=emotion_recognizer_input_height)

    # Run the inference the same way you did before
    inference_results = cls._emotion_recognizer.inference(prepared_frame)

    return inference_results


def blur(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    pixels_count = 16

    # Resize the image to pixels_count*pixels_count with interpolation to blur the image
    resized_image = cv2.resize(image, (pixels_count, pixels_count), interpolation=cv2.INTER_LINEAR)
    # Resize the image to original image size
    blurry_image = cv2.resize(resized_image, (width, height), interpolation=cv2.INTER_NEAREST)

    return blurry_image


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


def get_emoji_by_index(emotion_inference_result: np.ndarray) -> np.ndarray:
    emotions = ['neutral', 'happy', 'sad', 'surprised', 'angry']
    # Get the index of the emotion with the highest probability
    emotion_index = np.argmax(emotion_inference_result.flatten())
    emoji_path = f'./data/{emotions[emotion_index]}.png'
    return cv2.imread(emoji_path, -1)


def put_emoji_on_top_of_face(inference_results: np.ndarray, face: np.ndarray) -> np.ndarray:
    result_face = face.copy()

    # Get width and height of the face
    height, width, _ = face.shape

    # Get an emoji by inference results
    emoji = get_emoji_by_index(inference_results)

    # Resize the emoji to the face shape
    resized_emoji = cv2.resize(emoji, (width, height))

    # Put the emoji over the face
    alpha_s = resized_emoji[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        result_face[:, :, c] = alpha_s * resized_emoji[:, :, c] + alpha_l * face[:, :, c]

    return result_face


def parse_face_detection_results(inference_results: np.ndarray,
                                 original_image_width: int,
                                 original_image_height: int,
                                 prob_threshold: float = 0.6) -> list:
    # Prepare a list to save the detected faces
    detected_faces = []

    # Iterate through all the detected faces
    for inference_result in inference_results[0][0]:

        # Get the probability of the detected face and convert it to percent
        probability = inference_result[2]

        # If confidence is more than the specified threshold, draw and label the box
        if probability < prob_threshold:
            continue

        # Get coordinates of the box containing the detected object
        xmin = int(inference_result[3] * original_image_width)
        ymin = int(inference_result[4] * original_image_height)
        xmax = int(inference_result[5] * original_image_width)
        ymax = int(inference_result[6] * original_image_height)
        confidence = round(probability * 100, 1)

        detected_face = (xmin, ymin, xmax, ymax, confidence)
        detected_faces.append(detected_face)

    return detected_faces
