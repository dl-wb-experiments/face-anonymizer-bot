# Define the function to pre-process (resize, transpose) the input image
import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork

prob_threshold = 50
ie_core = IECore()
device = 'CPU'

face_detection_model_xml = '/Users/atugarev/Downloads/ubuntu18_deployment_package_CPU_python3.6_python3.7_python3.8_with_model_face-detection-adas-0001/face-detection-adas-0001/face-detection-adas-0001.xml'
face_detection_model_bin = '/Users/atugarev/Downloads/ubuntu18_deployment_package_CPU_python3.6_python3.7_python3.8_with_model_face-detection-adas-0001/face-detection-adas-0001/face-detection-adas-0001.bin'
face_detection_model = ie_core.read_network(model=face_detection_model_xml, weights=face_detection_model_bin)
face_detector_input_name = next(iter(face_detection_model.input_info))
face_detector_output_name = next(iter(face_detection_model.outputs))

# Read the input dimensions: n=batch size, c=number of channels, h=height, w=width
face_detector_input_shape = face_detection_model.input_info[face_detector_input_name].input_data.shape
*_, face_detector_input_height, face_detector_input_width = face_detector_input_shape
face_detector = ie_core.load_network(network=face_detection_model, device_name=device)


emotion_recognition_model_xml = '/Users/atugarev/Downloads/ubuntu18_deployment_package_CPU_python3.6_python3.7_python3.8_with_model_face-detection-adas-0001/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml'
emotion_recognition_model_bin = '/Users/atugarev/Downloads/ubuntu18_deployment_package_CPU_python3.6_python3.7_python3.8_with_model_face-detection-adas-0001/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.bin'
emotion_recognition_model = ie_core.read_network(model=emotion_recognition_model_xml, weights=emotion_recognition_model_bin)
emotion_recognizer_input_name = next(iter(emotion_recognition_model.input_info))

emotion_recognizer_input_shape = emotion_recognition_model.input_info[emotion_recognizer_input_name].input_data.shape
*_, emotion_recognizer_input_height, emotion_recognizer_input_width = emotion_recognizer_input_shape
emotion_recognizer = ie_core.load_network(emotion_recognition_model, device)
emotion_recognizer_output_name = next(iter(emotion_recognition_model.outputs))


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


def parse_face_detection_results(inference_results: np.ndarray, original_image_width: int,
                                 original_image_height: int) -> list:
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
        xmin = max(0, int(inference_result[3] * original_image_width))
        ymin = max(0, int(inference_result[4] * original_image_height))
        xmax = min(original_image_width, int(inference_result[5] * original_image_width))
        ymax = min(original_image_height, int(inference_result[6] * original_image_height))

        detected_face = (xmin, ymin, xmax, ymax, confidence)
        detected_faces.append(detected_face)

    return detected_faces


def blur(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    pixels_count = 16

    # Resize the image to pixels_count*pixels_count with interpolation to blur the image
    resized_image = cv2.resize(image, (pixels_count, pixels_count), interpolation=cv2.INTER_LINEAR)
    # Resize the image to original image size
    return cv2.resize(resized_image, (width, height), interpolation=cv2.INTER_NEAREST)


def face_detector_inference(image: np.ndarray) -> np.ndarray:
    # 1. Prepare the image
    input_frame = pre_process_input_image(image, target_width=face_detector_input_width, target_height=face_detector_input_height)

    # 2. Infer the model
    face_detection_inference_results = face_detector.infer(inputs={face_detector_input_name: input_frame})
    return face_detection_inference_results[face_detector_output_name]


def blur_faces(face_detection_inference_result: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    # Get the original image size
    original_image_height, original_image_width, _ = original_image.shape

    # Prepare the resulting image to not affect the original image
    processed_image = original_image.copy()

    # 1. Parse the model inference results
    detected_faces = parse_face_detection_results(face_detection_inference_result, original_image_width, original_image_height)

    # 2. Iterate through the faces and blur each face
    for detected_face in detected_faces:
        xmin, ymin, xmax, ymax, confidence = detected_face
        face = original_image[ymin:ymax, xmin:xmax]
        processed_image[ymin:ymax, xmin:xmax] = blur(face)

    return processed_image


def emotion_recognizer_inference(face_frame: np.ndarray) -> np.ndarray:
    prepared_frame = pre_process_input_image(face_frame, target_width=emotion_recognizer_input_width,
                                             target_height=emotion_recognizer_input_height)

    # Run the inference the same way you did before
    inference_results = emotion_recognizer.infer({
        emotion_recognizer_input_name: prepared_frame
    })

    return inference_results[emotion_recognizer_output_name]


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
