from openvino.inference_engine import IECore

from .models_paths import face_detection_model_xml, face_detection_model_bin, emotion_recognition_model_xml, emotion_recognition_model_bin

ie_core = IECore()
device = 'CPU'

face_detection_model = ie_core.read_network(model=face_detection_model_xml, weights=face_detection_model_bin)
face_detector_input_name = next(iter(face_detection_model.input_info))
face_detector_output_name = next(iter(face_detection_model.outputs))

# Read the input dimensions: n=batch size, c=number of channels, h=height, w=width
face_detector_input_shape = face_detection_model.input_info[face_detector_input_name].input_data.shape
*_, face_detector_input_height, face_detector_input_width = face_detector_input_shape
face_detector = ie_core.load_network(network=face_detection_model, device_name=device)


emotion_recognition_model = ie_core.read_network(model=emotion_recognition_model_xml, weights=emotion_recognition_model_bin)
emotion_recognizer_input_name = next(iter(emotion_recognition_model.input_info))

emotion_recognizer_input_shape = emotion_recognition_model.input_info[emotion_recognizer_input_name].input_data.shape
*_, emotion_recognizer_input_height, emotion_recognizer_input_width = emotion_recognizer_input_shape
emotion_recognizer = ie_core.load_network(emotion_recognition_model, device)
emotion_recognizer_output_name = next(iter(emotion_recognition_model.outputs))
