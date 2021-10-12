from app.config import Config, OpenVINOModelPaths
from app.engines.engine import IEngine


class EmotionRecognizer(IEngine):
    _model_path: OpenVINOModelPaths = Config.emotion_recognition_model
