from app.config import Config, OpenVINOModelPaths
from app.engines.engine import IEngine


class FaceDetector(IEngine):
    _model_path: OpenVINOModelPaths = Config.face_detection_model
