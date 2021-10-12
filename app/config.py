import pathlib
from pathlib import Path

from app.constants import Emotion
from typing import Dict
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class OpenVINOModelPaths(TypedDict):
    xml_path: Path
    bin_path: Path


class Config:
    _data_path: Path = pathlib.Path(__file__).parent.parent.resolve() / 'data'
    _models_path: Path = _data_path / 'models'

    face_detection_model = OpenVINOModelPaths(
        xml_path=_models_path / 'face-detection-adas-0001.xml',
        bin_path=_models_path / 'face-detection-adas-0001.bin',
    )

    emotion_recognition_model = OpenVINOModelPaths(
        xml_path=_models_path / 'emotions-recognition-retail-0003.xml',
        bin_path=_models_path / 'emotions-recognition-retail-0003.bin',
    )

    device: str = 'CPU'

    _emotion_images_path: Path = _data_path / 'images' / 'emotions'

    emotions_images: Dict[Emotion, Path] = {
        Emotion.anger: _emotion_images_path / 'anger.png',
        Emotion.happy: _emotion_images_path / 'happy.png',
        Emotion.neutral: _emotion_images_path / 'neutral.png',
        Emotion.sad: _emotion_images_path / 'sad.png',
        Emotion.surprise: _emotion_images_path / 'surprise.png',
    }
