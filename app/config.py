import os
import pathlib
from pathlib import Path

from app.constants import Emotion
from typing import Dict, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class OpenVINOModelPaths(TypedDict):
    xml_path: Optional[Path]
    bin_path: Optional[Path]


class Config:
    _data_path: Path = pathlib.Path(__file__).parent.parent.resolve() / 'data'

    face_detection_model = OpenVINOModelPaths(
        xml_path=os.environ.get('FACE_DETECTION_XML_FILE_PATH'),
        bin_path=os.environ.get('FACE_DETECTION_BIN_FILE_PATH'),
    )

    emotion_recognition_model = OpenVINOModelPaths(
        xml_path=os.environ.get('EMOTION_RECOGNITION_XML_FILE_PATH'),
        bin_path=os.environ.get('EMOTION_RECOGNITION_BIN_FILE_PATH'),
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
