
from openvino.inference_engine import IECore


class MetaSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class IECoreWrapper(metaclass=MetaSingleton):
    def __init__(self):
        self._ie_core = IECore()

    @property
    def ie_core(self) -> IECore:
        return self._ie_core
