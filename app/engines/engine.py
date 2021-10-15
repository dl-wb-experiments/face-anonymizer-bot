from typing import Tuple

import numpy as np

from app.config import OpenVINOModelPaths
from app.engines.ie_core_wrapper import IECoreWrapper, MetaSingleton

InputShapeType = Tuple[int, int]


class IEngine(metaclass=MetaSingleton):
    _model_path: OpenVINOModelPaths = None

    def __init__(self):
        ie_core = IECoreWrapper().ie_core

        self._network = None
        self._executable_network = None

        if self._model_path['xml_path'] and self._model_path['bin_path']:
            self._network = ie_core.read_network(
                self._model_path['xml_path'],
                self._model_path['bin_path']
            )

            self._executable_network = ie_core.load_network(self._network, 'CPU')
        else:
            print(f'IENetwork and ExecutableNetwork was not initialized for {self.__class__}')

    @property
    def is_ready(self) -> bool:
        return self._network and self._executable_network

    @property
    def input_shape(self) -> InputShapeType:
        input_item = self._network.input_info[self.input_layer_name]
        *_, height, width = input_item.input_data.shape
        return width, height

    @property
    def input_layer_name(self) -> str:
        return next(iter(self._network.input_info))

    @property
    def output_layer_name(self) -> str:
        return next(iter(self._network.outputs))

    def inference(self, input_data: np.ndarray) -> np.ndarray:
        inference_results = self._executable_network.infer({
            self.input_layer_name: input_data
        })
        return inference_results[self.output_layer_name]
