from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from typeguard import typechecked
from keras.models import Model
from keras.callbacks import Callback
import pandas as pd

class Config(ABC):
    def __init__(self, data: object) -> None:
        self._data = data

    @abstractmethod
    def get_config(self):
        pass

class MetricsConfig(Config):
    @typechecked
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)

    def get_config(self):
        config = {}

        for idx, row in self._data.iterrows():
            config[f'epoch_{idx}'] = {}
            for column in self._data.columns:
                config[f'epoch_{idx}'][column] = float(row.loc[column])
        
        return 'metrics', config

class ModelConfig(Config):
    @typechecked
    def __init__(self, model: Model) -> None:
        super().__init__(model)

    def get_config(self):
        model_architecture = self._data.get_config()
        optimizer_config = self._data.optimizer.get_config()

        # Numpy.float32 types are not json serializables so we must cast them to float64
        for key, value in optimizer_config.items():
            if isinstance(value, np.float32):
                optimizer_config[key] = float(value) # Numpy.float64 dtype is the same as Python built-in float

        config = {
            'model_config': model_architecture,
            'optimizer_config': optimizer_config
        }

        return 'model', config

class CallbackConfig(Config):
    @typechecked
    def __init__(self, data: Callback) -> None:
        super().__init__(data)

    def get_config(self):
        callback_attributes = self._data.__dict__
        public_attributes = {}

        for key,value in callback_attributes.items():
            # We don't want to serialize public attributes. Also, Callables
            # are not JSON serializable
            if key[0] != '_' and not isinstance(value, Callable):
                public_attributes[key] = value

        callback_name = self._data.__class__.__name__

        config = {
            callback_name: public_attributes
        }

        return 'callbacks', config