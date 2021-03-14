from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, Tuple, Union
import numpy as np
from typeguard import typechecked
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import pandas as pd

class Config(ABC):
    """
    Configuration base class of objects that the Experiment can log.

    Parameters
    ----------
    data : Any
        Data to be stored in an experiment.

    Attributes
    ----------
    _data : Any
        Data to be stored in an experiment.
    """
    def __init__(self, data: Any) -> None:
        if isinstance(data, Dict): # Creation from a config dictionary
            self._config = data
        else:
            self._config = self.get_config(data)

    @abstractmethod
    def get_config(self, data) -> Tuple[str, Dict]:
        pass

    @property
    def config(self) -> Tuple:
        return self._config

class MetricsConfig(Config):
    """
    Config subclass to store the model training metrics

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the model metrics

    Attributes
    ----------
    _data : pd.DataFrame
        DataFrame containing the model metrics
    """
    @typechecked
    def __init__(self, data: Union[pd.DataFrame, Dict]) -> None:
        super().__init__(data)

    def get_config(self, data):
        config = {}

        for idx, row in data.iterrows():
            config[f'epoch_{idx}'] = {}
            for column in data.columns:
                config[f'epoch_{idx}'][column] = float(row.loc[column])
        
        return ('metrics', config)

class ModelConfig(Config):
    @typechecked
    def __init__(self, model: Union[Model, Dict]) -> None:
        super().__init__(model)

    def get_config(self, data):
        model_architecture = data.get_config()
        optimizer_config = data.optimizer.get_config()

        config = {
            'model_config': model_architecture,
            'optimizer_config': optimizer_config
        }

        return ('model', config)

class CallbackConfig(Config):
    @typechecked
    def __init__(self, data: Union[Callback, Dict]) -> None:
        super().__init__(data)

    def get_config(self, data):
        callback_attributes = data.__dict__
        public_attributes = {}

        for key,value in callback_attributes.items():
            # We don't want to serialize public attributes. Also, Callables
            # are not JSON serializable
            if key[0] != '_' and not isinstance(value, Callable):
                public_attributes[key] = value

        callback_name = data.__class__.__name__

        config = {
            callback_name: public_attributes
        }

        return ('callbacks', config)