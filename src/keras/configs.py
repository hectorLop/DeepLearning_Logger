from abc import ABC, abstractmethod
import numpy as np

class Config(ABC):
    def __init__(self, data: object) -> None:
        self._data = data

    @abstractmethod
    def get_config(self):
        pass

class MetricsConfig(Config):
    def __init__(self, data: object) -> None:
        super().__init__(data)

    def get_config(self):
        config = {}

        for idx, row in self._data.iterrows():
            config[f'epoch_{idx}'] = {}
            for column in self._data.columns:
                config[f'epoch_{idx}'][column] = float(row.loc[column])
        
        return 'metrics', config

class ModelConfig(Config):
    def __init__(self, model: object) -> None:
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

class CheckpointConfig(Config):
    def __init__(self, data: object) -> None:
        super().__init__(data)

    def get_config(self):
        config = {
            'checkpoints_folder': self._data.filepath
        }

        return 'checkpoints', config