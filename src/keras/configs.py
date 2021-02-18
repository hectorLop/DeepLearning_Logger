from abc import ABC, abstractmethod

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
        config = {
            'model_config': self._data.get_config(),
            'optimizer_config': self._data.optimizer.get_config()
        }

        return 'model', config

class CheckpointConfig(Config):
    def __init__(self, data: object) -> None:
        super().__init__(data)

    def get_config(self):
        checkpoint_path = self._data.filepath.split('/')
        checkpoint_folder = '/'.join(checkpoint_path[:-1]) + '/'

        config = {
            'checkpoints_folder': checkpoint_folder
        }

        return 'checkpoints', config