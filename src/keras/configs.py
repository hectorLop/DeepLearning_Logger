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
        training_loss, validation_loss = self._data['loss'], self._data['val_loss']
        config = {}

        for idx, train_loss, val_loss in enumerate(zip(training_loss, validation_loss)):
            config[f'epoch_{idx + 1}'] = {
                'train_loss': train_loss,
                'val_loss': val_loss
            }

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