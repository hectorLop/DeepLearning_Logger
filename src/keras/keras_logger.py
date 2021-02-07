import json
import pandas as pd
import os

from datetime import datetime
from keras.callbacks import Callback

from typing import List

class Experiment():
    def __init__(self, name: str, model: object) -> None:
        self._model = model
        self._description = ''
        self._datetime = datetime.now()
        self._name = name

    def register_experiment(self, loss_history: pd.DataFrame=None, callbacks: List[Callback]=None,
                            path=os.path.dirname(os.path.realpath(__file__))):
        model_architecture = self._model.get_config()
        optimizer_config = self._model.optimizer.get_config()
        experiment_config = self._create_experiment_config()

        self._write_config_files(path, 'architecture.json', model_architecture)
        self._write_config_files(path, 'optimizer.json', optimizer_config)
        self._write_config_files(path, 'experiment_config.json', experiment_config)

    def _create_experiment_config(self):
        datetime_str = self._datetime.strftime('%Y-%m-%dT%H:%M:%S')

        experiment_config = {
            'name': self._name,
            'description': self._description,
            'datetime': datetime_str
        }

        return experiment_config

    def _write_config_files(self, path, filename, config_data):
        with open(path + filename, 'w') as outfile:  
            json.dump(config_data, outfile, indent=4)