import json
import pandas as pd
import os

from datetime import datetime
from typing import List
from src.keras.configs import Config

class Experiment():
    def __init__(self, experiment_path: str, name: str, configs: List[Config]) -> None:
        self._description = ''
        self._datetime = datetime.now()
        self._name = name
        self._experiment_path = experiment_path
        self._configs = configs

    def register_experiment(self):
        # Obtains the experiment configs
        experiment_config = self._create_experiment_config()
        experiment_data = {}

        for config in self._configs:
            name, data = config.get_config()
            experiment_data[name] = data

        # Write the experiments config
        self._write_config_files(self._experiment_path, 'experiment_config.json', experiment_config)
        self._write_config_files(self._experiment_path, 'experiment_data.json', experiment_data)

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

