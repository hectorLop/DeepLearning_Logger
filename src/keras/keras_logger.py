import json
import pandas as pd
import os

from datetime import datetime
from typing import Dict, List
from src.keras.configs import Config

class Experiment():
    """
    Deeplearning Experiment

    Parameters
    ----------
    experiment_path : str
        Path which contains the experiment files.
    name : str
        Experiment name.
    configs : List
        Configurations list.

    Attributes
    ----------
    _description : str
        Experiment description.
    _datatime : datetime
        Datetime the experiment was performed.
    _name : str
        Experiment name.
    _experiment_path : str
        Path which contains the experiment files.
    _configs : List
        Configurations list.
    """
    def __init__(self, experiment_path: str, name: str, configs: List[Config], description: str='') -> None:
        self._description = description
        self._datetime = datetime.now()
        self._name = name
        self._experiment_path = experiment_path
        self._configs = configs

    def register_experiment(self) -> None:
        """
        Registers the experiment data
        """
        # Obtains the experiment configs
        experiment_config = self._create_experiment_config()
        experiment_data = {}

        for config in self._configs:
            if not isinstance(config, Config):
                raise ValueError(f'{config.__class__.__name__} is not a Config object')
            
            name, data = config.get_config()
            experiment_data[name] = data

        # Write the experiments config
        self._write_json_file('experiment_config.json', experiment_config)
        self._write_json_file('experiment_data.json', experiment_data)

    def _create_experiment_config(self) -> Dict:
        """
        Creates the experiment configuration dictionary

        Returns
        -------
        experiment_config : Dict
            Dictionary containing the experiment description
        """
        datetime_str = self._datetime.strftime('%Y-%m-%dT%H:%M:%S')

        experiment_config = {
            'name': self._name,
            'description': self._description,
            'datetime': datetime_str
        }

        return experiment_config

    def _write_json_file(self, filename: str, config_data: Dict) -> None:
        """
        Write a given configuration data into a JSON file.

        Parameters
        ----------
        filename : str
            JSON filename
        config_data : Dict
            Data to write into the JSOn file
        """
        with open(self._experiment_path + filename, 'w') as outfile:  
            json.dump(config_data, outfile, indent=4)

    def add_description(self, description: str):
        if not isinstance(description, str):
            raise ValueError('Description must be a string')
        
        self._description = description