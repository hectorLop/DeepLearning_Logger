import json

from datetime import date, datetime
from typing import Dict, List
from deeplearning_logger.keras.configs import *
from deeplearning_logger.json import ConfigsJSONEncoder

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
    _CONFIGS_EQUIVALENCES = {
        'metrics': MetricsConfig,
        'model': ModelConfig,
        'callbacks': CallbackConfig
    }

    def __init__(self, experiment_path: str, name: str, configs: List[Config],
                 description: str='', datetime : date = None) -> None:
        self._description = description

        if datetime is None:
            self._datetime = datetime.now()
        else:
            self._datetime = datetime

        self._name = name
        self._experiment_path = experiment_path
        self._configs = configs

    @classmethod
    def by_config_files(cls, path: str, config_info_file: str,
                        config_data_file: str):
        experiment_data = cls._parse_config_file(config_data_file)
        experiment_config = cls._parse_config_file(config_info_file)

        name, description, datetime = cls._parse_experiment_config(
                                                            experiment_config)
        configs = cls._parse_experiment_data(experiment_data)

        return Experiment(experiment_path=path, name=name,
                        description=description, datetime=datetime,
                        configs=configs)

    def _parse_config_file(self, config_file):
        with open(config_file, 'r') as file:
            config = json.load(file)

        return config

    # TODO: Create config objects from parsed json file
    def _parse_experiment_data(self, experiment_data: Dict):
        configs_list = [(key, value) for key, value in experiment_data]

        configs = []
        for config in configs_list:
            configs.append(self._CONFIGS_EQUIVALENCES[config[0]](config[1]))

        return configs

    def _parse_experiment_config(self, experiment_config: Dict):
        name = experiment_config['name']
        description = experiment_config['description']
        datetime = datetime.strptime(experiment_config['datetime'],
                                        '%Y-%m-%dT%H:%M:%S')

        return name, description, datetime

    def register_experiment(self) -> None:
        """
        Registers the experiment data
        """
        # Obtains the experiment configs
        experiment_config = self._create_experiment_config()
        experiment_data = {}

        for config_element in self._configs:
            if not isinstance(config_element, Config):
                raise ValueError(
                        f'{config_element.__class__.__name__} is'\
                            ' not a Config object')
            
            name, data = config_element.config
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
            json.dump(config_data, outfile, indent=4, cls=ConfigsJSONEncoder)

    def add_description(self, description: str):
        if not isinstance(description, str):
            raise ValueError('Description must be a string')
        
        self._description = description