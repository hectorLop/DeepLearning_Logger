from src.keras.keras_logger import Experiment
from src.keras.configs import Config
from typing import List

import os

class Project():
    def __init__(self, project_name: str, project_path: str=os.path.dirname(os.path.realpath(__file__))) -> None:
        self._project_path = project_path
        self._project_name = project_name
        self._project_folder_path = self._create_folder([project_path, project_name])

    def create_experiment(self, experiment_name: str='', configs: List[Config]=[]):
        if not experiment_name:
            raise ValueError('Must use a non empty experiment name')
        if not configs:
            raise ValueError('The configurations list is empty, there are nothing to log')

        experiment_folder_path = self._create_folder([self._project_folder_path, experiment_name])
        experiment = Experiment(experiment_folder_path, experiment_name, configs=configs)
        experiment.register_experiment()

    def open_experiment(self):
        pass

    def list_experiments(self):
        pass

    def _create_folder(self, paths: List[str]=[]):
        if not paths:
            raise ValueError('Path lists cannot be empty')
        
        folder_path = ''

        for path in paths:
            if path[-1] == '/':
                folder_path += path
            else:
                folder_path += path + '/'

        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        return folder_path

        