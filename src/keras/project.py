from src.keras.keras_logger import Experiment
from src.keras.configs import Config
from typing import List

import os

class Project():
    def __init__(self, project_name: str, project_path: str=os.path.dirname(os.path.realpath(__file__))) -> None:
        self._project_path = project_path
        self._project_name = project_name
        self._project_folder_path = self._create_project_folder()

    def create_experiment(self, experiment_name: str, configs: List[Config]):
        if not experiment_name:
            raise ValueError('Must use a non empty experiment name')

        experiment_folder_path = self._create_experiment_folder(experiment_name)
        experiment = Experiment(experiment_folder_path, experiment_name, configs=configs)
        experiment.register_experiment()

    def open_experiment(self):
        pass

    def list_experiments(self):
        pass

    def _create_experiment_folder(self, experiment_name: str) -> str:
        experiment_folder_path = self._project_folder_path  + experiment_name + '/'

        os.makedirs(experiment_folder_path)

        return experiment_folder_path

    def _create_project_folder(self):
        project_folder_path = self._project_path + '/' + self._project_name + '/'

        if not os.path.isdir(project_folder_path):
            os.makedirs(project_folder_path)

        return project_folder_path

        