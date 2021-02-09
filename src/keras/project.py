from src.keras.keras_logger import Experiment

import os
import pandas as pd

class Project():
    def __init__(self, project_path: str, project_name: str) -> None:
        self._project_path = project_path
        self._project_name = project_name
        self._project_folder_path = self._create_project_folder()

    def create_experiment(self, experiment_name: str, model: object=None, metrics_history: pd.DataFrame=None):
        experiment = Experiment(self._project_folder_path, experiment_name, model, metrics_history)

        experiment.register_experiment()

    def _create_project_folder(self):
        project_folder_path = self._project_path + '/' + self._project_name

        if not os.path.isdir(project_folder_path):
            os.makedirs(project_folder_path)

        return project_folder_path

        