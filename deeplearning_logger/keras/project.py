from deeplearning_logger.keras.keras_logger import Experiment
from deeplearning_logger.keras.configs import Config
from typing import List

import os
import glob

class Project():
    """
    Deeplearning Project.

    Parameters
    ----------
    project_name : str
        Project name
    proejct_path : str
        Path where to create the project.

    Attributes
    ----------
    _project_path : str
        Path were to create the project
    _project_name : str
        Project name
    _project_folder_path : str
        Path to the project folder
    """
    def __init__(self, project_name: str, project_path: str = '') -> None:
        self._project_path = project_path
        self._project_name = project_name

        if not project_path:
            project_path = os.getcwd()

        self._project_folder_path = self._create_folder([project_path,
                                                        project_name])

    def create_experiment(self, experiment_name: str = '',
                        configs: List[Config] = [],
                        description: str = '') -> None:
        """
        Creates an experiment inside the project.

        Parameters
        ----------
        experiment_name : str
            Experiment's name. Default is an empty string.
        configs : List
            Configurations list. Default is an empty list.
        """
        if not experiment_name:
            raise ValueError('Must use a non empty experiment name')
        if not configs:
            raise ValueError('The configurations list is empty,' \
                            'there are nothing to log')
        
        # Create the experiment folder
        experiment_folder_path = self._create_folder([self._project_folder_path,
                                                    experiment_name])
        # Register the experiment
        experiment = Experiment(experiment_path=experiment_folder_path,
                                name=experiment_name,
                                configs=configs,
                                description=description)
        experiment.register_experiment()

    def open_experiment(self, experiment_name: str) -> Experiment:
        # Create the experiment folder path
        experiment_folder = self._project_folder_path + experiment_name
        # Create the experiment config files path
        experiment_data_file = experiment_folder + '/experiment_data.json'
        experiment_config_file = experiment_folder + '/experiment_config.json'

        experiment = Experiment.by_config_files(
                                path=experiment_folder,
                                config_info_file=experiment_config_file,
                                config_data_file=experiment_data_file)

        return experiment

    def list_experiments(self):
        """
        Get the list of experiments inside a project.

        Returns
        -------
        experiment_names : List[str] 
            List containing the experiment names
        """
        path = self._project_folder_path + '*/'
        # Search for directories in the project path and get the names
        experiment_names = [path.split('/')[-2] 
                            for path in glob.glob(path)]

        return experiment_names

    def _create_folder(self, paths: List[str]=[]) -> str:
        """
        Creates a folder from a list of paths.

        Parameters
        ----------
        paths : List[str]
            List of paths to concatenate. Default is an empty list

        Returns
        -------
        folder_path : str
            Path to the folder created
        """
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

        