from __future__ import annotations
from typing import Dict
from deeplearning_logger.json import ConfigsJSONEncoder
from deeplearning_logger.pytorch.experiment_data import MetricsData, ModelData,\
                                                        OptimizerData
import json
import os
from datetime import date, datetime

class PytorchLogger():
    def __init__(self, project_folder: str = '') -> None:
        if not project_folder:
            self.project_path = os.getcwd()
            self.project_path = os.path.join(self.project_path, '')
        else:
            # This method adds the '/' at the end if it is not already added
            self.project_path = os.path.join(project_folder, '')

    def save(self, data: ExperimentData, experiment_name: str) -> None:
        """
        Saves the experiment data into a JSON file

        Parameters
        ----------
        data : ExperimentData
            ExperimentData object which contains the data
        experiment_name : str
            JSON filename
        """
        if not isinstance(data, ExperimentData):
            raise ValueError('The data must be an ExperimentData object')

        if os.path.isfile(f'{self.project_path}{experiment_name}.json'):
            raise ValueError('There is already an experiment with that name')
        
        with open(f'{self.project_path}{experiment_name}.json', 'w') as outfile:    
            json.dump(data.get(), outfile, indent=4, cls=ConfigsJSONEncoder)

class ExperimentData():
    """
    This class defines the data related to an experiment.

    Parameters
    ----------
    model : ModelData
        Data object containing the model related data
    metrics : MetricsData
        Data object containing the metrics related data
    optimizer : OptimizerData
        Data object containing the optimizer related data

    Attributes
    ----------
    data : list
        List containig the experiment related data
    """
    def __init__(self, model: ModelData = ModelData(),
                 metrics: MetricsData = MetricsData(),
                 optimizer: OptimizerData = OptimizerData(),
                 annotations: str = '') -> None:
        if not isinstance(model, ModelData):
            raise TypeError(f'model parameter must be a ModelData object')

        if not isinstance(metrics, MetricsData):
            raise TypeError(f'metrics parameter must be a MetricsData object')

        if not isinstance(optimizer, OptimizerData):
            raise TypeError(f'optimizer parameter must be a OptimizerData object')
            
        self.data = [model, metrics, optimizer]
        self._annotations = {'annotations': annotations}
        self._datetime = self._get_datetime()

    def _get_datetime(self):
        """
        Get the experiment timestamp

        Returns
        -------
        datetime_data : dict
            Dictionary containing both the date and time where the experiment
            was done
        """
        now = datetime.now()

        date = now.strftime('%d/%m/%Y')
        time = now.strftime('%H:%M:%S')

        datetime_date = {
            'date': date,
            'time': time
        }

        return datetime_date

    def get(self) -> Dict:
        """
        Gets the experiment data as a dictionary

        Returns
        -------
        data : dict
            Dictionary containing the experiment data
        """
        data = {}  

        for element in self.data:
            data.update(element.get())

        # Adds to the dictionary both the annotations and the datetime
        data.update(self._annotations)
        data.update(self._datetime)

        return data    