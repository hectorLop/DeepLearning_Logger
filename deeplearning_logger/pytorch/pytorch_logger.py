from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn
from dataclasses import dataclass, field, fields
from deeplearning_logger.json import ConfigsJSONEncoder
import json
import os
import typing

from typing import List

class PytorchLogger():
    def __init__(self, project_folder: str = '') -> None:
        if not project_folder:
            self.project_path = os.getcwd()
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
        
        with open(f'{experiment_name}.json', 'w') as outfile:  
            json.dump(data.__dict__, outfile, indent=4, cls=ConfigsJSONEncoder)

@dataclass
class ExperimentData():
    """
    This class defines the data related to an experiment.

    Attributes
    ----------
    lr : float
        Learning rate used in the experiment. Default is 0.0
    optimizer : str
        Optimizer name used in the experiment.
    weight_decay : float
        L2 regularization term. Default is 0.0
    checkpoint : str
        Final checkpoint path
    architecture : str
        Model architecture used in the training
    epochs : str
        Number of epochs used in the experiment
    train_losses : list of float
        List containing the training loss for each epoch
    val_losses : list of float
        List containing the validation loss for each epoch
    train_metrics : list of dict
        List containing the training metrics dictionary for each epoch
    val_metrics : list of dict
        List containing tje validation metrics dictionary for each epoch
    test_metrics : dict
        Dictionary containing the test metrics
    """
    lr: float = 0.0
    optimizer: str = ''
    weight_decay: float = 0.0
    checkpoint: str = ''
    architecture: nn.Module = None
    epochs: int = 0
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    train_metrics: list = field(default_factory=list)
    val_metrics: list = field(default_factory=list)
    test_metrics: dict = field(default_factory=dict)

    def validate(self, instance):
        for field in fields(instance):
            attr = getattr(instance, field.name)
            attr_type = typing.get_type_hints(ExperimentData)[field.name]

            if not isinstance(attr, attr_type):
                msg = f'Field {field.name} is of type {type(attr)}, should be {attr_type}'

                raise ValueError(msg)

    def __post_init__(self):
        self.validate(self)
        # Gets the model architecture
        self.architecture = str(self.architecture)