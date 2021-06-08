from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
import torch.nn as nn
import typing

@dataclass
class Data(ABC):
    def get(self):
        return self.__dict__

    def validate_types(self, instance: Data, class_type: object):
        for field in fields(instance):
            attr = getattr(instance, field.name)
            attr_type = typing.get_type_hints(class_type)[field.name]

            if attr is not None and not isinstance(attr, attr_type):
                msg = (
                    f'Field {field.name} is of type {type(attr)}, it ',  
                    f'must be {attr_type}')

                raise ValueError(msg)

    @abstractmethod
    def __post_init__(self):
        pass

@dataclass
class OptimizerData(Data):
    """
    Attributes
    ----------
    lr : float
        Learning rate used in the experiment. Default is 0.0
    optimizer : str
        Optimizer name used in the experiment.
    weight_decay : float
        L2 regularization term. Default is 0.0
    """
    lr: float = 0.0
    optimizer: str = ''
    weight_decay: float = 0.0

    def __post_init__(self):
        self.validate_types(self, OptimizerData)

@dataclass
class MetricsData(Data):
    """
    Attributes
    ----------
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
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    train_metrics: list = field(default_factory=list)
    val_metrics: list = field(default_factory=list)
    test_metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.validate_types(self, MetricsData)

@dataclass
class ModelData(Data):
    """
    Attributes
    ----------
    checkpoint : str
        Final checkpoint path
    architecture : str
        Model architecture used in the training
    epochs : str
        Number of epochs used in the experiment
    """
    checkpoint: str = ''
    architecture: nn.Module = None
    epochs: int = 0

    def __post_init__(self):
        self.validate_types(self, ModelData)

        self.architecture = str(self.architecture)