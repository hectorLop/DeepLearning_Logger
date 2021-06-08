from deeplearning_logger.pytorch.pytorch_logger import PytorchLogger, \
                                                        ExperimentData
from deeplearning_logger.pytorch.experiment_data import ModelData, OptimizerData
from tests.pytorch.test_utils import CustomModel
import pytest
import json
import os

def test_default_pytorch_logger():
    """
    Test the PytorchLogger default behaviour.
    """
    logger = PytorchLogger()

    experiment = ExperimentData()

    logger.save(experiment, 'default_ex')

    with open('default_ex.json', 'r') as file:
        data = json.load(file)

    assert data['lr'] == 0.0
    assert data['architecture'] == 'None'
    assert data['optimizer'] == ''
    assert not data['train_losses']

    os.remove('default_ex.json')

def test_pytorch_logger():
    """
    Test a PytorchLogger which logs custom model and optimizer data
    """
    model = CustomModel()

    # Create the logger
    logger = PytorchLogger()

    # Model and optimizer data
    model_data = ModelData(architecture=model)
    optimizer_data = OptimizerData(lr=0.0001, optimizer='adam')

    # Experiment data
    ex_data = ExperimentData(
            model=model_data,
            optimizer=optimizer_data
            )

    logger.save(ex_data, 'prueba')

    with open('prueba.json', 'r') as file:
        data = json.load(file)

    assert data['lr'] == 0.0001
    assert data['architecture'] == str(model)
    assert data['optimizer'] == 'adam'

    os.remove('prueba.json')

def test_pytorch_logger_annotations():
    """
    Test a PytorchLogger with annotations
    """
    logger = PytorchLogger()

    annotations = 'This is a test'
    experiment = ExperimentData(annotations=annotations)

    logger.save(experiment, 'default_ex')

    with open('default_ex.json', 'r') as file:
        data = json.load(file)

    assert data['annotations'] == annotations

    os.remove('default_ex.json')

def test_pytorch_logger_same_filename_exception():
    """
    Test the PytorchLogger exception when an experiment with the same name
    already exists.
    """
    logger = PytorchLogger()

    annotations = 'This is a test'
    experiment = ExperimentData(annotations=annotations)

    logger.save(experiment, 'prueba')

    with pytest.raises(ValueError):
        logger.save(experiment, 'prueba')

    os.remove('prueba.json')