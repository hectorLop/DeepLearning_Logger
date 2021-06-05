from deeplearning_logger.pytorch.pytorch_logger import PytorchLogger, \
                                                        ExperimentData
from tests.pytorch.test_utils import CustomModel
import pytest
import torch
import torch.nn as nn
import json
import os

def test_pytorch_logger():
    model = CustomModel()

    logger = PytorchLogger()
    ex_data = ExperimentData(
            lr=0.0001,
            architecture=model,
            optimizer='adam'
            )

    logger.save(ex_data, 'prueba')

    with open('prueba.json', 'r') as file:
        data = json.load(file)

    assert data['lr'] == 0.0001
    assert data['architecture'] == str(model)
    assert data['optimizer'] == 'adam'

    os.remove('prueba.json')