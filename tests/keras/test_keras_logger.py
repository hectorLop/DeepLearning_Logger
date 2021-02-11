import pytest
import keras
import json
import numpy as np
import os
import pandas as pd

from src.keras.keras_logger import Experiment
from src.keras.configs import MetricsConfig, ModelConfig

@pytest.fixture
def get_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    optimizer = keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

@pytest.fixture
def get_metrics():
    val_loss = np.arange(0, 1000, 10)
    train_loss = np.arange(0, 1000, 10)

    metrics_values = np.c_[train_loss, val_loss]
    metrics_df = pd.DataFrame(metrics_values, columns=['train_loss', 'val_loss'])

    return metrics_df

def test_keras_logger_log_model(get_model):
    experiement_path = os.path.dirname(os.path.realpath(__file__)) + '/project_01/log_model/'

    configs = [ModelConfig(get_model)]
    experiment = Experiment(experiment_path=experiement_path, name='log_model', configs=configs)

    experiment.register_experiment()

    assert os.path.isfile(experiement_path + 'experiment_config.json')
    assert os.path.isfile(experiement_path + 'experiment_data.json')

def test_keras_logger_log_metrics(get_metrics):
    experiement_path = os.path.dirname(os.path.realpath(__file__)) + '/project_01/log_metrics/'

    configs = [MetricsConfig(get_metrics)]
    experiment = Experiment(experiment_path=experiement_path, name='log_metrics', configs=configs)

    experiment.register_experiment()

    assert os.path.isfile(experiement_path + 'experiment_config.json')
    assert os.path.isfile(experiement_path + 'experiment_data.json')