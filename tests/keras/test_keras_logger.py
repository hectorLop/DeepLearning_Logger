from keras.engine.training import Model
import pytest
import keras
import json
import numpy as np
import os
import pandas as pd

from src.keras.keras_logger import Experiment
from src.keras.configs import MetricsConfig, ModelConfig, CallbackConfig
from keras.callbacks import ModelCheckpoint

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
    """
    Tests logging an experiment containing a model
    """
    experiement_path = os.path.dirname(os.path.realpath(__file__)) + '/project_01/log_model/'
    configs = [ModelConfig(get_model)]
    experiment = Experiment(experiment_path=experiement_path, name='log_model', configs=configs)

    experiment.register_experiment()

    assert os.path.isfile(experiement_path + 'experiment_config.json')
    assert os.path.isfile(experiement_path + 'experiment_data.json')

    with open(experiement_path + 'experiment_config.json') as file:
        experiment_info = json.load(file)

    with open(experiement_path + 'experiment_data.json') as file:
        experiment_data = json.load(file)

    assert experiment_info['name'] == 'log_model'
    assert experiment_data['model']['model_config']['name'] == 'sequential_1'

def test_keras_logger_log_metrics(get_metrics):
    """
    Tests logging an experiment containing metrics
    """
    experiement_path = os.path.dirname(os.path.realpath(__file__)) + '/project_01/log_metrics/'

    configs = [MetricsConfig(get_metrics)]
    experiment = Experiment(experiment_path=experiement_path, name='log_metrics', configs=configs)

    experiment.register_experiment()

    assert os.path.isfile(experiement_path + 'experiment_config.json')
    assert os.path.isfile(experiement_path + 'experiment_data.json')

    with open(experiement_path + 'experiment_config.json') as file:
        experiment_info = json.load(file)

    with open(experiement_path + 'experiment_data.json') as file:
        experiment_data = json.load(file)

    assert experiment_info['name'] == 'log_metrics'
    assert experiment_data['metrics']['epoch_0']['train_loss'] == 0
    assert experiment_data['metrics']['epoch_0']['val_loss'] == 0

def test_keras_logger_log_experiment(get_metrics, get_model):
    """
    Tests logging an experiment containing metrics
    """
    experiement_path = os.path.dirname(os.path.realpath(__file__)) + '/project_01/log_experiment/'

    configs = [MetricsConfig(get_metrics), ModelConfig(get_model)]
    experiment = Experiment(experiment_path=experiement_path, name='log_experiment', configs=configs, description='experimento de prueba')

    experiment.register_experiment()

    assert os.path.isfile(experiement_path + 'experiment_config.json')
    assert os.path.isfile(experiement_path + 'experiment_data.json')

    with open(experiement_path + 'experiment_config.json') as file:
        experiment_info = json.load(file)

    with open(experiement_path + 'experiment_data.json') as file:
        experiment_data = json.load(file)

    assert experiment_info['name'] == 'log_experiment'
    assert experiment_data['model']['model_config']['name'] == 'sequential_2'

    assert experiment_info['name'] == 'log_experiment'
    assert experiment_data['metrics']['epoch_0']['train_loss'] == 0
    assert experiment_data['metrics']['epoch_0']['val_loss'] == 0

def test_keras_logger_log_callbacks():
    """
    Tests logging an experiment containing metrics
    """
    experiement_path = os.path.dirname(os.path.realpath(__file__)) + '/project_01/log_callbacks/'
    checkpoints_path = experiement_path + 'checkpoint.h5'

    configs = [CallbackConfig(ModelCheckpoint(checkpoints_path))]
    experiment = Experiment(experiment_path=experiement_path, name='log_callbacks', configs=configs, description='experimento de prueba')

    experiment.register_experiment()

    assert os.path.isfile(experiement_path + 'experiment_config.json')
    assert os.path.isfile(experiement_path + 'experiment_data.json')

    with open(experiement_path + 'experiment_config.json') as file:
        experiment_info = json.load(file)

    with open(experiement_path + 'experiment_data.json') as file:
        experiment_data = json.load(file)

    assert experiment_info['name'] == 'log_callbacks'
    assert experiment_info['description'] == 'experimento de prueba'

    assert experiment_data['callbacks']['ModelCheckpoint']['filepath'] == checkpoints_path

def test_keras_logger_log_non_config():
    """
    Tests logging an experiment containing a non Config object
    """
    experiement_path = os.path.dirname(os.path.realpath(__file__)) + '/project_01/log_checkpoint/'
    checkpoints_path = experiement_path + 'checkpoint.h5'

    # Creating the non config object
    configs = [ModelCheckpoint(checkpoints_path)]
    # Creating the experiment
    experiment = Experiment(experiment_path=experiement_path, name='log_checkpoint',
                            configs=configs, description='experimento de prueba')

    with pytest.raises(ValueError) as exception_info:
        experiment.register_experiment()

    assert str(exception_info.value) == 'ModelCheckpoint is not a Config object'