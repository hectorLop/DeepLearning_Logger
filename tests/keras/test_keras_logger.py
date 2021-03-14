import pytest
import json
import numpy as np
import os
import pandas as pd
import shutil

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from deeplearning_logger.keras.keras_logger import Experiment
from deeplearning_logger.keras.configs import MetricsConfig, ModelConfig, \
                                            CallbackConfig

from tests.keras.fixtures import get_trained_model

_PROJECT_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/project_01/'

@pytest.fixture
def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                metrics=['accuracy'])

    return model

@pytest.fixture
def get_metrics():
    val_loss = np.arange(0, 1000, 10)
    train_loss = np.arange(0, 1000, 10)

    metrics_values = np.c_[train_loss, val_loss]
    metrics_df = pd.DataFrame(metrics_values,
                            columns=['train_loss', 'val_loss'])

    return metrics_df

def test_logger_log_model(get_trained_model):
    """
    Tests logging an experiment containing a model configuration.
    """
    # Creates the project and experiment folder
    experiement_path = _PROJECT_FOLDER + 'log_model/'
    os.makedirs(experiement_path)
    
    # Gets the model and creates its configuration
    model, _ = get_trained_model
    model_config = ModelConfig(model)
    # Creates the experiment
    experiment = Experiment(experiment_path=experiement_path,
                            name='log_model', configs=[model_config])
    experiment.register_experiment()

    # Assert that the experiment files exist
    assert os.path.isfile(experiement_path + 'experiment_config.json')
    assert os.path.isfile(experiement_path + 'experiment_data.json')

    # Load the experiment files
    with open(experiement_path + 'experiment_config.json') as file:
        experiment_info = json.load(file)

    with open(experiement_path + 'experiment_data.json') as file:
        experiment_data = json.load(file)

    # Assert the experiment files content
    assert experiment_info['name'] == 'log_model'
    assert 'sequential' in experiment_data['model']['model_config']['name']

    # Removes the project and experiment folder
    shutil.rmtree(_PROJECT_FOLDER)

def test_logger_log_metrics(get_trained_model):
    """
    Tests logging an experiment which only contains metrics
    """
    # Create the project and experiment folder
    experiement_path = _PROJECT_FOLDER + 'log_metrics/'
    os.makedirs(experiement_path)

    # Get the model metrics
    _, metrics = get_trained_model
    metrics_configs = MetricsConfig(metrics)
    # Create the experiment
    experiment = Experiment(experiment_path=experiement_path,
                            name='log_metrics', configs=[metrics_configs])
    experiment.register_experiment()

    # Assert that the experiment files exist
    assert os.path.isfile(experiement_path + 'experiment_config.json')
    assert os.path.isfile(experiement_path + 'experiment_data.json')

    # Load the experiment files
    with open(experiement_path + 'experiment_config.json') as file:
        experiment_info = json.load(file)

    with open(experiement_path + 'experiment_data.json') as file:
        experiment_data = json.load(file)

    # Assert the experiment name
    assert experiment_info['name'] == 'log_metrics'
    # Assert that the experiments has logged the metrics correctly
    assert experiment_data['metrics']['epoch_0']['loss'] > 0.
    assert experiment_data['metrics']['epoch_0']['accuracy'] > 0.

    # Remove the project and experiment folder
    shutil.rmtree(_PROJECT_FOLDER)

def test_logger_log_callbacks():
    """
    Tests logging an experiment which only contains callbacks
    """
    # Create the project and experiment folder
    experiement_path = _PROJECT_FOLDER + 'log_callbacks/'
    os.makedirs(experiement_path)

    # Create the model checkpoint and the callback config
    checkpoints_path = experiement_path + 'checkpoint.h5'
    callbacks_configs = CallbackConfig(ModelCheckpoint(checkpoints_path))
    # Create the experiment
    experiment = Experiment(experiment_path=experiement_path,
                            name='log_callbacks',
                            configs=[callbacks_configs],
                            description='experimento de prueba')
    experiment.register_experiment()

    # Assert the experiment files exist
    assert os.path.isfile(experiement_path + 'experiment_config.json')
    assert os.path.isfile(experiement_path + 'experiment_data.json')

    # Load the experiment files
    with open(experiement_path + 'experiment_config.json') as file:
        experiment_info = json.load(file)

    with open(experiement_path + 'experiment_data.json') as file:
        experiment_data = json.load(file)

    # Assert the experiment info is right
    assert experiment_info['name'] == 'log_callbacks'
    assert experiment_info['description'] == 'experimento de prueba'
    # Assert the callback info is right
    assert experiment_data['callbacks']['ModelCheckpoint']['filepath'] \
            == checkpoints_path

    # Remove the project and experiment folder
    shutil.rmtree(_PROJECT_FOLDER)

def test_logger_log_experiment(get_trained_model):
    """
    Tests logging a full experiment
    """
    # Create the project and experiment folder
    experiement_path = _PROJECT_FOLDER + 'log_experiment/'
    os.makedirs(experiement_path)

    # Create model and metrics configs
    model, metrics = get_trained_model
    configs = [MetricsConfig(metrics), ModelConfig(model)]

    # Create the experiment
    experiment = Experiment(experiment_path=experiement_path,
                            name='log_experiment', configs=configs,
                            description='experimento de prueba')
    experiment.register_experiment()

    # Assert the experiment files exist
    assert os.path.isfile(experiement_path + 'experiment_config.json')
    assert os.path.isfile(experiement_path + 'experiment_data.json')

    # Load the experiment files
    with open(experiement_path + 'experiment_config.json') as file:
        experiment_info = json.load(file)

    with open(experiement_path + 'experiment_data.json') as file:
        experiment_data = json.load(file)

    # Assert the experiment name
    assert experiment_info['name'] == 'log_experiment'
    # Assert the experiment data
    assert 'sequential' in experiment_data['model']['model_config']['name']
    assert experiment_data['metrics']['epoch_0']['loss'] > 0.
    assert experiment_data['metrics']['epoch_0']['accuracy'] > 0.

    # Remove the project and experiment folder
    shutil.rmtree(experiement_path)

def test_logger_non_config_exception():
    """
    Tests logging an experiment with a non-config object
    """
    # Create the project and experiment folder
    experiement_path = _PROJECT_FOLDER + 'log_checkpoint/'
    os.makedirs(experiement_path)

    # Create the model checkpoint and the callback config
    checkpoints_path = experiement_path + 'checkpoint.h5'

    # Create the non config object
    checkpoint = ModelCheckpoint(checkpoints_path)
    # Create the experiment
    experiment = Experiment(experiment_path=experiement_path,
                            name='log_checkpoint',
                            configs=[checkpoint],
                            description='experimento de prueba')

    # Capture the exception generated
    with pytest.raises(ValueError) as exception_info:
        experiment.register_experiment()

    # Assert the exception message is right
    assert str(exception_info.value) == 'ModelCheckpoint is not a Config object'

    # Remove the project and experiment folder
    shutil.rmtree(_PROJECT_FOLDER)