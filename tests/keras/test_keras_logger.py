import pytest
import keras
import json

import os

from src.keras.keras_logger import Experiment

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

def test_keras_logger(get_model):
    model = get_model
    project_path = os.path.dirname(os.path.realpath(__file__)) + '/project/'
    experiment = Experiment(project_path=project_path, name='prueba', model=model)

    experiment.register_experiment()

    assert os.path.isdir(project_path + 'prueba')
    assert os.path.isfile(project_path + 'prueba/architecture.json')
    assert os.path.isfile(project_path + 'prueba/optimizer.json')
    assert os.path.isfile(project_path + 'prueba/experiment_config.json')