import pytest
import os
import keras

from src.keras.project import Project

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

def test_create_project():
    project_path = os.path.dirname(os.path.realpath(__file__))
    name = 'project_01'
    project = Project(project_path, name)

    assert os.path.isdir(project_path + '/' + name)

def test_create_experiment(get_model):
    project_path = os.path.dirname(os.path.realpath(__file__))
    name = 'project_01'

    project = Project(project_path, name)

    project.create_experiment('experiment_1', get_model)

    experiment_path = project_path + '/' + name + '/' + 'experiment_1'

    assert os.path.isdir(experiment_path)
    assert os.path.isfile(experiment_path + '/architecture.json')
    assert os.path.isfile(experiment_path + '/optimizer.json')
    assert os.path.isfile(experiment_path + '/experiment_config.json')