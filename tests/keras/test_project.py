import pytest
import os
import tensorflow as tf
import shutil

from deeplearning_logger.keras.project import Project
from deeplearning_logger.keras.configs import MetricsConfig, ModelConfig

from tests.keras.fixtures import get_trained_model

@pytest.fixture
def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

    return model

def test_create_project():
    """
    Tests the creation of an empty project
    """
    project_path = os.path.dirname(os.path.realpath(__file__))
    name = 'project_02'
    project = Project(project_name=name, project_path=project_path)

    assert os.path.isdir(project_path + '/' + name)

def test_create_experiment(get_model):
    """
    Tests the creation of an experiment
    """
    project_path = os.path.dirname(os.path.realpath(__file__))
    name = 'project_02'
    project = Project(project_name=name, project_path=project_path)
    configs = [ModelConfig(get_model)]
    project.create_experiment('experiment_1', configs=configs)

    experiment_path = project_path + '/' + name + '/' + 'experiment_1/'

    assert os.path.isfile(experiment_path + 'experiment_config.json')
    assert os.path.isfile(experiment_path + 'experiment_data.json')

def test_create_experiment_empty_name_exception():
    """
    Tests the creation of an experiment without name
    """
    project_path = os.path.dirname(os.path.realpath(__file__))
    name = 'project_02'
    project = Project(project_name=name, project_path=project_path)

    with pytest.raises(ValueError) as exception_info:
        project.create_experiment()

    assert str(exception_info.value) == 'Must use a non empty experiment name'

def test_create_experiment_empty_configs_exception():
    """
    Tests the creation of an experiment without configurations
    """
    project_path = os.path.dirname(os.path.realpath(__file__))
    name = 'project_02'
    project = Project(project_name=name, project_path=project_path)

    with pytest.raises(ValueError) as exception_info:
        project.create_experiment(experiment_name='prueba')

    assert str(exception_info.value) == 'The configurations list is empty,'\
                                        'there are nothing to log'

def test_create_project_empty_location():
    """
    Test a project creation with no location
    """
    project = Project(project_name='project_03')

    assert os.path.isdir(os.getcwd() + '/project_03/')

    shutil.rmtree(os.getcwd() + '/project_03/')

def test_list_experiments(get_trained_model):
    """
    Test the listing the experiments inside a project
    """
    # Project location
    project_path = os.path.dirname(os.path.realpath(__file__))
    # Model and metrics to log
    model, metrics = get_trained_model

    # Create the project
    project = Project(project_name='project_03', project_path=project_path)
    # Create the configs to be stored
    model_config, metric_config = ModelConfig(model), MetricsConfig(metrics)

    # Create the first experiment
    project.create_experiment(experiment_name='experiment_1', 
                            configs=[model_config, metric_config],
                            description='First experiment')

    # Create the second experiment
    project.create_experiment(experiment_name='experiment_2', 
                            configs=[model_config, metric_config],
                            description='Second experiment')

    # Get the list of experiments inside the project
    experiments = project.list_experiments()

    assert experiments
    assert 'experiment_1' in experiments
    assert 'experiment_2' in experiments

    # Remove the project and its files
    shutil.rmtree(project_path + '/project_03/')