# Deep Learning Logger
DeepLearning_Logger is a Deep Learning library developed to keep track of different deep learning experiments. With experiments we refer to the different trainings of neural networks, in which the architecture, optimizers, etc. can vary. At the moment, it only supports Keras, but in the future it could be used with other Deep Learning frameworks.

## Features
- Create a project to store all related experiments.
- Define an experiment by name, description and date.
- Saves the architecture and optimizer used in training.
- Keep track of model checkpoints created during training.
- Monitoring of the different metrics used to evaluate the training of the model.
- Save all data in JSON files.

## Future features
- PyTorch module
- Add more Configs to store
- ...

## Installation

DeepLearning_Logger requires Python 3.5 or higher.

```sh
git clone https://github.com/hectorLop/DeepLearning_Logger.git
pip install -r requirements.txt
```

## Example
First, you must import the necessary classes. In this example we are going to store the model config and the training metrics, so we have imported the MetricsConfig and ModelConfig classes.
```python
from src.keras.configs import MetricsConfig, ModelConfig
from src.keras.project import Project
```
Now you must create the project and assing a name and a location
```python
project = Project(project_name='name', project_path='path')
```
At this point, you must perform the model training and then create the experiment.

```python
history = model.fit(...)
metrics = pd.DataFrame(history.history)

metric_config = MetricsConfig(metrics)
model_config = ModelConfig(model)

project.create_experiment('experiment_name', configs=[metric_config, model_config])
```

This code will create the following file structure:
```
path/project_name/
--- experiment_name/
------- experiment_config.json
------- experiment_data.json
```
The experiment_config.json contains the experiment information and the experiment_data.json contains the model and metrics information.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Hector Lopez Almazan - <lopez.almazan.hector@gmail.com> - https://github.com/hectorLop
Project Link: https://github.com/hectorLop/DeepLearning_Logger