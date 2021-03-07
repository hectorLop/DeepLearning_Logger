import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

@pytest.fixture
def get_trained_model():
    X, y = get_iris_data()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss='sparse_categorical_crossentropy', 
                optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(X, y, epochs=3)
    history = pd.DataFrame(history.history)

    return model, history

def get_iris_data():
    X = []
    y = []

    with open('tests/iris.data', 'r') as file:
        for line in file.readlines()[:-1]:
            line = line.strip()
            line = line.split(',')

            X.append(tuple(map(float, line[:-1])))
            y.append(line[-1])
            
    X = np.array(X, dtype='float32')
    y = np.array(y)
    y = LabelEncoder().fit_transform(y)

    return X, y

    

    

    

    