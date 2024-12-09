import logging
import math
import numpy as np
from pandas import DataFrame
from sklearn.metrics import average_precision_score, mean_squared_error

import keras
from keras import layers


def log_metrics(model, X_test, y_test):
    if model:
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        # aps = average_precision_score(y_test, y_pred)

        logging.info(f'mean squared error is: {mse}')
        # logging.info(f'average precision score is: {aps}')
        return mse
    else:
        return None


def discretize_feature(y: DataFrame, n_bins: int):
    bins = list(np.linspace(math.floor(y.min()),
                math.ceil(y.max()), n_bins))
    binned_y = y.apply(lambda val: np.digitize(val, bins=bins))
    return binned_y

def create_model(optimizer='adagrad',
                  kernel_initializer='glorot_uniform', 
                  dropout=0.2):
    model = keras.Sequential(
    [
        # keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
    )

    return model