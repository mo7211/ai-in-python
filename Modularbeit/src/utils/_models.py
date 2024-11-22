import logging
import math
import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_squared_error


def log_mean_squared_error(model, X_test, y_test):
    if model:
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        logging.info(f'mean squared error is: {mse}')
        return mse
    else:
        return None


def discretize_feature(y: DataFrame, n_bins: int):
    bins = list(np.linspace(math.floor(y.min()),
                math.ceil(y.max()), n_bins))
    binned_y = y.apply(lambda val: np.digitize(val, bins=bins))
    return binned_y
