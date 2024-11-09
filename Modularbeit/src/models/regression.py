import logging
import pandas as pd

from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

import config


def sdg_regression(X: DataFrame, y: DataFrame):
    if config.TRAIN:

        logging.info('spit training data')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=42)

        sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                               n_iter_no_change=100, random_state=42)
        logging.info('train stochastic gradient descent regressor')
        # y.ravel() because fit() expects 1D targets, not a column vector
        sgd_reg.fit(X_train, y_train.values.ravel())

        y_pred = sgd_reg.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        logging.info(f'mean squared error is: {mse}')
