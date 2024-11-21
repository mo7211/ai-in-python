import logging
import pandas as pd

from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor


import config
from utils._logging import LogExecutionTime
from utils._visualization import visualize_sdg_regressor
from utils._train import define_target
from utils._models import log_mean_squared_error


@LogExecutionTime
def train_regression(X: DataFrame, y: DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=42)
    if config.METHOD == config.METHODS.SGDRegressor:
        model = sgd_regression(X_train, X_test, y_train, y_test)
    elif config.METHOD == config.METHODS.LinearRegression:
        model = regression(X_train, X_test, y_train, y_test)

        # The 'area' column for visualization
        column_name = 'area'  # 'area'
        # visualize_sdg_regressor(y, X, sdg_reg, column_name)

    log_mean_squared_error(model, X_test, y_test)


@LogExecutionTime
def sgd_regression(X_train: DataFrame, X_test: DataFrame, y_train: DataFrame, y_test: DataFrame) -> SGDRegressor:
    if config.TRAIN:
        logging.info('spit training data')

        sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                               n_iter_no_change=100, random_state=42)

        logging.info('train stochastic gradient descent regressor')
        sgd_reg.fit(X_train, y_train.values.ravel())

        return sgd_reg


@LogExecutionTime
def regression(X_train: DataFrame, X_test: DataFrame, y_train: DataFrame, y_test: DataFrame) -> LinearRegression:
    """
    Performs linear or stochastic gradient descent regression on the provided data.

    Args:
        X (DataFrame): Feature data.
        y (DataFrame): Target variable.

    Returns:
        None

    Notes:
        If TRAIN is set to True in config, this function will split the data into training and testing sets,
        train a linear regressor or stochastic gradient descent regressor on the training data, make predictions
        on the test data, and log the mean squared error.
    """
    if config.TRAIN:
        logging.info('train linear regressor')

        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train.values.ravel())

        y_pred = lin_reg.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        logging.info(f'mean squared error is: {mse}')

        return lin_reg
