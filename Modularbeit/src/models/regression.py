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

def train_regression(preprocessed_df):
    y, X = define_target(preprocessed_df, 'price')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=42)

    sdg_reg = sdg_regression(X_train, X_test, y_train, y_test)

    # The 'area' column for visualization
    column_name = 'area'  # 'area'
    visualize_sdg_regressor(y, X, sdg_reg, column_name)

    regression(X_train, X_test, y_train, y_test)




@LogExecutionTime
def sdg_regression(X_train: DataFrame, X_test: DataFrame, y_train: DataFrame, y_test: DataFrame) -> SGDRegressor:
    if config.TRAIN:
        logging.info('spit training data')

        sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                               n_iter_no_change=100, random_state=42)

        logging.info('train stochastic gradient descent regressor')
        sgd_reg.fit(X_train, y_train.values.ravel())

        y_pred = sgd_reg.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        logging.info(f'mean squared error is: {mse}')

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
