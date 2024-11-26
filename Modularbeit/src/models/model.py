import logging
import warnings
import pandas as pd

from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures


import config
from utils._logging import LogExecutionTime
from utils._models import log_metrics
from utils._hyperparams import create_hyperparam_model


@LogExecutionTime
def train_model(X: pd.DataFrame, y: pd.Series, pipeline:Pipeline, parameters:dict, test_size: float):
    logging.info('Training model started')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    try:
        logging.info(f'Hyperparameter method is {
                     config.HYPERPARAM_METHOD.name}')
        if config.HYPERPARAM_METHOD == config.HyperparamMethods.RandomizedSearchCV:
            logging.info(f'Parameters are {parameters}')
            clf = RandomizedSearchCV(
                pipeline, parameters, random_state=0, n_jobs=config.N_JOBS)

        elif config.HYPERPARAM_METHOD == config.HyperparamMethods.GridSearchCV:
            logging.info(f'Parameters are {config.POLY_REG_DISTRIBUTION_GRID}')
            clf = GridSearchCV(
                pipeline, config.POLY_REG_DISTRIBUTION_GRID, n_jobs=config.N_JOBS)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            search = clf.fit(X_train, y_train)

            # Iterate through caught warnings and log them
            for warning in caught_warnings:
                logging.warning(f'{warning.message}')

        logging.info(f'Best parameters: {search.best_params_}')

        model = search.best_estimator_

        log_metrics(model, X_test, y_test)

        # return model
    except Exception as e:
        logging.error(f'An error occurred: {e}')
        return None


@LogExecutionTime
def sgd_regression(X_train: DataFrame, y_train: DataFrame) -> SGDRegressor:
    logging.info('spit training data')

    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                           n_iter_no_change=100, random_state=42)

    logging.info('train stochastic gradient descent regressor')
    sgd_reg.fit(X_train, y_train.values.ravel())

    return sgd_reg


@LogExecutionTime
def linear_regression(X_train: DataFrame, y_train: DataFrame) -> LinearRegression:
    logging.info('train linear regressor')

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train.values.ravel())

    return lin_reg
