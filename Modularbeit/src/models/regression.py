import logging
import warnings
import pandas as pd

from pandas import DataFrame
from sklearn.discriminant_analysis import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


import config
from utils._logging import LogExecutionTime
from utils._models import log_metrics


@LogExecutionTime
def train_regression(X: DataFrame, y: DataFrame):
    # if config.MODEL_METHOD == config.ModellingMethods.SGDRegressor:
    #     model = sgd_regression(X_train, y_train)

    # elif config.MODEL_METHOD == config.ModellingMethods.LinearRegression:
    #     model = linear_regression(X_train, y_train)

    if config.MODEL_METHOD == config.ModellingMethods.PolynomialRegression:
        degrees = config.POLY_DEGREES
        poly_regression(X, y, degrees)

    # return model


@LogExecutionTime
def poly_regression(X: pd.DataFrame, y: pd.Series, degrees:list[int]):
    logging.info('Polynomial regression started')
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=42)
    for degree in degrees:            
        try:
            logging.info(f'Degree is {degree}')
            poly_features = PolynomialFeatures(degree, include_bias=False)
            X_poly = poly_features.fit_transform(X_train)

            model = SGDRegressor(max_iter=1000,
                                n_iter_no_change=100, random_state=42)
            # (max_iter=1000, tol=1e-5, penalty=None, eta0=0.01, n_iter_no_change=100, random_state=42)

            logging.info(f'Hyperparameter method is {config.HYPERPARAM_METHOD.name}')
            if config.HYPERPARAM_METHOD == config.HyperparamMethods.RandomizedSearchCV:
                logging.info('Fit hyperparameters with RandomizedSearchCV')
                logging.info(f'Parameters are {config.POLY_REG_DISTRIBUTION_RANDOM}')
                clf = RandomizedSearchCV(
                    model, config.POLY_REG_DISTRIBUTION_RANDOM, random_state=0)

            elif config.HYPERPARAM_METHOD == config.HyperparamMethods.GridSearchCV:
                logging.info('Fit hyperparameters with GridSearchCV')
                logging.info(f'Parameters are {config.POLY_REG_DISTRIBUTION_GRID}')
                clf = GridSearchCV(model, config.POLY_REG_DISTRIBUTION_GRID)

            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                search = clf.fit(X_poly, y_train)

                # Iterate through caught warnings and log them
                for warning in caught_warnings:
                    logging.warning(f'{warning.message}')

            logging.info(f'Best parameters: {search.best_params_}')

            model = search.best_estimator_

            poly_features = PolynomialFeatures(degree, include_bias=False)
            X_test = poly_features.fit_transform(X_test)

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
