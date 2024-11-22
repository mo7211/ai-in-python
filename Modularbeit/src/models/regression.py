import logging
import pandas as pd

from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures


import config
from utils._logging import LogExecutionTime
from utils._visualization import visualize_sdg_regressor
from utils._train import define_target
from utils._models import log_mean_squared_error


@LogExecutionTime
def train_regression(X: DataFrame, y: DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=42)
    if config.METHOD == config.ModellingMethods.SGDRegressor:
        model = sgd_regression(X_train, X_test, y_train, y_test)

    elif config.METHOD == config.ModellingMethods.LinearRegression:
        model = linear_regression(X_train, X_test, y_train, y_test)

    elif config.METHOD == config.ModellingMethods.PolynomialRegression:
        model = poly_regression(X_train, X_test, y_train, config.POLY_DEGREE)

        # The 'area' column for visualization
        column_name = 'area'  # 'area'
        # visualize_sdg_regressor(y, X, sdg_reg, column_name)

    log_mean_squared_error(model, X_test, y_test)

    return model


def poly_regression(X_train: pd.DataFrame, y_train: pd.Series, degree=2):
    logging.info('Polynomial regression started')
    poly_features = PolynomialFeatures(degree, include_bias=False)
    X_poly = poly_features.fit_transform(X_train)

    model = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                         n_iter_no_change=100, random_state=42)

    distributions = dict(penalty=[None, 'l2', 'l1', 'elasticnet'], eta0=[
                         0.5, 0.1, 0.02, 0.01])

    clf = RandomizedSearchCV(model, distributions, random_state=0, n_iter=10)

    search = clf.fit(X_poly, y_train)

    logging.info(f'Degree is {degree}')

    logging.info(f'Best parameters: {search.best_params_}')

    return search.best_estimator_


@LogExecutionTime
def sgd_regression(X_train: DataFrame, X_test: DataFrame, y_train: DataFrame, y_test: DataFrame) -> SGDRegressor:
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
