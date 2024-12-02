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


from utils._visualization import create_tree_plot
import config
from utils._logging import LogExecutionTime
from utils._models import log_metrics
from utils._hyperparams import create_hyperparam_model


@LogExecutionTime
def train_model(X: pd.DataFrame, y: pd.Series, pipeline: Pipeline, parameters: dict, test_size: float):
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
            logging.info(f'Parameters are {parameters}')
            clf = GridSearchCV(
                pipeline, parameters, n_jobs=config.N_JOBS)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            search = clf.fit(X_train, y_train)

            # Iterate through caught warnings and log them
            for warning in caught_warnings:
                logging.warning(f'{warning.message}')

        logging.info(f'Best parameters: {search.best_params_}')

        model = search.best_estimator_

        # create_tree_plot(model, X_train)

        log_metrics(model, X_test, y_test)

        return model
    except Exception as e:
        logging.error(f'An error occurred: {e}')
        return None


