import logging
import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor

import config
from utils._visualization import *
from utils._models import *
from utils._metrics import *
# import log_mean_squared_error, plot_tree, is_in_pipeline, plot_feature_importance, plot_learning_curve, log_silhouette


def measure_model(model: BaseEstimator, X_test: DataFrame, y_test: DataFrame, X: DataFrame, y: DataFrame) -> BaseEstimator:
    metrics = {}

    if model is not None and X is not None and y is not None and isinstance(model, Pipeline):
        logging.info('Generate metrics')
        if is_in_pipeline(model, DBSCAN) or is_in_pipeline(model, MiniBatchKMeans):
            log_mean_squared_error(model, X_test, y_test)

            labels = np.unique(y_test.values)
            logging.info(f'Unique labels are: {labels}')

            log_silhouette(model, X_test, metrics)
        else:

            if is_in_pipeline(model, DecisionTreeRegressor):
                plot_tree(model, X_test, 'Decision Tree Regressor')

                plot_feature_importance(model, X_test, 'tree')

                # log_average_precision_score(model, X_test, y_test, metrics)

            elif is_in_pipeline(model, RandomForestRegressor):
                # plot_tree(model, X_test, 'Random Forest Regressor')

                plot_feature_importance(model, X_test, 'random_forest')
                # plot_learning_curve(model, X, y)

                # log_average_precision_score(model, X_test, y_test, metrics)

            elif is_in_pipeline(model, LinearRegression) or is_in_pipeline(model, SGDRegressor):
                plot_learning_curve(model, X, y)

            # elif is_in_pipeline(model, SVC) or is_in_pipeline(model, SVR):
            #     # Lecture_05_Support_Vector_Machines_8_solution 2 Grafiken
            #     placeholder = 1

            log_mean_squared_error(model, X_test, y_test, metrics)
            log_max_error(model, X_test, y_test, metrics)
            log_explained_variance_score(model, X_test, y_test, metrics)

            write_run_metrics_to_csv(
                config.METRICS_PATH, config.MODEL_METHOD.name, metrics)

            # write those to csv and create diagram

        return model
    else:
        logging.info('Inputs not succifient to create metrics')
