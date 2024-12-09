import logging
import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns

import config
from utils import log_metrics, plot_tree, is_in_pipeline, plot_feature_importance


def measure_model(model: BaseEstimator, X_test: DataFrame, y_test: DataFrame, X: DataFrame, y: DataFrame) -> BaseEstimator:
    if model is not None and X is not None and y is not None and isinstance(model, Pipeline):
        logging.info('Generate metrics')
        if is_in_pipeline(model, DBSCAN) or is_in_pipeline(model, MiniBatchKMeans):
            log_metrics(model, X_test, y_test)

            labels = np.unique(y_test.values)
            logging.info(f'Unique labels are: {labels}')

        elif is_in_pipeline(model, DecisionTreeRegressor):
            plot_tree(model, X_test)

            plot_feature_importance(model, X_test)

            log_metrics(model, X_test, y_test)

        elif is_in_pipeline(model, SVC) or is_in_pipeline(model, SVR):
            # Lecture_05_Support_Vector_Machines_8_solution 2 Grafiken
            log_metrics(model, X_test, y_test)
        else:
            log_metrics(model, X_test, y_test)

        return model
    else:
        logging.info('Inputs not succifient to create metrics')
