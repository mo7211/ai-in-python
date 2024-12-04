from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from utils import log_metrics, plot_tree, is_in_pipeline


def measure_model(model: BaseEstimator, X: DataFrame, y: DataFrame) -> BaseEstimator:

    if isinstance(model, Pipeline):
        if is_in_pipeline(model, DBSCAN) or is_in_pipeline(model, MiniBatchKMeans):
            log_metrics(model, X, y)
        elif is_in_pipeline(model, DecisionTreeRegressor):
            plot_tree(model, X)






            log_metrics(model, X, y)
        else:
            log_metrics(model, X, y)

    return model


