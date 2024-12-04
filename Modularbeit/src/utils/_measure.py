from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from utils import log_metrics, plot_tree


def measure_model(model: BaseEstimator, X: DataFrame, y: DataFrame) -> BaseEstimator:

    if isinstance(model, Pipeline):
        if any(isinstance(step, DBSCAN) for _, step in model.steps) or any(isinstance(step, MiniBatchKMeans) for _, step in model.steps):
            log_metrics(model, X, y)
        elif any(isinstance(step, DecisionTreeRegressor) for _, step in model.steps):
            plot_tree(model, X)

            log_metrics(model, X, y)
        else:
            log_metrics(model, X, y)

    return model
