from enum import Enum
from pathlib import Path
import time

from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from utils._cleaning import SplitOption
from scikeras.wrappers import KerasRegressor


class ModellingMethods(Enum):
    pca_poly_regressor = 1
    poly_regressor = 1.1
    pca_decision_tree = 2
    pca_random_forest = 3
    pca_scaler_svr = 4
    scaler_svr = 5
    decision_tree = 6
    random_forest = 7

    pca_scaler_poly_regressor = 8

    mini_batch_kmeans = 10
    kmeans = 11
    dbscan = 12

    keras_regressor = 20

    pca = 99

    training_off = False


class HyperparamMethods(Enum):
    RandomizedSearchCV = 'RandomizedSearchCV'
    GridSearchCV = 'GridSearchCV'
    parameter_search_off = False


# Options
VISUALIZE = False

CLEAN = False
PREPROCESS = False
REDUCE_DIMENSIONS = False
TRAINING = True
HYPERPARAM_METHOD = HyperparamMethods.RandomizedSearchCV
MODEL_METHOD = ModellingMethods.poly_regressor
TARGET = 'price'  # 'condition'
SPLIT_OPTION = SplitOption.WITH_INDEX

PIPELINE = None
PARAMETERS = None

if not TRAINING:
    HYPERPARAM_METHOD = HyperparamMethods.parameter_search_off
    MODEL_METHOD = ModellingMethods.training_off


# Regression + dim reduction
if HYPERPARAM_METHOD == HyperparamMethods.RandomizedSearchCV:
    if MODEL_METHOD == ModellingMethods.pca_poly_regressor:
        PIPELINE = Pipeline([('pca', PCA()),
                            ('poly', PolynomialFeatures(include_bias=False)),
                            ('regressor', SGDRegressor(tol=1e-5, n_iter_no_change=100, random_state=42))])
        PARAMETERS = {'pca__n_components': [0.95],
                      'poly__degree': [1, 2, 3],
                      'regressor__max_iter': [1000, 2000, 3000],
                      'regressor__penalty': [None, 'l2', 'l1', 'elasticnet'],
                      'regressor__eta0': [0.5, 0.1, 0.05, 0.01],
                      }
    if MODEL_METHOD == ModellingMethods.poly_regressor:
        PIPELINE = Pipeline([('poly', PolynomialFeatures(include_bias=False)),
                            ('regressor', SGDRegressor(tol=1e-5, n_iter_no_change=100, random_state=42))])
        PARAMETERS = {
            'poly__degree': [1, 2, 3],
            'regressor__max_iter': [1000, 2000, 3000],
            'regressor__penalty': [None, 'l2', 'l1', 'elasticnet'],
            'regressor__eta0': [0.5, 0.1, 0.05, 0.01],
        }
    if MODEL_METHOD == ModellingMethods.pca_scaler_poly_regressor:
        PIPELINE = Pipeline([('pca', PCA()),
                             ('scaler', StandardScaler()),
                            ('poly', PolynomialFeatures(include_bias=False)),
                            ('regressor', SGDRegressor(tol=1e-5, n_iter_no_change=100, random_state=42))])
        PARAMETERS = {'pca__n_components': [0.95],
                      'poly__degree': [1, 2, 3],
                      'regressor__max_iter': [1000, 2000, 3000],
                      'regressor__penalty': [None, 'l2', 'l1', 'elasticnet'],
                      'regressor__eta0': [0.5, 0.1, 0.05, 0.01],
                      }
    elif MODEL_METHOD == ModellingMethods.decision_tree:
        # Decision tree + dim reduction
        PIPELINE = Pipeline([('tree', DecisionTreeRegressor(random_state=0))])
        PARAMETERS = {
            'tree__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, None]
        }
    elif MODEL_METHOD == ModellingMethods.pca_decision_tree:
        # Decision tree + dim reduction
        PIPELINE = Pipeline([('pca', PCA()),
                            ('tree', DecisionTreeRegressor(random_state=0))])
        PARAMETERS = {'pca__n_components': [0.95],
                      'tree__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, None]
                      }
    elif MODEL_METHOD == ModellingMethods.pca_random_forest:
        # Decision tree + dim reduction
        PIPELINE = Pipeline([('pca', PCA()),
                            ('random_forest', RandomForestRegressor(random_state=0))])
        PARAMETERS = {'pca__n_components': [0.95],
                      'random_forest__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, None],
                      'random_forest__n_jobs': [12]
                      }
    elif MODEL_METHOD == ModellingMethods.random_forest:
        # Decision tree + dim reduction
        PIPELINE = Pipeline(
            [('random_forest', RandomForestRegressor(random_state=0))])
        PARAMETERS = {
            'random_forest__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, None],
            'random_forest__n_jobs': [12]
        }
    elif MODEL_METHOD == ModellingMethods.scaler_svr:
        # Decision tree + dim reduction
        PIPELINE = Pipeline([('scaler', StandardScaler()),
                            ('svr', SVR())])
        PARAMETERS = {'svr__C': [1, 0.9, 0.5, 0.1, 2, 5, 10],
                      'svr__kernel': ['linear',
                                      'poly', 'rbf'],
                      'svr__gamma': [1]
                      }
    elif MODEL_METHOD == ModellingMethods.pca_scaler_svr:
        # Decision tree + dim reduction
        PIPELINE = Pipeline([('pca', PCA()),
                            ('scaler', StandardScaler()),
                            ('svr', SVR())])
        PARAMETERS = {'pca__n_components': [0.95],
                      'svr__C': [1, 0.9, 0.5, 0.1, 2, 5, 10],
                      'svr__kernel': ['linear',
                                      'poly', 'rbf'],
                      'svr__gamma': [1]
                      }
    elif MODEL_METHOD == ModellingMethods.mini_batch_kmeans:
        # Decision tree + dim reduction
        PIPELINE = Pipeline([('scaler', StandardScaler()),
                            ('minibatchkmeans', MiniBatchKMeans(random_state=0, batch_size=100))])
        PARAMETERS = {
            'minibatchkmeans__n_clusters': [2, 3, 4, 5, 6, 7, 8]
        }
    elif MODEL_METHOD == ModellingMethods.dbscan:
        # Decision tree + dim reduction
        PIPELINE = Pipeline([('scaler', StandardScaler()),
                            ('dbscan', DBSCAN(n_jobs=-1))])
        PARAMETERS = {
            'minibatchkmeans__n_clusters': [2, 3, 4, 5, 6, 7, 8]
        }
    elif MODEL_METHOD == ModellingMethods.pca:
        # Decision tree + dim reduction
        PIPELINE = Pipeline([('pca', PCA())])
        PARAMETERS = {
            'pca__n_components': [0.95]
        }
    elif MODEL_METHOD == ModellingMethods.keras_regressor:
        # Decision tree + dim reduction
        PIPELINE = Pipeline([('pca', PCA()),
                             ('clf', KerasRegressor())])
        PARAMETERS = {
            'pca__n_components': [0.95]
        }


SHOW_PLOTS = False
BINARIZE = False
TEST_SIZE = 0.3

# Hyperparameters
N_JOBS = 6

# Data

INPUT_DATA_PATH = 'Modularbeit/data/raw_data/Real Estate Dataset.csv'
CLEANED_DATA_PATH = 'Modularbeit/data/cleaned_data/re_cleaned.csv'
SPLITTED_DATA_PATH = 'Modularbeit/data/cleaned_data/re_cleaned_' + \
    SPLIT_OPTION.value + '.csv'
PREPROCESSED_DATA_PATH = 'Modularbeit/data/features/re_preprosessed_' + \
    SPLIT_OPTION.value

METRICS_PATH = 'Modularbeit/data/metrics/metrics.csv'

# Paths

IMAGES_PATH = Path('Modularbeit') / 'images'
IMAGES_PATH.mkdir(parents=True, exist_ok=True)
LOGGING_PATH = 'Modularbeit/logging/' + \
    SPLIT_OPTION.value

# Preprossesing

FEATURE_MAPPER = {'condition': {'Development project': 1,
                                'Under construction': 2,
                                'Original condition': 3,
                                'Partial reconstruction': 4,
                                'Complete reconstruction': 5
                                },
                  'certificate': {'Unknown': 1,
                                  'none': 2,
                                  'G': 3,
                                  'F': 4,
                                  'E': 5,
                                  'D': 6,
                                  'C': 7,
                                  'B': 8,
                                  'A': 9
                                  }
                  }
