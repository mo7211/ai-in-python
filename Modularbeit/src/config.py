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
from sklearn.tree import DecisionTreeRegressor
from scikeras.wrappers import KerasRegressor
from keras import Sequential
from keras.api.layers import Dense, Input, Dropout

from utils._cleaning import SplitOption

# def create_model(input_dim, optimizer='adam'):
#     model = Sequential()
#     model.add(Dense(64, input_dim=input_dim, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(1, activation='linear'))  # Adjust output layer for regression
#     model.compile(optimizer=optimizer, loss='mean_squared_error')
#     return model


def create_model(optimizer="adam", dropout=0.1, init='uniform', nbr_features=164, dense_nparams=256):
    model = Sequential()
    model.add(Dense(dense_nparams, activation='relu',
              input_shape=(nbr_features,), kernel_initializer=init,))
    model.add(Dropout(dropout), )
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=["accuracy"])
    return model


def create_model_pca(optimizer="adam", dropout=0.1, init='uniform', nbr_features=27, dense_nparams=256):
    model = Sequential()
    model.add(Dense(dense_nparams, activation='relu',
              input_shape=(nbr_features,), kernel_initializer=init,))
    model.add(Dropout(dropout), )
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=["accuracy"])
    return model


class ModellingMethods(Enum):
    training_off = (None, None)

    pca_keras_regressor = (
        # Decision tree + dim reduction
        Pipeline([('pca', PCA()),
                  ('regressor', KerasRegressor(model=create_model_pca, verbose=0))]),
        {
            'pca__n_components': [0.95],
            # Example: You can add these to the grid search
            'regressor__epochs': [50, 100],
            'regressor__batch_size': [10, 20],
            'regressor__optimizer': ['adam', 'rmsprop'],
        })

    keras_regressor = (
        # Decision tree + dim reduction
        Pipeline([
            ('regressor', KerasRegressor(model=create_model, verbose=0))]),
        {
            'regressor__epochs': [50, 100],
            'regressor__batch_size': [10, 20],
            'regressor__optimizer': ['adam', 'rmsprop'],
        })

    pca_poly_regressor = (
        Pipeline([('pca', PCA()),
                  ('poly', PolynomialFeatures(include_bias=False)),
                  ('regressor', SGDRegressor(tol=1e-5, n_iter_no_change=100, random_state=42))]),
        {'pca__n_components': [0.95],
         'poly__degree': [1, 2, 3],
         'regressor__max_iter': [1000, 2000, 3000],
         'regressor__penalty': [None, 'l2', 'l1', 'elasticnet'],
         'regressor__eta0': [0.5, 0.1, 0.05, 0.01],
         })
    poly_regressor = (
        Pipeline([('poly', PolynomialFeatures(include_bias=False)),
                  ('regressor', SGDRegressor(tol=1e-5, n_iter_no_change=100, random_state=42))]),
        {
            'poly__degree': [1, 2, 3],
            'regressor__max_iter': [1000, 2000, 3000],
            'regressor__penalty': [None, 'l2', 'l1', 'elasticnet'],
            'regressor__eta0': [0.5, 0.1, 0.05, 0.01],
        })
    pca_scaler_poly_regressor = (
        Pipeline([('pca', PCA()),
                  ('scaler', StandardScaler()),
                  ('poly', PolynomialFeatures(include_bias=False)),
                  ('regressor', SGDRegressor(tol=1e-5, n_iter_no_change=100, random_state=42))]),
        {'pca__n_components': [0.95],
         'poly__degree': [1, 2, 3],
         'regressor__max_iter': [1000, 2000, 3000],
         'regressor__penalty': [None, 'l2', 'l1', 'elasticnet'],
         'regressor__eta0': [0.5, 0.1, 0.05, 0.01],
         })

    decision_tree = (
        Pipeline([('tree', DecisionTreeRegressor(random_state=0))]),
        {
            'tree__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, None]
        })

    pca_decision_tree = (
        Pipeline([('pca', PCA()),
                  ('tree', DecisionTreeRegressor(random_state=0))]),
        {'pca__n_components': [0.95],
         'tree__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, None]
         })

    pca_random_forest = (
        Pipeline([('pca', PCA()),
                  ('random_forest', RandomForestRegressor(random_state=0))]),
        {'pca__n_components': [0.95],
         'random_forest__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, None],
         'random_forest__n_jobs': [12]
         })
    random_forest = (
        # Decision tree + dim reduction
        Pipeline(
            [('random_forest', RandomForestRegressor(random_state=0))]),
        {
            'random_forest__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, None],
            'random_forest__n_jobs': [12]
        })
    scaler_svr = (
        # Decision tree + dim reduction
        Pipeline([('scaler', StandardScaler()),
                  ('svr', SVR())]),
        {'svr__C': [1, 0.9, 0.5, 0.1, 2, 5, 10],
         'svr__kernel': ['linear',
                         'poly', 'rbf'],
         'svr__gamma': [1]
         })
    pca_scaler_svr = (
        # Decision tree + dim reduction
        Pipeline([('pca', PCA()),
                  ('scaler', StandardScaler()),
                  ('svr', SVR())]),
        {'pca__n_components': [0.95],
         'svr__C': [1, 0.9, 0.5, 0.1, 2, 5, 10],
         'svr__kernel': ['linear',
                         'poly', 'rbf'],
         'svr__gamma': [1]
         })
    mini_batch_kmeans = (
        # Decision tree + dim reduction
        Pipeline([('scaler', StandardScaler()),
                  ('minibatchkmeans', MiniBatchKMeans(random_state=0, batch_size=100))]),
        {
            'minibatchkmeans__n_clusters': [2, 3, 4, 5, 6, 7, 8]
        })
    dbscan = (
        # Decision tree + dim reduction
        Pipeline([('scaler', StandardScaler()),
                  ('dbscan', DBSCAN(n_jobs=-1))]),
        {
            'minibatchkmeans__n_clusters': [2, 3, 4, 5, 6, 7, 8]
        })

    pca = (
        # Decision tree + dim reduction
        Pipeline([('pca', PCA())]),
        {
            'pca__n_components': [0.95]
        })

    def __init__(self, pipeline, parameters):
        self.pipeline = pipeline
        self.parameters = parameters


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
MODEL_METHOD = ModellingMethods.pca_keras_regressor
TARGET = 'price'  # 'condition'
SPLIT_OPTION = SplitOption.WITH_INDEX

PIPELINE = MODEL_METHOD.pipeline
PARAMETERS = MODEL_METHOD.parameters

if not TRAINING:
    HYPERPARAM_METHOD = HyperparamMethods.parameter_search_off
    MODEL_METHOD = ModellingMethods.training_off

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
