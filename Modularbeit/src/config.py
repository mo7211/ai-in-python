from enum import Enum
from pathlib import Path
import time

from sklearn.linear_model import SGDRegressor
from utils._cleaning import SplitOption


class ModellingMethods(Enum):
    SGDRegressor = 'SGDRegressor'
    LinearRegression = 'LinearRegression'
    PolynomialRegression = 'PolynomialRegression'
    DecisionTree = 'DecisionTree'
    RandomForrest = 'RandomForrest'
    SVM = 'SVM'
    kmeans = 'kmeans'
    DBScan = 'DBScan'
    NeuralNetwork = 'NeuralNetwork'
    NoTraining = 'NoTraining'


class HyperparamMethods(Enum):
    RandomizedSearchCV = 'RandomizedSearchCV'
    GridSearchCV = 'GridSearchCV'


class HyperparametersFitting(Enum):
    On = True
    Off = False


# Options
NAME = 'DecisionTree'


CLEAN = False
PREPROCESS = False
MODEL_METHOD = ModellingMethods.PolynomialRegression
HYPERPARAM_METHOD = HyperparamMethods.RandomizedSearchCV

SPLIT_OPTION = SplitOption.WITH_INDEX
SHOW_PLOTS = False
TARGET = 'price'
TEST_SIZE = 0.3
POLY_DEGREES = [1, 2, 3]

# Hyperparameters

POLY_REG_DISTRIBUTION_RANDOM = dict(tol=[1e-5], penalty=[None, 'l2', 'l1', 'elasticnet'], eta0=[
    0.05, 0.01, 0.005])
POLY_REG_DISTRIBUTION_GRID = dict(max_iter=[1000], tol=[
                                  1e-5], penalty=[None], eta0=[0.01], n_iter_no_change=[100], random_state=[42])


# Data

INPUT_DATA_PATH = 'Modularbeit/data/raw_data/Real Estate Dataset.csv'
CLEANED_DATA_PATH = 'Modularbeit/data/cleaned_data/re_cleaned.csv'
SPLITTED_DATA_PATH = 'Modularbeit/data/cleaned_data/re_cleaned_' + \
    SPLIT_OPTION.value + '.csv'
PREPROCESSED_DATA_PATH = 'Modularbeit/data/features/re_preprosessed_' + \
    SPLIT_OPTION.value

# Paths

IMAGES_PATH = Path('Modularbeit') / 'images' / \
    (MODEL_METHOD.name + "_" + time.strftime("%Y-%m-%d_%H-%M-%S"))
IMAGES_PATH.mkdir(parents=True, exist_ok=True)
LOGGING_PATH = 'Modularbeit/logging'

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
