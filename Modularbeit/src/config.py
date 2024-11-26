from enum import Enum
from pathlib import Path
import time

from sklearn.decomposition import PCA
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from utils._cleaning import SplitOption


class ModellingMethods(Enum):
    PolynomialRegression_SGD = {'model': SGDRegressor(),
                                'search_params': dict(max_iter=[1000, 2000], tol=[1e-5], penalty=[None, 'l2', 'l1', 'elasticnet'], eta0=[
                                    0.5, 0.1, 0.05, 0.01], n_iter_no_change=[100], random_state=[0, 42]),
                                'train_params': dict(max_iter=1000, tol=1e-5, penalty='elasticnet', eta0=0.05, n_iter_no_change=100, random_state=0),
                                }

    LinearRegression = SGDRegressor()
    PolynomialRegression = SGDRegressor()
    DecisionTree = DecisionTreeClassifier()
    # RandomForrest = RandomForrest()
    # SVM = SVM()
    # kmeans = kmeans()
    # DBScan = DBScan()
    # NeuralNetwork = NeuralNetwork()
    Off = False


class HyperparamMethods(Enum):
    RandomizedSearchCV = 'RandomizedSearchCV'
    GridSearchCV = 'GridSearchCV'
    Off = False


# Options
NAME = 'DecisionTree'


CLEAN = False
VISUALIZE = False
PREPROCESS = False
REDUCE_DIMENSIONS = True
HYPERPARAM_METHOD = HyperparamMethods.RandomizedSearchCV

# Regression + dim reduction
PIPELINE = Pipeline([('pca', PCA())
                     ('poly', PolynomialFeatures(include_bias=False)),
                     ('regressor', SGDRegressor(tol=1e-5, n_iter_no_change=100, random_state=42))])
PARAMETERS = {'pca__n_components': 0.95,
              'poly__degree': [1, 2, 3],
              'regressor__max_iter': [1000, 2000, 3000],
              'regressor__penalty': [None, 'l2', 'l1', 'elasticnet'],
              'regressor__eta0': [0.5, 0.1, 0.05, 0.01],
              }



SPLIT_OPTION = SplitOption.WITH_INDEX
SHOW_PLOTS = False
TARGET = 'price'
TEST_SIZE = 0.3

# Hyperparameters
N_JOBS = 6

POLY_REG_DISTRIBUTION_RANDOM = dict(max_iter=[1000, 2000], tol=[1e-5], penalty=[None, 'l2', 'l1', 'elasticnet'], eta0=[
    0.5, 0.1, 0.05, 0.01], n_iter_no_change=[100], random_state=[42])
POLY_REG_DISTRIBUTION_GRID = dict(max_iter=[1000], tol=[
                                  1e-5], penalty=['elasticnet'], eta0=[0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09], n_iter_no_change=[100], random_state=[42])


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
