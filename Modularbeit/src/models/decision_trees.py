import logging
import math
from random import uniform
import numpy as np
from pandas import DataFrame


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils._models import log_mean_squared_error
import config
from utils._logging import LogExecutionTime
from utils._train import define_target


@LogExecutionTime
def train_decision_tree(X: DataFrame, y: DataFrame, n_bins=1000):

    if config.METHOD == config.METHODS.DecisionTree:
        # generate the classification object
        logging.info('Start training decision tree')

        bins = list(np.linspace(math.floor(y.min()), math.ceil(y.max()), n_bins))
        binned_y = y.apply(lambda val: np.digitize(val, bins=bins))

        X_train, X_test, y_train, y_test = train_test_split(
            X, binned_y, test_size=config.TEST_SIZE, random_state=42)

        # train the model
        decisiontree = DecisionTreeClassifier(random_state=0)
        model = decisiontree.fit(X_train, y_train)

    log_mean_squared_error(model, X_test, y_test)

    model = LogisticRegression(solver='saga', tol=1e-2, max_iter=200, random_state=0)

    distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])

    clf = RandomizedSearchCV(model, distributions, random_state=0)

    search = clf.fit(iris.data, iris.target)
    search.best_params_



