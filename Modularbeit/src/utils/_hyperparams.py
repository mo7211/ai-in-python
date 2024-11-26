import logging

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import config


def create_hyperparam_model(model, method, parameters):
    logging.info(f'Hyperparameter method is {method.name}')
    if method == config.HyperparamMethods.RandomizedSearchCV:
        method = config.POLY_REG_DISTRIBUTION_RANDOM
        logging.info(f'Parameters are {parameters}')
        clf = RandomizedSearchCV(
                    model, parameters, random_state=0, n_jobs=config.N_JOBS)
        return clf

    elif method == config.HyperparamMethods.GridSearchCV:
        logging.info(f'Parameters are {config.POLY_REG_DISTRIBUTION_GRID}')
        clf = GridSearchCV(model, config.POLY_REG_DISTRIBUTION_GRID, n_jobs=config.N_JOBS)
        return clf

    else:
        return model