import logging
from sklearn import metrics
from sklearn.metrics import average_precision_score, mean_squared_error


def is_in_pipeline(model, model_class):
    return any(isinstance(step, model_class) for _, step in model.steps)


def log_mean_squared_error(model, X_test, y_test):
    if model:
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        # aps = average_precision_score(y_test, y_pred)

        logging.info(f'Mean squared error is: {mse}')

        # logging.info(f'average precision score is: {aps}')
        return mse
    else:
        return None


def log_max_error(model, X_test, y_test):
    if model:
        y_pred = model.predict(X_test)

        max_error = max_error(y_test, y_pred)
        logging.info(f'Max error is: {max_error}')

        return max_error
    else:
        return None


def log_average_precision_score(model, X_test, y_test):
    if model:
        y_pred = model.predict(X_test)

        aps = average_precision_score(y_test, y_pred)

        logging.info(f'Average precision score is: {aps}')
        return aps
    else:
        return None


def log_silhouette(model, X):
    labels = model.labels_
    silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
    logging.info(f'Silhouette is: {silhouette}')
