import logging
from sklearn.metrics import mean_squared_error


def log_mean_squared_error(model, X_test, y_test):
    if model:
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        logging.info(f'mean squared error is: {mse}')
        return mse
    else:
        return None
