import logging
import os
import pandas as pd
from sklearn import metrics
from sklearn.metrics import average_precision_score, explained_variance_score, max_error, mean_squared_error

from utils._logging import get_time


def is_in_pipeline(model, model_class):
    return any(isinstance(step, model_class) for _, step in model.steps)


def log_mean_squared_error(model, X_test, y_test, metrics_: dict):
    if model:
        y_pred = model.predict(X_test)

        metrics_['mean squared error'] = mean_squared_error(y_test, y_pred)

        logging.info(f'Mean squared error is: {
                     metrics_['mean squared error']}')


def log_max_error(model, X_test, y_test, metrics_: dict):
    if model:
        y_pred = model.predict(X_test)

        metrics_['max error'] = max_error(y_test, y_pred)
        logging.info(f'Max error is: {metrics_['max error']}')


def log_average_precision_score(model, X_test, y_test, metrics_: dict):
    if model:
        y_pred = model.predict(X_test)

        metrics_['average precision score'] = average_precision_score(
            y_test, y_pred)

        logging.info(f'Average precision score is: {
                     metrics_['average precision score']}')


def log_silhouette(model, X: pd.DataFrame, metrics_: dict):
    labels = model.labels_
    metrics_['silhouette'] = metrics.silhouette_score(
        X, labels, metric='euclidean')
    logging.info(f'Silhouette is: {metrics_['silhouette']}')


def log_explained_variance_score(model, X_test, y_test, metrics_: dict):
    if model:
        y_pred = model.predict(X_test)

        metrics_['explained variance score'] = explained_variance_score(
            y_test, y_pred)

        logging.info(f'Explained variance score is: {
                     metrics_['explained variance score']}, 1.0 is best')


def ensure_directory_exists(filepath):
    # Get the directory name from the filepath
    directory = os.path.dirname(filepath)

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_run_metrics_to_csv(filepath, run_name, metrics):

    ensure_directory_exists(filepath)

    # Convert metrics dictionary to a DataFrame with the run name included
    metrics_df = pd.DataFrame([metrics])
    metrics_df['run name'] = run_name
    metrics_df['run time'] = get_time()

    # Check if the file already exists
    if os.path.exists(filepath):
        # Read the existing dataframe
        existing_df = pd.read_csv(filepath)

        # Ensure both DataFrames have the same columns
        combined_columns = existing_df.columns.union(metrics_df.columns)
        existing_df = existing_df.reindex(
            columns=combined_columns, fill_value=pd.NA)
        metrics_df = metrics_df.reindex(
            columns=combined_columns, fill_value=pd.NA)

        if run_name in existing_df['run name'].values:
            # Overwrite the existing row with the new metrics
            row_index = existing_df.index[existing_df['run name'] == run_name].tolist()[
                0]
            combined_df = existing_df.copy()
            combined_df.iloc[row_index] = metrics_df.iloc[0]
        else:
            # Append the new metrics as it doesn't already exist
            combined_df = pd.concat(
                [existing_df, metrics_df], ignore_index=True)
    else:
        # If the file doesn't exist, start fresh
        combined_df = metrics_df

    # Write the combined DataFrame to the CSV
    combined_df.to_csv(filepath, index=False)
