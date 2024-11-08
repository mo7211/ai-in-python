

import logging

from pandas import DataFrame
from sklearn.naive_bayes import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler


def binarize_labels(df: DataFrame, column: str):
    names = df[column]
    feature_name_nsi = names.values.reshape(-1, 1)

    one_hot = LabelBinarizer()
    df[column] = list(one_hot.fit_transform(feature_name_nsi))


def scale_minmax(df: DataFrame, columns: list[str]):
    minmax_scale = MinMaxScaler(feature_range=(0, 1))
    for c in columns:
        logging.info(f'Start scaling column {c}')
        if df.columns.str.contains(c).any():
            df[c] = minmax_scale.fit_transform(df[c].values.reshape(-1, 1))
            logging.info(f'Column {c} succesfully scaled')
        else:
            logging.info(f'No column {c} in dataframe')


def map_values(df: DataFrame, column: str, condition_mapper: dict):
    df[column] = df[column].replace(condition_mapper)


def has_column_with_title(df: DataFrame, title: str) -> bool:
    return df.columns.str.contains(title).any()

    # # ## Split data-set
    # minmax_scale = MinMaxScaler(feature_range=(0, 1))
