

import logging

from pandas import DataFrame
from sklearn.naive_bayes import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

import config


def binarize_labels(df: DataFrame, columns: list[str]) -> None:
    for c in columns:
        logging.info(f'Start binarizing column {c}')
        if df.columns.str.contains(c).any():
            names = df[c]
            config.FEATURE_MAPPER[c] = names.values.reshape(-1, 1)

            one_hot = LabelBinarizer()
            df[c] = list(one_hot.fit_transform(config.FEATURE_MAPPER[c]))


def inverse_binarize_labels(df: DataFrame, columns: list[str]) -> None:
    for c in columns:
        logging.info(f'Start inverse binarizing column {c}')
        if df.columns.str.contains(c).any():
            one_hot = LabelBinarizer()
            df[c] = list(one_hot.inverse_transform(config.FEATURE_MAPPER[c]))


def scale_minmax(df: DataFrame, columns: list[str]):
    minmax_scale = MinMaxScaler(feature_range=(0, 1))
    for c in columns:
        logging.info(f'Start scaling column {c}')
        if df.columns.str.contains(c).any():
            df[c] = minmax_scale.fit_transform(df[c].values.reshape(-1, 1))
            logging.info(f'Column {c} succesfully scaled')
        else:
            logging.info(f'No column {c} in dataframe')


def inverse_minmax(df: DataFrame, columns: list[str]):
    minmax_scale = MinMaxScaler(feature_range=(0, 1))
    for c in columns:
        logging.info(f'Start inverse scaling column {c}')
        if df.columns.str.contains(c).any():
            df[c] = minmax_scale.inverse_transform(df[c].values.reshape(-1, 1))
            logging.info(f'Column {c} succesfully scaled inverse')
        else:
            logging.info(f'No column {c} in dataframe')


def map_values(df: DataFrame, column: str, condition_mapper: dict):
    df[column] = df[column].replace(condition_mapper)


def inverse_mapper(mapper: dict):
    return {v: k for k, v in mapper.items()}
