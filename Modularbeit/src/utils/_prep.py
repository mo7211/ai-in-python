

import ast
import logging
from turtle import pd

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.naive_bayes import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

import config


def binarize_labels(df: DataFrame, columns: list[str]) -> None:
    df_out = df
    for c in columns:
        
        logging.info(f'Start binarizing column {c}')
        if df_out.columns.str.contains(c).any():
            names = df_out[c]
            config.FEATURE_MAPPER[c] = names.values.reshape(-1, 1)

            one_hot = LabelBinarizer()
            df_out[c] = list(one_hot.fit_transform(config.FEATURE_MAPPER[c]))
            df_out = expand_feature(df_out, c)

    return df_out


def inverse_binarize_labels(df: DataFrame, columns: list[str]) -> None:
    for c in columns:
        logging.info(f'Start inverse binarizing column {c}')
        inverse_expand_feature(df, c)
        if df.columns.str.contains(c).any():
            one_hot = LabelBinarizer()
            df[c] = list(one_hot.inverse_transform(config.FEATURE_MAPPER[c]))
            inverse_expand_feature(df, c)


def scale_minmax(df: DataFrame, columns: list[str]):
    minmax_scale = MinMaxScaler(feature_range=(0, 1))
    df_out = df
    for c in columns:
        logging.info(f'Start scaling column {c}')
        if df_out.columns.str.contains(c).any():
            df_out[c] = minmax_scale.fit_transform(df[c].values.reshape(-1, 1))
            logging.info(f'Column {c} succesfully scaled')
            
        else:
            logging.info(f'No column {c} in dataframe')
    return df_out


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


def read_prepped_data(path: str, columns: list[str]):
    def convert_to_array(array_string):
        # Return a numpy array from the evaluated string
        return np.array(ast.literal_eval(array_string.replace('\n', ' ').strip().replace(' ', ',')))
    df = pd.read_csv(path)
    for c in columns:
        df[c] = df[c].apply(convert_to_array)

    return df


def expand_feature(df: DataFrame, column_name: str):
    features_expanded = pd.DataFrame(df[column_name].tolist(), index=df.index)

    features_expanded.columns = [f'{column_name}_{i}\
                                 ' for i in features_expanded.columns]

    df_out = pd.concat([features_expanded, df.drop(columns=[column_name])], axis=1)

    return df_out


def inverse_expand_feature(df: DataFrame, column_name: str):
    feature_cols = [
        col for col in df.columns if col.startswith(column_name + '_')]

    df[column_name] = df[feature_cols].apply(
        lambda row: np.array(row.values), axis=1)

    df = df.drop(columns=feature_cols)