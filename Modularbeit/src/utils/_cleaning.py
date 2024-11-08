from enum import Enum
import logging
import numpy as np
from pandas import DataFrame
from typing import List, Union


def delete_duplicates(df: DataFrame):
    rows_before = df.shape[0]

    df.drop_duplicates()

    rows_after = df.shape[0]
    logging.info(str(rows_before - rows_after) + ' rows deleted.')


def drop_null_rows(df: DataFrame, columns: List[str] = None):
    rows_before = df.shape[0]

    df.dropna(axis=0, how='any', subset=columns, inplace=True)

    rows_after = df.shape[0]
    logging.info(str(rows_before - rows_after) + ' rows deleted.')


def clean_rows_floor(df: DataFrame):
    rows_before = df.shape[0]

    df.drop(df[df['floor'] < 0].index, inplace=True)

    rows_after = df.shape[0]
    logging.info(str(rows_before - rows_after) + ' rows deleted.')
    return df


def replace_value_in_column(df: DataFrame, column: str, value_old: str, value_new: str):
    rows_before = df.shape[0]

    df[column] = df[column].replace(value_old, value_new)

    rows_after = df.shape[0]
    logging.info(str(rows_before - rows_after) + ' rows deleted.')
    return df


def drop_column(df: DataFrame, columns: List[str] = None):
    rows_before = df.shape[0]

    df.drop(columns, axis=1, inplace=True)

    rows_after = df.shape[0]
    logging.info(str(rows_before - rows_after) + ' rows deleted.')
    return df


def clean_year_built(df: DataFrame):
    rows_before = df.shape[0]
    # Drop rows with invalid year_built values (before 1800 or not -1) or after 2023
    df.drop(df[((df['year_built'] < 1800) & (
        df['year_built'] != -1)) | (df['year_built'] > 2023)].index, inplace=True)
    df['year_built'] = df['year_built'].replace(
        -1, np.nan)

    # Replace NaN in year_built with median value of the column
    df['year_built'] = df['year_built'].replace(
        np.nan, df['year_built'].median())
    rows_after = df.shape[0]
    logging.info(str(rows_before - rows_after) + ' rows deleted.')
    return df


def clean_year_reconstructed(df: DataFrame):
    rows_before = df.shape[0]
    df['last_reconstruction'] = df['last_reconstruction'].fillna(
        df['year_built'])
    df.drop(df[df['last_reconstruction'] > 2023].index, inplace=True)

    rows_after = df.shape[0]
    logging.info(str(rows_before - rows_after) + ' rows deleted.')
    return df


def clean_high_prices(df: DataFrame):
    rows_before = df.shape[0]

    df.drop(df[df['price'] > 2000000].index, inplace=True)

    rows_after = df.shape[0]
    logging.info(str(rows_before - rows_after) + ' rows deleted.')
    return df


class SplitOption(Enum):
    WITH_INDEX = 'with_index'
    WITHOUT_INDEX = 'without_index'

def split_dataframe(df:DataFrame, option: Union[SplitOption, str]) -> None:
    if isinstance(option, str):
        option = SplitOption(option)

    if option == SplitOption.WITH_INDEX:
        logging.info('splitting dataframe with index')
        df_with_index = df.dropna(axis=0, how='any', subset=['quality_of_living', 'safety', 'transport', 'services', 'relax','environment'], inplace=False)
        return df_with_index
    
    elif option == SplitOption.WITHOUT_INDEX:
        logging.info('splitting dataframe without index')
        df_without_index = df.drop(['quality_of_living', 'safety', 'transport', 'services', 'relax','environment'], axis=1, inplace=False)
        return df_without_index
    else:
        raise ValueError("Ungültige Option ausgewählt.")

