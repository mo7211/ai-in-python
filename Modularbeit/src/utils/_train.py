import logging
from pandas import DataFrame

import config


def define_target(df: DataFrame, column_name: str):
    logging.info(f'Target is \'{config.TARGET}\'')
    y = df[column_name]

    # features
    X = df.drop(columns=[column_name])
    return y, X
