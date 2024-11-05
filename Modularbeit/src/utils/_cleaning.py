import logging

from pandas import DataFrame


def logg_cleaning_rows(name, rows_before, rows_after):
    logging.info(name)
    logging.info(str(rows_before - rows_after) + ' rows deleted.')


def delete_duplicates(df: DataFrame):
    rows_before = df.shape[0]

    df.drop_duplicates()

    rows_after = df.shape[0]
    logg_cleaning_rows('Delete duplicates', rows_before, rows_after)


def drop_null_rows(df: DataFrame, columns: list):
    rows_before = df.shape[0]

    df.dropna(axis=0, how='any', subset=columns, inplace=True)

    rows_after = df.shape[0]
    message = "Delete rows, if any of {}".format(", ".join(columns))
    logg_cleaning_rows(message, rows_before, rows_after)
