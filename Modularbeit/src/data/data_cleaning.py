import logging

from utils import *


def clean_data(df: DataFrame):
    logging.info("Start data cleaning")

    delete_duplicates(df)

    columns = ['rooms', 'price', 'area', 'condition', 'floor']

    drop_null_rows(df, columns)
