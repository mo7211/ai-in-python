import logging
from typing import List

from utils import *


def clean_data(df: DataFrame, split_option:SplitOption):
    logging.info(50*"=")
    logging.info("Start data cleaning")
    log_df_shape(df)

    cleaned_df = df

    delete_duplicates(cleaned_df)

    columns = ["rooms", "price", "area", "condition", "floor"]
    drop_null_rows(cleaned_df, columns)

    columns_to_drop = ['orientation', 'energy_costs',
                       'total_floors', 'balkonies', 'loggia', 'index', 'type']
    drop_column(cleaned_df, columns_to_drop)

    clean_year_built(cleaned_df)

    clean_year_reconstructed(cleaned_df)

    replace_value_in_column(
        cleaned_df, "condition", "New building", "Original condition")

    replace_value_in_column(
        cleaned_df, "construction_type", np.nan, "Unknown")

    replace_value_in_column(
        cleaned_df, "certificate", np.nan, "Unknown")

    clean_rows_floor(cleaned_df)

    clean_high_prices(cleaned_df)

    log_df_shape(cleaned_df)

    df.to_csv('Modularbeit/data/cleaned_data/re_cleaned.csv')

    splitted_df = split_dataframe(cleaned_df, split_option)

    splitted_df.to_csv('Modularbeit/data/cleaned_data/re_cleaned' + split_option + '.csv')

    return splitted_df
