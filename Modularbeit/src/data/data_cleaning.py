import logging

import config
from typing import List
from utils._logging import LogExecutionTime


from utils import *


@LogExecutionTime
def clean_data(df: DataFrame, split_option: SplitOption, run:bool):
    if run:
        logging.info('Clean data')

        logging.info(50*"=")
        logging.info("Start data cleaning")
        log_df_shape(df)

        cleaned_df = df

        delete_duplicates(cleaned_df)

        columns = ["rooms", "price", "area", "condition", "floor"]
        drop_null_rows(cleaned_df, columns)

        columns_to_drop = ['orientation', 'energy_costs',
                           'total_floors', 'balkonies', 'loggia', 'type']
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

        # # Clean types
        # columns_float = ['price,'
        #                  'area',
        #                  'environment',
        #                  'quality_of_living',
        #                  'safety',
        #                  'transport',
        #                  'services',
        #                  'index',
        #                  'relax']
        # convert_column_to_type(df, columns_float, float)

        clean_high_prices(cleaned_df)
        clean_big_area(cleaned_df)

        log_df_shape(cleaned_df)

    # columns_int = ['rooms', 'year_built', 'year_reconstructed', 'floor']
    # convert_column_to_type(cleaned_df, columns_int, int)

        cleaned_df.to_csv(config.CLEANED_DATA_PATH, index=False)

        splitted_df = split_dataframe(cleaned_df, split_option)

        log_df_shape(splitted_df)

        splitted_df.to_csv(
            config.SPLITTED_DATA_PATH, index=False)

        return splitted_df
