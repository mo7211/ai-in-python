import logging

import config
from utils import *


@LogExecutionTime
def prep_data(df: DataFrame, run:bool):
    if run:
        logging.info(50*"=")
        logging.info("Start data preprocessing")

        # scale Minmax
        columns_minmax = ['price',
                          'area',
                          'year_built',
                          'last_reconstruction',
                          'floor',
                          'rooms',
                          'environment',
                          'quality_of_living',
                          'safety',
                          'transport',
                          'services',
                          'relax',
                          'index']
        scaled_df = scale_minmax(df, columns_minmax)

        # binarize labels
        columns_binarize = ['name_nsi', 'district', 'construction_type']

        binarized_df = binarize_labels(scaled_df, columns_binarize)

        # Map
        condition_mapper = config.FEATURE_MAPPER['condition']
        map_values(binarized_df, 'condition', condition_mapper)

        certificates_mapper = config.FEATURE_MAPPER['certificate']
        map_values(binarized_df, 'certificate', certificates_mapper)

        logging.info("Export preprocessed data")
        # binarized_df.to_parquet(config.PREPROCESSED_DATA_PATH + '.parquet', index=False)
        binarized_df.to_csv(
            config.PREPROCESSED_DATA_PATH + '.csv', index=False)

        return df
