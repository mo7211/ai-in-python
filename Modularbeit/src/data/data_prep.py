import logging

import config
from utils import *


def prep_data(df: DataFrame):
    if config.PREPROCESS:
        logging.info(50*"=")
        logging.info("Start data preprocessing")

        # binarize labels
        columns_binarize = ['name_nsi', 'district', 'construction_type']

        binarize_labels(df, columns_binarize)

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
                          'relax']
        scale_minmax(df, columns_minmax)

        # Map

        condition_mapper = config.FEATURE_MAPPER['condition']
        map_values(df, 'condition', condition_mapper)


        certificates_mapper = config.FEATURE_MAPPER['certificate']
        map_values(df, 'certificate', certificates_mapper)

        split_option = config.SPLIT_OPTION

        logging.info("Export preprocessed data")
        df.to_csv(config.PREPROCESSED_DATA_PATH, index=False)
        
        df['price'].to_csv(config.TARGET_DATA_PATH, index=False)

        df.drop(columns=['price']).to_csv(config.FEATURE_DATA_PATH, index=False)

        return df
