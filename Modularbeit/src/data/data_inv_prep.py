import logging

import config
from utils import *


def inverse_prep_data(df: DataFrame):
    if config.PREPROCESS:
        logging.info(50*"=")
        logging.info("Start inverse preprocessed data")

        # binarize labels
        columns_binarize = ['name_nsi', 'district', 'construction_type']

        inverse_binarize_labels(df, columns_binarize)

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
        inverse_minmax(df, columns_minmax)

        # Map

        inverse_condition_mapper = inverse_mapper(
            config.FEATURE_MAPPER['condition'])
        map_values(df, 'condition', inverse_condition_mapper)

        inverse_certificates_mapper = inverse_mapper(
            config.FEATURE_MAPPER['certificate'])
        map_values(df, 'certificate', inverse_certificates_mapper)

        return df
