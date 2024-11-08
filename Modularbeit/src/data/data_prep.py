import logging

import config
from utils import *


def prep_data(df: DataFrame):
    if config.PREPROCESS:
        logging.info(50*"=")
        logging.info("Start data preprocessing")

        # binarize labels
        binarize_labels(df, "name_nsi")
        binarize_labels(df, "district")
        binarize_labels(df, 'construction_type')

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
        condition_mapper = {'Development project': 1,
                            'Under construction': 2,
                            'Original condition': 3,
                            'Partial reconstruction': 4,
                            'Complete reconstruction': 5
                            }

        map_values(df, 'condition', condition_mapper)

        certificates_mapper = {'Unknown': 1,
                               'none': 2,
                               'G': 3,
                               'F': 4,
                               'E': 5,
                               'D': 6,
                               'C': 7,
                               'B': 8,
                               'A': 9
                               }

        map_values(df, 'certificate', certificates_mapper)

        split_option = config.SPLIT_OPTION

        logging.info("Export preprocessed data")
        df.to_csv('Modularbeit/data/features/re_preprosessed_' +
                  split_option.value + '.csv', index=False)

        return df
