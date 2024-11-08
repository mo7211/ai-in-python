# coding: utf-8

import logging
import pandas as pd

from visualization import visualize_cleaning
from utils import configurize_logger
from data import clean_data, SplitOption


def main():

    config = {
        'clean': True,
        'preprocess': False,
        'train': False,
        'train_method': 'xx',
        'split_option': SplitOption.WITH_INDEX,
        'show_plots': False,
        'input_data': 'Modularbeit/data/raw_data/Real Estate Dataset.csv',
    }

    df = pd.read_csv(
        config['input_data'], sep=";")

    configurize_logger(__name__)

    logging.info('Creating visualization before cleaning')
    visualize_cleaning(df, "before cleaning", config['show_plots'])

    logging.info('Clean data')

    cleaned_df = clean_data(df, config['split_option'])

    logging.info('Creating visualization after cleaning')
    visualize_cleaning(cleaned_df, "after cleaning", config['show_plots'])

    # preprocessed_df = prep_data(cleaned_df)

    # visualize_cleaning(preprocessed_df)

if __name__ == "__main__":
    main()
