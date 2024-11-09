# coding: utf-8

import logging
import pandas as pd

from visualization import visualize_cleaning
from utils import configurize_logger
from data import clean_data, prep_data
import config


def main():
    configurize_logger(__name__)
    show_plots = config.SHOW_PLOTS


    logging.info('Start script')

    df = pd.read_csv(
        config.INPUT_DATA_PATH, sep=";")

    
    visualize_cleaning(df, "before cleaning", show_plots)
    clean_data(df, config.SPLIT_OPTION)

    cleaned_df = pd.read_csv(
        config.SPLITTED_DATA_PATH)
    visualize_cleaning(cleaned_df, "after cleaning", show_plots)

    prep_data(cleaned_df)

    preprocessed_df = pd.read_csv(
        config.PREPROCESSED_DATA_PATH)

    visualize_cleaning(
        preprocessed_df, "after preprocessing", show_plots)
    logging.info('Script succesfully ended')


if __name__ == "__main__":
    main()
