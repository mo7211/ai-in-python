# coding: utf-8

import logging
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import add_dummy_feature


from visualization import visualize_cleaning
from utils import configurize_logger, save_fig, visualize_sdg_regressor
from data import clean_data, prep_data
from models import *
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

    config.TARGET = 'price'

    # preprocessed_df = pd.read_parquet(
    #     config.PREPROCESSED_DATA_PATH + '.parquet')
    preprocessed_df = pd.read_csv(config.PREPROCESSED_DATA_PATH + '.csv')

    visualize_cleaning(
        preprocessed_df, "after preprocessing", show_plots)

    # target data
    train_regression(preprocessed_df)

    logging.info('Script succesfully ended')


if __name__ == "__main__":
    main()
