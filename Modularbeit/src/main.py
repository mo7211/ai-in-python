# coding: utf-8

import logging
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


from visualization import visualize_cleaning
from utils import configurize_logger, discretize_feature
from data import clean_data, prep_data
from models import *
import config
from config import ModellingMethods


def main():
    configurize_logger(config.MODEL_METHOD.name)
    show_plots = config.SHOW_PLOTS

    logging.info('Start script')

    # cleaning

    df = pd.read_csv(
        config.INPUT_DATA_PATH, sep=";")

    visualize_cleaning(df, "before cleaning", show_plots)
    clean_data(df, config.SPLIT_OPTION)

    cleaned_df = pd.read_csv(
        config.SPLITTED_DATA_PATH)
    visualize_cleaning(cleaned_df, "after cleaning", show_plots)

    prep_data(cleaned_df)

    # preprocessing

    preprocessed_df = pd.read_csv(config.PREPROCESSED_DATA_PATH + '.csv')

    visualize_cleaning(
        preprocessed_df, "after preprocessing", show_plots)

    # training
    logging.info('Start training')
    config.TARGET = 'price'

    y, X = define_target(preprocessed_df, config.TARGET)

    n_bins = 1000
    binned_y = discretize_feature(y, n_bins)

    train_regression(X, y)
    train_decision_tree(X, binned_y)

    logging.info('Script succesfully ended')


if __name__ == "__main__":
    main()
