# coding: utf-8

import logging
import pandas as pd

from visualization import visualize_cleaning, create_pairplot, create_heatmap, plot_tree
from utils import configurize_logger, visualize_model, define_target, log_df_shape, measure_model
from data import clean_data, prep_data, reduce_dimensions
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
    cleaned_df = clean_data(df, config.SPLIT_OPTION)

    cleaned_df = pd.read_csv(
        config.SPLITTED_DATA_PATH)

    # To-do refine pairplot
    # create_pairplot(cleaned_df)
    # To-do refine heatmap
    # TypeError: Image data of dtype object cannot be converted to float
    # create_heatmap(cleaned_df)

    visualize_cleaning(cleaned_df, "after cleaning", show_plots)

    # preprocessing

    preprocessed_df = prep_data(cleaned_df)

    preprocessed_df = pd.read_csv(config.PREPROCESSED_DATA_PATH + '.csv')

    visualize_cleaning(
        preprocessed_df, "after preprocessing", show_plots)

    # define target
    logging.info('Start training')

    y, X = define_target(preprocessed_df, config.TARGET)

    # reduce dimensions

    reduce_dimensions(X, config.REDUCE_DIMENSIONS)

    # train
    train_model(X, y, config.PIPELINE, config.PARAMETERS, config.TEST_SIZE)

    visualize_model()

    measure_model()

    logging.info('Script succesfully ended')


if __name__ == "__main__":
    main()
