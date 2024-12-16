# coding: utf-8

import logging
import pandas as pd

from visualization import visualize_dataframe
from utils import configurize_logger, define_target, read_data
from data import clean_data, prep_data, reduce_dimensions
from models import *

import config


def main():
    config.MODEL_METHOD = config.ModellingMethods.pca_keras_regressor
    configurize_logger(config.MODEL_METHOD.name)
    show_plots = config.SHOW_PLOTS

    logging.info('Start script')

    # cleaning

    df = read_data(config.INPUT_DATA_PATH, config.CLEAN)

    visualize_dataframe(df, "before cleaning", show_plots)

    cleaned_df = clean_data(df, config.SPLIT_OPTION, config.CLEAN)

    cleaned_df = pd.read_csv(
        config.SPLITTED_DATA_PATH)

    # plot_pairplot(cleaned_df, config.TARGET)

    visualize_dataframe(cleaned_df, "after cleaning", show_plots)

    # preprocessing

    preprocessed_df = prep_data(cleaned_df, config.PREPROCESS)

    visualize_dataframe(
        preprocessed_df, "after preprocessing", show_plots)

    preprocessed_df = pd.read_csv(config.PREPROCESSED_DATA_PATH + '.csv')

    # define target
    y, X = define_target(preprocessed_df, config.TARGET)

    # reduce dimensions
    reduce_dimensions(X, config.REDUCE_DIMENSIONS)

    # train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=42)

    config.DATA_SHAPE = X_train.shape

    model = train_model(X_train, y_train, config.PIPELINE, config.PARAMETERS)

    measure_model(model, X_test, y_test, X, y)

    logging.info('Script succesfully ended')


if __name__ == "__main__":
    main()
