# coding: utf-8

import logging
import pandas as pd

# visualize_cleaning, create_pairplot, create_heatmap, plot_tree, plot_pairplot
from visualization import *
from utils import configurize_logger, define_target, log_df_shape
from data import clean_data, prep_data, reduce_dimensions
from models import *
import config
from config import ModellingMethods


def main():
    config.TARGET = 'certificate'
    configurize_logger(config.MODEL_METHOD.name)
    show_plots = config.SHOW_PLOTS

    logging.info('Start script')

    # cleaning

    df = pd.read_csv(
        config.INPUT_DATA_PATH, sep=";")

    columns_float = ['area',
                     'environment',
                     'quality_of_living',
                     'safety',
                     'transport',
                     'services',
                     'index',
                     'relax']
    convert_column_to_type(df, columns_float, float)

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

    # plot_pairplot(preprocessed_df, config.TARGET)

    # define target
    logging.info('Start training')
    y, X = define_target(preprocessed_df, config.TARGET)

    # reduce dimensions
    reduce_dimensions(X, config.REDUCE_DIMENSIONS)

    # train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=42)

    model = train_model(X_train, y_train, config.PIPELINE, config.PARAMETERS)

    measure_model(model, X_test, y_test, X, y)

    logging.info('Script succesfully ended')


if __name__ == "__main__":
    main()
